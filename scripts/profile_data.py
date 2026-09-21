"""Profile the supplied InsureX CSV and Excel data dictionary.

This script intentionally keeps the raw files unchanged. It creates auditable JSON
and Markdown outputs under analysis/reports/ and analysis/data/derived/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pandas as pd


MONTH_ORDER = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

EXPECTED_LABELS = {0, 1, 2}
NUMERIC_DECLARATIONS = {"integer", "double"}


def json_value(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-safe values."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


def json_default(value: Any) -> Any:
    converted = json_value(value)
    if converted is value:
        return str(value)
    return converted


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_dictionary(path: Path) -> dict[str, Any]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=False)
    sheets: dict[str, Any] = {}
    definitions: list[dict[str, Any]] = []

    for worksheet in workbook.worksheets:
        rows = []
        for row in worksheet.iter_rows(values_only=True):
            values = [json_value(value) for value in row]
            rows.append(values)
            if len(values) >= 4 and values[1] not in (None, "Field"):
                definitions.append(
                    {
                        "field": values[1],
                        "declared_type": values[2],
                        "definition": values[3],
                        "sheet": worksheet.title,
                    }
                )
        sheets[worksheet.title] = {
            "max_row": worksheet.max_row,
            "max_column": worksheet.max_column,
            "rows": rows,
        }

    return {
        "sheet_names": workbook.sheetnames,
        "sheets": sheets,
        "definitions": definitions,
    }


def values_as_records(series: pd.Series, limit: int = 10) -> list[dict[str, Any]]:
    counts = series.value_counts(dropna=False).head(limit)
    records = []
    for value, count in counts.items():
        records.append(
            {
                "value": "<MISSING>" if pd.isna(value) else json_value(value),
                "count": int(count),
            }
        )
    return records


def profile_column(series: pd.Series, declared_type: str | None) -> dict[str, Any]:
    non_null = series.dropna()
    profile: dict[str, Any] = {
        "observed_dtype": str(series.dtype),
        "declared_type": declared_type,
        "rows": int(len(series)),
        "non_null": int(series.notna().sum()),
        "null_count": int(series.isna().sum()),
        "null_rate": float(series.isna().mean()),
        "distinct_non_null": int(series.nunique(dropna=True)),
        "cardinality_ratio": float(series.nunique(dropna=True) / len(series)),
        "top_values": values_as_records(series),
    }

    if pd.api.types.is_numeric_dtype(series):
        quantiles = series.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        profile["numeric"] = {
            "min": json_value(series.min()),
            "max": json_value(series.max()),
            "mean": json_value(series.mean()),
            "median": json_value(series.median()),
            "std": json_value(series.std()),
            "quantiles": {str(k): json_value(v) for k, v in quantiles.items()},
            "zero_count": int((series == 0).sum()),
            "negative_count": int((series < 0).sum()),
        }
    else:
        string_values = non_null.astype(str)
        profile["text"] = {
            "empty_string_count": int((string_values == "").sum()),
            "leading_or_trailing_whitespace_count": int(
                (string_values != string_values.str.strip()).sum()
            ),
            "min_length": int(string_values.str.len().min()) if len(string_values) else 0,
            "max_length": int(string_values.str.len().max()) if len(string_values) else 0,
            "mean_length": float(string_values.str.len().mean()) if len(string_values) else 0.0,
        }
    return profile


def group_metrics(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    labels = frame["label"]
    valid = labels.isin(EXPECTED_LABELS)
    grouping = frame[column].astype("object").where(frame[column].notna(), "Unknown")
    work = frame.assign(_group=grouping, _valid=valid, _accepted=labels.isin({1, 2}))
    grouped = work.groupby("_group", dropna=False, sort=False)
    rows = []
    for value, group in grouped:
        valid_count = int(group["_valid"].sum())
        accepted_count = int((group["_valid"] & group["_accepted"]).sum())
        rows.append(
            {
                "group": json_value(value),
                "records": int(len(group)),
                "valid_label_records": valid_count,
                "accepted_records": accepted_count,
                "acceptance_rate": (
                    float(accepted_count / valid_count) if valid_count else None
                ),
                "pa_count": int((group["label"] == 1).sum()),
                "life_count": int((group["label"] == 2).sum()),
            }
        )
    return rows


def add_band_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["age_band"] = pd.cut(
        result["age"],
        bins=[-np.inf, 29, 39, 49, 59, np.inf],
        labels=["<30", "30-39", "40-49", "50-59", "60+"],
        right=True,
    )
    result["income_band"] = result["income"].map(income_band).astype("category")
    return result


def income_band(value: Any) -> str:
    """Return a disjoint, exhaustive income band with explicit boundaries.

    Currency and time period are not inferred here; this is only a numeric
    feature band. Missing values remain explicitly visible as ``Missing``.
    """

    if pd.isna(value):
        return "Missing"
    numeric = float(value)
    if numeric < 0:
        return "Negative (<0)"
    if numeric == 0:
        return "Zero (=0)"
    if numeric <= 15000:
        return "Positive (0, 15,000]"
    if numeric <= 30000:
        return "Positive (15,000, 30,000]"
    if numeric <= 50000:
        return "Positive (30,000, 50,000]"
    if numeric <= 100000:
        return "Positive (50,000, 100,000]"
    return "Positive (>100,000)"


def pair_comparison(frame: pd.DataFrame, left: str, right: str) -> dict[str, int]:
    """Separate equal observed values from shared missing values."""

    both_present = frame[left].notna() & frame[right].notna()
    shared_missing = frame[left].isna() & frame[right].isna()
    equal_non_null = int((both_present & frame[left].eq(frame[right])).sum())
    shared_missing_count = int(shared_missing.sum())
    return {
        "equal_non_null": equal_non_null,
        "shared_missing": shared_missing_count,
        "equal_including_shared_missing": equal_non_null + shared_missing_count,
        "non_null_mismatch": int((both_present & ~frame[left].eq(frame[right])).sum()),
    }


def metric_snapshot(frame: pd.DataFrame) -> dict[str, Any]:
    valid = frame["label"].isin(EXPECTED_LABELS)
    accepted = valid & frame["label"].isin({1, 2})
    valid_count = int(valid.sum())
    accepted_count = int(accepted.sum())
    return {
        "records": int(len(frame)),
        "valid_label_records": valid_count,
        "accepted_records": accepted_count,
        "offer_acceptance_rate": (
            float(accepted_count / valid_count) if valid_count else None
        ),
        "pa_count": int((frame["label"] == 1).sum()),
        "life_count": int((frame["label"] == 2).sum()),
        "reject_count": int((frame["label"] == 0).sum()),
    }


def build_quality_flags(frame: pd.DataFrame) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []

    def add(severity: str, field: str, count: int, detail: str) -> None:
        flags.append(
            {"severity": severity, "field": field, "count": int(count), "detail": detail}
        )

    for field, count in frame.isna().sum().items():
        if count:
            rate = count / len(frame)
            severity = "alert" if rate > 0.20 else "warn" if rate > 0.05 else "info"
            add(severity, field, int(count), f"missing_rate={rate:.4%}")

    negative_spend = int((frame["dcspend_last_30d"] < 0).sum())
    if negative_spend:
        add("warn", "dcspend_last_30d", negative_spend, "negative financial feature values")

    for left, right in [
        ("inflow30d", "inflow1_15"),
        ("outflow30d", "outflow1_15"),
        ("net_flow_30d", "net_flow_15d"),
    ]:
        comparison = pair_comparison(frame, left, right)
        add(
            "review",
            f"{left}={right}",
            comparison["equal_non_null"],
            "equal among non-missing values; "
            f"shared_missing={comparison['shared_missing']}; "
            f"including_shared_missing={comparison['equal_including_shared_missing']}",
        )

    for net, inflow, outflow in [
        ("net_flow_30d", "inflow30d", "outflow30d"),
        ("net_flow_15d", "inflow1_15", "outflow1_15"),
    ]:
        calculated = frame[inflow] - frame[outflow]
        comparable = frame[net].notna() & calculated.notna()
        mismatch = int(
            (~np.isclose(frame.loc[comparable, net], calculated[comparable], atol=0.01, rtol=0)).sum()
        )
        add("info" if mismatch == 0 else "warn", net, mismatch, f"mismatch versus {inflow} - {outflow}")

    add("info", "exact_duplicates", int(frame.duplicated(keep="first").sum()), "excess rows beyond first exact copy")
    invalid_labels = int((~frame["label"].isin(EXPECTED_LABELS)).sum())
    add("info" if invalid_labels == 0 else "alert", "label", invalid_labels, "outside expected set {0,1,2}")
    return flags


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def build_report(profile: dict[str, Any], aggregates: dict[str, Any]) -> str:
    overall = aggregates["overall"]
    flags = profile["quality_flags"]
    warning_flags = [flag for flag in flags if flag["severity"] in {"warn", "alert"}]
    dedup = aggregates["sensitivity"]["exact_deduplicated"]
    segment_rows = aggregates["groups"]["customer_segment"]
    occupation_rows = aggregates["groups"]["main_occupation"]
    month_rows = aggregates["groups"]["campaign_month"]

    def best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        usable = [row for row in rows if row["acceptance_rate"] is not None]
        return max(usable, key=lambda row: row["acceptance_rate"]) if usable else None

    best_segment = best(segment_rows)
    best_occupation = best(occupation_rows)
    best_month = best(month_rows)

    lines = [
        "# InsureX supplied-data analysis report",
        "",
        "Status: generated from the supplied CSV and Excel dictionary; no raw input was modified.",
        "",
        "## Scope and grain",
        "",
        "The CSV is analysed at one supplied campaign-record row per record. It has no customer, policy, agent, or payment transaction identifier. The Excel file is used as a data dictionary. The rates below describe campaign response labels, not completed sales, issued policies, premium, payment, or revenue.",
        "",
        "## Baseline summary",
        "",
        f"- Records: {overall['records']:,}; fields: {profile['column_count']}",
        f"- Valid-label records: {overall['valid_label_records']:,}; accepted labels (1 or 2): {overall['accepted_records']:,}; offer acceptance rate: {overall['offer_acceptance_rate']:.4%}",
        f"- Reject: {overall['reject_count']:,}; PA response: {overall['pa_count']:,}; Life response: {overall['life_count']:,}",
        f"- Rows with at least one missing field: {profile['rows_with_any_missing']:,}",
        f"- Exact duplicate excess rows: {profile['exact_duplicate_excess']:,}",
        "",
        "## Findings supported by the supplied data",
        "",
    ]
    if best_segment:
        lines.append(
            f"1. Among customer segments, `{best_segment['group']}` has the highest observed offer acceptance rate ({best_segment['acceptance_rate']:.4%}, {best_segment['accepted_records']:,}/{best_segment['valid_label_records']:,} valid-label records). This is an association in campaign responses, not a causal effect."
        )
    if best_occupation:
        lines.append(
            f"2. Among occupations, `{best_occupation['group']}` has the highest observed offer acceptance rate ({best_occupation['acceptance_rate']:.4%}, {best_occupation['accepted_records']:,}/{best_occupation['valid_label_records']:,}). Small groups should be checked before targeting decisions."
        )
    if best_month:
        lines.append(
            f"3. The highest campaign-month response rate in this month-labelled dataset is `{best_month['group']}` ({best_month['acceptance_rate']:.4%}, {best_month['accepted_records']:,}/{best_month['valid_label_records']:,}). The source has month names without a year, so this is not a multi-year trend."
        )
    lines.extend(
        [
            f"4. Among non-missing values, the three 15-day/30-day pairs are equal for {aggregates['quality_checks']['pair_equal_counts']['inflow30d=inflow1_15']:,}, {aggregates['quality_checks']['pair_equal_counts']['outflow30d=outflow1_15']:,}, and {aggregates['quality_checks']['pair_equal_counts']['net_flow_30d=net_flow_15d']:,} rows respectively. Each pair also has {aggregates['quality_checks']['pair_shared_missing_counts']['inflow30d=inflow1_15']:,}, {aggregates['quality_checks']['pair_shared_missing_counts']['outflow30d=outflow1_15']:,}, and {aggregates['quality_checks']['pair_shared_missing_counts']['net_flow_30d=net_flow_15d']:,} shared-missing rows; including shared missingness, each total is {aggregates['quality_checks']['pair_equal_including_shared_missing_counts']['inflow30d=inflow1_15']:,}. The cause is unknown and the fields are retained separately.",
            f"5. `dcspend_last_30d` contains {aggregates['quality_checks']['negative_dcspend']:,} negative values. This is flagged for source/business review; it is not silently converted to zero or removed.",
        ]
    )
    lines.extend(
        [
            "",
            "## Data quality and sensitivity",
            "",
            f"- Exact-deduplicated sensitivity view has {dedup['records']:,} records and an offer acceptance rate of {dedup['offer_acceptance_rate']:.4%}. This view is comparison-only; the baseline keeps all source rows.",
            f"- Missingness over the 5% warning threshold: {sum(1 for flag in warning_flags if flag['field'] not in {'dcspend_last_30d'})} field-level warnings/alerts; missingness is still retained in the full profile.",
            "- Numeric missing values are excluded from numeric summaries; categorical missing values are shown as Unknown only in grouped presentation outputs.",
            "- No source field is renamed to premium or payment. Real agent KPI remains limited by the supplied inputs.",
            "",
            "## Required follow-up",
            "",
            "1. Confirm the semantic cause of the identical 15-day and 30-day fields and the negative debit-card spending values with the data owner.",
            "2. Obtain policy-level and agent-level transactions if real premium, policy count, or contract KPI reporting is required.",
            "3. Add a year or campaign date before interpreting month changes as a time trend.",
            "",
            "## Reproducibility",
            "",
            "Run `python -X utf8 scripts/profile_data.py` from the project root. Output files include `analysis/reports/data_profile.json`, `analysis/data/derived/analysis_aggregates.json`, and this report.",
            "",
        ]
    )
    return "\n".join(lines)


def profile_inputs(csv_path: Path, dictionary_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    dictionary = read_dictionary(dictionary_path)
    definitions = {item["field"]: item for item in dictionary["definitions"]}
    expected_columns = list(definitions)
    actual_columns = list(frame.columns)
    missing_from_dictionary = [column for column in actual_columns if column not in definitions]
    missing_from_csv = [column for column in expected_columns if column not in frame.columns]

    profile_columns = {
        column: profile_column(frame[column], definitions.get(column, {}).get("declared_type"))
        for column in actual_columns
    }
    exact_duplicate_excess = int(frame.duplicated(keep="first").sum())
    duplicate_groups = int(frame.duplicated(keep=False).sum())
    rows_with_any_missing = int(frame.isna().any(axis=1).sum())
    missing_patterns = (
        frame.isna()
        .astype(int)
        .astype(str)
        .agg("".join, axis=1)
        .value_counts()
        .head(20)
        .to_dict()
    )

    overall = metric_snapshot(frame)
    accepted_product_share = {
        "pa_share_of_accepted": float(overall["pa_count"] / overall["accepted_records"])
        if overall["accepted_records"]
        else None,
        "life_share_of_accepted": float(overall["life_count"] / overall["accepted_records"])
        if overall["accepted_records"]
        else None,
    }

    banded = add_band_columns(frame)
    aggregates = {
        "overall": {**overall, **accepted_product_share},
        "groups": {
            "campaign_month": group_metrics(frame, "campaign_month"),
            "customer_segment": group_metrics(frame, "customer_segment"),
            "main_occupation": group_metrics(frame, "main_occupation"),
            "gender": group_metrics(frame, "gender"),
            "age_band": group_metrics(banded, "age_band"),
            "income_band": group_metrics(banded, "income_band"),
        },
        "financial_summary": {},
        "quality_checks": {
            "negative_dcspend": int((frame["dcspend_last_30d"] < 0).sum()),
            "pair_equal_counts": {},
            "pair_shared_missing_counts": {},
            "pair_equal_including_shared_missing_counts": {},
            "pair_non_null_mismatch_counts": {},
            "net_flow_mismatch_counts": {},
        },
    }
    for left, right in [
        ("inflow30d", "inflow1_15"),
        ("outflow30d", "outflow1_15"),
        ("net_flow_30d", "net_flow_15d"),
    ]:
        comparison = pair_comparison(frame, left, right)
        pair_name = f"{left}={right}"
        aggregates["quality_checks"]["pair_equal_counts"][pair_name] = comparison["equal_non_null"]
        aggregates["quality_checks"]["pair_shared_missing_counts"][pair_name] = comparison["shared_missing"]
        aggregates["quality_checks"]["pair_equal_including_shared_missing_counts"][pair_name] = comparison["equal_including_shared_missing"]
        aggregates["quality_checks"]["pair_non_null_mismatch_counts"][pair_name] = comparison["non_null_mismatch"]
    for field in [
        "income",
        "maxosdc_last_30d",
        "dcspend_last_30d",
        "easypymt_last_30d",
        "savacc_bal",
        "currentacc_bal",
        "avg_savaccbal_30d",
        "avg_currentaccbal_30d",
        "inflow30d",
        "outflow30d",
        "net_flow_30d",
    ]:
        values = frame[field].dropna()
        aggregates["financial_summary"][field] = {
            "count": int(values.size),
            "median": json_value(values.median()),
            "q1": json_value(values.quantile(0.25)),
            "q3": json_value(values.quantile(0.75)),
            "min": json_value(values.min()),
            "max": json_value(values.max()),
            "zero_count": int((values == 0).sum()),
            "negative_count": int((values < 0).sum()),
        }
    for net, inflow, outflow in [
        ("net_flow_30d", "inflow30d", "outflow30d"),
        ("net_flow_15d", "inflow1_15", "outflow1_15"),
    ]:
        calc = frame[inflow] - frame[outflow]
        comparable = frame[net].notna() & calc.notna()
        aggregates["quality_checks"]["net_flow_mismatch_counts"][net] = int(
            (~np.isclose(frame.loc[comparable, net], calc[comparable], atol=0.01, rtol=0)).sum()
        )

    dedup = frame.drop_duplicates(keep="first")
    aggregates["sensitivity"] = {
        "exact_deduplicated": metric_snapshot(dedup),
        "removed_excess_rows": int(len(frame) - len(dedup)),
    }

    profile = {
        "generated_at": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
        "source_files": {
            "csv": {"path": str(csv_path), "bytes": csv_path.stat().st_size, "sha256": sha256(csv_path)},
            "dictionary": {"path": str(dictionary_path), "bytes": dictionary_path.stat().st_size, "sha256": sha256(dictionary_path)},
        },
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": actual_columns,
        "dictionary_sheet_names": dictionary["sheet_names"],
        "dictionary_definition_count": len(dictionary["definitions"]),
        "dictionary_coverage": {
            "csv_columns_missing_from_dictionary": missing_from_dictionary,
            "dictionary_fields_missing_from_csv": missing_from_csv,
            "all_csv_columns_documented": not missing_from_dictionary,
            "all_dictionary_fields_present": not missing_from_csv,
        },
        "grain": "one supplied campaign dataset record per row; no customer/policy/transaction identifier",
        "label_mapping": {
            "0": "Reject Offer",
            "1": "Accept Offer in PA Insurance Product",
            "2": "Accept Offer in Life Insurance Product",
        },
        "rows_with_any_missing": rows_with_any_missing,
        "missing_patterns_top20": [
            {"pattern": pattern, "rows": int(count)}
            for pattern, count in missing_patterns.items()
        ],
        "exact_duplicate_excess": exact_duplicate_excess,
        "duplicate_rows_in_groups": duplicate_groups,
        "columns_profile": profile_columns,
        "quality_flags": build_quality_flags(frame),
        "observed_business_limits": [
            "No policy-level, payment-transaction, premium, agent, or contract data is present.",
            "Month values have no year and cannot support a cross-year trend.",
            "Financial fields are customer features and are not premium or payment amounts.",
        ],
    }
    return profile, aggregates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data_dir = Path("Dataset Test Case") if Path("Dataset Test Case").exists() else Path("analysis/data/raw")
    parser.add_argument("--csv", type=Path, default=default_data_dir / "dsc_test_case.csv")
    parser.add_argument("--dictionary", type=Path, default=default_data_dir / "data definition.xlsx")
    parser.add_argument("--profile-out", type=Path, default=Path("analysis/reports/data_profile.json"))
    parser.add_argument("--aggregates-out", type=Path, default=Path("analysis/data/derived/analysis_aggregates.json"))
    parser.add_argument("--report-out", type=Path, default=Path("analysis/reports/analysis_report.md"))
    args = parser.parse_args()

    profile, aggregates = profile_inputs(args.csv, args.dictionary)
    write_json(args.profile_out, profile)
    write_json(args.aggregates_out, aggregates)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(build_report(profile, aggregates), encoding="utf-8")
    print(json.dumps({
        "rows": profile["row_count"],
        "columns": profile["column_count"],
        "dictionary_coverage": profile["dictionary_coverage"],
        "exact_duplicate_excess": profile["exact_duplicate_excess"],
        "offer_acceptance_rate": aggregates["overall"]["offer_acceptance_rate"],
        "outputs": [str(args.profile_out), str(args.aggregates_out), str(args.report_out)],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
