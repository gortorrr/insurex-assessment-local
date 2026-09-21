import json
from pathlib import Path
import unittest

import pandas as pd

from scripts.profile_data import add_band_columns, income_band, pair_comparison


ROOT = Path(__file__).resolve().parents[2]


class ProfileOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads((ROOT / "analysis/reports/data_profile.json").read_text(encoding="utf-8"))
        cls.aggregates = json.loads(
            (ROOT / "analysis/data/derived/analysis_aggregates.json").read_text(encoding="utf-8")
        )

    def test_dictionary_covers_all_columns(self):
        coverage = self.profile["dictionary_coverage"]
        self.assertTrue(coverage["all_csv_columns_documented"])
        self.assertTrue(coverage["all_dictionary_fields_present"])
        self.assertEqual(self.profile["column_count"], 26)

    def test_baseline_counts_and_denominator_reconcile(self):
        self.assertEqual(self.profile["row_count"], 215_993)
        overall = self.aggregates["overall"]
        self.assertEqual(overall["valid_label_records"], 215_993)
        self.assertEqual(
            overall["reject_count"] + overall["pa_count"] + overall["life_count"],
            overall["records"],
        )
        self.assertEqual(overall["accepted_records"], overall["pa_count"] + overall["life_count"])

    def test_quality_findings_are_retained(self):
        self.assertEqual(self.profile["exact_duplicate_excess"], 2_173)
        self.assertEqual(self.profile["rows_with_any_missing"], 5_535)
        self.assertEqual(self.aggregates["quality_checks"]["negative_dcspend"], 17)
        self.assertEqual(
            self.aggregates["quality_checks"]["net_flow_mismatch_counts"]["net_flow_30d"],
            0,
        )

    def test_income_bands_cover_negative_zero_boundaries_and_missing(self):
        cases = {
            -1: "Negative (<0)",
            0: "Zero (=0)",
            10_000: "Positive (0, 15,000]",
            15_000: "Positive (0, 15,000]",
            15_000.01: "Positive (15,000, 30,000]",
            30_000: "Positive (15,000, 30,000]",
            50_000: "Positive (30,000, 50,000]",
            100_000: "Positive (50,000, 100,000]",
            100_000.01: "Positive (>100,000)",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(income_band(value), expected)
        self.assertEqual(income_band(float("nan")), "Missing")

    def test_income_band_dataframe_has_no_unassigned_values(self):
        frame = pd.DataFrame({"age": [20] * 9, "income": [-1, 0, 10_000, 15_000, 15_000.01, 30_000, 50_000, 100_000, None]})
        banded = add_band_columns(frame)
        self.assertEqual(len(banded), banded["income_band"].notna().sum())

    def test_pair_metrics_separate_non_null_equality_from_shared_missing(self):
        frame = pd.DataFrame({"left": [1.0, 2.0, None, None], "right": [1.0, 3.0, None, None]})
        self.assertEqual(
            pair_comparison(frame, "left", "right"),
            {"equal_non_null": 1, "shared_missing": 2, "equal_including_shared_missing": 3, "non_null_mismatch": 1},
        )


if __name__ == "__main__":
    unittest.main()
