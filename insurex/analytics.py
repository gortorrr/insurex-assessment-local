"""Pure analytics helpers shared by the dashboard and regression tests."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def response_filter_view(frame: pd.DataFrame, selected_labels: Iterable[int] | None = None) -> pd.DataFrame:
    """Return the detail view while keeping the supplied frame unmodified."""

    labels = list(selected_labels or [])
    if not labels:
        return frame.copy()
    return frame[frame["label"].isin(labels)].copy()


def acceptance_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    """Calculate acceptance against the valid-label population of this frame."""

    valid = frame["label"].isin({0, 1, 2})
    accepted = valid & frame["label"].isin({1, 2})
    denominator = int(valid.sum())
    return {
        "records": int(len(frame)),
        "valid_label_records": denominator,
        "accepted_records": int(accepted.sum()),
        "rate": float(accepted.sum() / denominator) if denominator else None,
    }


def acceptance_by_group(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Calculate group acceptance using all valid labels in the supplied frame."""

    valid = frame["label"].isin({0, 1, 2})
    work = frame.assign(
        _group=frame[column].astype("object").where(frame[column].notna(), "Unknown"),
        _valid=valid,
        _accepted=valid & frame["label"].isin({1, 2}),
    )
    grouped = work.groupby("_group", dropna=False, sort=False).agg(
        records=("label", "size"),
        valid_label_records=("_valid", "sum"),
        accepted=("_accepted", "sum"),
    )
    grouped["acceptance_rate"] = grouped["accepted"].where(
        grouped["valid_label_records"] > 0
    ) / grouped["valid_label_records"].where(grouped["valid_label_records"] > 0)
    return grouped.reset_index(names=column)
