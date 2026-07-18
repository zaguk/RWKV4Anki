from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class LiveRetentionCategory(Enum):
    SAME_DAY = "same_day"
    YOUNG = "young"
    MATURE = "mature"


@dataclass(frozen=True)
class LiveRetentionRecord:
    review_id: int
    card_id: int
    category: LiveRetentionCategory
    predicted_retrievability: float
    remembered: bool
    fsrs_predicted_retrievability: float | None = None
    rating: int | None = None
    elapsed_days: float | None = None
    source_deck_id: int | None = None
    desired_retention: float | None = None
    active_desired_retention: float | None = None
    rwkv_stability_days: float | None = None
    fsrs_difficulty: float | None = None
    answered_at_ms: int | None = None


@dataclass(frozen=True)
class LiveRetentionSummaryRow:
    label: str
    review_count: int
    predicted_retention: float | None
    actual_retention: float | None
    remembered_count: int
    fsrs_predicted_retention: float | None = None
    fsrs_actual_retention: float | None = None
    fsrs_available_count: int = 0


@dataclass(frozen=True)
class LiveRetentionSummary:
    records: tuple[LiveRetentionRecord, ...]
    rows: tuple[LiveRetentionSummaryRow, ...]
    skipped_count: int = 0

    @property
    def review_count(self) -> int:
        return len(self.records)


def retention_record_for_answer(
    candidate,
    review_row: Mapping[str, Any] | None,
    *,
    fsrs_prediction: float | None = None,
) -> LiveRetentionRecord | None:
    if candidate is None or review_row is None:
        return None
    try:
        review_id = int(review_row["review_id"])
        card_id = int(review_row["card_id"])
        rating = int(review_row["rating"])
        elapsed_days = float(review_row["elapsed_days"])
        prediction = float(candidate.predicted_retrievability)
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(prediction):
        return None
    fsrs_prediction_value = _finite_optional_float(fsrs_prediction)
    category = retention_category_for_elapsed_days(elapsed_days)
    if category is None:
        return None
    return LiveRetentionRecord(
        review_id=review_id,
        card_id=card_id,
        category=category,
        predicted_retrievability=prediction,
        remembered=rating > 1,
        fsrs_predicted_retrievability=fsrs_prediction_value,
        rating=rating,
        elapsed_days=elapsed_days,
        source_deck_id=_optional_int(getattr(candidate, "source_deck_id", None)),
        desired_retention=_candidate_float(candidate, "desired_retention"),
        active_desired_retention=_candidate_float(candidate, "active_desired_retention"),
        rwkv_stability_days=_candidate_float(candidate, "rwkv_stability_days"),
        fsrs_difficulty=_candidate_float(candidate, "fsrs_difficulty"),
        answered_at_ms=review_id,
    )


def retention_category_for_elapsed_days(
    elapsed_days: float,
) -> LiveRetentionCategory | None:
    if not math.isfinite(elapsed_days) or elapsed_days < 0:
        return None
    if elapsed_days < 1:
        return LiveRetentionCategory.SAME_DAY
    if elapsed_days < 21:
        return LiveRetentionCategory.YOUNG
    return LiveRetentionCategory.MATURE


def summarize_live_retention_records(
    records: Iterable[LiveRetentionRecord],
    *,
    skipped_count: int = 0,
) -> LiveRetentionSummary:
    record_tuple = tuple(records)
    rows = (
        _summary_row("Same Day", record_tuple, {LiveRetentionCategory.SAME_DAY}),
        _summary_row("Young", record_tuple, {LiveRetentionCategory.YOUNG}),
        _summary_row("Mature", record_tuple, {LiveRetentionCategory.MATURE}),
        _summary_row(
            "Total",
            record_tuple,
            {LiveRetentionCategory.MATURE, LiveRetentionCategory.YOUNG},
        ),
        _summary_row(
            "Total + Same Day",
            record_tuple,
            {
                LiveRetentionCategory.MATURE,
                LiveRetentionCategory.YOUNG,
                LiveRetentionCategory.SAME_DAY,
            },
        ),
    )
    return LiveRetentionSummary(
        records=record_tuple,
        rows=rows,
        skipped_count=max(0, int(skipped_count)),
    )


def format_live_retention_percent(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def format_live_retention_delta(
    actual_retention: float | None,
    predicted_retention: float | None,
) -> str:
    if actual_retention is None or predicted_retention is None:
        return "N/A"
    delta = float(actual_retention) - float(predicted_retention)
    return f"{delta * 100:+.1f} pp"


def _summary_row(
    label: str,
    records: tuple[LiveRetentionRecord, ...],
    categories: set[LiveRetentionCategory],
) -> LiveRetentionSummaryRow:
    selected = [record for record in records if record.category in categories]
    if not selected:
        return LiveRetentionSummaryRow(
            label=label,
            review_count=0,
            predicted_retention=None,
            actual_retention=None,
            remembered_count=0,
        )
    remembered_count = sum(1 for record in selected if record.remembered)
    fsrs_selected = [
        record
        for record in selected
        if record.fsrs_predicted_retrievability is not None
    ]
    return LiveRetentionSummaryRow(
        label=label,
        review_count=len(selected),
        predicted_retention=sum(
            record.predicted_retrievability for record in selected
        )
        / len(selected),
        actual_retention=remembered_count / len(selected),
        remembered_count=remembered_count,
        fsrs_predicted_retention=_average_fsrs_prediction(fsrs_selected),
        fsrs_actual_retention=_average_actual_retention(fsrs_selected),
        fsrs_available_count=len(fsrs_selected),
    )


def _finite_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _candidate_float(candidate, name: str) -> float | None:
    try:
        return _finite_optional_float(getattr(candidate, name))
    except Exception:
        return None


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _average_fsrs_prediction(records: Iterable[LiveRetentionRecord]) -> float | None:
    values = [
        float(record.fsrs_predicted_retrievability)
        for record in records
        if record.fsrs_predicted_retrievability is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _average_actual_retention(records: Iterable[LiveRetentionRecord]) -> float | None:
    record_tuple = tuple(records)
    if not record_tuple:
        return None
    return sum(1 for record in record_tuple if record.remembered) / len(record_tuple)
