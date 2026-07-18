from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CalibrationReview:
    review_id: int
    card_id: int
    prediction: float
    actual: int


@dataclass(frozen=True)
class ReviewIntervalFilter:
    operator: str
    days: float

    def __post_init__(self) -> None:
        if self.operator not in {"<", "<=", "=", ">=", ">"}:
            raise ValueError("Review interval operator must be one of <, <=, =, >=, >.")
        days = float(self.days)
        if not math.isfinite(days) or days < 0:
            raise ValueError("Review interval must be a non-negative number of days.")
        object.__setattr__(self, "days", days)


@dataclass(frozen=True)
class CalibrationBin:
    index: int
    low: float
    high: float
    predicted_sum: float
    actual_sum: float
    count: int

    @property
    def predicted_average(self) -> float | None:
        return None if self.count <= 0 else self.predicted_sum / self.count

    @property
    def actual_average(self) -> float | None:
        return None if self.count <= 0 else self.actual_sum / self.count


@dataclass(frozen=True)
class CalibrationSummary:
    bins: tuple[CalibrationBin, ...]
    count: int
    skipped_count: int
    missing_prediction_count: int
    total_count: int
    average_prediction: float | None
    actual_recall: float | None

    @property
    def invalid_count(self) -> int:
        return max(0, self.skipped_count - self.missing_prediction_count)


def calibration_reviews(
    prediction_rows: Iterable[dict[str, Any]],
    *,
    card_ids: Iterable[int] | None = None,
    include_sameday: bool = False,
) -> list[CalibrationReview]:
    allowed_cards = None if card_ids is None else {int(card_id) for card_id in card_ids}
    reviews: list[CalibrationReview] = []
    for row in prediction_rows:
        card_id = _optional_int(row.get("card_id"))
        if card_id is None:
            continue
        if allowed_cards is not None and card_id not in allowed_cards:
            continue
        elapsed_days = _optional_float(row.get("elapsed_days", 0))
        if elapsed_days is None:
            continue
        if not _include_elapsed_days(elapsed_days, include_sameday=include_sameday):
            continue
        review = calibration_review_from_row(row)
        if review is not None:
            reviews.append(review)
    return reviews


def calibration_review_from_row(row: dict[str, Any]) -> CalibrationReview | None:
    card_id = _optional_int(row.get("card_id"))
    if card_id is None:
        return None
    review_id = _optional_int(row.get("review_id"))
    if review_id is None:
        return None

    try:
        prediction = float(row["prediction"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(prediction) or prediction < 0.0 or prediction > 1.0:
        return None

    try:
        rating = int(row["rating"])
    except (KeyError, TypeError, ValueError):
        return None
    if rating not in {1, 2, 3, 4}:
        return None

    return CalibrationReview(
        review_id=review_id,
        card_id=card_id,
        prediction=prediction,
        actual=0 if rating == 1 else 1,
    )


def filter_calibration_reviews(
    reviews: Iterable[CalibrationReview],
    card_ids: Iterable[int],
) -> list[CalibrationReview]:
    allowed_cards = {int(card_id) for card_id in card_ids}
    return [review for review in reviews if review.card_id in allowed_cards]


def filter_rows_by_review_interval(
    rows: Iterable[dict[str, Any]],
    review_interval: ReviewIntervalFilter | None,
) -> list[dict[str, Any]]:
    if review_interval is None:
        return list(rows)
    return [
        row
        for row in rows
        if row_matches_review_interval(row, review_interval)
    ]


def row_matches_review_interval(
    row: dict[str, Any],
    review_interval: ReviewIntervalFilter,
) -> bool:
    elapsed_days = _optional_float(row.get("elapsed_days"))
    if elapsed_days is None or elapsed_days < 0:
        return False
    if review_interval.operator == "<":
        return elapsed_days < review_interval.days
    if review_interval.operator == "<=":
        return elapsed_days <= review_interval.days
    if review_interval.operator == "=":
        return elapsed_days == review_interval.days
    if review_interval.operator == ">=":
        return elapsed_days >= review_interval.days
    return elapsed_days > review_interval.days


def summarize_calibration(
    reviews: Iterable[CalibrationReview],
    *,
    total_count: int | None = None,
    skipped_count: int = 0,
    missing_prediction_count: int = 0,
    bin_count: int = 20,
) -> CalibrationSummary:
    if bin_count <= 0:
        raise ValueError("bin_count must be positive.")

    bins = [
        {
            "predicted_sum": 0.0,
            "actual_sum": 0.0,
            "count": 0,
        }
        for _index in range(bin_count)
    ]
    count = 0
    predicted_total = 0.0
    actual_total = 0.0
    for review in reviews:
        index = calibration_bin_index(review.prediction, bin_count=bin_count)
        bucket = bins[index]
        bucket["predicted_sum"] += review.prediction
        bucket["actual_sum"] += review.actual
        bucket["count"] += 1
        count += 1
        predicted_total += review.prediction
        actual_total += review.actual

    summary_bins = tuple(
        CalibrationBin(
            index=index,
            low=calibration_bin_low(index, bin_count=bin_count),
            high=calibration_bin_high(index, bin_count=bin_count),
            predicted_sum=bucket["predicted_sum"],
            actual_sum=bucket["actual_sum"],
            count=int(bucket["count"]),
        )
        for index, bucket in enumerate(bins)
    )
    return CalibrationSummary(
        bins=summary_bins,
        count=count,
        skipped_count=int(skipped_count),
        missing_prediction_count=int(missing_prediction_count),
        total_count=count if total_count is None else int(total_count),
        average_prediction=None if count == 0 else predicted_total / count,
        actual_recall=None if count == 0 else actual_total / count,
    )


def summarize_calibration_rows(
    prediction_rows: Iterable[dict[str, Any]],
    *,
    card_ids: Iterable[int] | None = None,
    include_sameday: bool = False,
    bin_count: int = 20,
) -> CalibrationSummary:
    total = 0
    valid: list[CalibrationReview] = []
    missing_predictions = 0
    allowed_cards = None if card_ids is None else {int(card_id) for card_id in card_ids}
    for row in prediction_rows:
        card_id = _optional_int(row.get("card_id"))
        if allowed_cards is not None and (card_id is None or card_id not in allowed_cards):
            continue
        elapsed_days = _optional_float(row.get("elapsed_days", 0))
        if elapsed_days is None:
            total += 1
            continue
        if not _include_elapsed_days(elapsed_days, include_sameday=include_sameday):
            continue
        total += 1
        review = calibration_review_from_row(row)
        if review is not None:
            valid.append(review)
        elif _is_missing_prediction(row) and card_id is not None:
            missing_predictions += 1
    return summarize_calibration(
        valid,
        total_count=total,
        skipped_count=total - len(valid),
        missing_prediction_count=missing_predictions,
        bin_count=bin_count,
    )


def calibration_bin_index(prediction: float, *, bin_count: int = 20) -> int:
    if bin_count <= 0:
        raise ValueError("bin_count must be positive.")
    value = min(1.0, max(0.0, float(prediction)))
    index = math.floor(math.exp(math.log(bin_count + 1) * value)) - 1
    return min(bin_count - 1, max(0, index))


def calibration_bin_low(index: int, *, bin_count: int) -> float:
    return math.log(index + 1) / math.log(bin_count + 1)


def calibration_bin_high(index: int, *, bin_count: int) -> float:
    return math.log(index + 2) / math.log(bin_count + 1)


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _is_missing_prediction(row: dict[str, Any]) -> bool:
    return row.get("prediction") is None


def _include_elapsed_days(elapsed_days: float, *, include_sameday: bool) -> bool:
    if elapsed_days < 0:
        return False
    if elapsed_days == 0:
        return include_sameday
    return True
