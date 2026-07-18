from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean
from typing import Any

from .review_rows import ReviewData, prediction_rows_for_card_ids
from .rwkv_curve_predictions import curve_probabilities_for_prediction_rows
from .rwkv_modes import RetrievabilityMode as RetrievabilityMode


@dataclass(frozen=True)
class CardPrediction:
    card_id: int
    deck_id: int | None
    preset_id: int | None
    retrievability: float
    elapsed_days: float | None = None
    elapsed_seconds: float | None = None


@dataclass(frozen=True)
class RetrievabilitySummary:
    count: int
    average: float | None
    bins: list[tuple[float, float, int]]
    skipped_count: int = 0
    total_count: int = 0
    invalid_predictions: tuple[CardPrediction, ...] = ()


def predict_card_retrievability(
    *,
    card_ids: Iterable[int],
    review_data: ReviewData,
    target_timestamp_seconds: float,
    predictor,
) -> list[CardPrediction]:
    card_id_list = [int(card_id) for card_id in card_ids]
    rows = prediction_rows_for_card_ids(
        card_id_list,
        review_data,
        target_timestamp_seconds=target_timestamp_seconds,
    )
    probabilities = predictor(rows)
    predictions: list[CardPrediction] = []
    for card_id, row, probability in zip(card_id_list, rows, probabilities, strict=True):
        predictions.append(
            CardPrediction(
                card_id=card_id,
                deck_id=row.get("deck_id"),
                preset_id=row.get("preset_id"),
                retrievability=float(probability),
                elapsed_days=float(row["elapsed_days"]),
                elapsed_seconds=float(row["elapsed_seconds"]),
            )
        )
    return predictions


def predict_curve_retrievability(
    *,
    card_ids: Iterable[int],
    review_data: ReviewData,
    target_timestamp_seconds: float,
    curves_by_card: dict[int, Any],
    curve_predictor,
) -> list[CardPrediction]:
    card_id_list = [int(card_id) for card_id in card_ids]
    rows = prediction_rows_for_card_ids(
        card_id_list,
        review_data,
        target_timestamp_seconds=target_timestamp_seconds,
    )
    probabilities = curve_probabilities_for_prediction_rows(
        rows,
        curves_by_card=curves_by_card,
        curve_predictor=curve_predictor,
    )
    predictions: list[CardPrediction] = []
    for card_id, row, probability in zip(card_id_list, rows, probabilities, strict=True):
        predictions.append(
            CardPrediction(
                card_id=card_id,
                deck_id=row.get("deck_id"),
                preset_id=row.get("preset_id"),
                retrievability=probability,
                elapsed_days=float(row["elapsed_days"]),
                elapsed_seconds=float(row["elapsed_seconds"]),
            )
        )
    return predictions


def prediction_time_is_allowed(
    selected_timestamp_seconds: float,
    latest_review_timestamp_seconds: float | None,
) -> bool:
    return (
        latest_review_timestamp_seconds is None
        or selected_timestamp_seconds >= latest_review_timestamp_seconds
    )


def retrievability_bin_index(value: float, *, bin_size: float = 0.05) -> int | None:
    if not math.isfinite(value):
        return None
    bin_count = int(round(1.0 / bin_size))
    return min(bin_count - 1, max(0, int(value / bin_size)))


def card_ids_for_retrievability_bin(
    predictions: Iterable[CardPrediction],
    bucket_index: int,
    *,
    bin_size: float = 0.05,
) -> list[int]:
    return [
        int(prediction.card_id)
        for prediction in predictions
        if retrievability_bin_index(prediction.retrievability, bin_size=bin_size)
        == bucket_index
    ]


def card_id_search(card_ids: Iterable[int]) -> str:
    unique_ids: list[int] = []
    seen: set[int] = set()
    for card_id in card_ids:
        card_id = int(card_id)
        if card_id not in seen:
            unique_ids.append(card_id)
            seen.add(card_id)
    return "" if not unique_ids else "cid:" + ",".join(str(card_id) for card_id in unique_ids)


def summarize_retrievability(
    predictions: Iterable[CardPrediction],
    *,
    bin_size: float = 0.05,
) -> RetrievabilitySummary:
    rows = list(predictions)
    if not rows:
        return RetrievabilitySummary(count=0, average=None, bins=[], total_count=0)
    valid_rows = [row for row in rows if math.isfinite(row.retrievability)]
    invalid_rows = tuple(row for row in rows if not math.isfinite(row.retrievability))
    values = [row.retrievability for row in valid_rows]
    bin_count = int(round(1.0 / bin_size))
    counts = [0 for _ in range(bin_count)]
    for value in values:
        index = retrievability_bin_index(value, bin_size=bin_size)
        assert index is not None
        counts[index] += 1
    bins = [
        (index * bin_size, (index + 1) * bin_size, count)
        for index, count in enumerate(counts)
    ]
    return RetrievabilitySummary(
        count=len(values),
        average=mean(values) if values else None,
        bins=bins,
        skipped_count=len(invalid_rows),
        total_count=len(rows),
        invalid_predictions=invalid_rows,
    )


def filter_predictions(
    predictions: Iterable[CardPrediction],
    allowed_card_ids: Iterable[int],
) -> list[CardPrediction]:
    allowed = {int(card_id) for card_id in allowed_card_ids}
    return [prediction for prediction in predictions if prediction.card_id in allowed]
