from __future__ import annotations

import math
from typing import Any

RWKV_SRS_PREDICT_FIELDS = (
    "review_id",
    "card_id",
    "note_id",
    "deck_id",
    "preset_id",
    "day_offset",
    "elapsed_days",
    "elapsed_seconds",
)

RWKV_SRS_PROCESS_FIELDS = RWKV_SRS_PREDICT_FIELDS + (
    "rating",
    "duration",
    "state",
)

HISTORY_FINGERPRINT_FIELDS = RWKV_SRS_PROCESS_FIELDS

CACHED_PREDICTION_FIELDS = (
    "review_id",
    "prediction",
    "card_id",
    "deck_id",
    "preset_id",
    "rating",
    "elapsed_days",
    "elapsed_seconds",
    "review_count",
    "i",
    "prior_lapses",
    "rmse_bins_lapse",
)


def revlog_metadata_row(
    *,
    review_id: int,
    card_id: int,
    note_id: int | None,
    deck_id: int | None,
    preset_id: int | None,
    rating: int,
    duration: float,
    taken_millis: int,
    state: int,
    interval: int,
    last_interval: int,
    ease_factor: int,
    days_elapsed: int,
) -> dict[str, Any]:
    return {
        "review_id": int(review_id),
        "card_id": int(card_id),
        "note_id": optional_int(note_id),
        "deck_id": optional_int(deck_id),
        "preset_id": optional_int(preset_id),
        "rating": int(rating),
        "button_chosen": int(rating),
        "duration": max(0.0, float(duration)),
        "taken_millis": int(taken_millis),
        "review_kind": int(state),
        "state": int(state),
        "interval": int(interval),
        "last_interval": int(last_interval),
        "ease_factor": int(ease_factor),
        "days_elapsed": int(days_elapsed),
    }


def processing_review_row(
    *,
    review_id: int,
    card_id: int,
    note_id: int | None,
    deck_id: int | None,
    preset_id: int | None,
    raw_day_offset: int,
    day_offset: int,
    elapsed_days: int,
    elapsed_seconds: float,
    rating: int,
    duration: float,
    taken_millis: int,
    state: int,
    interval: int,
    last_interval: int,
    ease_factor: int,
    review_count: int,
    prior_lapses: int,
    review_kind: int | None = None,
) -> dict[str, Any]:
    return {
        "review_id": int(review_id),
        "card_id": int(card_id),
        "note_id": optional_int(note_id),
        "deck_id": optional_int(deck_id),
        "preset_id": optional_int(preset_id),
        "raw_day_offset": int(raw_day_offset),
        "day_offset": int(day_offset),
        "elapsed_days": float_or_int(elapsed_days),
        "elapsed_seconds": float_or_int(elapsed_seconds),
        "rating": int(rating),
        "button_chosen": int(rating),
        "duration": max(0.0, float(duration)),
        "taken_millis": int(taken_millis),
        "state": int(state),
        "review_kind": int(state if review_kind is None else review_kind),
        "interval": int(interval),
        "last_interval": int(last_interval),
        "ease_factor": int(ease_factor),
        "review_count": int(review_count),
        "i": int(review_count),
        "prior_lapses": int(prior_lapses),
        "rmse_bins_lapse": int(prior_lapses),
    }


def prediction_input_row(
    *,
    review_id: int,
    card_id: int,
    note_id: int | None,
    deck_id: int | None,
    preset_id: int | None,
    day_offset: int,
    elapsed_days: float,
    elapsed_seconds: float,
    raw_day_offset: int | None = None,
) -> dict[str, Any]:
    row = {
        "review_id": int(review_id),
        "card_id": int(card_id),
        "note_id": optional_int(note_id),
        "deck_id": optional_int(deck_id),
        "preset_id": optional_int(preset_id),
        "day_offset": int(day_offset),
        "elapsed_days": float_or_int(elapsed_days),
        "elapsed_seconds": float_or_int(elapsed_seconds),
    }
    if raw_day_offset is not None:
        row["raw_day_offset"] = int(raw_day_offset)
    return row


def cached_prediction_row(row: dict[str, Any], prediction: float) -> dict[str, Any]:
    review_count = optional_int(row.get("review_count", row.get("i")))
    prior_lapses = optional_int(row.get("prior_lapses", row.get("rmse_bins_lapse")))
    return {
        "review_id": required_int(row, "review_id"),
        "prediction": float(prediction),
        "card_id": required_int(row, "card_id"),
        "deck_id": optional_int(row.get("deck_id")),
        "preset_id": optional_int(row.get("preset_id")),
        "rating": required_int(row, "rating"),
        "elapsed_days": float(row["elapsed_days"]),
        "elapsed_seconds": float(row["elapsed_seconds"]),
        "review_count": 1 if review_count is None else review_count,
        "i": 1 if review_count is None else review_count,
        "prior_lapses": 0 if prior_lapses is None else prior_lapses,
        "rmse_bins_lapse": 0 if prior_lapses is None else prior_lapses,
    }


def required_int(row: dict[str, Any], key: str) -> int:
    return int(row[key])


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def float_or_int(value: float) -> float | int:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Review row numeric fields must be finite.")
    return int(number) if number.is_integer() else number
