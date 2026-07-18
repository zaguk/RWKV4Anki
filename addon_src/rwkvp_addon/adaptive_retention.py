from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .review_load_policy import is_same_day_elapsed
from .review_rows import SECONDS_PER_DAY

MAX_ADAPTIVE_DESIRED_RETENTION = 0.995
MIN_STABILITY_FOR_LOG = 1e-12


@dataclass(frozen=True)
class AdaptiveRetentionSettings:
    enabled: bool = False
    flat: float = 0.0
    s_multi: float = 0.0
    d_multi: float = 0.0


@dataclass(frozen=True)
class AdaptiveRetentionCardData:
    rwkv_stability_days: float | None = None
    fsrs_difficulty: float | None = None


def adaptive_desired_retention(
    *,
    rwkv_stability_days: float,
    fsrs_difficulty: float,
    settings: AdaptiveRetentionSettings,
) -> float:
    """Return ADR from RWKV curve stability and FSRS difficulty.

    ``rwkv_stability_days`` is the latest RWKV curve's 90% interval converted to
    days. ``fsrs_difficulty`` is Anki's raw FSRS difficulty value, not display
    normalized difficulty.
    """

    stability = _finite_float(rwkv_stability_days, "rwkv_stability_days")
    difficulty = _finite_float(fsrs_difficulty, "fsrs_difficulty")
    flat = _finite_float(settings.flat, "flat")
    s_multi = _finite_float(
        settings.s_multi,
        "s_multi",
    )
    d_multi = _finite_float(
        settings.d_multi,
        "d_multi",
    )

    safe_stability = max(stability, MIN_STABILITY_FOR_LOG)
    logit = flat + s_multi * math.log(safe_stability)
    logit += d_multi * difficulty
    logit = min(10.0, max(-10.0, logit))

    desired_retention = 1.0 / (1.0 + math.exp(-logit))
    return min(MAX_ADAPTIVE_DESIRED_RETENTION, max(0.0, desired_retention))


def active_desired_retention_with_adaptive(
    desired_retention: float,
    same_day_desired_retention: float | None = None,
    *,
    elapsed_days: float | None = None,
    adaptive_settings: AdaptiveRetentionSettings | None = None,
    rwkv_stability_days: float | None = None,
    fsrs_difficulty: float | None = None,
) -> float:
    base = float(desired_retention)
    if is_same_day_elapsed(elapsed_days):
        if same_day_desired_retention is not None:
            try:
                same_day = float(same_day_desired_retention)
            except (TypeError, ValueError):
                same_day = math.nan
            if math.isfinite(same_day):
                return same_day
        return base

    if adaptive_settings is None or not adaptive_settings.enabled:
        return base
    if rwkv_stability_days is None or fsrs_difficulty is None:
        return base
    try:
        return adaptive_desired_retention(
            rwkv_stability_days=rwkv_stability_days,
            fsrs_difficulty=fsrs_difficulty,
            settings=adaptive_settings,
        )
    except ValueError:
        return base


def rwkv_stability_days_from_curve(
    curve,
    *,
    interval_for_curve: Callable[[object, float], float],
) -> float | None:
    if curve is None:
        return None
    try:
        interval_seconds = float(interval_for_curve(curve, 0.90))
    except Exception:
        return None
    if not math.isfinite(interval_seconds) or interval_seconds < 0:
        return None
    return interval_seconds / SECONDS_PER_DAY


def adaptive_retention_card_data_for_card_ids(
    card_ids,
    *,
    latest_curves_by_card: Mapping[int, object],
    fsrs_difficulties_by_card: Mapping[int, float],
    interval_for_curve: Callable[[object, float], float],
) -> dict[int, AdaptiveRetentionCardData]:
    data: dict[int, AdaptiveRetentionCardData] = {}
    for raw_card_id in card_ids:
        card_id = int(raw_card_id)
        data[card_id] = AdaptiveRetentionCardData(
            rwkv_stability_days=rwkv_stability_days_from_curve(
                latest_curves_by_card.get(card_id),
                interval_for_curve=interval_for_curve,
            ),
            fsrs_difficulty=_optional_finite_float(
                fsrs_difficulties_by_card.get(card_id)
            ),
        )
    return data


def _finite_float(value: float, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def _optional_finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None
