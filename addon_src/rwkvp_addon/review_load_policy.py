from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

DEFAULT_MINIMUM_RETENTION_STEP = 0.02

CandidateT = TypeVar("CandidateT")


def prediction_below_retention(
    predicted_retrievability: float,
    desired_retention: float,
    *,
    extra_retention: float = 0.0,
) -> bool:
    prediction = float(predicted_retrievability)
    desired = float(desired_retention)
    if not math.isfinite(prediction) or not math.isfinite(desired):
        return False
    return prediction < widened_retention_threshold(
        desired,
        extra_retention=extra_retention,
    )


def widened_retention_threshold(
    desired_retention: float,
    *,
    extra_retention: float,
) -> float:
    desired = float(desired_retention)
    if not math.isfinite(desired):
        return 0.0
    return min(1.0, max(0.0, desired + max(0.0, float(extra_retention))))


def active_desired_retention(
    desired_retention: float,
    same_day_desired_retention: float | None = None,
    *,
    elapsed_days: float | None = None,
) -> float:
    if is_same_day_elapsed(elapsed_days) and same_day_desired_retention is not None:
        try:
            same_day = float(same_day_desired_retention)
        except (TypeError, ValueError):
            same_day = math.nan
        if math.isfinite(same_day):
            return same_day
    return float(desired_retention)


def is_same_day_elapsed(elapsed_days: float | None) -> bool:
    try:
        elapsed = float(elapsed_days)
    except (TypeError, ValueError):
        return False
    return math.isfinite(elapsed) and 0.0 <= elapsed < 1.0


def widened_retention_steps(
    *,
    extra_quantum: float = DEFAULT_MINIMUM_RETENTION_STEP,
) -> tuple[float, ...]:
    resolved_quantum = float(extra_quantum)
    if not math.isfinite(resolved_quantum) or resolved_quantum <= 0:
        return (0.0,)
    values = [0.0]
    current = 0.0
    while current < 1.0:
        current = min(1.0, current + resolved_quantum)
        values.append(current)
        if current >= 1.0:
            break
    return tuple(values)


def minimum_retention_extra_required(
    predicted_retrievability: float,
    desired_retention: float,
    *,
    extra_quantum: float = DEFAULT_MINIMUM_RETENTION_STEP,
) -> float | None:
    prediction = float(predicted_retrievability)
    desired = float(desired_retention)
    quantum = float(extra_quantum)
    if (
        not math.isfinite(prediction)
        or not math.isfinite(desired)
        or not math.isfinite(quantum)
        or quantum <= 0
        or prediction >= 1.0
    ):
        return None
    delta = prediction - desired
    if delta < 0:
        return 0.0
    required_units = math.floor(delta / quantum) + 1
    return min(1.0, max(0.0, required_units * quantum))


def can_satisfy_minimum_retention(
    predicted_retrievability: float,
    desired_retention: float,
    *,
    is_intraday: bool,
) -> bool:
    """Return whether a candidate can ever be selected by minimum-review widening."""

    return prediction_below_retention(
        predicted_retrievability,
        desired_retention,
        extra_retention=0.0 if is_intraday else 1.0,
    )


def select_with_minimum_retention(
    candidates: Iterable[CandidateT],
    *,
    limit: int,
    minimum: int,
    sort_key: Callable[[CandidateT], tuple],
    prediction: Callable[[CandidateT], float],
    desired_retention: Callable[[CandidateT], float],
    allow_widening: Callable[[CandidateT], bool] | None = None,
    extra_quantum: float = DEFAULT_MINIMUM_RETENTION_STEP,
) -> tuple[list[CandidateT], list[CandidateT], float]:
    """Select normal below-DR candidates first, then fill minimum in DR tiers.

    Returns ``(eligible_at_final_threshold, selected, final_extra_retention)``.
    The first tier, extra=0, is allowed to fill up to the maximum limit. Later
    widened tiers add only enough cards to satisfy the minimum, so a minimum
    load does not silently turn into "take every card below the widened DR."
    """

    candidate_list = list(candidates)
    resolved_limit = max(0, int(limit))
    resolved_minimum = min(resolved_limit, max(0, int(minimum)))
    if resolved_limit <= 0:
        return [], [], 0.0

    selected: list[CandidateT] = []
    selected_ids: set[int] = set()
    final_extra = 0.0
    can_widen = allow_widening or (lambda _candidate: True)

    for extra in widened_retention_steps(extra_quantum=extra_quantum):
        tier = [
            candidate
            for candidate in candidate_list
            if id(candidate) not in selected_ids
            and _candidate_below_active_threshold(
                candidate,
                prediction=prediction,
                desired_retention=desired_retention,
                extra_retention=extra,
                allow_widening=can_widen,
            )
        ]
        ordered_tier = sorted(tier, key=sort_key)
        if extra == 0.0:
            capacity = resolved_limit - len(selected)
        else:
            capacity = min(
                resolved_limit - len(selected),
                resolved_minimum - len(selected),
            )
        if capacity > 0:
            for candidate in ordered_tier[:capacity]:
                selected.append(candidate)
                selected_ids.add(id(candidate))
        if selected:
            final_extra = extra
        if len(selected) >= resolved_limit:
            break
        if len(selected) >= resolved_minimum:
            break

    eligible = candidates_below_retention(
        candidate_list,
        prediction=prediction,
        desired_retention=desired_retention,
        extra_retention=final_extra,
        allow_widening=can_widen,
    )
    return eligible, selected, final_extra


def candidates_below_retention(
    candidates: Sequence[CandidateT],
    *,
    prediction: Callable[[CandidateT], float],
    desired_retention: Callable[[CandidateT], float],
    extra_retention: float,
    allow_widening: Callable[[CandidateT], bool] | None = None,
) -> list[CandidateT]:
    can_widen = allow_widening or (lambda _candidate: True)
    return [
        candidate
        for candidate in candidates
        if _candidate_below_active_threshold(
            candidate,
            prediction=prediction,
            desired_retention=desired_retention,
            extra_retention=extra_retention,
            allow_widening=can_widen,
        )
    ]


def _candidate_below_active_threshold(
    candidate: CandidateT,
    *,
    prediction: Callable[[CandidateT], float],
    desired_retention: Callable[[CandidateT], float],
    extra_retention: float,
    allow_widening: Callable[[CandidateT], bool],
) -> bool:
    active_extra = float(extra_retention) if allow_widening(candidate) else 0.0
    return prediction_below_retention(
        prediction(candidate),
        desired_retention(candidate),
        extra_retention=active_extra,
    )
