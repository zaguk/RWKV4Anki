"""Historical FSRS calibration through Anki's public Card Info API.

Anki reconstructs *post-answer* memory states using each card's current preset.
Predict from the preceding retained answer's state, never the target's state.
No scheduling changes, parameter fitting, or separate FSRS implementation.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from .progress import ProgressReporter


def add_fsrs_calibration_predictions(
    col,
    prediction_rows: Sequence[dict[str, Any]],
    progress: ProgressReporter,
) -> list[dict[str, Any]]:
    """Return new aligned rows; leave RWKV's canonical/cache rows untouched."""
    card_stats = getattr(col, "card_stats_data", None)
    if not callable(card_stats):
        raise RuntimeError("This Anki version does not expose historical FSRS card statistics.")
    next_day_at = int(col.sched.day_cutoff)
    by_card: dict[int, list[int]] = defaultdict(list)
    result = []
    for index, row in enumerate(prediction_rows):
        if index % 1024 == 0:
            progress.check_cancelled()
        result.append(dict(row, fsrs_prediction=None))
        by_card[int(row["card_id"])].append(index)
    total = len(by_card)
    progress.update(0, total, "Reconstructing FSRS calibration predictions")
    for count, (card_id, indices) in enumerate(by_card.items(), 1):
        progress.check_cancelled()
        predictions = predictions_from_card_stats(
            card_stats(card_id), next_day_at=next_day_at,
            check_cancelled=progress.check_cancelled,
        )
        # Card Info exposes seconds, whereas RWKV's review IDs use milliseconds.
        # An ambiguous second must not be assigned to either answer by guessing.
        seconds = Counter(int(result[index]["review_id"]) // 1000 for index in indices)
        for index in indices:
            row = result[index]
            second = int(row["review_id"]) // 1000
            prediction = predictions.get(second)
            if (
                seconds[second] == 1 and prediction is not None
                and prediction[1] == int(row["rating"])
            ):
                row["fsrs_prediction"] = prediction[0]
        if count % 50 == 0 or count == total:
            progress.update(count, total, "Reconstructing FSRS calibration predictions")
    return result


def predictions_from_card_stats(
    stats,
    *,
    next_day_at: int,
    check_cancelled: Callable[[], None] = lambda: None,
) -> dict[int, tuple[float, int]]:
    """Map unambiguous review seconds to (pre-answer probability, rating)."""
    # Matches Anki Card Info's decay selection, including legacy preset weights.
    params = stats.fsrs_params
    decay = float(params[20]) if len(params) >= 21 else (0.5 if params else 0.1542)
    if not math.isfinite(decay) or decay <= 0:
        raise ValueError("FSRS preset decay must be a finite positive number.")
    factor = math.expm1(-math.log(0.9) / decay)
    entries = list(reversed(stats.revlog))  # Anki returns newest first.
    seconds = Counter(int(entry.time) for entry in entries)
    previous: tuple[int, float] | None = None
    predictions: dict[int, tuple[float, int]] = {}
    for index, entry in enumerate(entries):
        if index % 1024 == 0:
            check_cancelled()
        kind = int(entry.review_kind)
        rating = int(entry.button_chosen)
        if kind == 4 and int(entry.ease) == 0:  # Reset/forget.
            previous = None
        if rating not in (1, 2, 3, 4) or (kind == 3 and int(entry.ease) == 0):
            # Ignore manual/reschedule and preview/cram entries. Their displayed
            # memory states are interpolated by Anki, not actual state updates.
            continue
        if not entry.HasField("memory_state"):
            previous = None  # Before Anki's retained history/cutoff/reset.
            continue
        second = int(entry.time)
        stability = float(entry.memory_state.stability)
        if not math.isfinite(stability) or stability <= 0:
            previous = None
            continue
        if previous is not None and seconds[second] == 1:
            prior_second, prior_stability = previous
            # Use the same integer Anki day boundaries as historical FSRS replay.
            elapsed = max(0, (next_day_at - prior_second) // 86400) - max(
                0, (next_day_at - second) // 86400,
            )
            if elapsed >= 0:
                prediction = (1 + factor * elapsed / prior_stability) ** -decay
                if math.isfinite(prediction):
                    predictions[second] = (prediction, rating)
        previous = (second, stability)
    return predictions
