from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def curve_probabilities_for_prediction_rows(
    rows: Iterable[dict[str, Any]],
    *,
    curves_by_card: dict[int, Any],
    curve_predictor,
) -> list[float]:
    probabilities: list[float] = []
    for row in rows:
        card_id = int(row["card_id"])
        elapsed_seconds = float(row["elapsed_seconds"])
        curve = curves_by_card.get(card_id)
        if curve is None or elapsed_seconds < 0:
            probabilities.append(math.nan)
        else:
            probabilities.append(float(curve_predictor(curve, elapsed_seconds)))
    return probabilities
