"""Benchmark RMSE over elapsed-day, review-count and prior-lapse bins."""

from __future__ import annotations

import math


def rmse_bin_key(
    elapsed_days: float, review_count: int, prior_lapses: int,
) -> tuple[float, float, float]:
    """The benchmark's exact bin boundaries, shared by both paired forecasts."""
    return (
        round(2.48 * 3.62 ** math.floor(math.log(max(elapsed_days, 1e-6)) / math.log(3.62)), 2),
        round(1.99 * 1.89 ** math.floor(math.log(review_count) / math.log(1.89)), 0),
        0.0 if prior_lapses == 0 else round(
            1.65 * 1.73 ** math.floor(math.log(prior_lapses) / math.log(1.73)), 0,
        ),
    )


class RmseBinsAccumulator:
    def __init__(self) -> None:
        self._groups: dict[tuple[float, float, float], list[float]] = {}
        self.count = 0

    def add(
        self, prediction: float, actual: bool | int, *, elapsed_days: float,
        review_count: int, prior_lapses: int,
    ) -> None:
        self.add_to_bin(
            rmse_bin_key(elapsed_days, review_count, prior_lapses), prediction, actual,
        )

    def add_to_bin(
        self, key: tuple[float, float, float], prediction: float, actual: bool | int,
    ) -> None:
        """Accumulate in original review order using a previously computed key."""
        stats = self._groups.setdefault(key, [0.0, 0.0, 0.0])
        stats[0] += actual
        stats[1] += prediction
        stats[2] += 1.0
        self.count += 1

    def value(self) -> float | None:
        if not self.count:
            return None
        # Each bin's weight is its review count, as in srs-benchmark/utils.py::rmse_matrix.
        weighted_square_error = sum(
            count * (actual_sum / count - prediction_sum / count) ** 2
            for actual_sum, prediction_sum, count in self._groups.values()
        )
        return math.sqrt(weighted_square_error / self.count)
