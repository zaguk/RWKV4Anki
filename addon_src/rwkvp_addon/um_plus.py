"""Paired disagreement diagnostics, following JSchoreels' SSE UM+ definition.

This is an add-on-owned implementation of the statistic. Provenance, formulas,
and differences in historical prediction sources are documented in
docs/notes/UM_PLUS_GRAPH.md.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from .calibration import paired_calibration_reviews

BINS_PER_UNIT = 20
SMALL_BUCKET_THRESHOLD = 200


@dataclass(frozen=True)
class UmPlusBin:
    index: int
    count: int
    disagreement: float
    rwkv_error: float
    fsrs_error: float

    @property
    def small(self) -> bool:
        return self.count < SMALL_BUCKET_THRESHOLD

    @property
    def bounds(self) -> tuple[float, float]:
        center = (self.index - BINS_PER_UNIT) / BINS_PER_UNIT
        half_width = 0.5 / BINS_PER_UNIT
        return max(-1.0, center - half_width), min(1.0, center + half_width)


@dataclass(frozen=True)
class UmPlusSummary:
    bins: tuple[UmPlusBin, ...]
    count: int
    unpaired_count: int
    rwkv_score: float | None
    fsrs_score: float | None

    def visible_bins(self, *, show_small_buckets: bool = False) -> tuple[UmPlusBin, ...]:
        return tuple(bucket for bucket in self.bins if show_small_buckets or not bucket.small)


def summarize_um_plus_rows(
    rows: Iterable[dict[str, Any]], *, include_sameday: bool = False,
) -> UmPlusSummary:
    """41 fixed 5-pp buckets; scores include every paired review, even hidden bins."""
    size = 2 * BINS_PER_UNIT + 1
    counts = [0] * size
    differences = [0.0] * size
    rwkv_errors = [0.0] * size
    fsrs_errors = [0.0] * size
    unpaired = 0
    for rwkv, fsrs in paired_calibration_reviews(rows, include_sameday=include_sameday):
        if rwkv is None or fsrs is None:
            unpaired += 1
            continue
        difference = fsrs.prediction - rwkv.prediction
        # Halfway ties go toward +infinity, matching JavaScript Math.round.
        index = min(size - 1, max(0, math.floor(difference * BINS_PER_UNIT + 0.5) + BINS_PER_UNIT))
        counts[index] += 1
        differences[index] += difference
        rwkv_errors[index] += rwkv.prediction - rwkv.actual
        fsrs_errors[index] += fsrs.prediction - fsrs.actual
    bins = tuple(
        UmPlusBin(i, n, differences[i] / n, rwkv_errors[i] / n, fsrs_errors[i] / n)
        for i, n in enumerate(counts) if n
    )
    count = sum(counts)
    return UmPlusSummary(
        bins=bins,
        count=count,
        unpaired_count=unpaired,
        rwkv_score=(
            math.sqrt(math.fsum(b.count * b.rwkv_error ** 2 for b in bins) / count)
            if count else None
        ),
        fsrs_score=(
            math.sqrt(math.fsum(b.count * b.fsrs_error ** 2 for b in bins) / count)
            if count else None
        ),
    )


def regression_slope(bins: tuple[UmPlusBin, ...], model: Literal["rwkv", "fsrs"]) -> float | None:
    """Count-weighted least-squares slope of visible bin means (with an intercept)."""
    count = sum(b.count for b in bins)
    if not count:
        return None
    errors = [b.rwkv_error if model == "rwkv" else b.fsrs_error for b in bins]
    mean_x = math.fsum(b.count * b.disagreement for b in bins) / count
    mean_y = math.fsum(b.count * error for b, error in zip(bins, errors, strict=True)) / count
    variance = math.fsum(b.count * (b.disagreement - mean_x) ** 2 for b in bins)
    if variance <= 1e-12:
        return None
    return math.fsum(
        b.count * (b.disagreement - mean_x) * (error - mean_y)
        for b, error in zip(bins, errors, strict=True)
    ) / variance


def graph_domains(bins: tuple[UmPlusBin, ...]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Visible ranges rounded outward to 10 pp, keeping zero inside the error axis."""
    if not bins:
        return (-1.0, 1.0), (-0.1, 0.1)
    x_domain = _rounded_domain([b.disagreement for b in bins])
    low, high = _rounded_domain([0.0, *(b.rwkv_error for b in bins), *(b.fsrs_error for b in bins)])
    return x_domain, (min(low, -0.1), max(high, 0.1))


def _rounded_domain(values: list[float]) -> tuple[float, float]:
    low = max(-10, math.floor((min(values) + 1e-10) * 10))
    high = min(10, math.ceil((max(values) - 1e-10) * 10))
    if low == high:
        low, high = max(-10, low - 1), min(10, high + 1)
    return low / 10, high / 10
