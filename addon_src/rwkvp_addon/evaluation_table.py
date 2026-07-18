from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

TResult = TypeVar("TResult")

INSUFFICIENT_REVIEWS_LABEL = "insufficient reviews"

_INSUFFICIENT_REVIEW_MARKERS = (
    "no evaluable reviews after preprocessing",
    "no evaluable fsrs review items",
    "insufficient review history",
    "not enough history",
    "must have at least 400 reviews",
    "fsrsinsufficientdata",
    "fsrsinsufficientreviews",
)


@dataclass(frozen=True)
class ComparisonColors:
    better_background: str
    worse_background: str
    foreground: str


@dataclass(frozen=True)
class EvaluationResultSet(Generic[TResult]):
    results: dict[str, TResult]
    review_counts: dict[str, int]


class EvaluationResultCache(Generic[TResult]):
    def __init__(self) -> None:
        self._by_mode: dict[Hashable, EvaluationResultSet[TResult]] = {}

    def store(
        self,
        mode: Hashable,
        results: Mapping[str, TResult],
        review_counts: Mapping[str, int],
    ) -> None:
        self._by_mode[mode] = EvaluationResultSet(
            results=dict(results),
            review_counts=dict(review_counts),
        )

    def get(self, mode: Hashable) -> EvaluationResultSet[TResult]:
        cached = self._by_mode.get(mode)
        if cached is None:
            return EvaluationResultSet(results={}, review_counts={})
        return EvaluationResultSet(
            results=dict(cached.results),
            review_counts=dict(cached.review_counts),
        )

    def has(self, mode: Hashable) -> bool:
        return mode in self._by_mode

    def clear(self) -> None:
        self._by_mode.clear()


def format_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.4g}"


def format_error(error: str) -> str:
    return INSUFFICIENT_REVIEWS_LABEL if is_insufficient_reviews_error(error) else error


def is_insufficient_reviews_error(error: str | None) -> bool:
    if not error:
        return False
    normalized = error.lower()
    return any(marker in normalized for marker in _INSUFFICIENT_REVIEW_MARKERS)


def format_relative_ratio(fsrs_value: float | None, rwkv_value: float | None) -> str:
    """Format RWKV as a percentage of FSRS for a lower-is-better metric."""

    if fsrs_value is None or rwkv_value is None:
        return ""
    fsrs = float(fsrs_value)
    rwkv = float(rwkv_value)
    if not math.isfinite(fsrs) or not math.isfinite(rwkv) or fsrs <= 0 or rwkv < 0:
        return ""
    return f"{rwkv / fsrs:.0%}"


def comparison_states(fsrs_value: float | None, rwkv_value: float | None) -> tuple[str, str]:
    if fsrs_value is None or rwkv_value is None:
        return "", ""
    if fsrs_value < rwkv_value:
        return "better", "worse"
    if rwkv_value < fsrs_value:
        return "worse", "better"
    return "equal", "equal"


def comparison_colors(is_dark: bool) -> ComparisonColors:
    if is_dark:
        return ComparisonColors(
            better_background="#214d37",
            worse_background="#5a2a2e",
            foreground="#f3f6f4",
        )
    return ComparisonColors(
        better_background="#d8f3dc",
        worse_background="#f8d7da",
        foreground="#202124",
    )
