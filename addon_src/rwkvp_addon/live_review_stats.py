from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from .progress import ProgressReporter
from .rmse_bins import RmseBinsAccumulator, rmse_bin_key


class LiveRetentionCategory(Enum):
    SAME_DAY = "same_day"
    YOUNG = "young"
    MATURE = "mature"


@dataclass(frozen=True)
class LiveRetentionRecord:
    review_id: int
    card_id: int
    category: LiveRetentionCategory
    predicted_retrievability: float
    remembered: bool
    fsrs_predicted_retrievability: float | None = None
    rating: int | None = None
    elapsed_days: float | None = None
    source_deck_id: int | None = None
    desired_retention: float | None = None
    active_desired_retention: float | None = None
    rwkv_stability_days: float | None = None
    fsrs_difficulty: float | None = None
    answered_at_ms: int | None = None
    randomly_selected: bool = False


@dataclass(frozen=True)
class LiveLogLossSummary:
    paired_count: int = 0
    rwkv: float | None = None
    fsrs: float | None = None
    missing_rwkv_count: int = 0
    missing_fsrs_count: int = 0
    missing_both_count: int = 0

    @property
    def difference(self) -> float | None:
        """RWKV minus FSRS: negative values favor RWKV."""
        if self.rwkv is None or self.fsrs is None:
            return None
        return self.rwkv - self.fsrs


@dataclass(frozen=True)
class LiveRmseBinContext:
    elapsed_days: float
    review_count: int
    prior_lapses: int


@dataclass(frozen=True)
class LiveAucSummary:
    paired_count: int = 0
    rwkv: float | None = None
    fsrs: float | None = None

    @property
    def difference(self) -> float | None:
        """RWKV minus FSRS: positive values favor RWKV."""
        if self.rwkv is None or self.fsrs is None:
            return None
        return self.rwkv - self.fsrs


@dataclass(frozen=True)
class LiveRmseBinsSummary:
    paired_count: int = 0
    rwkv: float | None = None
    fsrs: float | None = None
    missing_context_count: int = 0
    same_day_count: int = 0

    @property
    def difference(self) -> float | None:
        if self.rwkv is None or self.fsrs is None:
            return None
        return self.rwkv - self.fsrs


@dataclass(frozen=True)
class LiveRetentionSummaryRow:
    label: str
    review_count: int
    predicted_retention: float | None
    actual_retention: float | None
    remembered_count: int
    fsrs_predicted_retention: float | None = None
    fsrs_actual_retention: float | None = None
    fsrs_available_count: int = 0
    log_loss: LiveLogLossSummary = LiveLogLossSummary()
    rmse_bins: LiveRmseBinsSummary = LiveRmseBinsSummary()
    auc: LiveAucSummary = LiveAucSummary()


@dataclass(frozen=True)
class LiveRetentionSummary:
    records: tuple[LiveRetentionRecord, ...]
    rows: tuple[LiveRetentionSummaryRow, ...]
    skipped_count: int = 0

    @property
    def review_count(self) -> int:
        return len(self.records)


def retention_record_for_answer(
    candidate,
    review_row: Mapping[str, Any] | None,
    *,
    fsrs_prediction: float | None = None,
    randomly_selected: bool = False,
) -> LiveRetentionRecord | None:
    if candidate is None or review_row is None:
        return None
    try:
        review_id = int(review_row["review_id"])
        card_id = int(review_row["card_id"])
        rating = int(review_row["rating"])
        elapsed_days = float(review_row["elapsed_days"])
        prediction = float(candidate.predicted_retrievability)
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(prediction):
        return None
    fsrs_prediction_value = _finite_optional_float(fsrs_prediction)
    category = retention_category_for_elapsed_days(elapsed_days)
    if category is None:
        return None
    return LiveRetentionRecord(
        review_id=review_id,
        card_id=card_id,
        category=category,
        predicted_retrievability=prediction,
        remembered=rating > 1,
        fsrs_predicted_retrievability=fsrs_prediction_value,
        rating=rating,
        elapsed_days=elapsed_days,
        source_deck_id=_optional_int(getattr(candidate, "source_deck_id", None)),
        desired_retention=_candidate_float(candidate, "desired_retention"),
        active_desired_retention=_candidate_float(candidate, "active_desired_retention"),
        rwkv_stability_days=_candidate_float(candidate, "rwkv_stability_days"),
        fsrs_difficulty=_candidate_float(candidate, "fsrs_difficulty"),
        answered_at_ms=review_id,
        randomly_selected=bool(randomly_selected),
    )


def retention_category_for_elapsed_days(
    elapsed_days: float,
) -> LiveRetentionCategory | None:
    if not math.isfinite(elapsed_days) or elapsed_days < 0:
        return None
    if elapsed_days < 1:
        return LiveRetentionCategory.SAME_DAY
    if elapsed_days < 21:
        return LiveRetentionCategory.YOUNG
    return LiveRetentionCategory.MATURE


def summarize_live_retention_records(
    records: Iterable[LiveRetentionRecord],
    *,
    skipped_count: int = 0,
    rmse_bin_context: Mapping[tuple[int, int], LiveRmseBinContext] | None = None,
    include_auc: bool = False,
    progress: ProgressReporter | None = None,
) -> LiveRetentionSummary:
    record_tuple = tuple(_cancellable_records(records, progress))
    labels = ("Same Day", "Young", "Mature", "Total", "Total + Same Day")
    groups = tuple(_LiveSummaryAccumulator() for _ in labels)
    membership = {
        LiveRetentionCategory.SAME_DAY: (0, 4),
        LiveRetentionCategory.YOUNG: (1, 3, 4),
        LiveRetentionCategory.MATURE: (2, 3, 4),
    }
    # Retain validated pairs only when requested. Rank one model at a time
    # after releasing the LogLoss scratch lists, to bound peak Python memory.
    auc_inputs: list[tuple[float, float, int, int]] = []
    for record in _cancellable_records(record_tuple, progress):
        indices = membership.get(record.category, ())
        if not indices:
            continue
        rwkv = _probability(record.predicted_retrievability)
        fsrs = _probability(record.fsrs_predicted_retrievability)
        paired = rwkv is not None and fsrs is not None
        key = None
        if paired:
            rwkv_loss = _binary_log_loss(rwkv, record.remembered)
            fsrs_loss = _binary_log_loss(fsrs, record.remembered)
            if rmse_bin_context is not None and indices[0] != 0:
                context = rmse_bin_context.get((record.review_id, record.card_id))
                if context is not None:
                    key = rmse_bin_key(
                        context.elapsed_days, context.review_count, context.prior_lapses,
                    )
            if include_auc:
                auc_inputs.append((rwkv, fsrs, int(record.remembered), indices[0]))
        for index in indices:
            group = groups[index]
            group.records.append(record)
            if not paired:
                if rwkv is None and fsrs is None:
                    group.missing_both += 1
                elif rwkv is None:
                    group.missing_rwkv += 1
                else:
                    group.missing_fsrs += 1
                continue
            group.rwkv_losses.append(rwkv_loss)
            group.fsrs_losses.append(fsrs_loss)
            group.paired_successes += bool(record.remembered)
            if rmse_bin_context is not None:
                if indices[0] == 0:
                    group.same_day += 1
                elif key is None:
                    group.missing_context += 1
                else:
                    group.rwkv_bins.add_to_bin(key, rwkv, record.remembered)
                    group.fsrs_bins.add_to_bin(key, fsrs, record.remembered)
    cohorts = tuple((len(g.rwkv_losses), g.paired_successes) for g in groups)
    rows = []
    for label, group in zip(labels, groups, strict=True):
        rows.append(_summary_row(
            label, group, progress, include_rmse_bins=rmse_bin_context is not None,
        ))
        group.records.clear()
        group.rwkv_losses.clear()
        group.fsrs_losses.clear()
    if include_auc:
        rows = [replace(row, auc=auc) for row, auc in zip(
            rows, _grouped_auc(cohorts, auc_inputs, progress), strict=True,
        )]
    return LiveRetentionSummary(
        records=record_tuple,
        rows=tuple(rows),
        skipped_count=max(0, int(skipped_count)),
    )


def format_live_retention_percent(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def summarize_live_log_loss(
    records: Iterable[LiveRetentionRecord], *, progress: ProgressReporter | None = None,
) -> LiveLogLossSummary:
    """Score recorded forecasts on the same reviews, including same-day reviews.

    Match Evaluate/srs-benchmark's binary LogLoss: natural logarithms, Again
    is failure, and probabilities are clipped to float64 epsilon bounds.
    Do not replay predictions or apply Evaluate's historical target filtering.
    """
    rwkv_losses: list[float] = []
    fsrs_losses: list[float] = []
    missing_rwkv = missing_fsrs = missing_both = 0
    for record in _cancellable_records(records, progress):
        rwkv = _probability(record.predicted_retrievability)
        fsrs = _probability(record.fsrs_predicted_retrievability)
        if rwkv is None or fsrs is None:
            if rwkv is None and fsrs is None:
                missing_both += 1
            elif rwkv is None:
                missing_rwkv += 1
            else:
                missing_fsrs += 1
            continue
        rwkv_losses.append(_binary_log_loss(rwkv, record.remembered))
        fsrs_losses.append(_binary_log_loss(fsrs, record.remembered))
    count = len(rwkv_losses)
    return LiveLogLossSummary(
        paired_count=count,
        rwkv=math.fsum(rwkv_losses) / count if count else None,
        fsrs=math.fsum(fsrs_losses) / count if count else None,
        missing_rwkv_count=missing_rwkv,
        missing_fsrs_count=missing_fsrs,
        missing_both_count=missing_both,
    )


def _probability(value: float | None) -> float | None:
    parsed = _finite_optional_float(value)
    return parsed if parsed is not None and 0.0 <= parsed <= 1.0 else None


def summarize_live_auc(
    records: Iterable[LiveRetentionRecord], *, progress: ProgressReporter | None = None,
) -> LiveAucSummary:
    """Use Evaluate's exact AUC math on Live's paired LogLoss cohort.

    Keep original probabilities and equal review weights, including same-day
    reviews. Historical bin context is unnecessary; do not replay either model.
    """
    from .metrics import roc_auc_from_counts

    rwkv_counts: dict[float, list[int]] = {}
    fsrs_counts: dict[float, list[int]] = {}
    paired_count = 0
    for record in _cancellable_records(records, progress):
        rwkv = _probability(record.predicted_retrievability)
        fsrs = _probability(record.fsrs_predicted_retrievability)
        if rwkv is None or fsrs is None:
            continue
        outcome = int(record.remembered)
        rwkv_counts.setdefault(rwkv, [0, 0])[outcome] += 1
        fsrs_counts.setdefault(fsrs, [0, 0])[outcome] += 1
        paired_count += 1
    return LiveAucSummary(
        paired_count=paired_count,
        rwkv=roc_auc_from_counts(rwkv_counts, progress=progress),
        fsrs=roc_auc_from_counts(fsrs_counts, progress=progress),
    )


def summarize_live_rmse_bins(
    records: Iterable[LiveRetentionRecord],
    context: Mapping[tuple[int, int], LiveRmseBinContext],
    *,
    progress: ProgressReporter | None = None,
) -> LiveRmseBinsSummary:
    """Pool paired forecasts in benchmark bins, using recovered historical counters."""
    rwkv_bins = RmseBinsAccumulator()
    fsrs_bins = RmseBinsAccumulator()
    missing_context = same_day = 0
    for record in _cancellable_records(records, progress):
        rwkv = _probability(record.predicted_retrievability)
        fsrs = _probability(record.fsrs_predicted_retrievability)
        if rwkv is None or fsrs is None:
            continue  # Already covered by the shared missing-prediction notice.
        if record.category is LiveRetentionCategory.SAME_DAY:
            same_day += 1
            continue
        bins = context.get((record.review_id, record.card_id))
        if bins is None:
            missing_context += 1
            continue
        key = rmse_bin_key(bins.elapsed_days, bins.review_count, bins.prior_lapses)
        rwkv_bins.add_to_bin(key, rwkv, record.remembered)
        fsrs_bins.add_to_bin(key, fsrs, record.remembered)
    return LiveRmseBinsSummary(
        paired_count=rwkv_bins.count, rwkv=rwkv_bins.value(), fsrs=fsrs_bins.value(),
        missing_context_count=missing_context, same_day_count=same_day,
    )


def _binary_log_loss(probability: float, remembered: bool) -> float:
    epsilon = 2.220446049250313e-16  # float64 epsilon, as in metrics.py
    clipped = min(max(probability, epsilon), 1.0 - epsilon)
    return -math.log(clipped if remembered else 1.0 - clipped)


def format_live_retention_delta(
    actual_retention: float | None,
    predicted_retention: float | None,
) -> str:
    if actual_retention is None or predicted_retention is None:
        return "N/A"
    delta = float(actual_retention) - float(predicted_retention)
    return f"{delta * 100:+.1f} pp"


@dataclass
class _LiveSummaryAccumulator:
    # References retain original order and Python's existing sum/fsum behavior;
    # forecasts and losses themselves are prepared only once per review.
    records: list[LiveRetentionRecord] = field(default_factory=list)
    rwkv_losses: list[float] = field(default_factory=list)
    fsrs_losses: list[float] = field(default_factory=list)
    missing_rwkv: int = 0
    missing_fsrs: int = 0
    missing_both: int = 0
    paired_successes: int = 0
    rwkv_bins: RmseBinsAccumulator = field(default_factory=RmseBinsAccumulator)
    fsrs_bins: RmseBinsAccumulator = field(default_factory=RmseBinsAccumulator)
    same_day: int = 0
    missing_context: int = 0


def _grouped_auc(
    cohorts: Sequence[tuple[int, int]], inputs: Sequence[tuple[float, float, int, int]],
    progress: ProgressReporter | None,
) -> tuple[LiveAucSummary, ...]:
    from .metrics import roc_auc_from_sorted_counts

    def scores(model_index):
        if progress is not None:
            progress.check_cancelled()
        paired_count, successes = cohorts[-1]
        if not 0 < successes < paired_count:
            return (None,) * len(cohorts)
        # Each distinct score stores lapse/success counts for three disjoint
        # categories. Total AUCs pool those counts, never category AUC values.
        counts: dict[float, list[int]] = {}
        for item in _cancellable_records(inputs, progress):
            counts.setdefault(item[model_index], [0, 0, 0, 0, 0, 0])[2 * item[3] + item[2]] += 1
        ordered = [counts[p] for p in _cancellable_records(sorted(counts), progress)]
        return tuple(
            roc_auc_from_sorted_counts(
                _auc_counts_for_group(ordered, index),
                negatives=count - successes, positives=successes, progress=progress,
            ) for index, (count, successes) in enumerate(cohorts)
        )

    return tuple(
        LiveAucSummary(cohort[0], rwkv, fsrs)
        for cohort, rwkv, fsrs in zip(cohorts, scores(0), scores(1), strict=True)
    )


def _auc_counts_for_group(
    ordered: Sequence[Sequence[int]], index: int,
) -> Iterable[tuple[int, int]]:
    if index < 3:
        offset = index * 2
        return ((c[offset], c[offset + 1]) for c in ordered)
    if index == 3:
        return ((c[2] + c[4], c[3] + c[5]) for c in ordered)
    return ((c[0] + c[2] + c[4], c[1] + c[3] + c[5]) for c in ordered)


def _summary_row(
    label: str, group: _LiveSummaryAccumulator, progress: ProgressReporter | None,
    *, include_rmse_bins: bool,
) -> LiveRetentionSummaryRow:
    selected = group.records
    if not selected:
        return LiveRetentionSummaryRow(
            label=label,
            review_count=0,
            predicted_retention=None,
            actual_retention=None,
            remembered_count=0,
        )
    remembered_count = sum(
        1 for record in _cancellable_records(selected, progress) if record.remembered
    )
    fsrs_selected = [
        record for record in _cancellable_records(selected, progress)
        if record.fsrs_predicted_retrievability is not None
    ]
    count = len(group.rwkv_losses)
    log_loss = LiveLogLossSummary(
        paired_count=count,
        rwkv=math.fsum(group.rwkv_losses) / count if count else None,
        fsrs=math.fsum(group.fsrs_losses) / count if count else None,
        missing_rwkv_count=group.missing_rwkv,
        missing_fsrs_count=group.missing_fsrs,
        missing_both_count=group.missing_both,
    )
    return LiveRetentionSummaryRow(
        label=label,
        review_count=len(selected),
        predicted_retention=(
            None
            if log_loss.missing_rwkv_count or log_loss.missing_both_count
            else sum(
                record.predicted_retrievability
                for record in _cancellable_records(selected, progress)
            ) / len(selected)
        ),
        actual_retention=remembered_count / len(selected),
        remembered_count=remembered_count,
        fsrs_predicted_retention=_average_fsrs_prediction(fsrs_selected),
        fsrs_actual_retention=_average_actual_retention(fsrs_selected),
        fsrs_available_count=len(fsrs_selected),
        log_loss=log_loss,
        rmse_bins=(
            LiveRmseBinsSummary(
                paired_count=group.rwkv_bins.count,
                rwkv=group.rwkv_bins.value(), fsrs=group.fsrs_bins.value(),
                missing_context_count=group.missing_context, same_day_count=group.same_day,
            ) if include_rmse_bins else LiveRmseBinsSummary()
        ),
    )


def _cancellable_records(records, progress: ProgressReporter | None):
    if progress is None:
        return records

    def checked():
        for index, record in enumerate(records):
            if index % 512 == 0:
                progress.check_cancelled()
            yield record
        progress.check_cancelled()

    return checked()


def _finite_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _candidate_float(candidate, name: str) -> float | None:
    try:
        return _finite_optional_float(getattr(candidate, name))
    except Exception:
        return None


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _average_fsrs_prediction(records: Iterable[LiveRetentionRecord]) -> float | None:
    values = [
        float(record.fsrs_predicted_retrievability)
        for record in records
        if record.fsrs_predicted_retrievability is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _average_actual_retention(records: Iterable[LiveRetentionRecord]) -> float | None:
    record_tuple = tuple(records)
    if not record_tuple:
        return None
    return sum(1 for record in record_tuple if record.remembered) / len(record_tuple)
