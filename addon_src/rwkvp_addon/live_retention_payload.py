"""Formatted, serializable Live report values for Svelte's shared table."""
from __future__ import annotations

from .live_retention_table_html import (
    LOG_LOSS_IMPROVEMENT_HELP,
    RETENTION_IMPROVEMENT_HELP,
    _format_improvement,
    _row_class,
    live_retention_delta_class,
    live_retention_improvement,
)
from .live_review_stats import (
    LiveAucSummary,
    LiveLogLossSummary,
    LiveRetentionSummary,
    LiveRetentionSummaryRow,
    LiveRmseBinsSummary,
    format_live_retention_delta,
    format_live_retention_percent,
)


def metric_cells(
    scores: LiveLogLossSummary | LiveRmseBinsSummary | LiveAucSummary,
    *, higher_is_better: bool = False,
) -> list[dict]:
    delta = scores.difference
    rounded = 0.0 if delta is None else round(delta, 6)
    better = rounded > 0 if higher_is_better else rounded < 0
    tone = "neutral" if rounded == 0 else "good" if better else "bad"
    improvement, improvement_tone = _format_improvement(
        scores.fsrs, scores.rwkv, higher_is_better=higher_is_better,
    )
    return [
        cell(str(scores.paired_count), "count group-start"),
        cell("N/A" if scores.rwkv is None else f"{scores.rwkv:.6f}", "total group-start"),
        cell("N/A" if scores.fsrs is None else f"{scores.fsrs:.6f}", "total"),
        cell("N/A" if delta is None else f"{delta:+.6f}", f"comparison-{tone} group-start"),
        cell(improvement, f"comparison-{improvement_tone}"),
    ]


def cell(text: str, css_class: str, *, partial_count: int | None = None) -> dict:
    return dict(text=text, cssClass=f"rwkv-number number {css_class}", partialCount=partial_count)


def retention_cells(
    row: LiveRetentionSummaryRow, *, include_fsrs: bool, session=False,
) -> list[dict]:
    row_class = "total" if session else _row_class(row.label)
    cells = [
        cell(str(row.review_count), "count"),
        cell(format_live_retention_percent(row.actual_retention),
             row_class + (" group-start" if session else "")),
        cell(format_live_retention_percent(row.predicted_retention),
             row_class + ("" if session else " group-start")),
        cell(format_live_retention_delta(row.actual_retention, row.predicted_retention),
             live_retention_delta_class(row.actual_retention, row.predicted_retention)),
    ]
    if include_fsrs:
        comparison, tone = live_retention_improvement(row)
        partial = (
            row.fsrs_available_count
            if row.fsrs_predicted_retention is not None
            and 0 < row.fsrs_available_count < row.review_count else None
        )
        cells.extend([
            cell(format_live_retention_percent(row.fsrs_predicted_retention),
                 row_class + " group-start", partial_count=partial),
            cell(format_live_retention_delta(
                row.fsrs_actual_retention, row.fsrs_predicted_retention,
            ), live_retention_delta_class(row.fsrs_actual_retention, row.fsrs_predicted_retention)),
            cell(comparison, f"comparison-{tone}" + ("" if session else " group-start")),
        ])
    return cells


def live_retention_table_data(
    summary: LiveRetentionSummary, *, include_fsrs=True, include_rmse_bins=False, rmse_message="",
    include_auc=False,
) -> dict:
    retention_notes = []
    if include_fsrs:
        retention_notes.append(RETENTION_IMPROVEMENT_HELP)
        if any(0 < row.fsrs_available_count < row.review_count for row in summary.rows):
            retention_notes.append(
                "FSRS columns use only reviews with available FSRS predictions; "
                "partial buckets show n."
            )
    metric_notes = [dict(text=(
        "Lower LogLoss is better. Difference = RWKV − FSRS; negative favors RWKV. "
        f"{LOG_LOSS_IMPROVEMENT_HELP} Only paired reviews are scored."
    ))]
    total = next(
        (row.log_loss for row in summary.rows if row.label == "Total + Same Day"),
        LiveLogLossSummary(),
    )
    for count, model in (
        (total.missing_rwkv_count, "RWKV"), (total.missing_fsrs_count, "FSRS"),
        (total.missing_both_count, "RWKV and FSRS"),
    ):
        if count:
            subject = "1 review was" if count == 1 else f"{count} reviews were"
            metric_notes.append(dict(role="status", text=(
                f"{subject} not considered due to missing or invalid {model} predictions."
            )))
    if summary.skipped_count:
        metric_notes.append(dict(text=(
            f"{summary.skipped_count} additional reviews were not recorded and could not be scored."
        )))
    rmse_notes = []
    if include_rmse_bins:
        rmse_notes.append(dict(text=(
            "RMSE(bins) groups reviews by elapsed days, review count and prior lapses, "
            "weighted by the number of reviews in each bin. It excludes same-day reviews "
            "and uses the same Difference and Improvement rules as LogLoss. "
            "Lower RMSE(bins) is better. Each metric shows its own paired review count."
        )))
        rmse = next((row.rmse_bins for row in summary.rows if row.label == "Total + Same Day"),
                    LiveRmseBinsSummary())
        if not rmse_message and rmse.missing_context_count:
            count = rmse.missing_context_count
            subject = "1 paired review was" if count == 1 else f"{count} paired reviews were"
            rmse_message = (
                f"{subject} excluded from RMSE(bins) because their historical bin data "
                "could not be recovered."
            )
        if rmse_message:
            rmse_notes.append(dict(role="status", text=rmse_message))
    return dict(
        includeFsrs=include_fsrs, includeRmse=include_rmse_bins, includeAuc=include_auc,
        aucHelp=(
            "Higher AUC is better. Difference = RWKV − FSRS; positive favors RWKV. "
            "Improvement = RWKV AUC / FSRS AUC × 100; above 100% favors RWKV. "
            "AUC uses Evaluation's original-probability ranking, equal review weights and "
            "half credit for ties, rounded to six decimals. Like Live LogLoss, it scores "
            "paired recorded forecasts, including same-day reviews; no historical bins "
            "or model replay are needed. AUC is N/A without both successes and lapses. "
            "Improvement is N/A when FSRS AUC is zero or either score is unavailable."
        ) if include_auc else "",
        retentionNotes=retention_notes, metricNotes=metric_notes, rmseNotes=rmse_notes,
        rows=[dict(
            label=row.label, cssClass=_row_class(row.label),
            retention=retention_cells(row, include_fsrs=include_fsrs),
            metrics=metric_cells(row.log_loss),
            rmse=metric_cells(row.rmse_bins) if include_rmse_bins else [],
            auc=metric_cells(row.auc, higher_is_better=True) if include_auc else [],
        ) for row in summary.rows],
    )
