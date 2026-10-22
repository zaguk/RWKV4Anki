from __future__ import annotations

import html
import math

from .evaluation_table import comparison_states, format_relative_ratio
from .live_review_stats import (
    LiveLogLossSummary,
    LiveRetentionSummary,
    LiveRetentionSummaryRow,
    LiveRmseBinsSummary,
    format_live_retention_delta,
    format_live_retention_percent,
)
from .modal_html import ModalField, render_field

RETENTION_IMPROVEMENT_HELP = (
    "Improvement = RWKV absolute retention error / FSRS absolute retention error × 100. "
    "Below 100% favors RWKV; above 100% favors FSRS. "
    "N/A when the FSRS error is zero or either error is unavailable."
)
LOG_LOSS_IMPROVEMENT_HELP = (
    "Improvement = RWKV LogLoss / FSRS LogLoss × 100. "
    "Below 100% favors RWKV; above 100% favors FSRS. "
    "N/A when FSRS LogLoss is zero or either score is unavailable."
)


def render_live_comparison_selector(name: str, *, log_loss: bool = False) -> str:
    field = render_field(
        ModalField(
            name=name,
            label="LogLoss",
            kind="switch",
            checked=log_loss,
            tooltip=(
                "Switch off for Retention; on for LogLoss. "
                "Retention compares average predicted and actual recall. "
                "LogLoss scores each recorded forecast on reviews with both predictions; "
                "lower is better. This choice applies to every table in this window."
            ),
            serialize=False,
        )
    )
    return (
        '<div class="rwkv-live-comparison-selector">'
        '<span class="rwkv-switch__label">Retention</span>'
        f"{field}</div>"
    )


def live_comparison_script() -> str:
    """Presentation-only switching; all scores and paired cohorts come from Python."""
    return """
<script data-rwkv-live-comparison-script>
document.addEventListener('DOMContentLoaded', () => {
  const root = document.querySelector('.rwkv-modal-shell');
  const selector = '.rwkv-live-comparison-selector input[type="checkbox"]';
  document.addEventListener('change', (event) => {
    const toggle = event.target.closest(selector);
    if (!toggle || !root) return;
    const logLoss = toggle.checked;
    root.classList.toggle('rwkv-live-log-loss', logLoss);
    for (const control of document.querySelectorAll(selector)) {
      control.checked = logLoss;
    }
  });
});
</script>
""".strip()


def render_live_retention_table(
    summary: LiveRetentionSummary,
    *,
    include_fsrs: bool = True,
    include_rmse_bins: bool = False,
    rmse_message: str = "",
) -> str:
    """Render the table shared by session completion and saved history."""

    rows = "\n".join(
        _render_live_retention_row(row, include_fsrs=include_fsrs) for row in summary.rows
    )
    fsrs_headers = _fsrs_headers() if include_fsrs else ""
    fsrs_note = _fsrs_partial_note(summary) if include_fsrs else ""
    improvement_note = (
        f'<p class="rwkv-live-retention-footnote">{RETENTION_IMPROVEMENT_HELP}</p>'
        if include_fsrs else ""
    )
    table_mode = "with-fsrs" if include_fsrs else "rwkv-only"
    column_layout = _column_layout(include_fsrs=include_fsrs)
    log_loss_rows = "\n".join(
        '<tr><th scope="row" class="row-header '
        f'{_row_class(row.label)}">{html.escape(row.label)}</th>'
        f"{render_live_metric_cells(row.log_loss)}"
        + (render_live_metric_cells(row.rmse_bins) if include_rmse_bins else "") + "</tr>"
        for row in summary.rows
    )
    total = next(
        (row.log_loss for row in summary.rows if row.label == "Total + Same Day"),
        LiveLogLossSummary(),
    )
    exclusions = render_live_log_loss_exclusions(total)
    rmse_note = ""
    if include_rmse_bins:
        rmse_note = (
            '<p class="rwkv-live-retention-footnote">'
            "RMSE(bins) groups reviews by elapsed days, review count and prior lapses, "
            "weighted by the number of reviews in each bin. It excludes same-day reviews "
            "and uses the same Difference and Improvement rules as LogLoss. "
            "Lower RMSE(bins) is better. "
            "Each metric shows its own paired review count.</p>"
        )
        rmse_total = next(
            (row.rmse_bins for row in summary.rows if row.label == "Total + Same Day"),
            LiveRmseBinsSummary(),
        )
        if rmse_message:
            rmse_note += (
                '<p class="rwkv-live-retention-footnote" role="status">'
                f"{html.escape(rmse_message)}</p>"
            )
        elif rmse_total.missing_context_count:
            count = rmse_total.missing_context_count
            subject = "1 paired review was" if count == 1 else f"{count} paired reviews were"
            rmse_note += (
                '<p class="rwkv-live-retention-footnote" role="status">'
                f"{subject} excluded from RMSE(bins) because their historical bin data "
                "could not be recovered.</p>"
            )
    metric_class = "rwkv-data-table rwkv-live-retention-table rwkv-live-retention-table--log-loss"
    if include_rmse_bins:
        metric_class += " rwkv-live-retention-table--summary-metrics"
    metric_headers = _render_live_metric_headers(include_rmse_bins=include_rmse_bins)
    if summary.skipped_count:
        exclusions += (
            '<p class="rwkv-live-retention-footnote">'
            f"{summary.skipped_count} additional reviews were not recorded and could not be scored."
            "</p>"
        )
    return f"""
<div class="rwkv-live-retention-table-component" data-rwkv-live-retention-table="true">
 <div data-rwkv-live-metric="retention_difference">
  <table class="rwkv-data-table rwkv-live-retention-table rwkv-live-retention-table--{table_mode}">
    <caption class="rwkv-sr-only">Live Session retention by review type</caption>
    {column_layout}
    <thead>
      <tr>
        <th scope="col" class="review-type-header">Review Type</th>
        <th scope="col" class="rwkv-number number count">Count</th>
        <th scope="col" class="rwkv-number number total">Actual</th>
        <th scope="col" class="rwkv-number number total group-start">RWKV Predicted</th>
        <th scope="col" class="rwkv-number number total">RWKV Difference</th>
        {fsrs_headers}
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  {improvement_note}
  {fsrs_note}
 </div>
 <div data-rwkv-live-metric="log_loss">
  <table class="{metric_class}">
    <caption class="rwkv-sr-only">Paired Live review metrics by review type</caption>
    <thead>{metric_headers}</thead>
    <tbody>{log_loss_rows}</tbody>
  </table>
  <p class="rwkv-live-retention-footnote">Lower LogLoss is better.
    Difference = RWKV − FSRS; negative favors RWKV.
    {LOG_LOSS_IMPROVEMENT_HELP}
    Only paired reviews are scored.</p>
  {exclusions}
  {rmse_note}
 </div>
</div>
""".strip()


def render_live_log_loss_headers() -> str:
    return """
<th scope="col" class="rwkv-number number count">Paired Reviews</th>
<th scope="col" class="rwkv-number number total group-start">RWKV LogLoss</th>
<th scope="col" class="rwkv-number number total">FSRS LogLoss</th>
<th scope="col" class="rwkv-number number total group-start">Difference</th>
<th scope="col" class="rwkv-number number total">Improvement</th>
""".strip()


def _render_live_metric_headers(*, include_rmse_bins: bool) -> str:
    if not include_rmse_bins:
        return (
            '<tr><th scope="col" class="review-type-header">Review Type</th>'
            f"{render_live_log_loss_headers()}</tr>"
        )
    columns = "".join(
        f'<th scope="col" class="rwkv-number number{group}">{label}</th>'
        for label, group in (
            ("Paired Reviews", " group-start"), ("RWKV", ""), ("FSRS", ""),
            ("Difference", ""), ("Improvement", ""),
        )
    )
    return (
        '<tr><th scope="col" rowspan="2" class="review-type-header">Review Type</th>'
        '<th scope="colgroup" colspan="5" class="group-start">LogLoss</th>'
        '<th scope="colgroup" colspan="5" class="group-start">RMSE(bins)</th></tr>'
        f"<tr>{columns}{columns}</tr>"
    )


def render_live_metric_cells(scores: LiveLogLossSummary | LiveRmseBinsSummary) -> str:
    delta = scores.difference
    formatted = "N/A" if delta is None else f"{delta:+.6f}"
    # Values equal at the displayed precision must not suggest a winner.
    rounded = 0.0 if delta is None else round(delta, 6)
    tone = "neutral" if rounded == 0 else ("good" if rounded < 0 else "bad")
    improvement_text, improvement_tone = _format_improvement(scores.fsrs, scores.rwkv)
    return (
        f'<td class="rwkv-number number count group-start">{scores.paired_count}</td>'
        f'<td class="rwkv-number number total group-start">{_format_score(scores.rwkv)}</td>'
        f'<td class="rwkv-number number total">{_format_score(scores.fsrs)}</td>'
        f'<td class="rwkv-number number comparison-{tone} group-start">{formatted}</td>'
        f'<td class="rwkv-number number comparison-{improvement_tone}">{improvement_text}</td>'
    )


def render_live_log_loss_exclusions(scores: LiveLogLossSummary) -> str:
    messages = []
    for count, model in (
        (scores.missing_rwkv_count, "RWKV"),
        (scores.missing_fsrs_count, "FSRS"),
        (scores.missing_both_count, "RWKV and FSRS"),
    ):
        if count:
            subject = "1 review was" if count == 1 else f"{count} reviews were"
            messages.append(
                '<p class="rwkv-live-retention-footnote" role="status">'
                f"{subject} not considered due to missing or invalid {model} predictions.</p>"
            )
    return "".join(messages)


def _format_score(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def live_retention_delta_class(
    actual_retention: float | None,
    predicted_retention: float | None,
) -> str:
    if actual_retention is None or predicted_retention is None:
        return "retention-difference-neutral"
    delta = float(actual_retention) - float(predicted_retention)
    if not math.isfinite(delta) or abs(delta) < 0.0005:
        return "retention-difference-neutral"
    return "retention-difference-positive" if delta > 0 else "retention-difference-negative"


def format_live_retention_fsrs_predicted(row: LiveRetentionSummaryRow) -> str:
    formatted = format_live_retention_percent(row.fsrs_predicted_retention)
    if row.fsrs_predicted_retention is not None and 0 < row.fsrs_available_count < row.review_count:
        return f'{formatted} <span class="fsrs-count">(n={row.fsrs_available_count})</span>'
    return formatted


def live_retention_improvement(row: LiveRetentionSummaryRow) -> tuple[str, str]:
    return _format_improvement(
        _absolute_error(row.fsrs_actual_retention, row.fsrs_predicted_retention),
        _absolute_error(row.actual_retention, row.predicted_retention),
    )


def _format_improvement(
    fsrs: float | None, rwkv: float | None, *, higher_is_better: bool = False,
) -> tuple[str, str]:
    """Use Immediate Evaluation's ratio, with no winner for displayed ties."""
    formatted = format_relative_ratio(fsrs, rwkv)
    if not formatted or formatted == "100%":
        return formatted or "N/A", "neutral"
    _fsrs_state, rwkv_state = comparison_states(fsrs, rwkv, higher_is_better=higher_is_better)
    return formatted, "good" if rwkv_state == "better" else "bad"


def _column_layout(*, include_fsrs: bool) -> str:
    prediction_span = 4 if include_fsrs else 2
    comparison = '<col class="comparison-column">' if include_fsrs else ""
    return (
        "<colgroup>"
        '<col class="review-type-column">'
        '<col class="count-column">'
        '<col class="actual-column">'
        f'<col class="prediction-column" span="{prediction_span}">'
        f"{comparison}</colgroup>"
    )


def _render_live_retention_row(
    row: LiveRetentionSummaryRow,
    *,
    include_fsrs: bool,
) -> str:
    row_class = _row_class(row.label)
    delta_class = live_retention_delta_class(
        row.actual_retention,
        row.predicted_retention,
    )
    fsrs_delta_class = live_retention_delta_class(
        row.fsrs_actual_retention,
        row.fsrs_predicted_retention,
    )
    actual = format_live_retention_percent(row.actual_retention)
    predicted = format_live_retention_percent(row.predicted_retention)
    fsrs_predicted = format_live_retention_fsrs_predicted(row)
    delta = format_live_retention_delta(row.actual_retention, row.predicted_retention)
    fsrs_delta = format_live_retention_delta(
        row.fsrs_actual_retention,
        row.fsrs_predicted_retention,
    )
    comparison, comparison_tone = live_retention_improvement(row)
    fsrs_cells = (
        f"""
        <td class="rwkv-number number {row_class} group-start">{fsrs_predicted}</td>
        <td class="rwkv-number number {fsrs_delta_class}">{fsrs_delta}</td>
        <td class="rwkv-number number comparison-{comparison_tone} group-start">{comparison}</td>
""".rstrip()
        if include_fsrs
        else ""
    )
    return f"""
      <tr>
        <th scope="row" class="row-header {row_class}">{html.escape(row.label)}</th>
        <td class="rwkv-number number count">{row.review_count}</td>
        <td class="rwkv-number number {row_class}">{actual}</td>
        <td class="rwkv-number number {row_class} group-start">{predicted}</td>
        <td class="rwkv-number number {delta_class}">{delta}</td>
{fsrs_cells}
      </tr>
""".strip()


def _row_class(label: str) -> str:
    normalized = label.strip().lower()
    if normalized == "same day":
        return "same-day"
    if normalized == "young":
        return "young"
    if normalized == "mature":
        return "mature"
    return "total"


def _fsrs_headers() -> str:
    return """
        <th scope="col" class="rwkv-number number total group-start">FSRS Predicted</th>
        <th scope="col" class="rwkv-number number total">FSRS Difference</th>
        <th scope="col" class="rwkv-number number total group-start">Improvement</th>
""".rstrip()


def _fsrs_partial_note(summary: LiveRetentionSummary) -> str:
    if not any(0 < row.fsrs_available_count < row.review_count for row in summary.rows):
        return ""
    return (
        '<p class="rwkv-live-retention-footnote">'
        "FSRS columns use only reviews with available FSRS predictions; "
        "partial buckets show n."
        "</p>"
    )


def _absolute_error(
    actual_retention: float | None,
    predicted_retention: float | None,
) -> float | None:
    if actual_retention is None or predicted_retention is None:
        return None
    error = abs(float(actual_retention) - float(predicted_retention))
    return error if math.isfinite(error) else None
