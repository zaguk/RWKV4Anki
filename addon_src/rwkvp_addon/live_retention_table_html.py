from __future__ import annotations

import html
import math

from .live_review_stats import (
    LiveRetentionSummary,
    LiveRetentionSummaryRow,
    format_live_retention_delta,
    format_live_retention_percent,
)


def render_live_retention_table(
    summary: LiveRetentionSummary,
    *,
    include_fsrs: bool = True,
) -> str:
    """Render the table shared by session completion and saved history."""

    rows = "\n".join(
        _render_live_retention_row(row, include_fsrs=include_fsrs) for row in summary.rows
    )
    fsrs_headers = _fsrs_headers() if include_fsrs else ""
    fsrs_note = _fsrs_partial_note(summary) if include_fsrs else ""
    table_mode = "with-fsrs" if include_fsrs else "rwkv-only"
    column_layout = _column_layout(include_fsrs=include_fsrs)
    return f"""
<div class="rwkv-live-retention-table-component" data-rwkv-live-retention-table="true">
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
  {fsrs_note}
</div>
""".strip()


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


def format_live_retention_error_ratio(row: LiveRetentionSummaryRow) -> str:
    ratio = _rwkv_fsrs_error_ratio(row)
    if ratio is None:
        return "N/A"
    if ratio == 0.0:
        return "RWKV perfect"
    if not math.isfinite(ratio):
        return "FSRS perfect"
    if abs(ratio - 1.0) < 0.005:
        return "Tie"
    return f"{ratio * 100:.0f}% error"


def live_retention_error_ratio_class(row: LiveRetentionSummaryRow) -> str:
    ratio = _rwkv_fsrs_error_ratio(row)
    if ratio is None or abs(ratio - 1.0) < 0.005:
        return "comparison-neutral"
    if not math.isfinite(ratio):
        return "comparison-bad"
    return "comparison-good" if ratio < 1.0 else "comparison-bad"


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
    comparison = format_live_retention_error_ratio(row)
    comparison_class = live_retention_error_ratio_class(row)
    fsrs_cells = (
        f"""
        <td class="rwkv-number number {row_class} group-start">{fsrs_predicted}</td>
        <td class="rwkv-number number {fsrs_delta_class}">{fsrs_delta}</td>
        <td class="rwkv-number number {comparison_class} group-start">{comparison}</td>
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
        <th scope="col" class="rwkv-number number total group-start">RWKV / FSRS Error</th>
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


def _rwkv_fsrs_error_ratio(row: LiveRetentionSummaryRow) -> float | None:
    rwkv_error = _absolute_error(row.actual_retention, row.predicted_retention)
    fsrs_error = _absolute_error(
        row.fsrs_actual_retention,
        row.fsrs_predicted_retention,
    )
    if rwkv_error is None or fsrs_error is None:
        return None
    if fsrs_error == 0.0:
        return 1.0 if rwkv_error == 0.0 else math.inf
    return rwkv_error / fsrs_error


def _absolute_error(
    actual_retention: float | None,
    predicted_retention: float | None,
) -> float | None:
    if actual_retention is None or predicted_retention is None:
        return None
    error = abs(float(actual_retention) - float(predicted_retention))
    return error if math.isfinite(error) else None
