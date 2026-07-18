from __future__ import annotations

import html
import math

from .evaluation_table import format_metric
from .modal_html import render_badge
from .report_html import (
    MetricDirection,
    MetricTone,
    relative_metric_percent,
    relative_metric_style,
    render_report_document,
    render_report_metric_cell,
    report_palette,
    semantic_metric_style,
)
from .state_comparison import (
    DELETED_REVIEWS_COMPARISON,
    ImmediateStateComparisonMeasurement,
    ImmediateStateComparisonResult,
)


def render_immediate_state_comparison_html(
    result: ImmediateStateComparisonResult,
    *,
    is_dark: bool = False,
    generation: int = 1,
) -> str:
    baseline = result.baseline
    rows = "\n".join(
        _measurement_row(
            measurement,
            baseline=baseline,
            is_dark=is_dark,
        )
        for measurement in result.measurements
    )
    title = (
        "Deleted-card History Comparison"
        if result.comparison == DELETED_REVIEWS_COMPARISON
        else "RWKV Model Comparison"
    )
    subtitle = _subtitle(result)
    notes = (
        "Only RWKV Immediate pre-answer predictions are evaluated; forgetting curves "
        "are never calculated for this comparison.",
        f"All variants are scored on the same {result.current_review_count:,} review-row "
        f"history from {result.current_card_count:,} reviewed cards that still exist. "
        "Same-day and otherwise non-evaluable rows are omitted by the "
        "RMSE(bins)/LogLoss calculation.",
        "Lower RMSE(bins) and LogLoss are better. Percentages compare each result with "
        "the setting currently selected in this window. Green is an improvement and "
        "red is a regression; color "
        "intensity reaches its maximum at 10%.",
        "Each variant is built as a disposable in-memory state. Its state and prediction "
        "array are released before the next variant starts; no checkpoint is written.",
        "State Building Mode is the mode active in this Anki process. A pending mode "
        "change that still requires a restart is not used.",
    )
    results = f"""
<table class="rwkv-data-table comparison-table">
    <thead><tr>
        <th class="variant">Configuration</th>
        <th>Reviews processed</th>
        <th>Reviews evaluated</th>
        <th>RMSE(bins)</th>
        <th>LogLoss</th>
    </tr></thead>
    <tbody>{rows}</tbody>
</table>
""".strip()
    return render_report_document(
        title=title,
        subtitle=subtitle,
        report_html=results,
        notes=notes,
        card_aria_label=f"{title} results",
        root_extra_classes="rwkv-comparison",
        is_dark=is_dark,
        generation=generation,
    )


def metric_improvement_percent(
    baseline_value: float | None,
    value: float | None,
) -> float | None:
    return relative_metric_percent(
        baseline_value,
        value,
        direction=MetricDirection.LOWER_IS_BETTER,
    )


def _measurement_row(
    measurement: ImmediateStateComparisonMeasurement,
    *,
    baseline: ImmediateStateComparisonMeasurement,
    is_dark: bool,
) -> str:
    current = measurement.key == baseline.key
    detail = (
        "Includes deleted-card history"
        if measurement.include_deleted_reviews
        else "Excludes deleted-card history"
    )
    badge = render_badge("Selected baseline", tone="accent") if current else ""
    return f"""
<tr data-variant="{html.escape(measurement.key, quote=True)}">
    <th scope="row" class="variant">
        {html.escape(measurement.label)}{badge}
        <span class="variant-detail">{html.escape(detail)}</span>
    </th>
    <td class="number rwkv-number">{measurement.processed_review_count:,}</td>
    <td class="number rwkv-number">{measurement.evaluated_review_count:,}</td>
    {
        _metric_cell(
            measurement.metrics.rmse_bins,
            baseline.metrics.rmse_bins,
            current=current,
            error=measurement.metrics.error,
            is_dark=is_dark,
        )
    }
    {
        _metric_cell(
            measurement.metrics.log_loss,
            baseline.metrics.log_loss,
            current=current,
            error=measurement.metrics.error,
            is_dark=is_dark,
        )
    }
</tr>
""".strip()


def _metric_cell(
    value: float | None,
    baseline_value: float | None,
    *,
    current: bool,
    error: str | None,
    is_dark: bool,
) -> str:
    palette = report_palette(is_dark=is_dark, neutral="gray")
    if error or value is None:
        detail = error or "No result"
        return render_report_metric_cell(
            "—",
            (detail,),
            classes=("metric",),
            style=semantic_metric_style(MetricTone.NEUTRAL, palette=palette),
        )
    change = 0.0 if current else metric_improvement_percent(baseline_value, value)
    style = relative_metric_style(change, palette=palette, full_intensity_at=10.0)
    if current:
        detail = "Selected-setting baseline"
    elif change is None:
        detail = "Baseline comparison unavailable"
    elif math.isclose(change, 0.0, abs_tol=0.0005):
        detail = "No change vs. selected"
    elif change > 0:
        detail = f"{change:,.1f}% better than selected"
    else:
        detail = f"{abs(change):,.1f}% worse than selected"
    data_attributes = (
        {} if change is None else {"data-change-percent": f"{change:.8g}"}
    )
    return render_report_metric_cell(
        format_metric(value),
        (detail,),
        classes=("metric",),
        style=style,
        data_attributes=data_attributes,
    )


def _subtitle(result: ImmediateStateComparisonResult) -> str:
    mode = "GPU" if result.process_many_mode == "gpu" else result.process_many_mode.title()
    if result.comparison == DELETED_REVIEWS_COMPARISON:
        return f"Model: {result.baseline.model_id} · State Building Mode: {mode}"
    history = (
        "with deleted-card history"
        if result.baseline.include_deleted_reviews
        else "without deleted-card history"
    )
    return f"State Building Mode: {mode} · Built {history}"
