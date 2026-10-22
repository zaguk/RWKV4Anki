from __future__ import annotations

import math

from .evaluation_table import format_metric
from .report_html import (
    MetricDirection,
    MetricTone,
    relative_metric_percent,
    relative_metric_style,
    report_palette,
    semantic_metric_style,
)
from .report_payload import metric_cell, report_document, value_cell
from .state_comparison import (
    DELETED_REVIEWS_COMPARISON,
    GENERATIONS_COMPARISON,
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
    rows = [
        _measurement_row(
            measurement,
            baseline=baseline,
            is_dark=is_dark,
            retained=measurement.key in result.retained_variant_keys,
        )
        for measurement in result.measurements
    ]
    title = (
        "Deleted-card History Comparison"
        if result.comparison == DELETED_REVIEWS_COMPARISON
        else "RWKV Generation Comparison"
        if result.comparison == GENERATIONS_COMPARISON
        else "RWKV Model Comparison"
    )
    subtitle = _subtitle(result)
    notes = (
        "Only RWKV Immediate pre-answer predictions are evaluated; forgetting curves "
        "are never calculated for this comparison.",
        f"All variants use a common scoring set from {result.current_card_count:,} "
        "reviewed cards that still exist. Each generation keeps its own state-building history. "
        "Same-day and otherwise non-evaluable rows are omitted by the "
        "RMSE(bins)/LogLoss calculation.",
        "Lower RMSE(bins) and LogLoss are better; higher AUC is better. "
        "AUC uses the same scoring rows and canonical srs-benchmark calculation. "
        "Percentages compare each result with "
        "the setting currently selected in this window. Green is an improvement and "
        "red is a regression; color "
        "intensity reaches its maximum at 10%.",
        (
            "Saved one checkpoint and Immediate evaluation cache per generation. "
            "Retained candidates are marked in the table. V1 curve features require "
            "rebuilding with curves enabled."
            if result.retained_variant_keys else
            "Each comparison uses disposable state and leaves your saved checkpoint unchanged."
        ),
        *(
            ("Each generation uses its default bundled model. "
             "The selected generation is the baseline.",)
            if result.comparison == GENERATIONS_COMPARISON else ()
        ),
        "State Building Mode is the mode active in this Anki process. A pending mode "
        "change that still requires a restart is not used.",
    )
    table = dict(
        cssClass="comparison-table", rows=rows,
        headers=[{"text": label, "cssClass": "variant" if index == 0 else ""}
                 for index, label in enumerate((
                     "Configuration", "Reviews processed", "Reviews evaluated", "Not scored",
                     "RMSE(bins)", "LogLoss", "AUC ↑",
                 ))],
    )
    return report_document(
        title=title,
        subtitle=subtitle,
        sections=[{"table": table}],
        notes=notes,
        card_label=f"{title} results",
        root_classes="rwkv-comparison",
        is_dark=is_dark,
        generation=generation,
        close_footer=True,
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
    retained: bool = False,
) -> dict:
    current = measurement.key == baseline.key
    detail = (
        "Includes deleted-card history"
        if measurement.include_deleted_reviews
        else "Excludes deleted-card history"
    )
    badges = [{"text": "Selected baseline", "tone": "accent"}] if current else []
    if retained:
        badges.append({"text": "Checkpoint saved", "tone": "neutral"})
    unscored = measurement.processed_review_count - measurement.evaluated_review_count
    return dict(variant=measurement.key, cells=[
        dict(header=True, text=measurement.label, cssClass="variant", badges=badges, detail=detail),
        value_cell(f"{measurement.processed_review_count:,}", css_class="number rwkv-number"),
        value_cell(f"{measurement.evaluated_review_count:,}", css_class="number rwkv-number"),
        value_cell(f"{unscored:,}", css_class="number rwkv-number"),
        _metric_cell(
            measurement.metrics.rmse_bins,
            baseline.metrics.rmse_bins,
            current=current,
            error=measurement.metrics.error,
            is_dark=is_dark,
        ),
        _metric_cell(
            measurement.metrics.log_loss,
            baseline.metrics.log_loss,
            current=current,
            error=measurement.metrics.error,
            is_dark=is_dark,
        ),
        _metric_cell(
            measurement.metrics.auc,
            baseline.metrics.auc,
            current=current,
            error=measurement.metrics.error,
            is_dark=is_dark,
            direction=MetricDirection.HIGHER_IS_BETTER,
        ),
    ])


def _metric_cell(
    value: float | None,
    baseline_value: float | None,
    *,
    current: bool,
    error: str | None,
    is_dark: bool,
    direction: MetricDirection = MetricDirection.LOWER_IS_BETTER,
) -> dict:
    palette = report_palette(is_dark=is_dark, neutral="gray")
    if error or value is None:
        detail = error or "No result"
        return metric_cell(
            "—",
            (detail,),
            classes=("metric",),
            style=semantic_metric_style(MetricTone.NEUTRAL, palette=palette),
        )
    change = 0.0 if current else relative_metric_percent(baseline_value, value, direction=direction)
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
    return metric_cell(
        f"{value:.6f}" if direction == MetricDirection.HIGHER_IS_BETTER else format_metric(value),
        (detail,),
        classes=("metric",),
        style=style,
        change_percent=None if change is None else f"{change:.8g}",
    )


def _subtitle(result: ImmediateStateComparisonResult) -> str:
    mode = "GPU" if result.process_many_mode == "gpu" else result.process_many_mode.title()
    if result.comparison == DELETED_REVIEWS_COMPARISON:
        return f"Model: {result.baseline.model_id} · State Building Mode: {mode}"
    if result.comparison == GENERATIONS_COMPARISON:
        return f"State Building Mode: {mode} · Default models · Common scoring cohort"
    history = (
        "with deleted-card history"
        if result.baseline.include_deleted_reviews
        else "without deleted-card history"
    )
    return f"State Building Mode: {mode} · Built {history}"
