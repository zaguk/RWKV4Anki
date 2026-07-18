from __future__ import annotations

import html
import math

from .report_html import (
    MetricDirection,
    MetricTone,
    ReportMetricStyle,
    ranked_metric_style,
    render_report_document,
    render_report_metric_cell,
    report_palette,
    semantic_metric_style,
)
from .speed_test import (
    LivePredictionSpeedTestResult,
    PredictManySpeedMeasurement,
    PredictManySpeedTestResult,
    ProcessManyCurveSpeedTestResult,
    ProcessManySpeedMeasurement,
    ProcessManySpeedTestResult,
)


def render_predict_many_speed_test_html(
    result: PredictManySpeedTestResult,
    *,
    is_dark: bool = False,
    generation: int = 1,
) -> str:
    modes = result.modes
    headers = "\n".join(
        _mode_header(
            mode,
            batch_size=result.measurement(result.card_counts[0], mode).batch_size,
            show_batch_detail=True,
        )
        for mode in modes
    )
    rows = "\n".join(
        _predict_row(
            card_count,
            tuple(result.measurement(card_count, mode) for mode in modes),
            is_dark=is_dark,
        )
        for card_count in result.card_counts
    )
    cap_note = (
        f"The collection has {result.collection_card_count:,} cards; "
        f"{result.eligible_card_count:,} have processed RWKV state and are eligible "
        f"for this comparison. The largest test used {max(result.card_counts):,}."
    )
    return _document(
        title="Prediction Throughput Speed Test",
        subtitle=f"Model: {result.model_id}",
        table=f"""
<table class="rwkv-data-table speed-table">
    <thead>
        <tr>
            <th scope="col" class="row-header">Cards</th>
            {headers}
        </tr>
    </thead>
    <tbody>
        {rows}
    </tbody>
</table>
""".strip(),
        notes=(
            cap_note,
            "Cards without processed RWKV state are excluded because RWKV-SRS routes "
            "them through the scalar Oracle fallback; including them would not measure "
            "Fast or GPU inference.",
            f"Each cell is the average of {result.repetitions} warmed native Live "
            "Session cycles. Checkpoint loading, native-session construction, its "
            "initial full prediction/index build, and one warm-up cycle are excluded.",
            "Each measured cycle processes and requeues one representative answer, "
            "refreshes the Rust-owned candidate index, and returns only the compact "
            "next-card selection used during live review. Synthetic answers are undone "
            "during untimed cleanup. GPU failure is reported instead of falling back "
            "to Fast under the GPU label.",
            "Batch size is shown under each mode. Automatic lets RWKV-SRS choose its "
            "optimized default.",
        ),
        is_dark=is_dark,
        generation=generation,
    )


def render_process_many_speed_test_html(
    result: ProcessManySpeedTestResult,
    *,
    is_dark: bool = False,
    generation: int = 1,
) -> str:
    headers = "\n".join(_mode_header(mode) for mode in result.modes)
    measurements = tuple(result.measurement(mode) for mode in result.modes)
    values = tuple(item.reviews_per_minute for item in measurements)
    cells = "\n".join(
        _process_cell(
            item,
            values=values,
            available_review_count=result.available_review_count,
            is_dark=is_dark,
        )
        for item in measurements
    )
    cap_note = (
        f"The collection has {result.available_review_count:,} processable reviews; "
        f"each available mode uses up to {result.review_count:,} reviews."
    )
    return _document(
        title="State Building Speed Test",
        subtitle=f"Model: {result.model_id}",
        table=f"""
<table class="rwkv-data-table speed-table">
    <thead>
        <tr>
            <th scope="col" class="row-header">Sample</th>
            {headers}
        </tr>
    </thead>
    <tbody>
        <tr>
            <th scope="row" class="row-header">One run</th>
            {cells}
        </tr>
    </tbody>
</table>
""".strip(),
        notes=(
            cap_note,
            "Each mode processes its stated review count once in a fresh runtime on "
            "Anki's background worker. "
            "Runtime construction and model loading are excluded from the timed result. "
            "Heat colors compare reviews per minute.",
            "The test uses process_many("
            f"return_curves={result.return_curves}, batch_size=10,000), matching the "
            "current checkpoint-processing setting. GPU timing includes initialization, "
            "release_gpu(), and materializing the final state back on the CPU.",
            "Full-history times are linear estimates of native processing only. They "
            "exclude collection export, model construction, checkpoint serialization, "
            "and evaluation-cache postprocessing.",
        ),
        is_dark=is_dark,
        generation=generation,
    )


def render_curve_speed_test_html(
    result: ProcessManyCurveSpeedTestResult,
    *,
    is_dark: bool = False,
    generation: int = 1,
) -> str:
    with_curves = result.measurement(True)
    without_curves = result.measurement(False)
    values = (with_curves.average_seconds, without_curves.average_seconds)
    comparison = _curve_speed_comparison(
        with_curves_seconds=with_curves.average_seconds,
        without_curves_seconds=without_curves.average_seconds,
    )
    cells = "\n".join(
        _curve_cell(item, values=values, is_dark=is_dark)
        for item in (with_curves, without_curves)
    )
    duration_comparison = _curve_duration_comparison(
        with_curves_seconds=with_curves.average_seconds,
        without_curves_seconds=without_curves.average_seconds,
    )
    comparison_style = semantic_metric_style(
        comparison.tone,
        palette=report_palette(
            is_dark=is_dark,
            neutral="warm",
            foreground="tone",
        ),
        css_class=comparison.css_class,
    )
    return _document(
        title="Forgetting Curve Speed Test",
        subtitle=(
            f"Model: {result.model_id} · "
            f"Mode: {_mode_label(result.mode)}"
        ),
        table=f"""
<table class="rwkv-data-table speed-table">
    <thead><tr>
        <th class="row-header">Reviews</th><th>With curves</th><th>Without curves</th>
    </tr></thead>
    <tbody><tr><th class="row-header number">{result.review_count:,}</th>{cells}</tr></tbody>
</table>
<div class="curve-comparison {comparison.css_class}"
     data-rwkv-metric-tone="{comparison_style.tone.value}"
     style="background-color: {comparison_style.background}; color: {comparison_style.foreground}">
    <span>Turning curves off</span>
    <strong>{comparison.percent_text} speed {comparison.direction}</strong>
    <small>based on average reviews/min</small>
</div>
""".strip(),
        notes=(
            f"The collection has {result.available_review_count:,} processable reviews; "
            f"this comparison used {result.review_count:,}.",
            f"Each result averages {result.repetitions} measurement"
            f"{'s' if result.repetitions != 1 else ''}, each using a fresh in-process "
            f"runtime. {duration_comparison}",
            "The percentage compares average processing throughput (reviews/min), "
            "using With curves as the baseline.",
            "This times native process_many() only. Runtime construction, model loading, "
            "collection export, and checkpoint serialization are excluded.",
        ),
        is_dark=is_dark,
        generation=generation,
    )


def render_live_prediction_speed_test_html(
    result: LivePredictionSpeedTestResult,
    *,
    is_dark: bool = False,
    generation: int = 1,
) -> str:
    style = _latency_style(
        result.average_seconds,
        is_dark=is_dark,
    )
    trials = ", ".join(_format_duration(value) for value in result.durations_seconds)
    cap_note = (
        f"The requested {result.requested_card_count:,} cards were tested."
        if result.card_count == result.requested_card_count
        else f"The requested {result.requested_card_count:,} cards were capped to "
        f"{result.card_count:,}, because {result.eligible_card_count:,} collection cards "
        "currently have processed RWKV state."
    )
    batch = "automatic" if result.batch_size is None else f"{result.batch_size:,}"
    metric_cell = render_report_metric_cell(
        _format_duration(result.average_seconds),
        (f"{_format_rate(result.cards_per_second)} cards/s",),
        classes=("metric", "heat"),
        style=style,
    )
    return _document(
        title="Between-Review Prediction Speed Test",
        subtitle=(
            f"Model: {result.model_id} · "
            f"Mode: {_mode_label(result.mode)} · Batch: {batch}"
        ),
        table=f"""
<table class="rwkv-data-table speed-table">
    <thead><tr>
        <th class="row-header">Cards</th>
        <th>{html.escape(_mode_label(result.mode))}</th>
    </tr></thead>
    <tbody><tr>
        <th class="row-header number">{result.card_count:,}</th>
        {metric_cell}
    </tr></tbody>
</table>
""".strip(),
        notes=(
            cap_note,
            f"The average covers {result.repetitions} warmed native Live Session "
            f"cycles. Individual cycles: {trials}.",
            "16.67 ms is one frame on a 60 Hz monitor. A prediction refresh below "
            "100 ms is likely to be unnoticeable during normal reviewing.",
            "This measures the recurring Rust-owned cycle, including one representative "
            "answer/requeue, candidate ordering, prediction, and its compact result. "
            "Native-session construction and its initial full prediction are excluded, "
            "as are Python/Anki bridge work and card display.",
        ),
        is_dark=is_dark,
        generation=generation,
    )


def _predict_row(
    card_count: int,
    measurements: tuple[PredictManySpeedMeasurement, ...],
    *,
    is_dark: bool,
) -> str:
    values = tuple(item.average_seconds for item in measurements)
    cells = "\n".join(
        _predict_cell(item, values=values, is_dark=is_dark) for item in measurements
    )
    return f"""
<tr>
    <th scope="row" class="row-header number">{card_count:,}</th>
    {cells}
</tr>
""".strip()


def _predict_cell(
    measurement: PredictManySpeedMeasurement,
    *,
    values: tuple[float, ...],
    is_dark: bool,
) -> str:
    style = ranked_metric_style(
        measurement.average_seconds,
        values,
        direction=MetricDirection.LOWER_IS_BETTER,
        palette=report_palette(is_dark=is_dark, neutral="warm"),
        best_class="fastest",
        worst_class="slowest",
    )
    return render_report_metric_cell(
        _format_duration(measurement.average_seconds),
        (f"{_format_rate(measurement.cards_per_second)} cards/s",),
        classes=("metric", "heat"),
        style=style,
    )


def _process_cell(
    measurement: ProcessManySpeedMeasurement,
    *,
    values: tuple[float, ...],
    available_review_count: int,
    is_dark: bool,
) -> str:
    style = ranked_metric_style(
        measurement.reviews_per_minute,
        values,
        direction=MetricDirection.HIGHER_IS_BETTER,
        palette=report_palette(is_dark=is_dark, neutral="warm"),
        best_class="fastest",
        worst_class="slowest",
    )
    estimated = _format_estimated_duration(
        measurement.estimated_seconds_for(available_review_count)
    )
    return render_report_metric_cell(
        _format_duration(measurement.duration_seconds),
        (
            f"{measurement.review_count:,} reviews",
            f"{_format_rate(measurement.reviews_per_minute)} reviews/min",
            f"Full history ≈ {estimated}",
        ),
        classes=("metric", "heat"),
        style=style,
    )


def _curve_cell(
    measurement,
    *,
    values: tuple[float, ...],
    is_dark: bool,
) -> str:
    style = ranked_metric_style(
        measurement.average_seconds,
        values,
        direction=MetricDirection.LOWER_IS_BETTER,
        palette=report_palette(is_dark=is_dark, neutral="warm"),
        best_class="fastest",
        worst_class="slowest",
    )
    return render_report_metric_cell(
        _format_duration(measurement.average_seconds),
        (f"{_format_rate(measurement.reviews_per_minute)} reviews/min",),
        classes=("metric", "heat"),
        style=style,
    )


def _mode_header(
    mode: str,
    *,
    batch_size: int | None = None,
    show_batch_detail: bool = False,
) -> str:
    detail = ""
    if show_batch_detail and batch_size is not None:
        detail = f'<span class="header-detail">batch {batch_size:,}</span>'
    elif show_batch_detail:
        detail = '<span class="header-detail">batch automatic</span>'
    return (
        '<th scope="col" class="mode-header">'
        f"{html.escape(_mode_label(mode))}{detail}</th>"
    )


def _document(
    *,
    title: str,
    subtitle: str,
    table: str,
    notes: tuple[str, ...],
    is_dark: bool,
    generation: int,
) -> str:
    return render_report_document(
        title=title,
        subtitle=subtitle,
        report_html=table,
        notes=notes,
        card_aria_label=f"{title} results",
        root_extra_classes="rwkv-speed-test",
        generation=generation,
        is_dark=is_dark,
        include_close_footer=False,
    )


def _format_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000.0:,.2f} ms"
    return f"{seconds:,.3f} s"


def _format_rate(value: float) -> str:
    return "∞" if math.isinf(value) else f"{value:,.0f}"


def _format_estimated_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "∞"
    if seconds < 60.0:
        return f"{seconds:,.1f} s"
    if seconds < 3600.0:
        minutes, remaining = divmod(round(seconds), 60)
        return f"{minutes:,} min {remaining:02d} s"
    hours = seconds / 3600.0
    return f"{hours:,.1f} h"


class _CurveSpeedComparison:
    def __init__(
        self,
        css_class: str,
        percent_text: str,
        direction: str,
        tone: MetricTone,
    ) -> None:
        self.css_class = css_class
        self.percent_text = percent_text
        self.direction = direction
        self.tone = tone


def _curve_speed_comparison(
    *,
    with_curves_seconds: float,
    without_curves_seconds: float,
) -> _CurveSpeedComparison:
    if with_curves_seconds == 0:
        percent = 0.0 if without_curves_seconds == 0 else -100.0
    elif without_curves_seconds == 0:
        percent = math.inf
    else:
        percent = (with_curves_seconds / without_curves_seconds - 1.0) * 100.0

    if math.isinf(percent):
        return _CurveSpeedComparison("gain", "∞%", "gain", MetricTone.GAIN)

    rounded = round(percent, 1)
    if rounded > 0:
        return _CurveSpeedComparison(
            "gain",
            f"{rounded:,.1f}%",
            "gain",
            MetricTone.GAIN,
        )
    if rounded < 0:
        return _CurveSpeedComparison(
            "loss",
            f"{abs(rounded):,.1f}%",
            "loss",
            MetricTone.LOSS,
        )
    return _CurveSpeedComparison(
        "no-change",
        "0.0%",
        "change",
        MetricTone.NEUTRAL,
    )


def _curve_duration_comparison(
    *,
    with_curves_seconds: float,
    without_curves_seconds: float,
) -> str:
    if math.isclose(
        with_curves_seconds,
        without_curves_seconds,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return "Both settings took the same average time in this sample."
    if without_curves_seconds == 0:
        return "Without curves completed below the timer's measurable resolution."
    if with_curves_seconds == 0:
        return "With curves completed below the timer's measurable resolution."
    if with_curves_seconds > without_curves_seconds:
        ratio = with_curves_seconds / without_curves_seconds
        return f"With curves took {ratio:,.2f}× as long in this sample."
    ratio = without_curves_seconds / with_curves_seconds
    return f"Without curves took {ratio:,.2f}× as long in this sample."


def _latency_style(seconds: float, *, is_dark: bool) -> ReportMetricStyle:
    milliseconds = max(0.0, float(seconds) * 1000.0)
    palette = report_palette(is_dark=is_dark, neutral="warm")
    if milliseconds <= 16.67:
        return semantic_metric_style(
            MetricTone.GAIN,
            palette=palette,
            css_class="under-frame",
        )
    if milliseconds < 100.0:
        return semantic_metric_style(
            MetricTone.NEUTRAL,
            palette=palette,
            css_class="under-100ms",
        )
    return semantic_metric_style(
        MetricTone.LOSS,
        palette=palette,
        css_class="over-100ms",
    )


def _mode_label(mode: str) -> str:
    return "GPU" if mode == "gpu" else mode.title()
