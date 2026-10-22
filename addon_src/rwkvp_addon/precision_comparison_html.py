from __future__ import annotations

from decimal import Decimal

from .evaluation_table import format_metric
from .model_generation import model_generation_label
from .precision_comparison import StatePrecisionComparisonResult
from .report_html import MetricTone, report_palette, semantic_metric_style
from .report_payload import metric_cell, report_document, value_cell


def render_state_precision_comparison_html(
    result: StatePrecisionComparisonResult, *, is_dark: bool = False, generation: int = 1,
) -> str:
    """Keep storage, speed and separate accuracy cohorts readable in narrow windows."""
    return report_document(
        **state_precision_comparison_payload(result, is_dark=is_dark),
        is_dark=is_dark, generation=generation, root_classes="rwkv-comparison",
    )


def state_precision_comparison_payload(
    result: StatePrecisionComparisonResult, *, is_dark: bool = False,
) -> dict:
    """Share the exact same comparison tables between Settings and Guided Setup."""
    measurements = result.measurements
    baseline = next((m for m in measurements if m.state_precision == "fp32"), None)
    batch_size = result.predict_batch_size if result.predict_batch_size is not None else "Automatic"
    performance_rows = [
        {"cells": [
            value_cell(m.state_precision.upper(), header=True, css_class=""),
            *(
                _compared_value(
                    getattr(m, attribute),
                    getattr(baseline, attribute) if baseline else None,
                    relative=m.state_precision == "fp16", is_dark=is_dark,
                    raw_text=formatter(getattr(m, attribute)),
                )
                for attribute, formatter in (
                    ("state_seconds", lambda value: f"{value:,.3f} s"),
                    ("checkpoint_bytes", _size),
                    ("canonical_state_bytes", _size),
                    ("card_note_state_bytes", _size),
                )
            ),
        ]}
        for m in measurements
    ]
    speed_rows = [
        {"cells": [
            value_cell(m.state_precision.upper(), header=True, css_class=""),
            _compared_value(
                m.average_prediction_seconds,
                baseline.average_prediction_seconds if baseline else None,
                relative=m.state_precision == "fp16", is_dark=is_dark,
                raw_text=f"{m.average_prediction_seconds * 1000:,.3f} ms",
            ),
        ]}
        for m in measurements
    ]
    accuracy_rows = [
        {"cells": [
            value_cell(f"{m.state_precision.upper()} · {label}", header=True, css_class=""),
            value_cell(f"{count:,}"),
            *(
                _compared_value(
                    getattr(metrics, attribute),
                    getattr(baseline_metrics, attribute) if baseline_metrics else None,
                    relative=m.state_precision == "fp16", is_dark=is_dark,
                    raw_text=format_metric(getattr(metrics, attribute)),
                    error=metrics.error or (
                        baseline_metrics.error
                        if m.state_precision == "fp16" and baseline_metrics else None
                    ),
                )
                for attribute in ("log_loss", "rmse_bins")
            ),
        ]}
        for label, metric_key, count_key in (
            ("All reviews", "all_metrics", "all_review_count"),
            ("Most recent 25% of reviews", "recent_metrics", "recent_review_count"),
        )
        for m in measurements
        for metrics, count in [(getattr(m, metric_key), getattr(m, count_key))]
        for baseline_metrics in [getattr(baseline, metric_key) if baseline else None]
    ]
    sections = [
        _table(
            (
                "Precision", "Calculation time ↓", "Checkpoint file ↓",
                "All state tensors ↓", "Card/note tensors ↓",
            ),
            performance_rows, "State calculation and size", "State calculation and size",
        ),
        _table(
            ("Precision / review period", "Reviews scored", "LogLoss ↓", "RMSE(bins) ↓"),
            accuracy_rows, "Immediate accuracy for each chronological review period",
            "Immediate prediction accuracy",
        ),
        _table(
            ("Precision", "Mean query pass ↓"),
            speed_rows, "Warmed predict_many speed", "Warmed predict_many() speed",
        ),
    ]
    return dict(
        title="FP16 vs FP32 State Comparison",
        subtitle=(
            f"{model_generation_label(result.model_generation)} · {result.model_id} · "
            f"{result.processed_review_count:,} processed reviews · "
            f"State Building: {result.process_many_mode.upper()} · "
            f"Prediction: {result.predict_many_mode.upper()}"
        ),
        sections=sections,
        notes=(
            "FP32 shows raw baseline values. FP16 shows FP16 / FP32 × 100%: "
            "99% means 1% better; 101% means 1% worse. Lower is better for every "
            "percentage, including query time. Green: ≤100%; yellow: >100% and "
            "<100.1%; red: ≥100.1%. Review counts remain raw counts.",
            "FP16 halves the card/note tensors, not total memory or checkpoint size. "
            "Shared state and calculations remain FP32. Reduced precision can change "
            "forecasts; speed benefits depend on hardware and workload.",
            "Both states process identical complete chronological history. Immediate "
            "pre-answer metrics score the same eligible reviews for both "
            "precisions; same-day and otherwise ineligible reviews are omitted. Lower "
            "LogLoss and RMSE(bins) are better. These are retrospective measurements, "
            "not a forecast of future accuracy.",
            f"Most recent 25% begins at review {result.recent_start_index + 1:,} of "
            f"{result.processed_review_count:,} before eligibility filtering. Both "
            "states still process all earlier reviews; neither is reset at that boundary.",
            "Calculation time excludes model loading, metrics and checkpoint file writing. "
            f"Forgetting curves were {'enabled' if result.calculate_curves else 'disabled'} "
            "to match the Settings draft. Checkpoint file size includes metadata; state "
            "tensor bytes exclude metadata and are not total process RAM or GPU memory.",
            f"Prediction uses the same {result.query_card_count:,} current state-backed "
            f"cards and fixed query timestamps in each run, with {result.repetitions:,} "
            "timed repetitions after warm-up. Native batch size is "
            f"{batch_size} "
            "for both precisions. Each pass totals the native predict_many() calls "
            "over cancellable chunks, excluding input preparation and progress updates. "
            "This times predict_many(), not Live "
            "Session ordering, rendering, model loading or cold GPU startup.",
            *(
                (f"FP32 baseline throughput: {baseline.predictions_per_second:,.0f} "
                 "cards per second. FP16 speed is expressed as relative query time, "
                 "not a throughput ratio; 80% means 20% less time.",)
                if baseline else ()
            ),
            "Temporary states were discarded. Your saved checkpoint and settings are unchanged.",
        ),
        card_label="FP16 and FP32 comparison results",
    )


def _table(headers: tuple[str, ...], rows: list[dict], label: str, heading: str) -> dict:
    return dict(heading=heading, wrap=True, table=dict(
        cssClass="comparison-table", label=label,
        headers=[{"text": header} for header in headers], rows=rows,
    ))


def _size(value: int) -> str:
    return f"{value / (1024 * 1024):,.2f} MiB"


def _compared_value(
    value: float | int | None, baseline: float | int | None, *,
    relative: bool, is_dark: bool, raw_text: str, error: str | None = None,
) -> dict:
    if error or value is None:
        return value_cell("—", tooltip=error or "No eligible reviews")
    measured = Decimal(str(value))
    if not measured.is_finite() or measured < 0:
        return value_cell("—", tooltip="Measurement is unavailable or invalid")
    if not relative:
        return value_cell(raw_text)
    if baseline is None:
        return value_cell("—", tooltip="FP32 baseline is unavailable")
    reference = Decimal(str(baseline))
    if not reference.is_finite() or reference < 0:
        return value_cell("—", tooltip="FP32 baseline is unavailable or invalid")
    if reference == 0 and measured != 0:
        return value_cell("—", tooltip="FP32 baseline is zero; relative percentage is undefined")
    percent = measured / reference * 100 if reference else Decimal(100)
    tone = _percentage_tone(percent)
    # Do not round a yellow cell across either of its narrow color boundaries.
    for digits in range(3, 29):
        if _percentage_tone(Decimal(f"{percent:.{digits}f}")) == tone:
            break
    detail = (
        "Better than FP32" if percent < 100
        else "Equal to FP32" if percent == 100
        else "Worse than FP32"
    )
    return metric_cell(
        f"{percent:,.{digits}f}%", [detail],
        style=semantic_metric_style(tone, palette=report_palette(is_dark=is_dark, neutral="warm")),
    )


def _percentage_tone(percent: Decimal) -> MetricTone:
    if percent <= 100:
        return MetricTone.GAIN
    if percent >= Decimal("100.1"):
        return MetricTone.LOSS
    return MetricTone.NEUTRAL
