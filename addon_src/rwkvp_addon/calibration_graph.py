from __future__ import annotations

import html
from dataclasses import dataclass

from .calibration import CalibrationSummary
from .svg_chart import (
    DEFAULT_CHART_GEOMETRY,
    PERCENT_TICKS,
    finite_number_or_none,
    nice_count_ticks,
    render_chart_axes,
    render_chart_fragment,
    render_chart_tooltip,
    render_count_y_ticks,
    render_empty_chart_state,
    render_percent_x_ticks,
    render_percent_y_ticks,
    render_plot_background,
)

GEOMETRY = DEFAULT_CHART_GEOMETRY


@dataclass(frozen=True)
class CalibrationGraphBin:
    index: int
    low: float
    high: float
    predicted_average: float | None
    actual_average: float | None
    count: int
    x: float
    y: float
    width: float
    height: float
    tooltip: str


@dataclass(frozen=True)
class CalibrationGraphModel:
    count: int
    skipped_count: int
    missing_prediction_count: int
    invalid_count: int
    total_count: int
    average_prediction: float | None
    actual_recall: float | None
    max_count: int
    y_axis_max: int
    bins: list[CalibrationGraphBin]
    line_points: list[tuple[float, float]]
    x_ticks: list[int]
    actual_y_ticks: list[int]
    count_y_ticks: list[int]


def calibration_graph_model(summary: CalibrationSummary) -> CalibrationGraphModel:
    counts = [max(0, int(bucket.count)) for bucket in summary.bins]
    max_count = max(counts, default=0)
    count_ticks = nice_count_ticks(max_count)
    y_axis_max = max(count_ticks)
    bins: list[CalibrationGraphBin] = []
    line_points: list[tuple[float, float]] = []
    for bucket, count in zip(summary.bins, counts, strict=True):
        low_x = GEOMETRY.x_for_unit(bucket.low)
        high_x = GEOMETRY.x_for_unit(bucket.high, fallback=bucket.low)
        width = max(0.0, high_x - low_x - 1)
        y = GEOMETRY.y_for_count(count, y_axis_max)
        height = GEOMETRY.plot_bottom - y
        predicted = finite_number_or_none(bucket.predicted_average)
        actual = finite_number_or_none(bucket.actual_average)
        if predicted is not None and actual is not None:
            line_points.append(
                (GEOMETRY.x_for_unit(predicted), GEOMETRY.y_for_unit(actual))
            )
        bins.append(
            CalibrationGraphBin(
                index=bucket.index,
                low=bucket.low,
                high=bucket.high,
                predicted_average=predicted,
                actual_average=actual,
                count=count,
                x=low_x,
                y=y,
                width=width,
                height=height,
                tooltip=_tooltip(
                    bucket.low,
                    bucket.high,
                    predicted,
                    actual,
                    count,
                ),
            )
        )
    return CalibrationGraphModel(
        count=summary.count,
        skipped_count=summary.skipped_count,
        missing_prediction_count=summary.missing_prediction_count,
        invalid_count=summary.invalid_count,
        total_count=summary.total_count,
        average_prediction=finite_number_or_none(summary.average_prediction),
        actual_recall=finite_number_or_none(summary.actual_recall),
        max_count=max_count,
        y_axis_max=y_axis_max,
        bins=bins,
        line_points=line_points,
        x_ticks=list(PERCENT_TICKS),
        actual_y_ticks=list(PERCENT_TICKS),
        count_y_ticks=count_ticks,
    )


def render_calibration_graph_fragment(
    summary: CalibrationSummary,
    *,
    title: str,
) -> str:
    """Render the calibration chart inside the owning unified analysis document."""

    model = calibration_graph_model(summary)
    bars = "\n".join(_bin_svg(bucket) for bucket in model.bins)
    calibration_line = _polyline_svg(model.line_points)
    no_data = "" if model.count else render_empty_chart_state(GEOMETRY)
    skipped = (
        ""
        if not model.invalid_count
        else f"<span>Skipped invalid: {model.invalid_count:,}</span>"
    )
    missing_prediction = (
        ""
        if not model.missing_prediction_count
        else f"<span>No prior prediction: {model.missing_prediction_count:,}</span>"
    )
    actual_y_ticks = render_percent_y_ticks(
        GEOMETRY,
        ticks=model.actual_y_ticks,
    )
    count_y_ticks = render_count_y_ticks(
        model.count_y_ticks,
        model.y_axis_max,
        GEOMETRY,
        side="right",
    )
    x_ticks = render_percent_x_ticks(GEOMETRY, ticks=model.x_ticks)
    axes = render_chart_axes(GEOMETRY, right_axis=True)
    markup = f"""
<section class="rwkvp-embedded-graph rwkvp-calibration-graph"
         data-rwkv-chart aria-labelledby="rwkvp-calibration-title">
  <h2 class="rwkv-section-title" id="rwkvp-calibration-title">{html.escape(title)}</h2>
  <p class="rwkv-section-intro rwkvp-subtitle">Actual recall rate by predicted probability.</p>
  <svg class="rwkv-chart" viewBox="{GEOMETRY.view_box}" role="img"
       aria-label="RWKV calibration graph">
    {render_plot_background(GEOMETRY)}
    <g class="rwkvp-count-bars">
{bars}
    </g>
    <path class="rwkvp-diagonal"
          d="M {GEOMETRY.x_for_unit(0):.2f} {GEOMETRY.y_for_unit(0):.2f}
             L {GEOMETRY.x_for_unit(1):.2f} {GEOMETRY.y_for_unit(1):.2f}" />
{calibration_line}
{actual_y_ticks}
{count_y_ticks}
{x_ticks}
    {axes}
    <text class="rwkv-chart-axis-title"
          x="{(GEOMETRY.plot_left + GEOMETRY.plot_right) / 2:.2f}"
          y="{GEOMETRY.height - 5:.2f}" text-anchor="middle">Predicted recall</text>
    <text class="rwkv-chart-axis-title" transform="rotate(-90)"
          x="{-(GEOMETRY.plot_top + GEOMETRY.plot_bottom) / 2:.2f}"
          y="18" text-anchor="middle">Actual recall</text>
    <text class="rwkv-chart-axis-title" transform="rotate(90)"
          x="{(GEOMETRY.plot_top + GEOMETRY.plot_bottom) / 2:.2f}"
          y="{-GEOMETRY.width + 22:.2f}" text-anchor="middle">Count</text>
{no_data}
  </svg>
  {render_chart_tooltip()}
  <div class="rwkv-chart-summary">
    <div><span>Average predicted</span><strong>{_percent(model.average_prediction)}</strong></div>
    <div><span>Actual recall</span><strong>{_percent(model.actual_recall)}</strong></div>
    <div><span>Reviews</span><strong>{model.count:,}</strong></div>
    {missing_prediction}
    {skipped}
  </div>
</section>
"""
    return render_chart_fragment(markup, feature_css=_CALIBRATION_CSS)


def _tooltip(
    low: float,
    high: float,
    predicted: float | None,
    actual: float | None,
    count: int,
) -> str:
    reviews = "review" if count == 1 else "reviews"
    return (
        f"Predicted range {_percent(low)}-{_percent(high)}\n"
        f"Average predicted: {_percent(predicted)}\n"
        f"Actual recall: {_percent(actual)}\n"
        f"Count: {count:,} {reviews}"
    )


def _bin_svg(bucket: CalibrationGraphBin) -> str:
    tooltip = html.escape(bucket.tooltip, quote=True)
    return f"""
      <rect class="rwkvp-count-bar" x="{bucket.x:.2f}" y="{bucket.y:.2f}"
            width="{bucket.width:.2f}" height="{bucket.height:.2f}" />
      <rect class="rwkv-chart-hit rwkvp-hit-bar" x="{bucket.x:.2f}"
            y="{GEOMETRY.plot_top:.2f}" width="{bucket.width:.2f}"
            height="{GEOMETRY.plot_height:.2f}"
            data-rwkv-chart-tooltip-text="{tooltip}" aria-label="{tooltip}" />"""


def _polyline_svg(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    data = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    circles = "\n".join(
        f'    <circle class="rwkvp-calibration-point" cx="{x:.2f}" cy="{y:.2f}" r="3" />'
        for x, y in points
    )
    return f"""
    <polyline class="rwkvp-calibration-line" points="{data}" />
{circles}"""


def _percent(value: float | None) -> str:
    finite = finite_number_or_none(value)
    return "N/A" if finite is None else f"{finite * 100:.1f}%"


_CALIBRATION_CSS = """
.rwkvp-calibration-graph .rwkv-chart-plot-bg {
  fill: rgba(127, 127, 127, 0.07);
}
.rwkvp-count-bar {
  fill: #4f7ecb;
  opacity: 0.34;
}
.rwkvp-hit-bar {
  fill: transparent;
}
.rwkvp-diagonal {
  fill: none;
  stroke: #f59e0b;
  stroke-width: 1.5;
}
.rwkvp-calibration-line {
  fill: none;
  stroke: #3b82f6;
  stroke-linejoin: round;
  stroke-width: 2;
}
.rwkvp-calibration-point {
  fill: #3b82f6;
  stroke: var(--canvas, #fff);
  stroke-width: 1;
}
html.night-mode .rwkvp-calibration-point,
body.night-mode .rwkvp-calibration-point {
  stroke: #1f2937;
}
.rwkvp-calibration-graph .rwkv-chart-axis--secondary {
  opacity: 0.55;
}
.rwkvp-calibration-graph .rwkv-chart-grid {
  stroke-dasharray: 2 4;
  stroke-width: 0.6;
  opacity: 0.25;
}
.rwkvp-calibration-graph .rwkv-chart-tick,
.rwkvp-calibration-graph .rwkv-chart-axis-title {
  opacity: 0.74;
}
.rwkvp-calibration-graph .rwkv-chart-summary {
  gap: 4px;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
}
.rwkvp-calibration-graph .rwkv-chart-summary div,
.rwkvp-calibration-graph .rwkv-chart-summary span {
  background: rgba(127, 127, 127, 0.09);
  border-radius: 6px;
  padding: 6px 8px;
}
.rwkvp-calibration-graph .rwkv-chart-summary div {
  display: flex;
  justify-content: space-between;
}
.rwkvp-calibration-graph .rwkv-chart-summary span {
  background: transparent;
  opacity: 0.75;
  padding: 0;
}
.rwkvp-calibration-graph .rwkv-chart-empty rect {
  fill: rgba(127, 127, 127, 0.14);
  opacity: 1;
}
.rwkvp-calibration-graph .rwkv-chart-empty text {
  opacity: 0.65;
}
"""
