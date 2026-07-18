from __future__ import annotations

import html
import math
from dataclasses import dataclass

from .retrievability import RetrievabilitySummary
from .svg_chart import (
    PERCENT_TICKS,
    ChartGeometry,
    ChartMargins,
    clamp_unit,
    finite_number_or_none,
    nice_count_ticks,
    render_chart_axes,
    render_chart_fragment,
    render_chart_tooltip,
    render_count_y_ticks,
    render_empty_chart_state,
    render_percent_x_ticks,
    render_plot_background,
)

GEOMETRY = ChartGeometry(margins=ChartMargins(bottom=25))

RDYLGN = (
    (165, 0, 38),
    (215, 48, 39),
    (244, 109, 67),
    (253, 174, 97),
    (254, 224, 139),
    (255, 255, 191),
    (217, 239, 139),
    (166, 217, 106),
    (102, 189, 99),
    (26, 152, 80),
    (0, 104, 55),
)


@dataclass(frozen=True)
class GraphBar:
    bucket_index: int
    low_percent: int
    high_percent: int
    count: int
    x: float
    y: float
    width: float
    height: float
    color: str
    tooltip: str
    clickable: bool


@dataclass(frozen=True)
class GraphModel:
    average_percent: float | None
    count: int
    skipped_count: int
    total_count: int
    max_count: int
    y_axis_max: int
    bars: list[GraphBar]
    x_ticks: list[int]
    y_ticks: list[int]


def graph_model(summary: RetrievabilitySummary) -> GraphModel:
    counts = [max(0, int(count)) for _low, _high, count in summary.bins]
    max_count = max(counts, default=0)
    y_ticks = nice_count_ticks(max_count)
    y_axis_max = max(y_ticks)
    bars: list[GraphBar] = []
    for bucket_index, ((low, high, _count), count) in enumerate(
        zip(summary.bins, counts, strict=True)
    ):
        low_value = clamp_unit(low)
        high_value = max(low_value, clamp_unit(high, fallback=low_value))
        low_percent = int(round(low_value * 100))
        high_percent = int(round(high_value * 100))
        x = GEOMETRY.x_for_unit(low_value)
        high_x = GEOMETRY.x_for_unit(high_value)
        width = max(0.0, high_x - x - 1)
        y = GEOMETRY.y_for_count(count, y_axis_max)
        height = GEOMETRY.plot_bottom - y
        percent = f"{low_percent}%-{high_percent}%"
        cards = "card" if count == 1 else "cards"
        bars.append(
            GraphBar(
                bucket_index=bucket_index,
                low_percent=low_percent,
                high_percent=high_percent,
                count=count,
                x=x,
                y=y,
                width=width,
                height=height,
                color=rdylgn_color(high_percent),
                tooltip=f"{count} {cards} with {percent} retrievability",
                clickable=count > 0,
            )
        )
    average = finite_number_or_none(summary.average)
    return GraphModel(
        average_percent=None if average is None else average * 100,
        count=summary.count,
        skipped_count=summary.skipped_count,
        total_count=summary.total_count,
        max_count=max_count,
        y_axis_max=y_axis_max,
        bars=bars,
        x_ticks=list(PERCENT_TICKS),
        y_ticks=y_ticks,
    )


def render_retrievability_graph_fragment(summary: RetrievabilitySummary) -> str:
    """Render the histogram inside the owning unified analysis document."""

    model = graph_model(summary)
    bars = "\n".join(_bar_svg(bar) for bar in model.bars)
    average = "N/A" if model.average_percent is None else f"{model.average_percent:.0f}%"
    invalid = (
        ""
        if not model.skipped_count
        else f"<span>Skipped non-finite: {model.skipped_count:,}</span>"
    )
    no_data = "" if model.count else render_empty_chart_state(GEOMETRY)
    y_ticks = render_count_y_ticks(
        model.y_ticks,
        model.y_axis_max,
        GEOMETRY,
        include_grid=True,
    )
    x_ticks = render_percent_x_ticks(GEOMETRY, ticks=model.x_ticks)
    axes = render_chart_axes(GEOMETRY)
    markup = f"""
<section class="rwkvp-embedded-graph rwkvp-retrievability-graph"
         data-rwkv-chart aria-labelledby="rwkvp-retrievability-title">
  <h2 class="rwkv-section-title" id="rwkvp-retrievability-title">Card Retrievability</h2>
  <p class="rwkv-section-intro rwkvp-subtitle">
    The probability of recalling a card at the selected time.
  </p>
  <svg class="rwkv-chart" viewBox="{GEOMETRY.view_box}" role="img"
       aria-label="Card retrievability histogram">
    {render_plot_background(GEOMETRY)}
    <g class="rwkvp-bars">
{bars}
    </g>
{y_ticks}
{x_ticks}
    {axes}
{no_data}
  </svg>
  {render_chart_tooltip()}
  <div class="rwkv-chart-summary">
    <div><span>Average retrievability</span><strong>{average}</strong></div>
    <div><span>Cards</span><strong>{model.count:,}</strong></div>
    {invalid}
  </div>
</section>
"""
    return render_chart_fragment(
        markup,
        feature_css=_RETRIEVABILITY_CSS,
        feature_script=_BUCKET_INTERACTION_SCRIPT,
    )


def rdylgn_color(percent: float) -> str:
    value = clamp_unit(float(percent) / 100.0)
    scaled = value * (len(RDYLGN) - 1)
    lower = int(math.floor(scaled))
    upper = min(len(RDYLGN) - 1, lower + 1)
    fraction = scaled - lower
    rgb = tuple(
        round(
            RDYLGN[lower][index]
            + (RDYLGN[upper][index] - RDYLGN[lower][index]) * fraction
        )
        for index in range(3)
    )
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _bar_svg(bar: GraphBar) -> str:
    tooltip = html.escape(bar.tooltip, quote=True)
    clickable_attrs = (
        f' data-bucket-index="{bar.bucket_index}" tabindex="0" role="button"'
        if bar.clickable
        else ""
    )
    return f"""
      <rect class="rwkvp-bar" x="{bar.x:.2f}" y="{bar.y:.2f}" width="{bar.width:.2f}"
            height="{bar.height:.2f}" fill="{bar.color}" />
      <rect class="rwkv-chart-hit rwkvp-bar-hit" x="{bar.x:.2f}"
            y="{GEOMETRY.plot_top:.2f}" width="{bar.width:.2f}"
            height="{GEOMETRY.plot_height:.2f}"
            data-rwkv-chart-tooltip-text="{tooltip}" aria-label="{tooltip}"
            {clickable_attrs} />"""


_RETRIEVABILITY_CSS = """
.rwkvp-retrievability-graph .rwkv-chart {
  height: min(330px, 48vh);
}
.rwkvp-retrievability-graph .rwkv-chart-grid {
  stroke-opacity: 0.10;
}
.rwkvp-retrievability-graph .rwkv-chart-axis {
  stroke-opacity: 0.28;
}
.rwkvp-retrievability-graph .rwkv-chart-tick {
  font-size: 10px;
  opacity: 0.55;
}
.rwkvp-bar {
  rx: 1px;
  shape-rendering: crispEdges;
}
.rwkvp-bar-hit {
  cursor: default;
  fill: transparent;
  pointer-events: all;
}
.rwkvp-bar-hit[data-bucket-index] {
  cursor: pointer;
}
.rwkvp-bar-hit[data-bucket-index]:focus {
  outline: none;
  stroke: currentColor;
  stroke-opacity: 0.55;
  stroke-width: 1.5;
}
.rwkvp-retrievability-graph .rwkv-chart-summary {
  border-top: 1px solid var(--rwkv-border);
  font-size: 13px;
  gap: 6px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  padding-top: 8px;
}
.rwkvp-retrievability-graph .rwkv-chart-summary div,
.rwkvp-retrievability-graph .rwkv-chart-summary span {
  align-items: baseline;
  display: flex;
  gap: 8px;
  justify-content: center;
}
.rwkvp-retrievability-graph .rwkv-chart-summary span {
  opacity: 0.72;
}
.rwkvp-retrievability-graph .rwkv-chart-empty rect {
  opacity: 0.04;
}
"""


_BUCKET_INTERACTION_SCRIPT = """
(function() {
  function openBucket(event) {
    const index = event.currentTarget.getAttribute("data-bucket-index");
    if (index === null) {
      return;
    }
    if (!window.RWKVModal || typeof window.RWKVModal.send !== "function") {
      return;
    }
    window.RWKVModal.send("open-bucket", { bucket_index: Number(index) }).catch(() => {});
  }
  function openBucketWithKeyboard(event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openBucket(event);
    }
  }
  document.querySelectorAll(".rwkvp-bar-hit[data-bucket-index]").forEach((bar) => {
    bar.addEventListener("click", openBucket);
    bar.addEventListener("keydown", openBucketWithKeyboard);
  });
})();
"""
