from __future__ import annotations

import html
import math
from dataclasses import dataclass

from .svg_chart import (
    ChartGeometry,
    ChartMargins,
    nice_count_ticks,
    render_chart_axes,
    render_chart_fragment,
    render_chart_tooltip,
    render_count_y_ticks,
    render_empty_chart_state,
    render_model_comparison_legend,
    render_plot_background,
)
from .um_plus import UmPlusBin, UmPlusSummary, graph_domains, regression_slope

GEOMETRY = ChartGeometry(width=700, height=345, margins=ChartMargins(bottom=45))


@dataclass(frozen=True)
class UmPlusGraphModel:
    bins: tuple[UmPlusBin, ...]
    x_domain: tuple[float, float]
    y_domain: tuple[float, float]
    count_ticks: list[int]
    rwkv_slope: float | None
    fsrs_slope: float | None

    def x(self, value: float) -> float:
        low, high = self.x_domain
        return GEOMETRY.x_for_unit((value - low) / (high - low))

    def y(self, value: float) -> float:
        low, high = self.y_domain
        return GEOMETRY.y_for_unit((value - low) / (high - low))


def um_plus_graph_model(
    summary: UmPlusSummary,
    *,
    show_small_buckets: bool = False,
) -> UmPlusGraphModel:
    bins = summary.visible_bins(show_small_buckets=show_small_buckets)
    x_domain, y_domain = graph_domains(bins)
    return UmPlusGraphModel(
        bins,
        x_domain,
        y_domain,
        nice_count_ticks(max((b.count for b in bins), default=0)),
        regression_slope(bins, "rwkv"),
        regression_slope(bins, "fsrs"),
    )


def render_um_plus_graph_fragment(
    summary: UmPlusSummary,
    *,
    title: str,
    show_small_buckets: bool = False,
    has_result_set: bool = True,
    active_label: str = "RWKV",
    comparison_label: str = "FSRS",
    comparison_is_fsrs: bool = True,
) -> str:
    model = um_plus_graph_model(summary, show_small_buckets=show_small_buckets)
    legend = render_model_comparison_legend(
        rwkv_detail=f"UM+ {_score(summary.rwkv_score)} · slope {_slope(model.rwkv_slope)}",
        fsrs_detail=f"UM+ {_score(summary.fsrs_score)} · slope {_slope(model.fsrs_slope)}",
        active_label=active_label,
        comparison_label=comparison_label,
    )
    bars, hits = [], []
    for bucket in model.bins:
        low, high = bucket.bounds
        x = model.x(low)
        width = max(0.0, model.x(high) - x)
        y = GEOMETRY.y_for_count(bucket.count, max(model.count_ticks))
        small_class = " rwkvp-um-small" if bucket.small else ""
        bars.append(
            f'<rect class="rwkvp-um-volume{small_class}" x="{x:.2f}" y="{y:.2f}" '
            f'width="{width:.2f}" height="{GEOMETRY.plot_bottom - y:.2f}" />'
        )
        tooltip = html.escape(
            f"{comparison_label} − {active_label}: {_pp(bucket.disagreement)}\n"
            f"{active_label} error: {_pp(bucket.rwkv_error)}\n"
            f"{comparison_label} error: {_pp(bucket.fsrs_error)}\n"
            f"Reviews: {bucket.count:,} "
            f"({100 * bucket.count / summary.count:.1f}% of paired reviews)",
            quote=True,
        )
        hits.append(
            f'<rect class="rwkv-chart-hit rwkvp-um-hit" tabindex="0" '
            f'x="{x:.2f}" y="{GEOMETRY.plot_top:.2f}" width="{width:.2f}" '
            f'height="{GEOMETRY.plot_height:.2f}" '
            f'data-rwkv-chart-tooltip-text="{tooltip}" aria-label="{tooltip}" />'
        )
    series = []
    for name in ("rwkv", "fsrs"):
        points = [
            (bucket, model.x(bucket.disagreement), model.y(getattr(bucket, f"{name}_error")))
            for bucket in model.bins
        ]
        lines = []
        for (a, ax, ay), (b, bx, by) in zip(points, points[1:], strict=False):
            small_class = " rwkvp-um-small" if a.small or b.small else ""
            lines.append(
                f'<line class="rwkvp-um-line{small_class}" '
                f'x1="{ax:.2f}" y1="{ay:.2f}" x2="{bx:.2f}" y2="{by:.2f}" />'
            )
        for bucket, x, y in points:
            radius = max(2.5, math.sqrt(1024 * bucket.count / summary.count / math.pi))
            small_class = " rwkvp-um-small" if bucket.small else ""
            lines.append(
                f'<circle class="rwkvp-um-point{small_class}" '
                f'cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" />'
            )
        series.append(f'<g class="rwkvp-um-series rwkvp-um-{name}">{"".join(lines)}</g>')
    zero_x = ""
    if model.x_domain[0] <= 0 <= model.x_domain[1]:
        zero_x = (
            f'<line class="rwkvp-um-zero-disagreement" x1="{model.x(0):.2f}" '
            f'x2="{model.x(0):.2f}" y1="{GEOMETRY.plot_top:.2f}" y2="{GEOMETRY.plot_bottom:.2f}" />'
        )
    empty = ""
    if not model.bins:
        label = (
            f"Click Calculate to compare {active_label} and {comparison_label}"
            if not has_result_set
            else "No paired predictions for these reviews"
            if not summary.count
            else "No buckets with 200 reviews. Enable Show small buckets."
        )
        empty = render_empty_chart_state(GEOMETRY, label=label)
    omitted = (
        f" Omitted without paired predictions: {summary.unpaired_count:,}."
        if summary.unpaired_count
        else ""
    )
    hidden = sum(b.count for b in summary.bins if b.small) if not show_small_buckets else 0
    prediction_note = (
        "RWKV uses saved review-time predictions; FSRS is reconstructed with current "
        "preset parameters."
        if comparison_is_fsrs
        else "Both generations use saved review-time predictions."
    )
    aria_label = f"UM+ comparison of {active_label} and {comparison_label} prediction errors"
    axis_label = f"{comparison_label} − {active_label} prediction (pp)"
    markup = f'''
<section class="rwkvp-embedded-graph rwkvp-um-plus-graph rwkvp-model-comparison"
         data-rwkv-chart aria-labelledby="rwkvp-um-plus-title">
  <h2 class="rwkv-section-title" id="rwkvp-um-plus-title">{html.escape(title)}</h2>
  <p class="rwkv-section-intro rwkvp-subtitle">Prediction error where the models disagree.
    Closer to zero is better; positive means overprediction, negative means underprediction.</p>
  {legend}
  <svg class="rwkv-chart" viewBox="{GEOMETRY.view_box}" role="img"
       aria-label="{html.escape(aria_label, quote=True)}">
    {render_plot_background(GEOMETRY)}
    {"".join(bars)}
    {_signed_ticks(model, axis="x")}
    {_signed_ticks(model, axis="y")}
    {render_count_y_ticks(model.count_ticks, max(model.count_ticks), GEOMETRY, side="right")}
    <line class="rwkvp-um-zero-error" x1="{GEOMETRY.plot_left:.2f}" x2="{GEOMETRY.plot_right:.2f}"
          y1="{model.y(0):.2f}" y2="{model.y(0):.2f}" />
    {zero_x}
    {"".join(series)}
    {render_chart_axes(GEOMETRY, right_axis=True)}
    <text class="rwkv-chart-axis-title" x="{GEOMETRY.width / 2:.2f}"
          y="{GEOMETRY.height - 6:.2f}" text-anchor="middle">{html.escape(axis_label)}</text>
    <text class="rwkv-chart-axis-title" transform="rotate(-90)"
          x="{-(GEOMETRY.plot_top + GEOMETRY.plot_bottom) / 2:.2f}"
          y="16" text-anchor="middle">Predicted − actual recall (pp)</text>
    <text class="rwkv-chart-axis-title" transform="rotate(90)"
          x="{(GEOMETRY.plot_top + GEOMETRY.plot_bottom) / 2:.2f}"
          y="{-GEOMETRY.width + 14:.2f}" text-anchor="middle">Reviews</text>
    {"".join(hits)}
    {empty}
  </svg>
  {render_chart_tooltip()}
  <p class="rwkv-section-intro">Paired reviews: {summary.count:,}
    · Hidden small-bucket reviews: {hidden:,}.
    {omitted}</p>
  <p class="rwkv-section-intro">Lower UM+ is better. Scores use all buckets;
    slopes use visible buckets. Bubble area shows each bucket’s share of paired reviews;
    bars show counts. pp = percentage points.</p>
  <p class="rwkv-section-intro">{prediction_note}
    This is a historical comparison, not a held-out test.</p>
</section>'''
    return render_chart_fragment(markup, feature_css=_UM_PLUS_CSS)


def _signed_ticks(model: UmPlusGraphModel, *, axis: str) -> str:
    domain = model.x_domain if axis == "x" else model.y_domain
    ticks = []
    for tenth in range(round(domain[0] * 10), round(domain[1] * 10) + 1):
        label = f"{tenth * 10:+d}" if tenth else "0"
        if axis == "x":
            x = model.x(tenth / 10)
            ticks.append(
                f'<line class="rwkv-chart-grid" x1="{x:.2f}" x2="{x:.2f}" '
                f'y1="{GEOMETRY.plot_top:.2f}" y2="{GEOMETRY.plot_bottom:.2f}" />'
                f'<text class="rwkv-chart-tick" x="{x:.2f}" y="{GEOMETRY.plot_bottom + 18:.2f}" '
                f'text-anchor="middle">{label}</text>'
            )
        else:
            y = model.y(tenth / 10)
            ticks.append(
                f'<line class="rwkv-chart-grid" x1="{GEOMETRY.plot_left:.2f}" '
                f'x2="{GEOMETRY.plot_right:.2f}" y1="{y:.2f}" y2="{y:.2f}" />'
                f'<text class="rwkv-chart-tick" x="{GEOMETRY.plot_left - 10:.2f}" y="{y + 4:.2f}" '
                f'text-anchor="end">{label}</text>'
            )
    return f'<g class="rwkvp-um-{axis}-ticks">{"".join(ticks)}</g>'


def _score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _slope(value: float | None) -> str:
    if value is None:
        return "n/a"
    return "0.000" if abs(value) < 0.0005 else f"{value:+.3f}"


def _pp(value: float) -> str:
    return "0.0 pp" if abs(value) < 0.0005 else f"{value * 100:+.1f} pp"


_UM_PLUS_CSS = """
.rwkvp-um-plus-graph .rwkv-chart-grid {
  stroke-dasharray: 2 4;
  stroke-width: 0.6;
  opacity: 0.25;
}
.rwkvp-um-volume {
  fill: currentColor;
  opacity: 0.12;
}
.rwkvp-um-volume.rwkvp-um-small { opacity: 0.04; }
.rwkvp-um-zero-error, .rwkvp-um-zero-disagreement { stroke: currentColor; }
.rwkvp-um-zero-error { stroke-width: 1.5; }
.rwkvp-um-zero-disagreement { opacity: 0.35; }
.rwkvp-um-rwkv { color: #3b82f6; }
.rwkvp-um-fsrs { color: var(--rwkvp-fsrs-color); }
.rwkvp-um-line { stroke: currentColor; stroke-width: 2; }
.rwkvp-um-fsrs .rwkvp-um-line { stroke-width: 1.5; }
.rwkvp-um-line.rwkvp-um-small { stroke-dasharray: 3 3; opacity: 0.35; }
.rwkvp-um-point { fill: currentColor; opacity: 0.45; }
.rwkvp-um-point.rwkvp-um-small { opacity: 0.2; }
.rwkvp-um-hit { fill: transparent; }
.rwkvp-um-hit:focus { outline: 2px solid var(--rwkv-accent); }
.rwkvp-um-plus-graph .rwkv-chart-axis--secondary { opacity: 0.55; }
"""
