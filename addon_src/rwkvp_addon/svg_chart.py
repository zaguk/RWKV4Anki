from __future__ import annotations

import html
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

__all__ = (
    "DEFAULT_CHART_GEOMETRY",
    "PERCENT_TICKS",
    "ChartGeometry",
    "ChartMargins",
    "clamp_unit",
    "finite_number",
    "finite_number_or_none",
    "nice_count_ticks",
    "render_chart_axes",
    "render_chart_fragment",
    "render_chart_tooltip",
    "render_count_y_ticks",
    "render_empty_chart_state",
    "render_percent_x_ticks",
    "render_percent_y_ticks",
    "render_plot_background",
)

PERCENT_TICKS = (0, 20, 40, 60, 80, 100)


def finite_number(value: object, *, fallback: float = 0.0) -> float:
    """Return a finite float, substituting a deliberately finite fallback."""

    try:
        fallback_value = float(fallback)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Finite-number fallback must be numeric.") from error
    if not math.isfinite(fallback_value):
        raise ValueError("Finite-number fallback must be finite.")
    result = finite_number_or_none(value)
    return fallback_value if result is None else result


def finite_number_or_none(value: object) -> float | None:
    """Return a finite float, or ``None`` when the input is not finite numeric data."""

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def clamp_unit(value: object, *, fallback: float = 0.0) -> float:
    """Return a finite value constrained to the inclusive unit interval."""

    return min(1.0, max(0.0, finite_number(value, fallback=fallback)))


@dataclass(frozen=True)
class ChartMargins:
    left: float = 70
    right: float = 70
    top: float = 20
    bottom: float = 35

    def __post_init__(self) -> None:
        for name in ("left", "right", "top", "bottom"):
            value = finite_number(getattr(self, name), fallback=-1)
            if value < 0:
                raise ValueError(f"Chart margin {name} must be a finite non-negative number.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ChartGeometry:
    width: float = 600
    height: float = 250
    margins: ChartMargins = ChartMargins()

    def __post_init__(self) -> None:
        width = finite_number(self.width, fallback=-1)
        height = finite_number(self.height, fallback=-1)
        if width <= 0 or height <= 0:
            raise ValueError("Chart width and height must be finite positive numbers.")
        if self.margins.left + self.margins.right >= width:
            raise ValueError("Chart horizontal margins must leave a positive plot width.")
        if self.margins.top + self.margins.bottom >= height:
            raise ValueError("Chart vertical margins must leave a positive plot height.")
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    @property
    def plot_left(self) -> float:
        return self.margins.left

    @property
    def plot_right(self) -> float:
        return self.width - self.margins.right

    @property
    def plot_top(self) -> float:
        return self.margins.top

    @property
    def plot_bottom(self) -> float:
        return self.height - self.margins.bottom

    @property
    def plot_width(self) -> float:
        return self.plot_right - self.plot_left

    @property
    def plot_height(self) -> float:
        return self.plot_bottom - self.plot_top

    @property
    def view_box(self) -> str:
        return f"0 0 {_svg_number(self.width)} {_svg_number(self.height)}"

    def x_for_unit(self, value: float, *, fallback: float = 0.0) -> float:
        return self.plot_left + self.plot_width * clamp_unit(value, fallback=fallback)

    def y_for_unit(self, value: float, *, fallback: float = 0.0) -> float:
        return self.plot_bottom - self.plot_height * clamp_unit(value, fallback=fallback)

    def y_for_count(self, count: int, axis_maximum: int) -> float:
        if axis_maximum <= 0:
            return self.plot_bottom
        ratio = finite_number(count / axis_maximum, fallback=0.0)
        return self.y_for_unit(ratio)


DEFAULT_CHART_GEOMETRY = ChartGeometry()


def nice_count_ticks(max_count: int, *, target_intervals: int = 4) -> list[int]:
    """Return human-readable count ticks that cover ``max_count``.

    Integer arithmetic keeps the result stable even for counts too large to
    round-trip through a float.
    """

    maximum = max(0, int(max_count))
    intervals = int(target_intervals)
    if intervals <= 0:
        raise ValueError("target_intervals must be positive.")
    if maximum <= 0:
        return [0]
    if maximum <= intervals:
        return list(range(0, maximum + 1))

    required_step = (maximum + intervals - 1) // intervals
    magnitude = 10 ** (len(str(required_step)) - 1)
    if required_step <= magnitude:
        step = magnitude
    elif required_step <= 2 * magnitude:
        step = 2 * magnitude
    elif required_step <= 5 * magnitude:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    top = ((maximum + step - 1) // step) * step
    return list(range(0, top + step, step))


def render_plot_background(geometry: ChartGeometry = DEFAULT_CHART_GEOMETRY) -> str:
    return (
        '<rect class="rwkv-chart-plot-bg" '
        f'x="{_svg_number(geometry.plot_left)}" '
        f'y="{_svg_number(geometry.plot_top)}" '
        f'width="{_svg_number(geometry.plot_width)}" '
        f'height="{_svg_number(geometry.plot_height)}" />'
    )


def render_chart_axes(
    geometry: ChartGeometry = DEFAULT_CHART_GEOMETRY,
    *,
    right_axis: bool = False,
) -> str:
    axes = [
        _axis_line(
            geometry.plot_left,
            geometry.plot_bottom,
            geometry.plot_right,
            geometry.plot_bottom,
        ),
        _axis_line(
            geometry.plot_left,
            geometry.plot_top,
            geometry.plot_left,
            geometry.plot_bottom,
        ),
    ]
    if right_axis:
        axes.append(
            _axis_line(
                geometry.plot_right,
                geometry.plot_top,
                geometry.plot_right,
                geometry.plot_bottom,
                extra_classes="rwkv-chart-axis--secondary",
            )
        )
    return "\n".join(axes)


def render_percent_x_ticks(
    geometry: ChartGeometry = DEFAULT_CHART_GEOMETRY,
    *,
    ticks: Iterable[int] = PERCENT_TICKS,
    group_classes: str = "",
) -> str:
    rows: list[str] = []
    for tick in ticks:
        value = int(tick)
        x = geometry.x_for_unit(value / 100)
        rows.append(
            f'''      <line class="rwkv-chart-grid" x1="{x:.2f}"
            y1="{_svg_number(geometry.plot_top)}" x2="{x:.2f}"
            y2="{_svg_number(geometry.plot_bottom)}" />
      <text class="rwkv-chart-tick" x="{x:.2f}"
            y="{geometry.plot_bottom + 18:.2f}" text-anchor="middle">{value}%</text>'''
        )
    return _tick_group("rwkv-chart-x-ticks", group_classes, rows)


def render_percent_y_ticks(
    geometry: ChartGeometry = DEFAULT_CHART_GEOMETRY,
    *,
    ticks: Iterable[int] = PERCENT_TICKS,
    group_classes: str = "",
) -> str:
    rows: list[str] = []
    for tick in ticks:
        value = int(tick)
        y = geometry.y_for_unit(value / 100)
        rows.append(
            f'''      <line class="rwkv-chart-grid"
            x1="{_svg_number(geometry.plot_left)}"
            y1="{y:.2f}" x2="{_svg_number(geometry.plot_right)}" y2="{y:.2f}" />
      <text class="rwkv-chart-tick"
            x="{geometry.plot_left - 10:.2f}" y="{y + 4:.2f}"
            text-anchor="end">{value}%</text>'''
        )
    return _tick_group("rwkv-chart-y-ticks", group_classes, rows)


def render_count_y_ticks(
    ticks: Iterable[int],
    axis_maximum: int,
    geometry: ChartGeometry = DEFAULT_CHART_GEOMETRY,
    *,
    side: Literal["left", "right"] = "left",
    include_grid: bool = False,
    group_classes: str = "",
) -> str:
    if side not in {"left", "right"}:
        raise ValueError("Count ticks must use the left or right side.")
    x = geometry.plot_left - 10 if side == "left" else geometry.plot_right + 8
    anchor = "end" if side == "left" else "start"
    rows: list[str] = []
    for tick in ticks:
        value = max(0, int(tick))
        y = geometry.y_for_count(value, axis_maximum)
        grid = ""
        if include_grid:
            grid = (
                f'      <line class="rwkv-chart-grid" '
                f'x1="{_svg_number(geometry.plot_left)}" y1="{y:.2f}" '
                f'x2="{_svg_number(geometry.plot_right)}" y2="{y:.2f}" />\n'
            )
        rows.append(
            f'''{grid}      <text class="rwkv-chart-tick" x="{x:.2f}"
            y="{y + 4:.2f}" text-anchor="{anchor}">{value:,}</text>'''
        )
    return _tick_group("rwkv-chart-count-ticks", group_classes, rows)


def render_empty_chart_state(
    geometry: ChartGeometry = DEFAULT_CHART_GEOMETRY,
    *,
    label: str = "No data",
) -> str:
    return f'''    <g class="rwkv-chart-empty">
      <rect x="{_svg_number(geometry.plot_left)}" y="{_svg_number(geometry.plot_top)}"
            width="{_svg_number(geometry.plot_width)}"
            height="{_svg_number(geometry.plot_height)}" />
      <text x="{(geometry.plot_left + geometry.plot_right) / 2:.2f}"
            y="{(geometry.plot_top + geometry.plot_bottom) / 2:.2f}"
            text-anchor="middle">{html.escape(label)}</text>
    </g>'''


def render_chart_tooltip() -> str:
    return (
        '<div class="rwkv-chart-tooltip" '
        'data-rwkv-chart-tooltip hidden></div>'
    )


def render_chart_fragment(
    markup: str,
    *,
    feature_css: str = "",
    feature_script: str = "",
) -> str:
    css = _BASE_CHART_CSS.strip()
    if feature_css.strip():
        css += "\n" + feature_css.strip()
    script = _IMMEDIATE_TOOLTIP_SCRIPT.strip()
    if feature_script.strip():
        script += "\n" + feature_script.strip()
    return f"""
<style>
{css}
</style>
{markup.strip()}
<script>
{script}
</script>
"""


def _axis_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    extra_classes: str = "",
) -> str:
    classes = _classes("rwkv-chart-axis", extra_classes)
    return (
        f'<line class="{classes}" x1="{_svg_number(x1)}" y1="{_svg_number(y1)}" '
        f'x2="{_svg_number(x2)}" y2="{_svg_number(y2)}" />'
    )


def _tick_group(base_classes: str, extra_classes: str, rows: list[str]) -> str:
    classes = _classes(base_classes, extra_classes)
    return f'''    <g class="{classes}">
{chr(10).join(rows)}
    </g>'''


def _classes(*values: str) -> str:
    return " ".join(part for value in values for part in value.split() if part)


def _svg_number(value: float) -> str:
    number = finite_number(value)
    return str(int(number)) if number.is_integer() else f"{number:.2f}"


_BASE_CHART_CSS = """
.rwkvp-embedded-graph {
  color: var(--rwkv-fg);
  padding: 4px 6px 2px;
}
.rwkvp-subtitle {
  font-size: 13px;
  margin-bottom: 8px;
}
.rwkv-chart {
  display: block;
  height: auto;
  max-width: 100%;
  overflow: visible;
  width: 100%;
}
.rwkv-chart-plot-bg {
  fill: transparent;
}
.rwkv-chart-grid {
  stroke: currentColor;
  stroke-width: 1;
}
.rwkv-chart-axis {
  stroke: currentColor;
  stroke-width: 1;
}
.rwkv-chart-tick,
.rwkv-chart-axis-title {
  fill: currentColor;
  font-size: 11px;
  opacity: 0.72;
}
.rwkv-chart-tooltip {
  background: rgba(32, 33, 36, 0.95);
  border-radius: 5px;
  color: #fff;
  font-size: 12px;
  left: 0;
  line-height: 1.35;
  max-width: 260px;
  padding: 6px 8px;
  pointer-events: none;
  position: fixed;
  top: 0;
  white-space: pre-line;
  z-index: 9999;
}
.rwkv-chart-summary {
  display: grid;
  margin-top: 8px;
}
.rwkv-chart-summary strong {
  font-weight: 600;
}
.rwkv-chart-empty rect {
  fill: currentColor;
  opacity: 0.07;
}
.rwkv-chart-empty text {
  fill: currentColor;
  font-size: 16px;
  opacity: 0.72;
}
"""


_IMMEDIATE_TOOLTIP_SCRIPT = """
(function() {
  document.querySelectorAll("[data-rwkv-chart]").forEach((chart) => {
    const tooltip = chart.querySelector("[data-rwkv-chart-tooltip]");
    if (!tooltip) {
      return;
    }
    function coordinates(event) {
      const bounds = event.currentTarget.getBoundingClientRect();
      return {
        x: Number.isFinite(event.clientX) ? event.clientX : bounds.right,
        y: Number.isFinite(event.clientY) ? event.clientY : bounds.bottom,
      };
    }
    function move(event) {
      const point = coordinates(event);
      const tooltipBounds = tooltip.getBoundingClientRect();
      const viewportWidth = Number(window.innerWidth) || 0;
      const viewportHeight = Number(window.innerHeight) || 0;
      let left = point.x + 12;
      let top = point.y + 12;
      if (viewportWidth > 0) {
        left = Math.max(8, Math.min(left, viewportWidth - tooltipBounds.width - 8));
      }
      if (viewportHeight > 0) {
        top = Math.max(8, Math.min(top, viewportHeight - tooltipBounds.height - 8));
      }
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${top}px`;
    }
    function show(event) {
      const text = event.currentTarget.getAttribute("data-rwkv-chart-tooltip-text") || "";
      if (!text) {
        return;
      }
      tooltip.textContent = text;
      tooltip.hidden = false;
      move(event);
    }
    function hide() {
      tooltip.hidden = true;
    }
    chart.querySelectorAll("[data-rwkv-chart-tooltip-text]").forEach((point) => {
      point.addEventListener("mouseenter", show);
      point.addEventListener("mousemove", move);
      point.addEventListener("mouseleave", hide);
      point.addEventListener("blur", hide);
    });
  });
})();
"""
