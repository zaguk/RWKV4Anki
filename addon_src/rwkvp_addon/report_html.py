from __future__ import annotations

import html
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .modal_html import (
    render_card,
    render_close_footer,
    render_footnotes,
    render_modal_document,
)

__all__ = (
    "MetricDirection",
    "MetricTone",
    "ReportColor",
    "ReportMetricStyle",
    "ReportPalette",
    "interpolate_hex_color",
    "ranked_metric_style",
    "relative_metric_percent",
    "relative_metric_style",
    "render_report_document",
    "render_report_footnotes",
    "render_report_metric_cell",
    "report_palette",
    "semantic_metric_style",
)

_ATTRIBUTE_NAME = re.compile(r"(?:aria|data)-[a-z0-9_.:-]+\Z")


class MetricDirection(str, Enum):
    """Declare which numeric direction represents a better result."""

    LOWER_IS_BETTER = "lower-is-better"
    HIGHER_IS_BETTER = "higher-is-better"


class MetricTone(str, Enum):
    """Semantic result color independent of the numeric sign."""

    GAIN = "gain"
    NEUTRAL = "neutral"
    LOSS = "loss"


@dataclass(frozen=True)
class ReportColor:
    background: str
    foreground: str


@dataclass(frozen=True)
class ReportPalette:
    gain: ReportColor
    neutral: ReportColor
    loss: ReportColor

    def color(self, tone: MetricTone) -> ReportColor:
        if tone is MetricTone.GAIN:
            return self.gain
        if tone is MetricTone.LOSS:
            return self.loss
        return self.neutral


@dataclass(frozen=True)
class ReportMetricStyle:
    tone: MetricTone
    css_class: str
    background: str
    foreground: str


def report_palette(
    *,
    is_dark: bool,
    neutral: Literal["gray", "warm"] = "gray",
    foreground: Literal["metric", "tone"] = "metric",
) -> ReportPalette:
    """Return one of the deliberately supported report palettes.

    A warm neutral is appropriate for an ordered slow-to-fast heat scale. A
    gray neutral is appropriate for baseline comparisons. Tone foregrounds are
    reserved for larger explanatory panels; compact metric cells use the
    higher-contrast common metric foreground.
    """

    if neutral not in {"gray", "warm"}:
        raise ValueError(f"Unsupported report neutral palette: {neutral!r}.")
    if foreground not in {"metric", "tone"}:
        raise ValueError(f"Unsupported report foreground palette: {foreground!r}.")

    if is_dark:
        backgrounds = {
            MetricTone.GAIN: "#1c5631",
            MetricTone.NEUTRAL: "#454a51" if neutral == "gray" else "#5b4e20",
            MetricTone.LOSS: "#69262a",
        }
        metric_foreground = "#f6f7f8"
        tone_foregrounds = {
            MetricTone.GAIN: "#e0f7e6",
            MetricTone.NEUTRAL: "#e8eaed" if neutral == "gray" else "#fff1bc",
            MetricTone.LOSS: "#ffe1e1",
        }
    else:
        backgrounds = {
            MetricTone.GAIN: "#c7eecf",
            MetricTone.NEUTRAL: "#eceff2" if neutral == "gray" else "#f6ebbe",
            MetricTone.LOSS: "#f7c7c7",
        }
        metric_foreground = "#1d2329"
        tone_foregrounds = {
            MetricTone.GAIN: "#174f29",
            MetricTone.NEUTRAL: "#222222" if neutral == "gray" else "#665713",
            MetricTone.LOSS: "#762024",
        }
    def foreground_for(tone: MetricTone) -> str:
        if foreground == "tone":
            return tone_foregrounds[tone]
        if neutral == "gray" and tone is MetricTone.NEUTRAL:
            # State-comparison baselines historically use ordinary foreground
            # text; only a meaningful gain/loss switches to the metric color.
            return tone_foregrounds[tone]
        return metric_foreground

    return ReportPalette(
        gain=ReportColor(backgrounds[MetricTone.GAIN], foreground_for(MetricTone.GAIN)),
        neutral=ReportColor(
            backgrounds[MetricTone.NEUTRAL],
            foreground_for(MetricTone.NEUTRAL),
        ),
        loss=ReportColor(backgrounds[MetricTone.LOSS], foreground_for(MetricTone.LOSS)),
    )


def relative_metric_percent(
    baseline_value: float | None,
    value: float | None,
    *,
    direction: MetricDirection,
) -> float | None:
    """Return a signed semantic change where positive always means better."""

    baseline = _finite_float(baseline_value)
    measured = _finite_float(value)
    if baseline is None or measured is None:
        return None
    if baseline == 0.0:
        return 0.0 if measured == 0.0 else None
    numeric_change = (measured - baseline) / abs(baseline) * 100.0
    return (
        -numeric_change
        if direction is MetricDirection.LOWER_IS_BETTER
        else numeric_change
    )


def semantic_metric_style(
    tone: MetricTone,
    *,
    palette: ReportPalette,
    css_class: str | None = None,
) -> ReportMetricStyle:
    color = palette.color(tone)
    return ReportMetricStyle(
        tone=tone,
        css_class=tone.value if css_class is None else css_class,
        background=color.background,
        foreground=color.foreground,
    )


def relative_metric_style(
    improvement_percent: float | None,
    *,
    palette: ReportPalette,
    full_intensity_at: float = 10.0,
    neutral_tolerance: float = 0.0005,
) -> ReportMetricStyle:
    """Color a semantic improvement/regression relative to the neutral color."""

    change = _finite_float(improvement_percent)
    if change is None or math.isclose(change, 0.0, abs_tol=neutral_tolerance):
        return semantic_metric_style(MetricTone.NEUTRAL, palette=palette)
    tone = MetricTone.GAIN if change > 0 else MetricTone.LOSS
    limit = float(full_intensity_at)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("full_intensity_at must be a finite positive number.")
    amount = min(1.0, abs(change) / limit)
    target = palette.color(tone)
    return ReportMetricStyle(
        tone=tone,
        css_class=tone.value,
        background=interpolate_hex_color(
            palette.neutral.background,
            target.background,
            amount,
        ),
        foreground=target.foreground,
    )


def ranked_metric_style(
    value: float,
    values: Sequence[float],
    *,
    direction: MetricDirection,
    palette: ReportPalette,
    best_class: str = "best",
    worst_class: str = "worst",
    intermediate_class: str = "intermediate",
    equal_class: str = "equal",
) -> ReportMetricStyle:
    """Color one value within an ordered set using an explicit quality direction."""

    measured = _finite_float(value)
    finite_values = tuple(item for raw in values if (item := _finite_float(raw)) is not None)
    if measured is None or not finite_values:
        return semantic_metric_style(
            MetricTone.NEUTRAL,
            palette=palette,
            css_class=equal_class,
        )
    minimum = min(finite_values)
    maximum = max(finite_values)
    if math.isclose(minimum, maximum, rel_tol=1e-12, abs_tol=1e-12):
        normalized = 0.5
        css_class = equal_class
    else:
        lower_to_higher = min(1.0, max(0.0, (measured - minimum) / (maximum - minimum)))
        normalized = (
            lower_to_higher
            if direction is MetricDirection.LOWER_IS_BETTER
            else 1.0 - lower_to_higher
        )
        if math.isclose(normalized, 0.0, rel_tol=1e-12, abs_tol=1e-12):
            css_class = best_class
        elif math.isclose(normalized, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            css_class = worst_class
        else:
            css_class = intermediate_class

    if normalized <= 0.5:
        tone = MetricTone.GAIN if normalized < 0.5 else MetricTone.NEUTRAL
        background = interpolate_hex_color(
            palette.gain.background,
            palette.neutral.background,
            normalized * 2.0,
        )
    else:
        tone = MetricTone.LOSS
        background = interpolate_hex_color(
            palette.neutral.background,
            palette.loss.background,
            (normalized - 0.5) * 2.0,
        )
    return ReportMetricStyle(
        tone=tone,
        css_class=css_class,
        background=background,
        foreground=palette.color(tone).foreground,
    )


def interpolate_hex_color(start: str, end: str, amount: float) -> str:
    """Interpolate two ``#RRGGBB`` colors with a clamped amount."""

    start_rgb = _parse_hex_color(start)
    end_rgb = _parse_hex_color(end)
    numeric_amount = float(amount)
    if not math.isfinite(numeric_amount):
        raise ValueError("Color interpolation amount must be finite.")
    bounded = min(1.0, max(0.0, numeric_amount))
    channels = tuple(
        round(start_rgb[index] + (end_rgb[index] - start_rgb[index]) * bounded)
        for index in range(3)
    )
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def render_report_metric_cell(
    primary: str,
    details: Sequence[str] = (),
    *,
    classes: Sequence[str] = ("metric",),
    style: ReportMetricStyle | None = None,
    data_attributes: Mapping[str, str] | None = None,
) -> str:
    """Render a primary metric with zero or more escaped secondary detail rows."""

    class_names = [part for value in classes for part in value.split() if part]
    if style is not None and style.css_class and style.css_class not in class_names:
        class_names.append(style.css_class)
    if "rwkv-report-metric" not in class_names:
        class_names.append("rwkv-report-metric")
    attributes = [f'class="{html.escape(" ".join(class_names), quote=True)}"']
    if style is not None:
        attributes.append(f'data-rwkv-metric-tone="{style.tone.value}"')
        attributes.append(
            'style="background-color: '
            f'{html.escape(style.background, quote=True)}; color: '
            f'{html.escape(style.foreground, quote=True)}"'
        )
    for name, value in (data_attributes or {}).items():
        if not _ATTRIBUTE_NAME.fullmatch(name):
            raise ValueError(f"Unsupported report metric attribute: {name!r}.")
        attributes.append(f'{name}="{html.escape(str(value), quote=True)}"')
    secondary = "".join(f"<span>{html.escape(str(detail))}</span>" for detail in details)
    return (
        f"<td {' '.join(attributes)}><strong>{html.escape(str(primary))}</strong>"
        f"{secondary}</td>"
    )


def render_report_footnotes(notes: Sequence[str]) -> str:
    """Render report notes through the canonical modal footnote component."""

    return render_footnotes(tuple(notes))


def render_report_document(
    *,
    title: str,
    subtitle: str,
    report_html: str,
    notes: Sequence[str],
    card_aria_label: str,
    root_extra_classes: str,
    is_dark: bool = False,
    generation: int = 1,
    include_close_footer: bool = True,
) -> str:
    """Render the canonical report document and result card shell.

    Immutable reports normally retain an explicit Close action.  Compact reports
    that are naturally dismissed through the title bar or Escape can omit that
    otherwise redundant footer.
    """

    body_html = "\n".join(
        (
            render_card(
                f'<div class="rwkv-table-wrap">{report_html}</div>',
                aria_label=card_aria_label,
            ),
            render_report_footnotes(notes),
        )
    )
    return render_modal_document(
        title=title,
        intro=subtitle,
        body_html=body_html,
        generation=generation,
        footer_html=render_close_footer() if include_close_footer else "",
        is_dark=is_dark,
        width="wide",
        root_extra_classes=root_extra_classes,
        enter_action="dialog-close" if include_close_footer else None,
        enter_payload={"outcome": "accept"} if include_close_footer else None,
    )


def _parse_hex_color(value: str) -> tuple[int, int, int]:
    text = str(value)
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", text):
        raise ValueError(f"Report color must use #RRGGBB: {value!r}.")
    return tuple(int(text[index : index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None
