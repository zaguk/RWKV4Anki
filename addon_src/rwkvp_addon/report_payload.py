"""Plain presentation data for Svelte-owned reports; no browser or Anki dependency."""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .report_html import ReportMetricStyle
from .svelte_ui import render_svelte_document


def chart_markup(fragment: str) -> str:
    """Retain trusted pure-Python SVG geometry; Svelte owns chart event lifecycle."""
    return re.sub(r"<script>.*?</script>", "", fragment, flags=re.DOTALL)


def metric_cell(
    primary: str, details: Sequence[str], *, style: ReportMetricStyle,
    classes: Sequence[str] = ("metric",), change_percent: str | None = None,
) -> dict[str, Any]:
    names = list(classes)
    if style.css_class and style.css_class not in names:
        names.append(style.css_class)
    names.append("rwkv-report-metric")
    return dict(
        kind="metric", text=primary, details=list(details), cssClass=" ".join(names),
        tone=style.tone.value, background=style.background, foreground=style.foreground,
        changePercent=change_percent,
    )


def value_cell(
    text: str, *, header: bool = False, css_class: str = "rwkv-number",
    tooltip: str | None = None,
) -> dict[str, Any]:
    return dict(text=text, header=header, cssClass=css_class, tooltip=tooltip)


def report_document(
    *, title: str, subtitle: str, sections: Sequence[dict[str, Any]], notes: Sequence[str],
    card_label: str, root_classes: str, is_dark: bool, generation: int,
    close_footer: bool = False,
) -> str:
    return render_svelte_document(
        "report", dict(title=title, intro=subtitle, cardLabel=card_label,
                       sections=list(sections), notes=list(notes), closeFooter=close_footer),
        title=title, intro=subtitle, generation=generation, is_dark=is_dark, width="wide",
        root_extra_classes=root_classes,
        enter_action="dialog-close" if close_footer else None,
        enter_payload={"outcome": "accept"} if close_footer else None,
    )
