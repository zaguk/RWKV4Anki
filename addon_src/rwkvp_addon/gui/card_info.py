from __future__ import annotations

import json
from typing import Any

from ..addon_config import (
    addon_config_for_mw,
    calculate_forgetting_curves,
    card_info_forgetting_curve_graph_enabled,
    card_info_forgetting_curve_graph_lower_bound_percent,
    card_info_intervals_enabled,
    card_info_retrievability_auto_refresh_enabled,
    card_info_retrievability_enabled,
    card_info_rwkv_enabled,
)
from ..card_info_retrievability import (
    CardInfoForgettingCurveGraph,
    CardInfoRetrievabilityValues,
    format_retrievability_percent,
    format_rwkv_interval,
    rwkv_card_info_retrievability,
)
from ..runtime import manager_for_mw
from ..vendor_bootstrap import require_rwkv_interval, require_rwkv_probability

_MESSAGE_PREFIX = "rwkvpCardInfo:"
_IMMEDIATE_REFRESH_MESSAGE_PREFIX = "rwkvpCardInfoImmediate:"
_IMMEDIATE_REFRESH_INTERVAL_MILLISECONDS = 5_000
_CARD_INFO_KIND = "browser card info"


def inject_card_info_rwkv_rows(webview) -> None:
    if _webview_kind_value(webview) != _CARD_INFO_KIND:
        return
    try:
        from aqt import mw

        config = addon_config_for_mw(mw)
        if not card_info_rwkv_enabled(config):
            return
    except Exception:
        return
    webview.eval(
        _injected_card_info_script(
            _anchor_labels(),
            auto_refresh_immediate=card_info_retrievability_auto_refresh_enabled(config),
        )
    )


def handle_card_info_rwkv_message(
    handled: tuple[bool, Any],
    message: str,
    _context: Any,
) -> tuple[bool, Any]:
    immediate_refresh = message.startswith(_IMMEDIATE_REFRESH_MESSAGE_PREFIX)
    if immediate_refresh:
        prefix = _IMMEDIATE_REFRESH_MESSAGE_PREFIX
    elif message.startswith(_MESSAGE_PREFIX):
        prefix = _MESSAGE_PREFIX
    else:
        return handled

    try:
        card_id = int(message[len(prefix) :])
    except ValueError:
        return True, _unavailable_card_info_payload(immediate_refresh)

    try:
        from aqt import mw

        config = addon_config_for_mw(mw)
        if immediate_refresh and not card_info_retrievability_auto_refresh_enabled(config):
            return True, {"enabled": False}
        if not card_info_rwkv_enabled(config):
            return True, {"available": False}
        runtime = _active_card_info_runtime()
        if runtime is None:
            return True, _unavailable_card_info_payload(immediate_refresh)
        include_immediate = card_info_retrievability_enabled(config)
        include_forgetting_curve = calculate_forgetting_curves(config)
        include_intervals = card_info_intervals_enabled(config)
        include_graph = card_info_forgetting_curve_graph_enabled(config)
        if not _runtime_contains_card(runtime, card_id):
            if immediate_refresh:
                return True, _immediate_refresh_payload(None)
            return True, _missing_card_info_payload(
                include_immediate=include_immediate,
                include_forgetting_curve=include_forgetting_curve,
                include_intervals=include_intervals,
            )
        manager = manager_for_mw(mw)
        if immediate_refresh:
            values = rwkv_card_info_retrievability(
                mw.col,
                manager,
                card_id,
                runtime=runtime,
                include_immediate=True,
                include_forgetting_curve=False,
                include_intervals=False,
                include_forgetting_curve_graph=False,
            )
            if values is None:
                return True, _unavailable_card_info_payload(True)
            return True, _immediate_refresh_payload(values.immediate)
        graph_lower_bound = card_info_forgetting_curve_graph_lower_bound_percent(config) / 100.0
        interval_predictor = require_rwkv_interval() if include_intervals or include_graph else None
        values = rwkv_card_info_retrievability(
            mw.col,
            manager,
            card_id,
            runtime=runtime,
            curve_predictor=(require_rwkv_probability() if include_forgetting_curve else None),
            interval_predictor=interval_predictor,
            include_immediate=include_immediate,
            include_forgetting_curve=include_forgetting_curve,
            include_intervals=include_intervals,
            include_forgetting_curve_graph=include_graph,
            forgetting_curve_graph_lower_bound=graph_lower_bound,
        )
    except Exception:
        values = None

    if values is None:
        return True, _unavailable_card_info_payload(immediate_refresh)

    return True, _card_info_payload(
        values,
        include_immediate=include_immediate,
        include_forgetting_curve=include_forgetting_curve,
        include_intervals=include_intervals,
    )


def _immediate_refresh_payload(value: float | None) -> dict[str, Any]:
    return {
        "enabled": True,
        "available": True,
        "value": format_retrievability_percent(value),
    }


def _unavailable_card_info_payload(immediate_refresh: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {"available": False}
    if immediate_refresh:
        payload["enabled"] = True
    return payload


def _card_info_payload(
    values: CardInfoRetrievabilityValues,
    *,
    include_immediate: bool = True,
    include_forgetting_curve: bool = True,
    include_intervals: bool = False,
) -> dict[str, Any]:
    rows = []
    if include_immediate:
        rows.append(
            {
                "key": "immediate",
                "label": "RWKV Immediate Retrievability",
                "value": format_retrievability_percent(values.immediate),
            }
        )
    if include_forgetting_curve:
        rows.append(
            {
                "key": "forgetting_curve",
                "label": "RWKV Forgetting Curve Retrievability",
                "value": format_retrievability_percent(values.forgetting_curve),
            }
        )
    if include_intervals or values.stability_interval_seconds is not None:
        rows.append(
            {
                "key": "stability",
                "label": "RWKV Forgetting Curve Stability",
                "value": format_rwkv_interval(values.stability_interval_seconds),
            }
        )
    if include_intervals or values.desired_retention_interval_seconds is not None:
        rows.append(
            {
                "key": "interval",
                "label": "RWKV Forgetting Curve Interval",
                "value": format_rwkv_interval(values.desired_retention_interval_seconds),
            }
        )

    return {
        "available": True,
        "rows": rows,
        "curveGraph": _curve_graph_payload(values.forgetting_curve_graph),
    }


def _missing_card_info_payload(
    *,
    include_immediate: bool,
    include_forgetting_curve: bool,
    include_intervals: bool,
) -> dict[str, Any]:
    return _card_info_payload(
        CardInfoRetrievabilityValues(
            immediate=None,
            forgetting_curve=None,
        ),
        include_immediate=include_immediate,
        include_forgetting_curve=include_forgetting_curve,
        include_intervals=include_intervals,
    )


def _active_live_review_runtime():
    try:
        from .live_review_bridge import active_live_review_session

        session = active_live_review_session()
    except Exception:
        return None
    if session is None or not bool(getattr(session, "active", False)):
        return None
    runtime = getattr(session, "runtime_session", None)
    if runtime is None or bool(getattr(runtime, "closed", False)):
        return None
    return runtime


def _active_card_info_runtime():
    runtime = _active_live_review_runtime()
    if runtime is not None:
        return runtime
    try:
        from .browser_card_info import active_browser_card_info_runtime

        return active_browser_card_info_runtime()
    except Exception:
        return None


def _runtime_contains_card(runtime, card_id: int) -> bool:
    contains_card = getattr(runtime, "contains_card", None)
    if not callable(contains_card):
        return False
    try:
        return bool(contains_card(int(card_id)))
    except Exception:
        return False


def _curve_graph_payload(graph: CardInfoForgettingCurveGraph | None):
    if graph is None:
        return None
    return {
        "title": "RWKV Forgetting Curve",
        "desiredRetention": float(graph.desired_retention),
        "minimumRetrievability": float(graph.minimum_retrievability),
        "desiredRetentionIntervalSeconds": graph.desired_retention_interval_seconds,
        "lastReviewTimestampSeconds": float(graph.last_review_timestamp_seconds),
        "nowTimestampSeconds": float(graph.now_timestamp_seconds),
        "points": [
            {
                "elapsedSeconds": float(point.elapsed_seconds),
                "retrievability": float(point.retrievability),
            }
            for point in graph.points
        ],
    }


def _webview_kind_value(webview) -> str | None:
    kind = getattr(webview, "kind", None)
    value = getattr(kind, "value", kind)
    return str(value) if value is not None else None


def _anchor_labels() -> dict[str, str]:
    try:
        from anki.lang import without_unicode_isolation
        from aqt.utils import tr

        return {
            "primary": without_unicode_isolation(tr.card_stats_fsrs_retrievability()),
            "fallback": without_unicode_isolation(tr.card_stats_interval()),
        }
    except Exception:
        return {
            "primary": "Retrievability",
            "fallback": "Interval",
        }


def _injected_card_info_script(
    anchor_labels: dict[str, str],
    *,
    auto_refresh_immediate: bool = False,
) -> str:
    labels_json = json.dumps(anchor_labels, ensure_ascii=False)
    prefix_json = json.dumps(_MESSAGE_PREFIX)
    immediate_prefix_json = json.dumps(_IMMEDIATE_REFRESH_MESSAGE_PREFIX)
    auto_refresh_json = json.dumps(bool(auto_refresh_immediate))
    curve_graph_script = _rwkv_curve_graph_script()
    return f"""
(function() {{
    const anchorLabels = {labels_json};
    const messagePrefix = {prefix_json};
    const immediateMessagePrefix = {immediate_prefix_json};
    const immediateRefreshIntervalMs = {_IMMEDIATE_REFRESH_INTERVAL_MILLISECONDS};
    const rowAttr = "data-rwkvp-card-info";
    const graphAttr = "data-rwkvp-card-info-curve";
    const maxInsertRetries = 40;
    let renderToken = 0;
    let renderedCardId = null;
    let autoRefreshImmediate = {auto_refresh_json};
    let immediateRefreshTimer = null;
    let immediateRefreshInFlight = false;

    function normalize(text) {{
        return String(text || "")
            .replace(/[\\u2066-\\u2069]/g, "")
            .replace(/\\s+/g, " ")
            .trim();
    }}

    function currentCardId() {{
        const match = window.location.pathname.match(/\\/card-info\\/([^/]+)/);
        if (!match) {{
            return null;
        }}
        const value = decodeURIComponent(match[1]);
        return /^\\d+$/.test(value) ? value : null;
    }}

    function removeRows() {{
        renderedCardId = null;
        document.querySelectorAll(`tr[${{rowAttr}}]`).forEach((row) => row.remove());
    }}

    function removeCurveGraph() {{
        document.querySelectorAll(`[${{graphAttr}}]`).forEach((node) => node.remove());
        const tooltip = document.getElementById("rwkvp-card-info-curve-tooltip");
        if (tooltip) {{
            tooltip.remove();
        }}
    }}

    function installRowStyle() {{
        if (document.getElementById("rwkvp-card-info-row-style")) {{
            return;
        }}
        const style = document.createElement("style");
        style.id = "rwkvp-card-info-row-style";
        style.textContent = `
            tr[${{rowAttr}}] th {{
                color: var(--fg-link, currentColor);
                color: color-mix(in srgb, var(--fg, currentColor) 74%, var(--fg-link, #2f6f9f));
                font-weight: 600;
            }}
            tr[${{rowAttr}}] td {{
                color: var(--fg-link, currentColor);
                color: color-mix(in srgb, var(--fg, currentColor) 74%, var(--fg-link, #2f6f9f));
                font-weight: 500;
            }}
            tr.rwkvp-card-info-spacer td {{
                padding: 0.42em 0;
                line-height: 0;
            }}
        `;
        document.head.appendChild(style);
    }}

    function createSpacerRow() {{
        const row = document.createElement("tr");
        row.setAttribute(rowAttr, "spacer");
        row.className = "rwkvp-card-info-spacer";
        row.setAttribute("aria-hidden", "true");
        const cell = document.createElement("td");
        cell.colSpan = 2;
        row.appendChild(cell);
        return row;
    }}

    function insertAfterOrAppend(tbody, previous, row) {{
        if (previous && previous.parentNode === tbody) {{
            previous.insertAdjacentElement("afterend", row);
        }} else {{
            tbody.appendChild(row);
        }}
        return row;
    }}

{curve_graph_script}

    function statsTableBody() {{
        const table = document.querySelector("table.stats-table");
        return table ? table.querySelector("tbody") : null;
    }}

    function anchorRow(tbody, label) {{
        const target = normalize(label);
        if (!target) {{
            return null;
        }}
        for (const row of tbody.querySelectorAll("tr")) {{
            if (row.hasAttribute(rowAttr)) {{
                continue;
            }}
            const header = row.querySelector("th");
            if (header && normalize(header.textContent) === target) {{
                return row;
            }}
        }}
        return null;
    }}

    function nativeStatsRows(tbody) {{
        return Array.from(tbody.querySelectorAll("tr"))
            .filter((row) => !row.hasAttribute(rowAttr));
    }}

    function insertRows(payload, token, cardId, attempt = 0) {{
        if (token !== renderToken || currentCardId() !== cardId) {{
            return;
        }}
        removeRows();
        if (!payload || !payload.available || !Array.isArray(payload.rows)) {{
            removeCurveGraph();
            return;
        }}

        const tbody = statsTableBody();
        if (!tbody || nativeStatsRows(tbody).length === 0) {{
            if (attempt < maxInsertRetries) {{
                setTimeout(() => insertRows(payload, token, cardId, attempt + 1), 100);
            }}
            return;
        }}

        let anchor = anchorRow(tbody, anchorLabels.primary);
        if (!anchor) {{
            anchor = anchorRow(tbody, anchorLabels.fallback);
        }}

        installRowStyle();
        let previous = anchor;
        previous = insertAfterOrAppend(tbody, previous, createSpacerRow());
        for (const item of payload.rows) {{
            const row = document.createElement("tr");
            row.setAttribute(rowAttr, item.key || "1");

            const header = document.createElement("th");
            header.className = "align-start";
            header.textContent = item.label || "";

            const value = document.createElement("td");
            value.textContent = item.value || "-";

            row.appendChild(header);
            row.appendChild(value);

            previous = insertAfterOrAppend(tbody, previous, row);
        }}
        insertAfterOrAppend(tbody, previous, createSpacerRow());
        renderedCardId = cardId;
        insertCurveGraph(payload, token, cardId);
    }}

    function renderRwkvRows() {{
        const cardId = currentCardId();
        renderToken += 1;
        renderedCardId = null;
        const token = renderToken;
        if (!cardId || typeof pycmd !== "function") {{
            removeRows();
            removeCurveGraph();
            return;
        }}
        pycmd(messagePrefix + cardId, (payload) => insertRows(payload, token, cardId));
    }}

    function applyImmediateRefresh(payload, token, cardId) {{
        if (
            token !== renderToken
            || currentCardId() !== cardId
            || renderedCardId !== cardId
        ) {{
            return;
        }}
        if (payload && payload.enabled === false) {{
            autoRefreshImmediate = false;
            renderRwkvRows();
            return;
        }}
        if (!payload || !payload.available) {{
            removeRows();
            removeCurveGraph();
            return;
        }}
        const value = document.querySelector(`tr[${{rowAttr}}="immediate"] td`);
        if (!value) {{
            renderRwkvRows();
            return;
        }}
        value.textContent = payload.value || "-";
    }}

    function scheduleImmediateRefresh() {{
        if (!autoRefreshImmediate || immediateRefreshTimer !== null) {{
            return;
        }}
        immediateRefreshTimer = window.setTimeout(
            refreshImmediate,
            immediateRefreshIntervalMs,
        );
    }}

    function refreshImmediate() {{
        immediateRefreshTimer = null;
        if (!autoRefreshImmediate || immediateRefreshInFlight) {{
            return;
        }}
        if (document.hidden) {{
            scheduleImmediateRefresh();
            return;
        }}
        const cardId = currentCardId();
        if (!cardId || typeof pycmd !== "function") {{
            scheduleImmediateRefresh();
            return;
        }}
        if (renderedCardId !== cardId) {{
            renderRwkvRows();
            scheduleImmediateRefresh();
            return;
        }}
        const token = renderToken;
        immediateRefreshInFlight = true;
        try {{
            pycmd(immediateMessagePrefix + cardId, (payload) => {{
                immediateRefreshInFlight = false;
                applyImmediateRefresh(payload, token, cardId);
                scheduleImmediateRefresh();
            }});
        }} catch (_error) {{
            immediateRefreshInFlight = false;
            scheduleImmediateRefresh();
        }}
    }}

    window.addEventListener("pagehide", () => {{
        autoRefreshImmediate = false;
        if (immediateRefreshTimer !== null) {{
            window.clearTimeout(immediateRefreshTimer);
            immediateRefreshTimer = null;
        }}
    }}, {{ once: true }});

    function installUpdateHook() {{
        globalThis.anki ||= {{}};
        if (globalThis.anki.__rwkvpCardInfoHooked) {{
            return;
        }}
        const originalUpdateCard = globalThis.anki.updateCard;
        if (typeof originalUpdateCard === "function") {{
            globalThis.anki.updateCard = async function(...args) {{
                const result = await originalUpdateCard.apply(this, args);
                setTimeout(renderRwkvRows, 100);
                return result;
            }};
            globalThis.anki.__rwkvpCardInfoHooked = true;
        }} else {{
            setTimeout(installUpdateHook, 100);
        }}
    }}

    installUpdateHook();
    renderRwkvRows();
    scheduleImmediateRefresh();
}})();
"""


def _rwkv_curve_graph_script() -> str:
    return r"""
    function installCurveGraphStyle() {
        if (document.getElementById("rwkvp-card-info-curve-style")) {
            return;
        }
        const style = document.createElement("style");
        style.id = "rwkvp-card-info-curve-style";
        style.textContent = `
            .rwkvp-card-info-curve {
                width: 100%;
                max-width: 50em;
                margin-bottom: 1.5em;
            }
            .rwkvp-card-info-curve-controls {
                display: flex;
                justify-content: space-around;
                margin-bottom: 1em;
                width: 100%;
                max-width: 50em;
            }
            .rwkvp-card-info-curve-controls label {
                display: flex;
                align-items: center;
                gap: 0.5em;
            }
            .rwkvp-card-info-curve-title {
                font-weight: 600;
                margin: 1.5em 0 0.5em;
            }
            .rwkvp-card-info-curve-graph {
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .rwkvp-card-info-curve svg {
                width: 100%;
                max-width: 600px;
                height: auto;
                overflow: visible;
            }
            .rwkvp-card-info-curve .tick text {
                opacity: 0.55;
                font-size: 10px;
                fill: currentColor;
            }
            .rwkvp-card-info-curve .tick line {
                opacity: 0.12;
                stroke: currentColor;
            }
            .rwkvp-card-info-curve .axis-domain {
                opacity: 0.08;
                stroke: currentColor;
            }
            #rwkvp-card-info-curve-tooltip {
                position: fixed;
                z-index: 99999;
                pointer-events: none;
                padding: 0.45em 0.6em;
                border-radius: 0.25em;
                color: var(--fg, #fff);
                background: var(--tooltip-bg, rgba(32, 32, 32, 0.92));
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.28);
                font-size: 0.9em;
                line-height: 1.35;
                opacity: 0;
            }
        `;
        document.head.appendChild(style);
    }

    function graphInsertionTarget() {
        const table = document.querySelector("table.stats-table");
        if (!table) {
            return null;
        }
        const statsRow = table.closest(".row") || table.parentElement;
        if (statsRow && statsRow.parentElement) {
            return {
                parent: statsRow.parentElement,
                beforeNode: statsRow.parentElement.firstChild,
                fallback: statsRow,
            };
        }
        if (table.parentElement) {
            return {
                parent: table.parentElement,
                beforeNode: table,
                fallback: table,
            };
        }
        return null;
    }

    function insertGraphAtCardInfoTop(wrapper, target) {
        try {
            target.parent.insertBefore(wrapper, target.beforeNode || null);
        } catch (_error) {
            target.fallback.insertAdjacentElement("beforebegin", wrapper);
        }
    }

    function insertCurveGraph(payload, token, cardId, attempt = 0) {
        if (token !== renderToken || currentCardId() !== cardId) {
            return;
        }
        removeCurveGraph();
        const graph = payload ? payload.curveGraph : null;
        if (!graph || !Array.isArray(graph.points) || graph.points.length < 2) {
            return;
        }
        const target = graphInsertionTarget();
        if (!target) {
            if (attempt < maxInsertRetries) {
                setTimeout(() => insertCurveGraph(payload, token, cardId, attempt + 1), 100);
            }
            return;
        }

        installCurveGraphStyle();
        const wrapper = document.createElement("div");
        wrapper.setAttribute(graphAttr, "1");
        wrapper.className = "rwkvp-card-info-curve";
        const controls = document.createElement("div");
        controls.className = "rwkvp-card-info-curve-controls";
        const title = document.createElement("div");
        title.className = "rwkvp-card-info-curve-title";
        title.textContent = graph.title || "RWKV Forgetting Curve";
        const graphContainer = document.createElement("div");
        graphContainer.className = "rwkvp-card-info-curve-graph";
        const svg = createSvg("svg");
        svg.setAttribute("viewBox", "0 0 600 250");
        graphContainer.appendChild(svg);
        wrapper.appendChild(controls);
        wrapper.appendChild(title);
        wrapper.appendChild(graphContainer);
        insertGraphAtCardInfoTop(wrapper, target);
        renderCurveGraph(wrapper, controls, svg, graph);
    }

    function createSvg(name) {
        return document.createElementNS("http://www.w3.org/2000/svg", name);
    }

    function renderCurveGraph(wrapper, controls, svg, graph) {
        const points = normalizeCurvePoints(graph.points);
        if (points.length < 2) {
            return;
        }
        const maxElapsed = Math.max(...points.map((point) => point.elapsedSeconds));
        const ranges = availableCurveRanges(maxElapsed);
        let selectedRange = defaultCurveRange(maxElapsed, ranges);
        renderControls(controls, ranges, selectedRange, (range) => {
            selectedRange = range;
            drawCurveGraph(svg, graph, points, selectedRange);
        });
        drawCurveGraph(svg, graph, points, selectedRange);
    }

    function normalizeCurvePoints(points) {
        const byElapsed = new Map();
        for (const point of points) {
            const elapsedSeconds = Number(point.elapsedSeconds);
            const retrievability = Number(point.retrievability);
            if (
                Number.isFinite(elapsedSeconds)
                && elapsedSeconds >= 0
                && Number.isFinite(retrievability)
            ) {
                byElapsed.set(elapsedSeconds, {
                    elapsedSeconds,
                    retrievability: Math.max(0, Math.min(1, retrievability)),
                });
            }
        }
        return Array.from(byElapsed.values()).sort(
            (left, right) => left.elapsedSeconds - right.elapsedSeconds,
        );
    }

    function availableCurveRanges(maxElapsed) {
        const ranges = [
            { key: "week", label: "First Week", seconds: 7 * 86400 },
            { key: "month", label: "First Month", seconds: 30 * 86400 },
            { key: "year", label: "First Year", seconds: 365 * 86400 },
            { key: "all", label: "All Time", seconds: Infinity },
        ];
        return ranges.filter((range) => range.key === "all" || maxElapsed > range.seconds);
    }

    function defaultCurveRange(maxElapsed, ranges) {
        const preferred = preferredCurveRange(maxElapsed);
        if (ranges.some((range) => range.key === preferred)) {
            return preferred;
        }
        if (ranges.some((range) => range.key === "all")) {
            return "all";
        }
        return ranges.length ? ranges[0].key : "all";
    }

    function preferredCurveRange(maxElapsed) {
        if (maxElapsed > 365 * 86400) {
            return "all";
        }
        if (maxElapsed > 30 * 86400) {
            return "year";
        }
        if (maxElapsed > 7 * 86400) {
            return "month";
        }
        return "week";
    }

    function renderControls(controls, ranges, selectedRange, onChange) {
        controls.replaceChildren();
        if (ranges.length <= 1) {
            return;
        }
        const group = `rwkvp-card-info-range-${Math.random().toString(36).slice(2)}`;
        for (const range of ranges) {
            const label = document.createElement("label");
            const input = document.createElement("input");
            input.type = "radio";
            input.name = group;
            input.value = range.key;
            input.checked = range.key === selectedRange;
            input.addEventListener("change", () => onChange(range.key));
            label.appendChild(input);
            label.appendChild(document.createTextNode(range.label));
            controls.appendChild(label);
        }
    }

    function rangeSeconds(range, maxElapsed) {
        if (range === "week") {
            return Math.min(maxElapsed, 7 * 86400);
        }
        if (range === "month") {
            return Math.min(maxElapsed, 30 * 86400);
        }
        if (range === "year") {
            return Math.min(maxElapsed, 365 * 86400);
        }
        return maxElapsed;
    }

    function drawCurveGraph(svg, graph, sourcePoints, selectedRange) {
        svg.replaceChildren();
        const bounds = {
            width: 600,
            height: 250,
            marginLeft: 70,
            marginRight: 70,
            marginTop: 20,
            marginBottom: 25,
        };
        const maxElapsed = Math.max(...sourcePoints.map((point) => point.elapsedSeconds));
        const xMax = Math.max(1, rangeSeconds(selectedRange, maxElapsed));
        const yMin = graphLowerBoundPercent(graph);
        const desiredPercent = Number(graph.desiredRetention) * 100;
        const nowElapsed = Math.max(
            0,
            Number(graph.nowTimestampSeconds) - Number(graph.lastReviewTimestampSeconds),
        );
        const points = visibleCurvePoints(sourcePoints, xMax);
        const x = (elapsedSeconds) =>
            bounds.marginLeft
            + (elapsedSeconds / xMax) * (bounds.width - bounds.marginLeft - bounds.marginRight);
        const y = (retrievabilityPercent) =>
            bounds.height - bounds.marginBottom
            - ((retrievabilityPercent - yMin) / (100 - yMin))
                * (bounds.height - bounds.marginTop - bounds.marginBottom);

        drawAxes(svg, bounds, xMax, yMin, x, y);
        const gradientId = `rwkvp-card-info-line-gradient-${Math.random().toString(36).slice(2)}`;
        drawGradient(svg, gradientId, bounds, y, yMin);

        const nowPoint = pointAtElapsed(sourcePoints, Math.min(nowElapsed, xMax));
        let pastData = points.filter((point) => point.elapsedSeconds <= nowElapsed);
        let futureData = points.filter((point) => point.elapsedSeconds >= nowElapsed);
        if (nowElapsed > 0 && nowElapsed < xMax && nowPoint) {
            pastData = withBoundaryPoint(pastData, nowPoint);
            futureData = withBoundaryPoint([nowPoint, ...futureData], nowPoint);
        }
        if (nowElapsed >= xMax) {
            pastData = points;
            futureData = [];
        }
        if (nowElapsed <= 0) {
            pastData = [];
            futureData = points;
        }

        drawPath(svg, pastData, x, y, `url(#${gradientId})`, "");
        drawPath(svg, futureData, x, y, `url(#${gradientId})`, "4 4");
        if (desiredPercent >= yMin && desiredPercent <= 100) {
            drawDesiredRetention(svg, bounds, desiredPercent, x, y);
        }
        drawHover(svg, graph, sourcePoints, bounds, xMax, yMin, x, y, desiredPercent);
    }

    function visibleCurvePoints(points, xMax) {
        const visible = points.filter((point) => point.elapsedSeconds <= xMax);
        const boundary = pointAtElapsed(points, xMax);
        if (
            boundary
            && !visible.some((point) => point.elapsedSeconds === boundary.elapsedSeconds)
        ) {
            visible.push(boundary);
        }
        return visible.sort((left, right) => left.elapsedSeconds - right.elapsedSeconds);
    }

    function pointAtElapsed(points, elapsedSeconds) {
        if (!Number.isFinite(elapsedSeconds) || elapsedSeconds < 0 || points.length === 0) {
            return null;
        }
        if (elapsedSeconds <= points[0].elapsedSeconds) {
            return { ...points[0], elapsedSeconds };
        }
        for (let index = 1; index < points.length; index += 1) {
            const previous = points[index - 1];
            const next = points[index];
            if (elapsedSeconds <= next.elapsedSeconds) {
                const span = next.elapsedSeconds - previous.elapsedSeconds;
                const ratio = span <= 0 ? 0 : (elapsedSeconds - previous.elapsedSeconds) / span;
                return {
                    elapsedSeconds,
                    retrievability: previous.retrievability
                        + (next.retrievability - previous.retrievability) * ratio,
                };
            }
        }
        return { ...points[points.length - 1], elapsedSeconds };
    }

    function withBoundaryPoint(points, boundary) {
        if (points.some((point) => point.elapsedSeconds === boundary.elapsedSeconds)) {
            return points;
        }
        return [...points, boundary].sort(
            (left, right) => left.elapsedSeconds - right.elapsedSeconds,
        );
    }

    function drawAxes(svg, bounds, xMax, yMin, x, y) {
        const xAxis = createSvg("line");
        xAxis.setAttribute("class", "axis-domain");
        xAxis.setAttribute("x1", String(bounds.marginLeft));
        xAxis.setAttribute("x2", String(bounds.width - bounds.marginRight));
        xAxis.setAttribute("y1", String(bounds.height - bounds.marginBottom));
        xAxis.setAttribute("y2", String(bounds.height - bounds.marginBottom));
        svg.appendChild(xAxis);

        const yAxis = createSvg("line");
        yAxis.setAttribute("class", "axis-domain");
        yAxis.setAttribute("x1", String(bounds.marginLeft));
        yAxis.setAttribute("x2", String(bounds.marginLeft));
        yAxis.setAttribute("y1", String(bounds.marginTop));
        yAxis.setAttribute("y2", String(bounds.height - bounds.marginBottom));
        svg.appendChild(yAxis);

        for (let index = 0; index <= 5; index += 1) {
            const elapsed = (xMax * index) / 5;
            const tick = createSvg("g");
            tick.setAttribute("class", "tick");
            const line = createSvg("line");
            line.setAttribute("x1", String(x(elapsed)));
            line.setAttribute("x2", String(x(elapsed)));
            line.setAttribute("y1", String(bounds.marginTop));
            line.setAttribute("y2", String(bounds.height - bounds.marginBottom + 4));
            const text = createSvg("text");
            text.setAttribute("x", String(x(elapsed)));
            text.setAttribute("y", String(bounds.height - 5));
            text.setAttribute("text-anchor", "middle");
            text.textContent = formatDurationShort(elapsed);
            tick.appendChild(line);
            tick.appendChild(text);
            svg.appendChild(tick);
        }

        for (const percent of curvePercentTicks(yMin)) {
            const tick = createSvg("g");
            tick.setAttribute("class", "tick");
            const line = createSvg("line");
            line.setAttribute("x1", String(bounds.marginLeft - 4));
            line.setAttribute("x2", String(bounds.width - bounds.marginRight));
            line.setAttribute("y1", String(y(percent)));
            line.setAttribute("y2", String(y(percent)));
            const text = createSvg("text");
            text.setAttribute("x", String(bounds.marginLeft - 8));
            text.setAttribute("y", String(y(percent) + 3));
            text.setAttribute("text-anchor", "end");
            text.textContent = `${percent}%`;
            tick.appendChild(line);
            tick.appendChild(text);
            svg.appendChild(tick);
        }
    }

    function drawGradient(svg, gradientId, bounds, y, yMin) {
        const defs = createSvg("defs");
        const gradient = createSvg("linearGradient");
        gradient.setAttribute("id", gradientId);
        gradient.setAttribute("gradientUnits", "userSpaceOnUse");
        gradient.setAttribute("x1", "0");
        gradient.setAttribute("x2", "0");
        gradient.setAttribute("y1", String(y(yMin)));
        gradient.setAttribute("y2", String(y(100)));
        for (const stop of [
            { offset: "0%", color: "tomato" },
            { offset: "75%", color: "steelblue" },
            { offset: "100%", color: "green" },
        ]) {
            const node = createSvg("stop");
            node.setAttribute("offset", stop.offset);
            node.setAttribute("stop-color", stop.color);
            gradient.appendChild(node);
        }
        defs.appendChild(gradient);
        svg.appendChild(defs);
    }

    function graphLowerBoundPercent(graph) {
        const value = Number(graph.minimumRetrievability);
        if (!Number.isFinite(value)) {
            return 60;
        }
        const percent = value > 1 ? value : value * 100;
        return Math.max(0, Math.min(99, percent));
    }

    function curvePercentTicks(yMin) {
        const min = Math.max(0, Math.min(99, Number(yMin) || 0));
        const step = 100 - min > 60 ? 20 : 10;
        const ticks = [];
        const firstRoundedTick = Math.ceil(min / step) * step;
        if (Math.abs(firstRoundedTick - min) > 0.001) {
            ticks.push(min);
        }
        for (let percent = firstRoundedTick; percent <= 100; percent += step) {
            ticks.push(percent);
        }
        if (!ticks.some((percent) => Math.abs(percent - 100) <= 0.001)) {
            ticks.push(100);
        }
        return Array.from(new Set(ticks)).sort((left, right) => left - right);
    }

    function drawPath(svg, points, x, y, stroke, dashArray) {
        if (points.length < 2) {
            return;
        }
        const path = createSvg("path");
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", stroke);
        path.setAttribute("stroke-width", "1.5");
        if (dashArray) {
            path.setAttribute("stroke-dasharray", dashArray);
        }
        path.setAttribute(
            "d",
            points.map((point, index) => {
                const command = index === 0 ? "M" : "L";
                const xPoint = x(point.elapsedSeconds).toFixed(2);
                const yPoint = y(point.retrievability * 100).toFixed(2);
                return `${command}${xPoint},${yPoint}`;
            }).join(" "),
        );
        svg.appendChild(path);
    }

    function drawDesiredRetention(svg, bounds, desiredPercent, x, y) {
        const line = createSvg("line");
        line.setAttribute("x1", String(bounds.marginLeft));
        line.setAttribute("x2", String(bounds.width - bounds.marginRight));
        line.setAttribute("y1", String(y(desiredPercent)));
        line.setAttribute("y2", String(y(desiredPercent)));
        line.setAttribute("stroke", "steelblue");
        line.setAttribute("stroke-dasharray", "4 4");
        line.setAttribute("stroke-width", "1.2");
        svg.appendChild(line);
    }

    function drawHover(svg, graph, points, bounds, xMax, yMin, x, y, desiredPercent) {
        const focusLine = createSvg("line");
        focusLine.setAttribute("y1", String(bounds.marginTop));
        focusLine.setAttribute("y2", String(bounds.height - bounds.marginBottom));
        focusLine.setAttribute("stroke", "currentColor");
        focusLine.setAttribute("stroke-width", "1");
        focusLine.style.opacity = "0";
        svg.appendChild(focusLine);

        const overlay = createSvg("rect");
        overlay.setAttribute("x", String(bounds.marginLeft));
        overlay.setAttribute("y", String(bounds.marginTop));
        overlay.setAttribute(
            "width",
            String(bounds.width - bounds.marginLeft - bounds.marginRight),
        );
        overlay.setAttribute(
            "height",
            String(bounds.height - bounds.marginTop - bounds.marginBottom),
        );
        overlay.setAttribute("fill", "transparent");
        overlay.style.pointerEvents = "all";
        overlay.addEventListener("mousemove", (event) => {
            const rect = svg.getBoundingClientRect();
            const localX = (event.clientX - rect.left) * bounds.width / rect.width;
            const localY = (event.clientY - rect.top) * bounds.height / rect.height;
            const ratio = Math.max(
                0,
                Math.min(
                    1,
                    (localX - bounds.marginLeft)
                        / (bounds.width - bounds.marginLeft - bounds.marginRight),
                ),
            );
            const elapsed = ratio * xMax;
            const point = pointAtElapsed(points, elapsed);
            if (!point) {
                return;
            }
            const xPos = x(elapsed);
            focusLine.setAttribute("x1", String(xPos));
            focusLine.setAttribute("x2", String(xPos));
            focusLine.style.opacity = "1";
            showCurveTooltip(
                graph,
                point,
                event.clientX,
                event.clientY,
                desiredPercent,
                Math.abs(localY - y(desiredPercent)) <= 10,
            );
        });
        overlay.addEventListener("mouseout", () => {
            focusLine.style.opacity = "0";
            hideCurveTooltip();
        });
        svg.appendChild(overlay);
    }

    function showCurveTooltip(
        graph,
        point,
        clientX,
        clientY,
        desiredPercent,
        nearDesiredRetention,
    ) {
        let tooltip = document.getElementById("rwkvp-card-info-curve-tooltip");
        if (!tooltip) {
            tooltip = document.createElement("div");
            tooltip.id = "rwkvp-card-info-curve-tooltip";
            document.body.appendChild(tooltip);
        }
        const date = new Date(
            (Number(graph.lastReviewTimestampSeconds) + point.elapsedSeconds) * 1000,
        );
        let html = `Date Time: ${date.toLocaleString()}<br>`;
        html += `Elapsed Time: ${formatDurationLong(point.elapsedSeconds)}<br>`;
        html += `RWKV Retrievability: ${(point.retrievability * 100).toFixed(2)}%`;
        if (nearDesiredRetention) {
            html += `<br>Desired Retention: ${desiredPercent.toFixed(0)}%`;
            if (Number.isFinite(Number(graph.desiredRetentionIntervalSeconds))) {
                html += ` after ${
                    formatDurationLong(Number(graph.desiredRetentionIntervalSeconds))
                }`;
            }
        }
        tooltip.innerHTML = html;
        tooltip.style.left = `${clientX + 12}px`;
        tooltip.style.top = `${clientY + 12}px`;
        tooltip.style.opacity = "1";
    }

    function hideCurveTooltip() {
        const tooltip = document.getElementById("rwkvp-card-info-curve-tooltip");
        if (tooltip) {
            tooltip.style.opacity = "0";
        }
    }

    function formatDurationShort(seconds) {
        const value = Math.max(0, Number(seconds) || 0);
        if (value < 60) {
            return `${Math.round(value)}s`;
        }
        if (value < 3600) {
            return `${Math.round(value / 60)}m`;
        }
        if (value < 86400) {
            return `${Math.round(value / 3600)}h`;
        }
        if (value < 86400 * 30) {
            return `${Math.round(value / 86400)}d`;
        }
        if (value < 86400 * 365) {
            return `${Math.round(value / (86400 * 30))}mo`;
        }
        return `${Math.round(value / (86400 * 365))}y`;
    }

    function formatDurationLong(seconds) {
        const value = Math.max(0, Number(seconds) || 0);
        const units = [
            ["second", 1, 60],
            ["minute", 60, 60],
            ["hour", 3600, 24],
            ["day", 86400, 30],
            ["month", 86400 * 30, 12],
            ["year", 86400 * 365, Infinity],
        ];
        let amount = value;
        let name = "second";
        for (const [unitName, scale, nextThreshold] of units) {
            amount = value / scale;
            name = unitName;
            if (amount < nextThreshold) {
                break;
            }
        }
        const rounded = Math.max(0, Math.round(amount));
        return `${rounded} ${name}${rounded === 1 ? "" : "s"}`;
    }
"""
