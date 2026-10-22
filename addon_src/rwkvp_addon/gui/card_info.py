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
    # Anki owns this document, so mount only our rows and graph. Re-injection
    # replaces the controller while retaining the one compiled Svelte runtime.
    from ..svelte_ui import shared_frontend_script

    labels_json = json.dumps(anchor_labels, ensure_ascii=False)
    refresh_json = json.dumps(bool(auto_refresh_immediate))
    return (
        "if (!window.RWKVFrontend) {\n"
        + shared_frontend_script()
        + "\n}\n"
        + f"window.RWKVFrontend.installCardInfo({labels_json}, {refresh_json});"
    )
