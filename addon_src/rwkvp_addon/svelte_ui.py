"""Data-only bridge to the precompiled, self-contained Svelte presentation layer.

No Node, compiler, package manager or network access is needed inside Anki.
Python owns workflow state and validates bridge actions; Svelte owns its DOM.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from .modal_html import shared_modal_script_tag
from .modal_style import modal_root_classes, shared_style_tag
from .web_dialog_bridge import WEB_DIALOG_BRIDGE_PREFIX, require_web_dialog_action

_FRONTEND = Path(__file__).with_name("frontend")


@lru_cache(maxsize=1)
def shared_frontend_script() -> str:
    return (_FRONTEND / "ui.js").read_text(encoding="utf-8")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )


def render_svelte_document(
    screen: str,
    props: Mapping[str, Any],
    *,
    title: str,
    generation: int = 1,
    is_dark: bool = False,
    width: str = "standard",
    intro: str | None = None,
    head_html: str = "",
    root_extra_classes: str | None = None,
    escape_action: str = "dialog-close",
    escape_payload: Mapping[str, Any] | None = None,
    enter_action: str | None = None,
    enter_payload: Mapping[str, Any] | None = None,
    layout: str = "document",
    bootstrap: Mapping[str, Any] | None = None,
    title_id: str | None = "rwkv-dialog-title",
    root_attributes: Mapping[str, str] | None = None,
) -> str:
    """Compose a native-host document from a typed screen payload.

    ``head_html`` accepts only trusted add-on CSS, never user data. Screens use
    the existing shared CSS and bridge attributes so the Windows popup adapter,
    keyboard routing, progress ownership and Python validation remain intact.
    """
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise ValueError("generation must be a positive integer")
    if width not in {"compact", "standard", "wide"}:
        raise ValueError(f"unsupported modal width: {width!r}")
    if layout not in {"document", "custom"}:
        raise ValueError(f"unsupported modal layout: {layout!r}")
    page = {"title": title, "intro": intro, "width": width, **props}
    payload = _json({"screen": screen, "props": page})
    bridge = _json(
        {"bridgePrefix": WEB_DIALOG_BRIDGE_PREFIX, **(bootstrap or {}), "generation": generation}
    )
    attrs = {
        "id": "rwkv-svelte-root",
        "class": modal_root_classes(is_dark=is_dark, extra=root_extra_classes or ""),
        "role": "document",
        "data-rwkv-generation": str(generation),
        "data-rwkv-escape-action": require_web_dialog_action(escape_action),
        "data-rwkv-escape-payload": _json(dict(escape_payload or {"outcome": "reject"})),
    }
    attrs["aria-labelledby" if title_id else "aria-label"] = title_id or title
    for key, value in (root_attributes or {}).items():
        if not key.startswith(("data-rwkv-", "aria-")) or any(
            character not in "abcdefghijklmnopqrstuvwxyz-" for character in key
        ):
            raise ValueError(f"unsupported Svelte root attribute: {key!r}")
        attrs[key] = str(value)
    if intro:
        attrs["aria-describedby"] = "rwkv-dialog-intro"
    if enter_action:
        attrs["data-rwkv-enter-action"] = require_web_dialog_action(enter_action)
        attrs["data-rwkv-enter-payload"] = _json(dict(enter_payload or {}))
    attributes = " ".join(
        f'{key}="{html.escape(value, quote=True)}"' for key, value in attrs.items()
    )
    return "\n".join(
        (
            shared_style_tag(),
            head_html,
            f"<div {attributes}></div>",
            f'<script id="rwkv-svelte-payload" type="application/json">{payload}</script>',
            f"<script>window.RWKV_MODAL_BOOTSTRAP = {bridge};</script>",
            f'<script data-rwkv-svelte="true">{shared_frontend_script()}</script>',
            shared_modal_script_tag(),
        )
    )
