from __future__ import annotations

import html
from functools import lru_cache
from pathlib import Path

_STYLE_PATH = Path(__file__).with_name("modal_styles.css")


@lru_cache(maxsize=1)
def _shared_style_source() -> str:
    return _STYLE_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def shared_modal_css() -> str:
    """Return the packaged shared stylesheet used by RWKV webview windows."""

    return _shared_style_source()


def shared_style_tag() -> str:
    """Return the shared stylesheet as an inline tag for Anki ``stdHtml`` views."""

    return f'<style data-rwkv-shared-modal-style="true">{shared_modal_css()}</style>'


def modal_root_classes(*, is_dark: bool = False, extra: str = "") -> str:
    classes = ["rwkv-modal-shell"]
    if is_dark:
        classes.append("rwkv-dark")
    classes.extend(part for part in extra.split() if part)
    return html.escape(" ".join(classes), quote=True)
