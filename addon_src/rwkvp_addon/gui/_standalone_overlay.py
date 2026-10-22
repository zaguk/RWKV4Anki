from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..svelte_ui import render_svelte_document
from ..web_dialog_controller import BaseWebDialogController

StatusState = Literal["empty", "loading", "disabled", "error", "success"]


@dataclass(frozen=True)
class StandaloneOverlayController(BaseWebDialogController):
    """Presentation-only document shared by standalone progress and messages."""

    title: str
    status_title: str
    status_message: str
    status_state: StatusState
    is_dark: bool
    intro: str | None = None
    inline_progress: bool = False

    def render_html(self, generation: int) -> str:
        return render_svelte_document(
            "standalone",
            {
                "status_title": self.status_title,
                "status_message": self.status_message,
                "status_state": self.status_state,
                "inline_progress": self.inline_progress,
            },
            title=self.title,
            intro=self.intro,
            generation=generation,
            is_dark=self.is_dark,
            width="compact",
            root_extra_classes=(
                "rwkv-standalone-progress" if self.inline_progress else None
            ),
        )


def create_standalone_overlay_host(
    *,
    parent: Any,
    title: str,
    status_title: str,
    status_message: str,
    status_state: StatusState,
    size: tuple[int, int],
    intro: str | None = None,
    inline_progress: bool = False,
):
    """Open the one shared window-modal host used outside an existing WebView."""

    # Keep Qt and WebDialogHost lazy: WebDialogHost imports the message and
    # progress owners, and presentation-neutral unit tests import those owners
    # without a running Anki application.
    from aqt.qt import Qt

    from .web_dialog import WebDialogHost, widget_uses_dark_palette

    controller = StandaloneOverlayController(
        title=str(title),
        status_title=str(status_title),
        status_message=str(status_message),
        status_state=status_state,
        is_dark=widget_uses_dark_palette(parent),
        intro=None if intro is None else str(intro),
        inline_progress=bool(inline_progress),
    )
    dialog = WebDialogHost(
        parent,
        title=title,
        controller=controller,
        size=(int(size[0]), int(size[1])),
        web_minimum_height=250,
        modality=Qt.WindowModality.WindowModal,
        requires_collection=False,
    )
    dialog.open()
    return dialog
