from __future__ import annotations

from collections.abc import Callable, Collection
from enum import Enum
from typing import Any, Protocol

from .web_dialog_bridge import BridgePayloadError, WebDialogCommand


class CloseReason(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    TITLE_BAR = "title-bar"
    PROFILE_TEARDOWN = "profile-teardown"
    PARENT_TEARDOWN = "parent-teardown"
    RENDER_FAILURE = "render-failure"
    DESTROYED = "destroyed"


class WebDialogController(Protocol):
    """Qt-independent controller boundary consumed by the native web host."""

    @property
    def actions(self) -> Collection[str]: ...

    def render_html(self, generation: int) -> str: ...

    def handle_command(self, command: WebDialogCommand) -> Any: ...

    def is_action_enabled(self, action: str) -> bool: ...

    def can_close(self, reason: CloseReason) -> bool: ...

    def on_dialog_closed(self, reason: CloseReason) -> None: ...


class BaseWebDialogController:
    """Convenience defaults for workflow controllers introduced in later steps."""

    actions: Collection[str] = ()

    def render_html(self, generation: int) -> str:
        raise NotImplementedError

    def handle_command(self, command: WebDialogCommand) -> Any:
        raise BridgePayloadError(f"Unhandled dialog action: {command.action}")

    def is_action_enabled(self, action: str) -> bool:
        del action
        return True

    def can_close(self, reason: CloseReason) -> bool:
        del reason
        return True

    def on_dialog_closed(self, reason: CloseReason) -> None:
        del reason


class CloseOnlyReportController(BaseWebDialogController):
    """Qt-independent controller for an immutable HTML report with Close only."""

    actions: Collection[str] = ()

    def __init__(self, renderer: Callable[[int], str]) -> None:
        if not callable(renderer):
            raise TypeError("report renderer must be callable")
        self._renderer = renderer

    def render_html(self, generation: int) -> str:
        rendered = self._renderer(generation)
        if not isinstance(rendered, str):
            raise TypeError("report renderer must return HTML text")
        return rendered
