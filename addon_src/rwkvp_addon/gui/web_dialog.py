from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any

import aqt
from aqt import gui_hooks
from aqt.qt import QDialog, Qt, QVBoxLayout
from aqt.utils import qconnect
from aqt.webview import AnkiWebView

from ..addon_config import addon_config_for_mw, webview_popup_control_mode
from ..modal_html import document_with_popup_control_mode
from ..web_dialog_bridge import (
    BridgePayloadError,
    WebDialogBridge,
    WebDialogCommand,
)
from ..web_dialog_controller import CloseReason, WebDialogController
from .web_message import (
    WEB_MESSAGE_CHECKBOX_ACTION,
    WEB_MESSAGE_RESPONSE_ACTION,
    WebMessageOwner,
    WebMessageSession,
    WebMessageSpec,
)
from .web_progress import (
    WEB_PROGRESS_CANCEL_ACTION,
    WebProgressOwner,
    WebProgressSession,
)

DIALOG_CLOSE_ACTION = "dialog-close"


_retained_web_dialogs: set[WebDialogHost] = set()


def _configured_popup_control_mode() -> str:
    return webview_popup_control_mode(
        addon_config_for_mw(getattr(aqt, "mw", None))
    )


class WebDialogHost(QDialog):
    """One-QDialog/one-WebView host for add-on-owned HTML workflows.

    The host owns only native window lifecycle. Workflow state, validation, and
    persistence stay in the controller; presentation stays in its renderer.
    """

    def __init__(
        self,
        parent,
        *,
        title: str,
        controller: WebDialogController,
        size: tuple[int, int] = (760, 560),
        web_minimum_height: int | None = None,
        modality: Any | None = None,
        requires_collection: bool = False,
        close_policy: Callable[[CloseReason], bool] | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self._cleaned_up = False
        self._delete_scheduled = False
        self._close_reason: CloseReason | None = None
        self._close_policy = close_policy
        self._profile_hook_registered = False
        self._popup_control_mode = _configured_popup_control_mode()

        self.setWindowTitle(str(title))
        self.resize(int(size[0]), int(size[1]))
        if modality is not None:
            self.setWindowModality(modality)

        layout = QVBoxLayout(self)
        margins = getattr(layout, "setContentsMargins", None)
        if callable(margins):
            margins(0, 0, 0, 0)
        spacing = getattr(layout, "setSpacing", None)
        if callable(spacing):
            spacing(0)

        self.web = AnkiWebView(self, title=self.windowTitle())
        self.web.requiresCol = bool(requires_collection)
        if web_minimum_height is not None:
            self.web.setMinimumHeight(max(0, int(web_minimum_height)))
        actions = frozenset(str(action) for action in controller.actions) | {
            DIALOG_CLOSE_ACTION,
            WEB_PROGRESS_CANCEL_ACTION,
            WEB_MESSAGE_RESPONSE_ACTION,
            WEB_MESSAGE_CHECKBOX_ACTION,
        }
        self._bridge = WebDialogBridge(
            generation=1,
            allowed_actions=actions,
            handler=self._handle_command,
            is_action_enabled=self._is_action_enabled,
        )
        self.web.set_bridge_command(self._bridge.dispatch, self)
        self.web.stdHtml(self._render_html(self._bridge.generation), context=self)
        self._progress_owner = WebProgressOwner(
            eval_js=self.web.eval,
            generation=lambda: self._bridge.generation,
            is_closed=lambda: self._cleaned_up,
        )
        self._message_owner = WebMessageOwner(
            eval_js=self.web.eval,
            generation=lambda: self._bridge.generation,
            is_closed=lambda: self._cleaned_up,
            can_start=lambda: not self._progress_owner.active,
        )
        layout.addWidget(self.web, 1)

        qconnect(self.finished, self._on_finished)
        destroyed = getattr(self, "destroyed", None)
        if destroyed is not None:
            with suppress(AttributeError, RuntimeError, TypeError):
                qconnect(destroyed, self._on_destroyed)
        parent_destroyed = getattr(parent, "destroyed", None)
        if parent_destroyed is not None:
            with suppress(AttributeError, RuntimeError, TypeError):
                qconnect(parent_destroyed, self._on_parent_destroyed)
        profile_hook = getattr(gui_hooks, "profile_will_close", None)
        if profile_hook is not None:
            profile_hook.append(self._on_profile_will_close)
            self._profile_hook_registered = True

    @property
    def generation(self) -> int:
        return self._bridge.generation

    @property
    def cleaned_up(self) -> bool:
        return self._cleaned_up

    def rerender(self) -> int:
        """Install a new document, then atomically commit its bridge generation.

        A failed WebEngine installation closes the host deterministically. This
        avoids leaving either a partially installed new page or the old visible
        page connected to a bridge generation it cannot use.
        """

        if self._cleaned_up:
            raise RuntimeError("cannot render a closed web dialog")
        if self._progress_owner.active:
            raise RuntimeError("cannot rerender a web dialog while progress is active")
        next_generation = self._bridge.generation + 1
        rendered = self._render_html(next_generation)
        try:
            self.web.stdHtml(rendered, context=self)
        except Exception as install_error:
            try:
                self._complete_finish(
                    CloseReason.RENDER_FAILURE,
                    _dialog_code("Rejected", 0),
                )
            except Exception as cleanup_error:
                install_error.add_note(
                    "Web dialog cleanup also failed after document installation failed: "
                    f"{cleanup_error!r}"
                )
            raise
        actual_generation = self._bridge.advance_generation()
        if actual_generation != next_generation:
            raise RuntimeError("web dialog generation changed unexpectedly")
        self._message_owner.document_rerendered(actual_generation)
        return actual_generation

    def _render_html(self, generation: int) -> str:
        return document_with_popup_control_mode(
            self.controller.render_html(generation),
            mode=self._popup_control_mode,
        )

    def start_web_progress(
        self,
        *,
        title: str,
        label: str,
        schedule_on_main: Callable[[Callable[[], None]], None],
        on_cancel: Callable[[], None],
        on_finished: Callable[[], None] | None = None,
    ) -> WebProgressSession:
        if self._message_owner.active:
            raise RuntimeError("cannot show progress while a message is active")
        return self._progress_owner.start(
            title=title,
            label=label,
            schedule_on_main=schedule_on_main,
            on_cancel=on_cancel,
            on_finished=on_finished,
        )

    def start_web_message(
        self,
        spec: WebMessageSpec,
        *,
        on_result: Callable[[str], None],
        on_checkbox_changed: Callable[[bool], None] | None = None,
    ) -> WebMessageSession:
        return self._message_owner.start(
            spec,
            on_result=on_result,
            on_checkbox_changed=on_checkbox_changed,
        )

    def show(self) -> None:
        self._retain()
        super().show()

    def open(self) -> None:
        self._retain()
        super().open()

    def accept(self) -> None:
        self._request_finish(CloseReason.ACCEPT, _dialog_code("Accepted", 1))

    def reject(self) -> None:
        self._request_finish(CloseReason.REJECT, _dialog_code("Rejected", 0))

    def done(self, result: int) -> None:
        reason = (
            CloseReason.ACCEPT if int(result) == _dialog_code("Accepted", 1) else CloseReason.REJECT
        )
        self._request_finish(reason, int(result))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._progress_owner.active:
            self._progress_owner.request_active_cancel()
            ignore = getattr(event, "ignore", None)
            if callable(ignore):
                ignore()
            return
        if self._message_owner.active:
            self._message_owner.request_escape()
            ignore = getattr(event, "ignore", None)
            if callable(ignore):
                ignore()
            return
        if not self._close_is_allowed(CloseReason.TITLE_BAR):
            ignore = getattr(event, "ignore", None)
            if callable(ignore):
                ignore()
            return
        accept = getattr(event, "accept", None)
        if callable(accept):
            accept()
        self._complete_finish(CloseReason.TITLE_BAR, _dialog_code("Rejected", 0))

    def force_teardown(self, reason: CloseReason = CloseReason.PROFILE_TEARDOWN) -> None:
        """Close during owner/profile teardown, bypassing a critical-work policy."""

        if reason not in {
            CloseReason.PROFILE_TEARDOWN,
            CloseReason.PARENT_TEARDOWN,
        }:
            raise ValueError("force_teardown requires a teardown close reason")
        if reason == CloseReason.PARENT_TEARDOWN:
            try:
                self._cleanup(reason)
            finally:
                _retained_web_dialogs.discard(self)
            return
        self._complete_finish(reason, _dialog_code("Rejected", 0))

    def _handle_command(self, command: WebDialogCommand) -> Any:
        if command.action == WEB_PROGRESS_CANCEL_ACTION:
            unsupported = set(command.payload) - {"token"}
            if unsupported:
                raise BridgePayloadError("Progress cancellation contains unsupported fields.")
            token = command.payload.get("token")
            if isinstance(token, bool) or not isinstance(token, int):
                raise BridgePayloadError("Progress cancellation requires an operation token.")
            return {"cancel_requested": self._progress_owner.request_cancel(token)}
        if command.action == WEB_MESSAGE_RESPONSE_ACTION:
            unsupported = set(command.payload) - {"token", "outcome"}
            if unsupported:
                raise BridgePayloadError("Message response contains unsupported fields.")
            token = command.payload.get("token")
            outcome = command.payload.get("outcome")
            if isinstance(token, bool) or not isinstance(token, int):
                raise BridgePayloadError("Message response requires an operation token.")
            if not isinstance(outcome, str):
                raise BridgePayloadError("Message response requires an outcome.")
            return {"response_accepted": self._message_owner.respond(token, outcome)}
        if command.action == WEB_MESSAGE_CHECKBOX_ACTION:
            unsupported = set(command.payload) - {"token", "checked"}
            if unsupported:
                raise BridgePayloadError("Message checkbox change contains unsupported fields.")
            token = command.payload.get("token")
            checked = command.payload.get("checked")
            if isinstance(token, bool) or not isinstance(token, int):
                raise BridgePayloadError("Message checkbox change requires an operation token.")
            if not isinstance(checked, bool):
                raise BridgePayloadError("Message checkbox change requires a checked state.")
            return {
                "checkbox_updated": self._message_owner.update_checkbox(token, checked)
            }
        if command.action != DIALOG_CLOSE_ACTION:
            return self.controller.handle_command(command)
        unsupported = set(command.payload) - {"outcome"}
        if unsupported:
            raise BridgePayloadError("Close command contains unsupported fields.")
        outcome = command.payload.get("outcome", "reject")
        if outcome == "accept":
            closed = self._request_finish(CloseReason.ACCEPT, _dialog_code("Accepted", 1))
        elif outcome == "reject":
            closed = self._request_finish(CloseReason.REJECT, _dialog_code("Rejected", 0))
        else:
            raise BridgePayloadError("Close outcome must be 'accept' or 'reject'.")
        return {"closed": closed}

    def _is_action_enabled(self, action: str) -> bool:
        if action == WEB_PROGRESS_CANCEL_ACTION:
            return self._progress_owner.cancel_enabled
        if action == WEB_MESSAGE_RESPONSE_ACTION:
            return self._message_owner.response_enabled
        if action == DIALOG_CLOSE_ACTION:
            return True
        return bool(self.controller.is_action_enabled(action))

    def _request_finish(self, reason: CloseReason, result: int) -> bool:
        if self._cleaned_up:
            return False
        if self._progress_owner.active:
            self._progress_owner.request_active_cancel()
            return False
        if self._message_owner.active:
            self._message_owner.request_escape()
            return False
        if not self._close_is_allowed(reason):
            return False
        self._complete_finish(reason, result)
        return True

    def _complete_finish(self, reason: CloseReason, result: int) -> None:
        if self._cleaned_up:
            return
        self._close_reason = reason
        try:
            self._cleanup(reason)
        finally:
            QDialog.done(self, int(result))

    def _close_is_allowed(self, reason: CloseReason) -> bool:
        if self._progress_owner.active:
            return False
        if self._close_policy is not None:
            return bool(self._close_policy(reason))
        return bool(self.controller.can_close(reason))

    def _cleanup(self, reason: CloseReason) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self._close_reason = reason
        self._progress_owner.shutdown()
        self._message_owner.shutdown()
        self._bridge.invalidate()
        self._remove_profile_hook()
        try:
            with suppress(RuntimeError, ValueError):
                self.web.cleanup()
        finally:
            self.controller.on_dialog_closed(reason)

    def _remove_profile_hook(self) -> None:
        if not self._profile_hook_registered:
            return
        self._profile_hook_registered = False
        profile_hook = getattr(gui_hooks, "profile_will_close", None)
        if profile_hook is not None:
            with suppress(ValueError):
                profile_hook.remove(self._on_profile_will_close)

    def _retain(self) -> None:
        if not self._cleaned_up:
            _retained_web_dialogs.add(self)

    def _on_finished(self, result: int) -> None:
        try:
            if not self._cleaned_up:
                reason = (
                    CloseReason.ACCEPT
                    if int(result) == _dialog_code("Accepted", 1)
                    else CloseReason.REJECT
                )
                self._cleanup(reason)
        finally:
            _retained_web_dialogs.discard(self)
            with suppress(AttributeError, RuntimeError):
                self.setWindowModality(Qt.WindowModality.NonModal)
            if not self._delete_scheduled:
                self._delete_scheduled = True
                with suppress(RuntimeError):
                    self.deleteLater()

    def _on_profile_will_close(self) -> None:
        self.force_teardown(CloseReason.PROFILE_TEARDOWN)

    def _on_parent_destroyed(self, *_args: Any) -> None:
        self.force_teardown(CloseReason.PARENT_TEARDOWN)

    def _on_destroyed(self, *_args: Any) -> None:
        try:
            if not self._cleaned_up:
                self._cleanup(CloseReason.DESTROYED)
        finally:
            _retained_web_dialogs.discard(self)


def retained_web_dialog_count() -> int:
    """Expose the modeless retention count for lifecycle tests/diagnostics."""

    return len(_retained_web_dialogs)


def widget_uses_dark_palette(widget: Any) -> bool:
    """Best-effort dark-palette detection without binding controllers to Qt."""

    try:
        palette = widget.palette()
        base_brush = palette.base() if callable(getattr(palette, "base", None)) else None
        color = (
            base_brush.color()
            if base_brush is not None
            else palette.color(getattr(palette, "Base", None))
        )
        return bool(color.lightness() < 128)
    except Exception:
        return False


def _dialog_code(name: str, fallback: int) -> int:
    enum = getattr(QDialog, "DialogCode", None)
    value = getattr(enum, name, fallback) if enum is not None else fallback
    return int(value)
