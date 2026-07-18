from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal

from ._standalone_overlay import create_standalone_overlay_host

WEB_MESSAGE_RESPONSE_ACTION = "message-response"
WEB_MESSAGE_CHECKBOX_ACTION = "message-checkbox-change"

MessageTone = Literal["info", "success", "warning", "error"]
MessageButtonVariant = Literal["primary", "secondary", "quiet", "destructive"]


@dataclass(frozen=True)
class WebMessageButton:
    outcome: str
    label: str
    variant: MessageButtonVariant = "secondary"


@dataclass(frozen=True)
class WebMessageCheckbox:
    label: str
    checked: bool = True

    def __post_init__(self) -> None:
        if not str(self.label).strip():
            raise ValueError("web message checkbox labels must not be empty")


@dataclass(frozen=True)
class WebMessageSpec:
    title: str
    message: str
    tone: MessageTone
    buttons: tuple[WebMessageButton, ...]
    escape_outcome: str
    initial_outcome: str
    details: str | None = None
    # This field is deliberately explicit: only add-on-owned, pre-escaped
    # renderer output may be passed here. Ordinary messages always use
    # ``message`` and are assigned through textContent in JavaScript.
    trusted_message_html: str | None = None
    checkbox: WebMessageCheckbox | None = None

    def __post_init__(self) -> None:
        if self.tone not in {"info", "success", "warning", "error"}:
            raise ValueError(f"unsupported web message tone: {self.tone}")
        if not self.buttons or len(self.buttons) > 3:
            raise ValueError("web messages require between one and three buttons")
        outcomes = tuple(button.outcome for button in self.buttons)
        if len(set(outcomes)) != len(outcomes):
            raise ValueError("web message button outcomes must be unique")
        if any(not _valid_outcome(outcome) for outcome in outcomes):
            raise ValueError("web message outcomes must be short action identifiers")
        if self.escape_outcome not in outcomes:
            raise ValueError("escape_outcome must name one of the message buttons")
        if self.initial_outcome not in outcomes:
            raise ValueError("initial_outcome must name one of the message buttons")
        for button in self.buttons:
            if button.variant not in {"primary", "secondary", "quiet", "destructive"}:
                raise ValueError(f"unsupported web message button variant: {button.variant}")


class WebMessageSession:
    """One token-scoped alert or confirmation owned by a WebView document."""

    def __init__(
        self,
        owner: WebMessageOwner,
        *,
        token: int,
        generation: int,
        spec: WebMessageSpec,
        on_result: Callable[[str], None],
        on_checkbox_changed: Callable[[bool], None] | None,
    ) -> None:
        self._owner = owner
        self.token = int(token)
        self.generation = int(generation)
        self.spec = spec
        self._on_result = on_result
        self._on_checkbox_changed = on_checkbox_changed
        self._checkbox_checked = None if spec.checkbox is None else bool(spec.checkbox.checked)
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def checkbox_checked(self) -> bool | None:
        return self._checkbox_checked

    def _resolve(self, outcome: str) -> bool:
        if self._finished:
            return False
        self._finished = True
        callback = self._on_result
        self._on_result = lambda _outcome: None
        self._on_checkbox_changed = None
        callback(str(outcome))
        return True

    def _update_checkbox(self, checked: bool) -> bool:
        if self._finished or self.spec.checkbox is None:
            return False
        checked = bool(checked)
        if checked == self._checkbox_checked:
            return True
        self._checkbox_checked = checked
        callback = self._on_checkbox_changed
        if callback is not None:
            callback(checked)
        return True

    def _shutdown(self) -> None:
        self._finished = True
        self._on_result = lambda _outcome: None
        self._on_checkbox_changed = None


class WebMessageOwner:
    """Host-local coordinator for in-page alerts and confirmations."""

    def __init__(
        self,
        *,
        eval_js: Callable[[str], None],
        generation: Callable[[], int],
        is_closed: Callable[[], bool],
        can_start: Callable[[], bool] | None = None,
    ) -> None:
        self._eval_js = eval_js
        self._generation = generation
        self._is_closed = is_closed
        self._can_start = can_start or (lambda: True)
        self._lock = RLock()
        self._next_token = 0
        self._active: WebMessageSession | None = None
        self._disposed = False

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active is not None

    @property
    def response_enabled(self) -> bool:
        return self.active

    def start(
        self,
        spec: WebMessageSpec,
        *,
        on_result: Callable[[str], None],
        on_checkbox_changed: Callable[[bool], None] | None = None,
    ) -> WebMessageSession:
        if not callable(on_result):
            raise TypeError("web message result callback must be callable")
        with self._lock:
            if self._disposed or self._is_closed():
                raise RuntimeError("cannot show a message in a closed web dialog")
            if self._active is not None:
                raise RuntimeError("this web dialog already owns an active message")
            if not self._can_start():
                raise RuntimeError("cannot show a message while another overlay is active")
            self._next_token += 1
            session = WebMessageSession(
                self,
                token=self._next_token,
                generation=int(self._generation()),
                spec=spec,
                on_result=on_result,
                on_checkbox_changed=on_checkbox_changed,
            )
            self._active = session
        self._eval("show", _message_payload(session))
        return session

    def respond(self, token: int, outcome: str) -> bool:
        with self._lock:
            session = self._active
            if (
                session is None
                or session.token != int(token)
                or session.generation != int(self._generation())
                or str(outcome) not in {button.outcome for button in session.spec.buttons}
            ):
                return False
            self._active = None
            can_render = not self._disposed and not self._is_closed()
        if can_render:
            self._eval("hide", {"token": session.token})
        session._resolve(str(outcome))
        return True

    def update_checkbox(self, token: int, checked: bool) -> bool:
        with self._lock:
            session = self._active
            if (
                session is None
                or session.token != int(token)
                or session.generation != int(self._generation())
            ):
                return False
        return session._update_checkbox(bool(checked))

    def request_escape(self) -> bool:
        with self._lock:
            session = self._active
        if session is None:
            return False
        return self.respond(session.token, session.spec.escape_outcome)

    def shutdown(self) -> None:
        with self._lock:
            if self._disposed:
                return
            self._disposed = True
            session = self._active
            self._active = None
        if session is not None:
            session._shutdown()

    def document_rerendered(self, generation: int) -> bool:
        """Rebind an active message after its underlying page was replaced."""

        with self._lock:
            session = self._active
            if session is None or self._disposed or self._is_closed():
                return False
            session.generation = int(generation)
        self._eval("show", _message_payload(session))
        return True

    def _eval(self, method: str, payload: dict[str, object]) -> None:
        script = (
            "window.RWKVMessage && "
            f"window.RWKVMessage.{method}({json.dumps(payload, ensure_ascii=False)});"
        )
        with suppress(AttributeError, RuntimeError):
            self._eval_js(script)


def alert_spec(
    *,
    title: str,
    message: str,
    tone: MessageTone = "error",
    details: str | None = None,
) -> WebMessageSpec:
    return WebMessageSpec(
        title=str(title),
        message=str(message),
        tone=tone,
        details=None if details is None else str(details),
        buttons=(WebMessageButton("dismiss", "OK", "primary"),),
        escape_outcome="dismiss",
        initial_outcome="dismiss",
    )


def confirmation_spec(
    *,
    title: str,
    message: str,
    confirm_label: str,
    cancel_label: str = "Cancel",
    destructive: bool = False,
    trusted_message_html: str | None = None,
    checkbox_label: str | None = None,
    checkbox_checked: bool = True,
) -> WebMessageSpec:
    return WebMessageSpec(
        title=str(title),
        message=str(message),
        trusted_message_html=trusted_message_html,
        tone="warning" if destructive else "info",
        buttons=(
            WebMessageButton("cancel", str(cancel_label), "secondary"),
            WebMessageButton(
                "confirm",
                str(confirm_label),
                "destructive" if destructive else "primary",
            ),
        ),
        escape_outcome="cancel",
        initial_outcome="cancel",
        checkbox=(
            WebMessageCheckbox(str(checkbox_label), bool(checkbox_checked))
            if checkbox_label is not None
            else None
        ),
    )


def choice_spec(
    *,
    title: str,
    message: str,
    choices: Sequence[WebMessageButton],
    tone: MessageTone = "warning",
    trusted_message_html: str | None = None,
    initial_outcome: str = "cancel",
) -> WebMessageSpec:
    buttons = tuple(choices)
    if not any(button.outcome == "cancel" for button in buttons):
        raise ValueError("web choices require an explicit cancel outcome")
    return WebMessageSpec(
        title=str(title),
        message=str(message),
        trusted_message_html=trusted_message_html,
        tone=tone,
        buttons=buttons,
        escape_outcome="cancel",
        initial_outcome=str(initial_outcome),
    )


def show_web_alert(
    *,
    parent: Any,
    title: str,
    message: str,
    tone: MessageTone = "error",
    details: str | None = None,
    on_closed: Callable[[], None] | None = None,
) -> WebMessageSession:
    callback = on_closed or (lambda: None)
    return show_web_message(
        parent=parent,
        spec=alert_spec(
            title=title,
            message=message,
            tone=tone,
            details=details,
        ),
        on_result=lambda _outcome: callback(),
    )


def show_web_warning(
    message: str,
    *,
    title: str,
    parent: Any,
    details: str | None = None,
) -> WebMessageSession:
    return show_web_alert(
        parent=parent,
        title=title,
        message=message,
        tone="error",
        details=details,
    )


def show_web_info(
    message: str,
    *,
    title: str,
    parent: Any,
    on_closed: Callable[[], None] | None = None,
) -> WebMessageSession:
    return show_web_alert(
        parent=parent,
        title=title,
        message=message,
        tone="info",
        on_closed=on_closed,
    )


def ask_web_confirmation(
    *,
    parent: Any,
    title: str,
    message: str,
    on_result: Callable[[bool], None],
    confirm_label: str = "Continue",
    cancel_label: str = "Cancel",
    destructive: bool = False,
    trusted_message_html: str | None = None,
    checkbox_label: str | None = None,
    checkbox_checked: bool = True,
    on_checkbox_changed: Callable[[bool], None] | None = None,
) -> WebMessageSession:
    return show_web_message(
        parent=parent,
        spec=confirmation_spec(
            title=title,
            message=message,
            confirm_label=confirm_label,
            cancel_label=cancel_label,
            destructive=destructive,
            trusted_message_html=trusted_message_html,
            checkbox_label=checkbox_label,
            checkbox_checked=checkbox_checked,
        ),
        on_result=lambda outcome: on_result(outcome == "confirm"),
        on_checkbox_changed=on_checkbox_changed,
    )


def ask_web_choice(
    *,
    parent: Any,
    title: str,
    message: str,
    choices: Sequence[WebMessageButton],
    on_result: Callable[[str], None],
    tone: MessageTone = "warning",
    trusted_message_html: str | None = None,
    initial_outcome: str = "cancel",
) -> WebMessageSession:
    return show_web_message(
        parent=parent,
        spec=choice_spec(
            title=title,
            message=message,
            choices=choices,
            tone=tone,
            trusted_message_html=trusted_message_html,
            initial_outcome=initial_outcome,
        ),
        on_result=on_result,
    )


def show_web_message(
    *,
    parent: Any,
    spec: WebMessageSpec,
    on_result: Callable[[str], None],
    on_checkbox_changed: Callable[[bool], None] | None = None,
) -> WebMessageSession:
    """Show in the parent's WebView, or create one asynchronous standalone host."""

    starter = getattr(parent, "start_web_message", None)
    if callable(starter):
        if on_checkbox_changed is None:
            return starter(spec, on_result=on_result)
        return starter(spec, on_result=on_result, on_checkbox_changed=on_checkbox_changed)
    return _show_standalone_message(
        parent=parent,
        spec=spec,
        on_result=on_result,
        on_checkbox_changed=on_checkbox_changed,
    )


def _show_standalone_message(
    *,
    parent: Any,
    spec: WebMessageSpec,
    on_result: Callable[[str], None],
    on_checkbox_changed: Callable[[bool], None] | None,
) -> WebMessageSession:
    from aqt import mw
    actual_parent = parent or mw
    dialog = create_standalone_overlay_host(
        parent=actual_parent,
        title=spec.title,
        status_title="Message",
        status_message="Review the message before returning to Anki.",
        status_state="disabled",
        size=(640, 390 if spec.details else 330),
    )

    def finish(outcome: str) -> None:
        if outcome in {"confirm", "dismiss"}:
            dialog.accept()
        else:
            dialog.reject()
        _defer_after_standalone_cleanup(lambda: on_result(outcome))

    if on_checkbox_changed is None:
        return dialog.start_web_message(spec, on_result=finish)
    return dialog.start_web_message(
        spec, on_result=finish, on_checkbox_changed=on_checkbox_changed
    )


def _defer_after_standalone_cleanup(callback: Callable[[], None]) -> None:
    try:
        from aqt import mw

        single_shot = getattr(getattr(mw, "progress", None), "single_shot", None)
        if callable(single_shot):
            single_shot(50, callback, True)
            return
    except Exception:
        pass
    callback()


def _message_payload(session: WebMessageSession) -> dict[str, object]:
    spec = session.spec
    return {
        "token": session.token,
        "title": spec.title,
        "message": spec.message,
        "messageHtml": spec.trusted_message_html,
        "details": spec.details,
        "tone": spec.tone,
        "buttons": [
            {
                "outcome": button.outcome,
                "label": button.label,
                "variant": button.variant,
            }
            for button in spec.buttons
        ],
        "escapeOutcome": spec.escape_outcome,
        "initialOutcome": spec.initial_outcome,
        "checkbox": (
            None
            if spec.checkbox is None
            else {
                "label": spec.checkbox.label,
                "checked": session.checkbox_checked,
            }
        ),
    }


def _valid_outcome(value: str) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= 32:
        return False
    return value[0].isalpha() and all(
        character.isalnum() or character == "-" for character in value
    )


__all__ = [
    "WEB_MESSAGE_RESPONSE_ACTION",
    "WEB_MESSAGE_CHECKBOX_ACTION",
    "WebMessageButton",
    "WebMessageCheckbox",
    "WebMessageOwner",
    "WebMessageSession",
    "WebMessageSpec",
    "alert_spec",
    "ask_web_choice",
    "ask_web_confirmation",
    "choice_spec",
    "confirmation_spec",
    "show_web_alert",
    "show_web_info",
    "show_web_message",
    "show_web_warning",
]
