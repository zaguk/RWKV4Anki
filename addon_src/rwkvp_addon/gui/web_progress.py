from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from threading import Lock, RLock
from typing import Any

from ..progress import format_eta
from ._standalone_overlay import create_standalone_overlay_host

WEB_PROGRESS_CANCEL_ACTION = "progress-cancel"


@dataclass(frozen=True)
class _ProgressUpdate:
    current: int
    total: int
    label: str
    eta: float | None


class WebProgressSession:
    """One cancellable operation displayed by a :class:`WebProgressOwner`.

    Worker threads may call :meth:`post_update`. UI mutation remains on the
    supplied main-thread scheduler, and multiple queued reports are coalesced
    to the newest value. The owner/token/generation checks make every late
    callback harmless after completion, rerender, or teardown.
    """

    def __init__(
        self,
        owner: WebProgressOwner,
        *,
        token: int,
        generation: int,
        title: str,
        label: str,
        schedule_on_main: Callable[[Callable[[], None]], None],
        on_cancel: Callable[[], None],
        on_finished: Callable[[], None] | None,
    ) -> None:
        self._owner = owner
        self.token = int(token)
        self.generation = int(generation)
        self.title = str(title)
        self._schedule_on_main = schedule_on_main
        self._on_cancel = on_cancel
        self._on_finished = on_finished
        self._state_lock = Lock()
        self._latest = _ProgressUpdate(0, 0, str(label), None)
        self._pending: _ProgressUpdate | None = None
        self._update_scheduled = False
        self._cancel_pending = False
        self._finished = False

    @property
    def cancel_pending(self) -> bool:
        with self._state_lock:
            return self._cancel_pending

    @property
    def finished(self) -> bool:
        with self._state_lock:
            return self._finished

    def post_update(
        self,
        current: int,
        total: int,
        label: str,
        eta: float | None,
    ) -> None:
        should_schedule = False
        with self._state_lock:
            if self._finished:
                return
            update = _ProgressUpdate(
                max(0, int(current)),
                max(0, int(total)),
                str(label or self._latest.label),
                eta,
            )
            self._latest = update
            self._pending = update
            if not self._update_scheduled:
                self._update_scheduled = True
                should_schedule = True
        if not should_schedule:
            return
        try:
            self._schedule_on_main(self._flush_update)
        except Exception:
            with self._state_lock:
                self._update_scheduled = False
                self._pending = None

    def finish(self) -> bool:
        with self._state_lock:
            if self._finished:
                return False
            self._finished = True
            self._pending = None
        return self._owner._finish_session(self)

    def _flush_update(self) -> None:
        with self._state_lock:
            update = self._pending
            self._pending = None
            self._update_scheduled = False
            if self._finished or update is None:
                return
            cancel_pending = self._cancel_pending
        self._owner._apply_update(self, update, cancel_pending=cancel_pending)

    def _request_cancel(self) -> bool:
        with self._state_lock:
            if self._finished or self._cancel_pending:
                return False
            self._cancel_pending = True
            update = self._latest
        self._owner._apply_update(self, update, cancel_pending=True)
        self._on_cancel()
        return True

    def _shutdown(self) -> None:
        callback: Callable[[], None] | None = None
        with self._state_lock:
            if self._finished:
                return
            self._finished = True
            self._pending = None
            if not self._cancel_pending:
                self._cancel_pending = True
                callback = self._on_cancel
        if callback is not None:
            callback()

    def _take_finished_callback(self) -> Callable[[], None] | None:
        with self._state_lock:
            callback = self._on_finished
            self._on_finished = None
            return callback


class WebProgressOwner:
    """Host-local progress coordinator shared by workflow and Settings WebViews."""

    def __init__(
        self,
        *,
        eval_js: Callable[[str], None],
        generation: Callable[[], int],
        is_closed: Callable[[], bool],
    ) -> None:
        self._eval_js = eval_js
        self._generation = generation
        self._is_closed = is_closed
        self._lock = RLock()
        self._next_token = 0
        self._active: WebProgressSession | None = None
        self._disposed = False

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active is not None

    @property
    def cancel_enabled(self) -> bool:
        with self._lock:
            session = self._active
        return bool(session is not None and not session.cancel_pending)

    def start(
        self,
        *,
        title: str,
        label: str,
        schedule_on_main: Callable[[Callable[[], None]], None],
        on_cancel: Callable[[], None],
        on_finished: Callable[[], None] | None = None,
    ) -> WebProgressSession:
        with self._lock:
            if self._disposed or self._is_closed():
                raise RuntimeError("cannot show progress in a closed web dialog")
            if self._active is not None:
                raise RuntimeError("this web dialog already owns an active operation")
            self._next_token += 1
            session = WebProgressSession(
                self,
                token=self._next_token,
                generation=int(self._generation()),
                title=title,
                label=label,
                schedule_on_main=schedule_on_main,
                on_cancel=on_cancel,
                on_finished=on_finished,
            )
            self._active = session
        self._show(session)
        return session

    def request_cancel(self, token: int) -> bool:
        with self._lock:
            session = self._active
            if session is None or session.token != int(token):
                return False
        return session._request_cancel()

    def request_active_cancel(self) -> bool:
        with self._lock:
            session = self._active
        return False if session is None else session._request_cancel()

    def shutdown(self) -> None:
        with self._lock:
            if self._disposed:
                return
            self._disposed = True
            session = self._active
            self._active = None
        if session is not None:
            session._shutdown()

    def _show(self, session: WebProgressSession) -> None:
        self._eval(
            "show",
            {
                "token": session.token,
                "title": session.title,
                "label": session._latest.label,
                "current": 0,
                "total": 0,
                "eta": format_eta(None),
                "cancellable": True,
                "cancelPending": False,
            },
        )

    def _apply_update(
        self,
        session: WebProgressSession,
        update: _ProgressUpdate,
        *,
        cancel_pending: bool,
    ) -> bool:
        with self._lock:
            if (
                self._disposed
                or self._is_closed()
                or self._active is not session
                or int(self._generation()) != session.generation
            ):
                return False
        self._eval(
            "update",
            {
                "token": session.token,
                "title": session.title,
                "label": update.label,
                "current": update.current,
                "total": update.total,
                "eta": format_eta(update.eta),
                "cancellable": True,
                "cancelPending": bool(cancel_pending),
            },
        )
        return True

    def _finish_session(self, session: WebProgressSession) -> bool:
        with self._lock:
            if self._active is not session:
                return False
            self._active = None
            can_render = (
                not self._disposed
                and not self._is_closed()
                and int(self._generation()) == session.generation
            )
        if can_render:
            self._eval("hide", {"token": session.token})
        callback = session._take_finished_callback()
        if callback is not None:
            callback()
        return can_render

    def _eval(self, method: str, payload: dict[str, object]) -> None:
        script = (
            "window.RWKVProgress && "
            f"window.RWKVProgress.{method}({json.dumps(payload, ensure_ascii=False)});"
        )
        with suppress(AttributeError, RuntimeError):
            self._eval_js(script)


def start_web_progress(
    *,
    parent: Any,
    title: str,
    label: str,
    schedule_on_main: Callable[[Callable[[], None]], None],
    on_cancel: Callable[[], None],
) -> WebProgressSession:
    """Use the parent's WebView when possible, otherwise open one web host."""

    starter = getattr(parent, "start_web_progress", None)
    if callable(starter):
        return starter(
            title=title,
            label=label,
            schedule_on_main=schedule_on_main,
            on_cancel=on_cancel,
        )

    dialog = create_standalone_overlay_host(
        parent=parent,
        title=title,
        status_title="Preparing",
        status_message=label,
        status_state="loading",
        size=(620, 330),
        intro="This operation can be cancelled without blocking Anki.",
        inline_progress=True,
    )
    return dialog.start_web_progress(
        title=title,
        label=label,
        schedule_on_main=schedule_on_main,
        on_cancel=on_cancel,
        on_finished=dialog.accept,
    )


__all__ = [
    "WEB_PROGRESS_CANCEL_ACTION",
    "WebProgressOwner",
    "WebProgressSession",
    "start_web_progress",
]
