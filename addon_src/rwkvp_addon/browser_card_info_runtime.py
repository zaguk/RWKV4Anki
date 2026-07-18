from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BrowserRuntimeLoadToken:
    """Identity for one explicit Browser runtime-load request."""

    browser_key: object
    generation: int


class BrowserCardInfoRuntime:
    """Card Info view over a scoped lease with an explicit visible-card scope."""

    def __init__(
        self,
        lease: object,
        *,
        visible_card_ids: set[int] | frozenset[int] | None,
    ) -> None:
        self._lease = lease
        self._visible_card_ids = (
            None
            if visible_card_ids is None
            else frozenset(int(card_id) for card_id in visible_card_ids)
        )

    @property
    def closed(self) -> bool:
        return bool(getattr(self._lease, "closed", False))

    def contains_card(self, card_id: int) -> bool:
        card_id = int(card_id)
        if self._visible_card_ids is not None and card_id not in self._visible_card_ids:
            return False
        contains_card = getattr(self._lease, "contains_card", None)
        return bool(callable(contains_card) and contains_card(card_id))

    def close(self) -> None:
        close = getattr(self._lease, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lease, name)


class BrowserCardInfoRuntimeOwner:
    """Own the one retained Browser Card Info runtime for a profile.

    Loading happens asynchronously. Generation tokens prevent an old load from
    being installed after a replacement request or after its Browser has closed.
    Lease closing is deliberately left to the GUI layer so it can run off the Qt
    thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._browser_key: object | None = None
        self._pending_token: BrowserRuntimeLoadToken | None = None
        self._runtime: object | None = None

    def begin_load(
        self,
        browser_key: object,
    ) -> tuple[BrowserRuntimeLoadToken, object | None]:
        with self._lock:
            self._generation += 1
            token = BrowserRuntimeLoadToken(browser_key, self._generation)
            previous = self._runtime
            self._runtime = None
            self._browser_key = browser_key
            self._pending_token = token
            return token, previous

    def complete_load(
        self,
        token: BrowserRuntimeLoadToken,
        runtime: object,
    ) -> bool:
        with self._lock:
            if token != self._pending_token or token.browser_key is not self._browser_key:
                return False
            self._runtime = runtime
            self._pending_token = None
            return True

    def cancel_load(self, token: BrowserRuntimeLoadToken) -> None:
        with self._lock:
            if token != self._pending_token:
                return
            self._pending_token = None
            self._browser_key = None

    def close_browser(self, browser_key: object) -> object | None:
        with self._lock:
            if browser_key is not self._browser_key:
                return None
            return self._clear_locked()

    def close_all(self) -> object | None:
        with self._lock:
            return self._clear_locked()

    def active_runtime(self) -> object | None:
        with self._lock:
            runtime = self._runtime
            if runtime is not None and bool(getattr(runtime, "closed", False)):
                self._clear_locked()
                return None
            return runtime

    def _clear_locked(self) -> object | None:
        self._generation += 1
        runtime = self._runtime
        self._runtime = None
        self._browser_key = None
        self._pending_token = None
        return runtime
