from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

T = TypeVar("T")


class ButtonLike(Protocol):
    def isEnabled(self) -> bool: ...

    def setEnabled(self, enabled: bool) -> None: ...


class ProgressActionLifecycle:
    def __init__(
        self,
        button: ButtonLike,
        *,
        before_start: Callable[[], None] | None = None,
        on_finish: Callable[[], None] | None = None,
    ) -> None:
        self.button = button
        self.before_start = before_start
        self.on_finish = on_finish
        self._started = False
        self._finished = False

    def begin(self) -> bool:
        if not self.button.isEnabled():
            return False
        if self.before_start:
            self.before_start()
        self.button.setEnabled(False)
        self._started = True
        return True

    def finish(self) -> bool:
        if not self._started or self._finished:
            return False
        self._finished = True
        if self.on_finish:
            self.on_finish()
        else:
            self.button.setEnabled(True)
        return True

    def wrap_success(self, callback: Callable[[T], None]) -> Callable[[T], None]:
        def wrapped(result: T) -> None:
            if self.finish():
                callback(result)

        return wrapped

    def wrap_failure(
        self,
        callback: Callable[[Exception], None],
    ) -> Callable[[Exception], None]:
        def wrapped(exception: Exception) -> None:
            if self.finish():
                callback(exception)

        return wrapped

    def wrap_cancel(self, callback: Callable[[], None] | None = None) -> Callable[[], None]:
        def wrapped() -> None:
            if self.finish() and callback:
                callback()

        return wrapped
