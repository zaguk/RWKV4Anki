from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

ProgressCallback = Callable[[int, int, str, float | None], None]


class CoalescedProgressCallback:
    """Keep at most one queued UI delivery, replacing it with the latest report.

    Posting is thread-safe; delivery runs only through the supplied scheduler.
    Closing discards pending and later reports without blocking the worker.
    """

    def __init__(
        self,
        callback: ProgressCallback,
        *,
        schedule: Callable[[Callable[[], None]], None],
    ) -> None:
        self._callback = callback
        self._schedule = schedule
        self._lock = Lock()
        self._pending: tuple[int, int, str, float | None] | None = None
        self._scheduled = False
        self._closed = False

    def __call__(self, current: int, total: int, label: str, eta: float | None) -> None:
        with self._lock:
            if self._closed:
                return
            self._pending = (current, total, label, eta)
            if self._scheduled:
                return
            self._scheduled = True
        try:
            # Scheduling outside the lock also supports immediate schedulers.
            self._schedule(self._flush)
        except Exception:
            with self._lock:
                self._scheduled = False
                self._pending = None
            raise

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._pending = None

    def _flush(self) -> None:
        with self._lock:
            pending = self._pending
            self._pending = None
            self._scheduled = False
            if self._closed or pending is None:
                return
        self._callback(*pending)


class CancelledError(RuntimeError):
    pass


@dataclass
class ProgressState:
    current: int = 0
    total: int = 0
    label: str = ""


class ProgressReporter:
    def __init__(self) -> None:
        self.state = ProgressState()

    def update(self, current: int, total: int, label: str = "") -> None:
        self.state = ProgressState(current=current, total=total, label=label)

    def check_cancelled(self) -> None:
        return


class TimedProgressReporter(ProgressReporter):
    def __init__(self, callback=None) -> None:
        super().__init__()
        self._callback = callback
        self._cancelled = False
        self._started = time.monotonic()

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def check_cancelled(self) -> None:
        if self._cancelled:
            raise CancelledError("Operation cancelled.")

    def update(self, current: int, total: int, label: str = "") -> None:
        previous = self.state
        now = time.monotonic()
        # Background workflows reuse one reporter for several independently
        # sized stages. Reset the ETA clock when a new stage starts instead of
        # extrapolating from all of the work that preceded it.
        if previous.total > 0 and (
            int(total) != int(previous.total) or int(current) < int(previous.current)
        ):
            self._started = now
        super().update(current, total, label)
        if self._callback:
            elapsed = max(0.001, now - self._started)
            eta = None
            if current > 0 and total > 0:
                eta = max(0.0, (total - current) * (elapsed / current))
            self._callback(current, total, label, eta)
        self.check_cancelled()


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "ETA unknown"
    seconds = int(seconds)
    if seconds < 60:
        return f"ETA {seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"ETA {minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"ETA {hours}h {minutes}m"
