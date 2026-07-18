from __future__ import annotations

import time
from dataclasses import dataclass


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
