from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence, Set
from enum import Enum
from threading import RLock
from typing import Any

DEFAULT_LIVE_REVIEW_TRACE_EVENT_LIMIT = 16
_MAX_JSON_DEPTH = 8


class LiveReviewLifecycleTrace:
    """Keep a small, dependency-free trace of Live Review lifecycle events.

    The trace deliberately stores only recent transitions. It is intended to be
    cheap enough to leave enabled and safe to snapshot while a session is being
    torn down. Values supplied as event details are copied into a JSON-safe form
    when they are recorded, so a diagnostic renderer cannot accidentally retain
    Anki or runtime objects.
    """

    def __init__(
        self,
        max_events: int = DEFAULT_LIVE_REVIEW_TRACE_EVENT_LIMIT,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        normalized_limit = int(max_events)
        if normalized_limit < 1:
            raise ValueError("max_events must be at least 1")
        self._max_events = normalized_limit
        self._clock = clock or time.monotonic
        self._lock = RLock()
        self._events: deque[dict[str, Any]] = deque(maxlen=normalized_limit)
        self._phase = "inactive"
        self._maximum_minimum_retention_extra = 0.0
        self._recorded_event_count = 0
        self._started_at = 0.0
        self.reset()

    def reset(self, *, phase: str = "inactive") -> None:
        """Clear the previous session and begin a new trace in ``phase``."""

        with self._lock:
            self._events.clear()
            self._phase = _normalized_label(phase, fallback="inactive")
            self._maximum_minimum_retention_extra = 0.0
            self._recorded_event_count = 0
            self._started_at = float(self._clock())

    def record(
        self,
        event: str,
        *,
        phase: str | None = None,
        minimum_retention_extra: float | None = None,
        **details: Any,
    ) -> None:
        """Record one transition and optionally update the current phase.

        Invalid or non-finite retention-extra values are ignored. This is
        diagnostic instrumentation, so malformed optional data must never be
        able to interrupt a review session.
        """

        with self._lock:
            if phase is not None:
                self._phase = _normalized_label(phase, fallback=self._phase)

            normalized_extra = _finite_nonnegative_float(minimum_retention_extra)
            if normalized_extra is not None:
                self._maximum_minimum_retention_extra = max(
                    self._maximum_minimum_retention_extra,
                    normalized_extra,
                )

            self._recorded_event_count += 1
            entry: dict[str, Any] = {
                "sequence": self._recorded_event_count,
                "elapsed_ms": self._elapsed_ms(),
                "event": _normalized_label(event, fallback="unknown"),
                "phase": self._phase,
            }
            if normalized_extra is not None:
                entry["minimum_retention_extra"] = normalized_extra
            if details:
                entry["details"] = _json_safe(details)
            self._events.append(entry)

    def snapshot(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable diagnostic snapshot."""

        with self._lock:
            events = [_json_safe(event) for event in self._events]
            return {
                "phase": self._phase,
                "maximum_minimum_retention_extra": (
                    self._maximum_minimum_retention_extra
                ),
                "recorded_event_count": self._recorded_event_count,
                "dropped_event_count": max(0, self._recorded_event_count - len(events)),
                "recent_events": events,
            }

    def _elapsed_ms(self) -> float:
        elapsed = max(0.0, float(self._clock()) - self._started_at)
        return round(elapsed * 1_000.0, 3)


def _normalized_label(value: object, *, fallback: str) -> str:
    normalized = str(value).strip()
    return normalized or fallback


def _finite_nonnegative_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(normalized):
        return None
    return max(0.0, normalized)


def _json_safe(
    value: Any,
    *,
    depth: int = 0,
    ancestors: frozenset[int] = frozenset(),
) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return _json_safe(value.value, depth=depth, ancestors=ancestors)
    if depth >= _MAX_JSON_DEPTH:
        return "<maximum nesting reached>"

    identity = id(value)
    if identity in ancestors:
        return "<recursive reference>"
    nested_ancestors = ancestors | {identity}

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(
                item,
                depth=depth + 1,
                ancestors=nested_ancestors,
            )
            for key, item in value.items()
        }
    if isinstance(value, Set):
        normalized = [
            _json_safe(item, depth=depth + 1, ancestors=nested_ancestors)
            for item in value
        ]
        return sorted(normalized, key=repr)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_safe(item, depth=depth + 1, ancestors=nested_ancestors)
            for item in value
        ]
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return str(value)
