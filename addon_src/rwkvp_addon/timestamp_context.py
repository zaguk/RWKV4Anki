"""Exact Anki clock conversion shared by retained-history and Live inputs."""

from __future__ import annotations

import math
import operator
from typing import Any

MILLISECONDS_PER_DAY = 86_400_000
SIGNED_I64_MIN = -(1 << 63)
SIGNED_I64_MAX = (1 << 63) - 1


def timestamp_ms_from_seconds(value: Any) -> int:
    """Return ``value`` truncated to the millisecond used by Anki rows.

    Values that are already the canonical float representation of an integer
    millisecond are preserved before truncation.  For example, IEEE-754 stores
    ``1.001 * 1000`` just below ``1001`` even though ``1.001`` is exactly the
    value produced by ``1001 / 1000`` for the public seconds-shaped APIs.
    """

    seconds = float(_finite_number(value, "target_timestamp_seconds"))
    milliseconds_float = seconds * 1_000.0
    if not math.isfinite(milliseconds_float):
        raise OverflowError("target_timestamp_ms must fit a signed 64-bit integer")
    nearest_millisecond = round(milliseconds_float)
    milliseconds = (
        nearest_millisecond
        if nearest_millisecond / 1_000.0 == seconds
        else math.trunc(milliseconds_float)
    )
    return _signed_i64(milliseconds, "target_timestamp_ms")


def millisecond_aligned_timestamp_seconds(value: Any) -> float:
    """Return ``value`` aligned to the exact millisecond used by exact-timestamp rows.

    Anki review ids and the exact-timestamp timestamp fields are signed integer
    milliseconds. Add-on-facing clocks still use seconds, so normalize those
    values once before constructing query rows. Native V5 Live calls receive
    the resulting integer timestamp directly.
    """

    return timestamp_ms_from_seconds(value) / 1_000.0


def _seconds_to_milliseconds(value: Any, field: str) -> int:
    seconds = _signed_i64(value, field)
    return _signed_i64(seconds * 1_000, f"{field}_ms")


def rollover_phase_ms_from_next_day_at(next_day_at: Any) -> int:
    """Return the canonical UTC epoch-day phase for an Anki cutoff."""

    return _seconds_to_milliseconds(next_day_at, "next_day_at") % MILLISECONDS_PER_DAY


def _signed_i64(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError:
        if not isinstance(value, float) or not math.isfinite(value) or not value.is_integer():
            raise TypeError(f"{field} must be an exact integer") from None
        result = int(value)
    result = int(result)
    if result < SIGNED_I64_MIN or result > SIGNED_I64_MAX:
        raise OverflowError(f"{field} must fit a signed 64-bit integer")
    return result


def _finite_number(value: Any, field: str) -> int | float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a finite number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return int(number) if number.is_integer() else number
