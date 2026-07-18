from __future__ import annotations

import struct
from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Keep this layout identical to rwkv_srs.ReviewBatch.RECORD_FORMAT.  The add-on
# deliberately owns the format string as well so loading review history does
# not import the platform-specific native extension on Anki's UI thread.
NATIVE_PROCESS_REVIEW_FORMAT = "<qqqqqqqqdddqdd"
NATIVE_PROCESS_REVIEW_RECORD = struct.Struct(NATIVE_PROCESS_REVIEW_FORMAT)

_PROCESS_AUXILIARY_RECORD = struct.Struct("<qqqqqqqq")
_REVLOG_RECORD = struct.Struct("<qqqqqqqqqdqqqqqq")

PROCESS_REVIEW_KEYS = (
    "review_id",
    "card_id",
    "note_id",
    "deck_id",
    "preset_id",
    "raw_day_offset",
    "day_offset",
    "elapsed_days",
    "elapsed_seconds",
    "rating",
    "button_chosen",
    "duration",
    "taken_millis",
    "state",
    "review_kind",
    "interval",
    "last_interval",
    "ease_factor",
    "review_count",
    "i",
    "prior_lapses",
    "rmse_bins_lapse",
)

REVLOG_KEYS = (
    "review_id",
    "card_id",
    "note_id",
    "deck_id",
    "preset_id",
    "rating",
    "button_chosen",
    "duration",
    "taken_millis",
    "review_kind",
    "state",
    "interval",
    "last_interval",
    "ease_factor",
    "days_elapsed",
)


@dataclass
class _ProcessSegment:
    native: bytearray
    auxiliary: bytearray
    count: int


@dataclass
class _RevlogSegment:
    records: bytearray
    count: int


class PackedProcessReviewRows(Sequence[Mapping[str, Any]]):
    """Compact process rows with a direct RWKV-SRS ``ReviewBatch`` buffer.

    The native 112-byte record is retained exactly once.  Eight add-on-only
    integer columns occupy a second 64-byte record.  Dictionary-shaped access
    remains available through short-lived immutable mapping views, while Rust
    state replay can consume the native bytes without materializing mappings.

    Concatenated incremental histories retain their existing immutable
    segments.  This mirrors the old list copy's non-mutating behavior without
    copying the complete packed history for every newly appended tail.
    """

    __slots__ = (
        "_appendable",
        "_ends",
        "_raw_day_offset_overrides",
        "_segments",
        "_start",
        "_stop",
    )

    def __init__(self) -> None:
        self._segments: tuple[_ProcessSegment, ...] = (
            _ProcessSegment(bytearray(), bytearray(), 0),
        )
        self._ends = (0,)
        self._start = 0
        self._stop = 0
        self._appendable = True
        self._raw_day_offset_overrides: dict[int, int] = {}

    @classmethod
    def _from_parts(
        cls,
        segments: tuple[_ProcessSegment, ...],
        *,
        start: int,
        stop: int,
        raw_day_offset_overrides: Mapping[int, int] | None = None,
    ) -> PackedProcessReviewRows:
        rows = object.__new__(cls)
        rows._segments = segments
        total = 0
        ends: list[int] = []
        for segment in segments:
            total += int(segment.count)
            ends.append(total)
        rows._ends = tuple(ends)
        rows._start = max(0, min(total, int(start)))
        rows._stop = max(rows._start, min(total, int(stop)))
        rows._appendable = False
        rows._raw_day_offset_overrides = dict(raw_day_offset_overrides or {})
        return rows

    @classmethod
    def concatenate(
        cls,
        *parts: PackedProcessReviewRows,
    ) -> PackedProcessReviewRows:
        segments: list[_ProcessSegment] = []
        overrides: dict[int, int] = {}
        offset = 0
        for part in parts:
            if part._start != 0 or part._stop != part._total_segment_count:
                raise ValueError("Only complete packed histories can be concatenated.")
            part.seal()
            segments.extend(part._segments)
            for index, value in part._raw_day_offset_overrides.items():
                overrides[offset + int(index)] = int(value)
            offset += len(part)
        return cls._from_parts(
            tuple(segment for segment in segments if segment.count),
            start=0,
            stop=offset,
            raw_day_offset_overrides=overrides,
        )

    @property
    def _total_segment_count(self) -> int:
        return self._ends[-1] if self._ends else 0

    @property
    def retained_bytes(self) -> int:
        return sum(len(segment.native) + len(segment.auxiliary) for segment in self._segments)

    def append(self, row: Mapping[str, Any]) -> None:
        self.append_values(
            review_id=int(row["review_id"]),
            card_id=int(row["card_id"]),
            note_id=_optional_int(row.get("note_id")),
            deck_id=_optional_int(row.get("deck_id")),
            preset_id=_optional_int(row.get("preset_id")),
            raw_day_offset=int(row.get("raw_day_offset", row["day_offset"])),
            day_offset=float(row["day_offset"]),
            elapsed_days=float(row["elapsed_days"]),
            elapsed_seconds=float(row["elapsed_seconds"]),
            rating=int(row["rating"]),
            duration=float(row["duration"]),
            taken_millis=int(row.get("taken_millis", round(float(row["duration"])))),
            state=int(row["state"]),
            review_kind=int(row.get("review_kind", row["state"])),
            interval=int(row.get("interval", 0)),
            last_interval=int(row.get("last_interval", 0)),
            ease_factor=int(row.get("ease_factor", 0)),
            review_count=int(row.get("review_count", row.get("i", 1))),
            prior_lapses=int(row.get("prior_lapses", row.get("rmse_bins_lapse", 0))),
        )

    def append_values(
        self,
        *,
        review_id: int,
        card_id: int,
        note_id: int | None,
        deck_id: int | None,
        preset_id: int | None,
        raw_day_offset: int,
        day_offset: float,
        elapsed_days: float,
        elapsed_seconds: float,
        rating: int,
        duration: float,
        taken_millis: int,
        state: int,
        review_kind: int,
        interval: int,
        last_interval: int,
        ease_factor: int,
        review_count: int,
        prior_lapses: int,
    ) -> None:
        if not self._appendable or len(self._segments) != 1:
            raise TypeError("Packed review history is sealed and cannot be appended.")
        segment = self._segments[0]
        segment.native.extend(
            NATIVE_PROCESS_REVIEW_RECORD.pack(
                int(review_id),
                int(card_id),
                int(note_id is not None),
                0 if note_id is None else note_id,
                int(deck_id is not None),
                0 if deck_id is None else deck_id,
                int(preset_id is not None),
                0 if preset_id is None else preset_id,
                float(day_offset),
                float(elapsed_days),
                float(elapsed_seconds),
                int(rating),
                float(duration),
                float(state),
            )
        )
        segment.auxiliary.extend(
            _PROCESS_AUXILIARY_RECORD.pack(
                int(raw_day_offset),
                int(taken_millis),
                int(review_kind),
                int(interval),
                int(last_interval),
                int(ease_factor),
                int(review_count),
                int(prior_lapses),
            )
        )
        segment.count += 1
        self._ends = (segment.count,)
        self._stop = segment.count

    def seal(self) -> PackedProcessReviewRows:
        self._appendable = False
        return self

    def clear(self) -> None:
        # Drop this sequence's references rather than mutating shared segment
        # bytearrays that may belong to a newer incremental history.
        self._segments = ()
        self._ends = ()
        self._start = 0
        self._stop = 0
        self._appendable = False
        self._raw_day_offset_overrides = {}

    def set_first_raw_day_offset(self, value: int) -> None:
        if self:
            self._raw_day_offset_overrides[self._start] = int(value)

    def to_native_review_batch(self, review_batch_type: Any) -> Any:
        """Create an immutable native batch, using the packed fast path when possible."""

        native_buffer = self._single_segment_native_buffer()
        if native_buffer is not None:
            record_format = getattr(review_batch_type, "RECORD_FORMAT", None)
            record_size = getattr(review_batch_type, "RECORD_SIZE", None)
            if (
                record_format != NATIVE_PROCESS_REVIEW_FORMAT
                or int(record_size or 0) != NATIVE_PROCESS_REVIEW_RECORD.size
            ):
                raise RuntimeError(
                    "RWKV-SRS ReviewBatch uses an incompatible packed review format."
                )
            return review_batch_type.from_buffer(native_buffer)
        # An incremental history can span a small number of shared segments.
        # Falling back to native mapping parsing avoids a full-history packed
        # copy solely to make those segments contiguous.
        return review_batch_type(self)

    def iter_native_review_buffers(self) -> Iterator[memoryview]:
        """Yield the canonical native record bytes covered by this sequence.

        Evaluation-cache revisions can hash these already-normalized buffers
        directly. Slices and segmented incremental histories expose only their
        selected byte ranges and never require a contiguous full-history copy.
        """

        absolute_start = self._start
        absolute_stop = self._stop
        segment_start = 0
        for segment in self._segments:
            segment_stop = segment_start + segment.count
            overlap_start = max(absolute_start, segment_start)
            overlap_stop = min(absolute_stop, segment_stop)
            if overlap_start < overlap_stop:
                local_start = overlap_start - segment_start
                local_stop = overlap_stop - segment_start
                byte_start = local_start * NATIVE_PROCESS_REVIEW_RECORD.size
                byte_stop = local_stop * NATIVE_PROCESS_REVIEW_RECORD.size
                yield memoryview(segment.native)[byte_start:byte_stop]
            if segment_stop >= absolute_stop:
                break
            segment_start = segment_stop

    def iter_last_review_values(
        self,
    ) -> Iterator[tuple[int, int, int, float, int, int]]:
        """Yield the scalar subset used to build ``LastReviewInfo`` values."""

        for absolute in range(self._start, self._stop):
            segment_index, local_index = self._locate(absolute)
            segment = self._segments[segment_index]
            native = NATIVE_PROCESS_REVIEW_RECORD.unpack_from(
                segment.native,
                local_index * NATIVE_PROCESS_REVIEW_RECORD.size,
            )
            auxiliary = _PROCESS_AUXILIARY_RECORD.unpack_from(
                segment.auxiliary,
                local_index * _PROCESS_AUXILIARY_RECORD.size,
            )
            yield (
                int(native[0]),
                int(native[1]),
                int(native[8]),
                float(native[9]),
                int(native[11]),
                int(auxiliary[3]),
            )

    def iter_review_identity_values(self) -> Iterator[tuple[int, int]]:
        """Yield review/card identifiers without constructing mapping views."""

        for absolute in range(self._start, self._stop):
            segment_index, local_index = self._locate(absolute)
            native = NATIVE_PROCESS_REVIEW_RECORD.unpack_from(
                self._segments[segment_index].native,
                local_index * NATIVE_PROCESS_REVIEW_RECORD.size,
            )
            yield int(native[0]), int(native[1])

    def iter_history_fingerprint_values(self) -> Iterator[tuple[Any, ...]]:
        """Yield values in ``HISTORY_FINGERPRINT_FIELDS`` order."""

        for absolute in range(self._start, self._stop):
            segment_index, local_index = self._locate(absolute)
            segment = self._segments[segment_index]
            native = NATIVE_PROCESS_REVIEW_RECORD.unpack_from(
                segment.native,
                local_index * NATIVE_PROCESS_REVIEW_RECORD.size,
            )
            yield (
                int(native[0]),
                int(native[1]),
                int(native[3]) if native[2] else None,
                int(native[5]) if native[4] else None,
                int(native[7]) if native[6] else None,
                _float_or_int(native[8]),
                _float_or_int(native[9]),
                _float_or_int(native[10]),
                int(native[11]),
                float(native[12]),
                int(native[13]),
            )

    def iter_cache_record_values(self) -> Iterator[tuple[Any, ...]]:
        """Yield values in the evaluation cache's base-column order."""

        for absolute in range(self._start, self._stop):
            segment_index, local_index = self._locate(absolute)
            segment = self._segments[segment_index]
            native = NATIVE_PROCESS_REVIEW_RECORD.unpack_from(
                segment.native,
                local_index * NATIVE_PROCESS_REVIEW_RECORD.size,
            )
            auxiliary = _PROCESS_AUXILIARY_RECORD.unpack_from(
                segment.auxiliary,
                local_index * _PROCESS_AUXILIARY_RECORD.size,
            )
            review_count = int(auxiliary[6])
            prior_lapses = int(auxiliary[7])
            yield (
                int(native[0]),
                int(native[1]),
                int(native[5]) if native[4] else None,
                int(native[7]) if native[6] else None,
                int(native[11]),
                float(native[9]),
                float(native[10]),
                review_count,
                review_count,
                prior_lapses,
                prior_lapses,
            )

    def iter_context_values(
        self,
    ) -> Iterator[tuple[int, int, int, float, int, int, int, bool]]:
        """Yield the scalar subset used to reconstruct incremental context."""

        for absolute in range(self._start, self._stop):
            segment_index, local_index = self._locate(absolute)
            segment = self._segments[segment_index]
            native = NATIVE_PROCESS_REVIEW_RECORD.unpack_from(
                segment.native,
                local_index * NATIVE_PROCESS_REVIEW_RECORD.size,
            )
            auxiliary = _PROCESS_AUXILIARY_RECORD.unpack_from(
                segment.auxiliary,
                local_index * _PROCESS_AUXILIARY_RECORD.size,
            )
            yield (
                int(native[0]),
                int(native[1]),
                int(native[8]),
                float(native[9]),
                int(native[11]),
                int(auxiliary[2]),
                int(native[13]),
                True,
            )

    def __len__(self) -> int:
        return self._stop - self._start

    def __getitem__(
        self,
        index: int | slice,
    ) -> Mapping[str, Any] | PackedProcessReviewRows | list[Mapping[str, Any]]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return [self[position] for position in range(start, stop, step)]
            return self._from_parts(
                self._segments,
                start=self._start + start,
                stop=self._start + stop,
                raw_day_offset_overrides=self._raw_day_offset_overrides,
            )
        normalized = _normalized_index(index, len(self))
        absolute = self._start + normalized
        segment_index, local_index = self._locate(absolute)
        return _PackedProcessReviewRow(self, absolute, segment_index, local_index)

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        for absolute in range(self._start, self._stop):
            segment_index, local_index = self._locate(absolute)
            yield _PackedProcessReviewRow(self, absolute, segment_index, local_index)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Sequence) or len(self) != len(other):
            return False
        return all(left == right for left, right in zip(self, other, strict=True))

    def __repr__(self) -> str:
        return f"PackedProcessReviewRows(count={len(self):,}, bytes={self.retained_bytes:,})"

    def _locate(self, absolute_index: int) -> tuple[int, int]:
        segment_index = bisect_right(self._ends, absolute_index)
        segment_start = 0 if segment_index == 0 else self._ends[segment_index - 1]
        return segment_index, absolute_index - segment_start

    def _single_segment_native_buffer(self) -> memoryview | None:
        if not self:
            return memoryview(b"")
        start_segment, start_local = self._locate(self._start)
        end_segment, end_local = self._locate(self._stop - 1)
        if start_segment != end_segment:
            return None
        segment = self._segments[start_segment]
        byte_start = start_local * NATIVE_PROCESS_REVIEW_RECORD.size
        byte_stop = (end_local + 1) * NATIVE_PROCESS_REVIEW_RECORD.size
        return memoryview(segment.native)[byte_start:byte_stop]


class _PackedProcessReviewRow(Mapping[str, Any]):
    __slots__ = ("_absolute", "_auxiliary", "_local", "_native", "_segment", "_table")

    def __init__(
        self,
        table: PackedProcessReviewRows,
        absolute: int,
        segment: int,
        local: int,
    ) -> None:
        self._table = table
        self._absolute = absolute
        self._segment = segment
        self._local = local
        self._native: tuple[Any, ...] | None = None
        self._auxiliary: tuple[Any, ...] | None = None

    def __getitem__(self, key: str) -> Any:
        native = self._native_values
        if key == "review_id":
            return native[0]
        if key == "card_id":
            return native[1]
        if key == "note_id":
            return native[3] if native[2] else None
        if key == "deck_id":
            return native[5] if native[4] else None
        if key == "preset_id":
            return native[7] if native[6] else None
        if key == "day_offset":
            return _float_or_int(native[8])
        if key == "elapsed_days":
            return _float_or_int(native[9])
        if key == "elapsed_seconds":
            return _float_or_int(native[10])
        if key in {"rating", "button_chosen"}:
            return native[11]
        if key == "duration":
            return native[12]
        if key == "state":
            return int(native[13])
        auxiliary = self._auxiliary_values
        if key == "raw_day_offset":
            return self._table._raw_day_offset_overrides.get(self._absolute, auxiliary[0])
        if key == "taken_millis":
            return auxiliary[1]
        if key == "review_kind":
            return auxiliary[2]
        if key == "interval":
            return auxiliary[3]
        if key == "last_interval":
            return auxiliary[4]
        if key == "ease_factor":
            return auxiliary[5]
        if key in {"review_count", "i"}:
            return auxiliary[6]
        if key in {"prior_lapses", "rmse_bins_lapse"}:
            return auxiliary[7]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(PROCESS_REVIEW_KEYS)

    def __len__(self) -> int:
        return len(PROCESS_REVIEW_KEYS)

    @property
    def _native_values(self) -> tuple[Any, ...]:
        values = self._native
        if values is None:
            segment = self._table._segments[self._segment]
            values = NATIVE_PROCESS_REVIEW_RECORD.unpack_from(
                segment.native,
                self._local * NATIVE_PROCESS_REVIEW_RECORD.size,
            )
            self._native = values
        return values

    @property
    def _auxiliary_values(self) -> tuple[Any, ...]:
        values = self._auxiliary
        if values is None:
            segment = self._table._segments[self._segment]
            values = _PROCESS_AUXILIARY_RECORD.unpack_from(
                segment.auxiliary,
                self._local * _PROCESS_AUXILIARY_RECORD.size,
            )
            self._auxiliary = values
        return values


class PackedRevlogRows(Sequence[Mapping[str, Any]]):
    """Compact immutable dictionary-shaped rows used by evaluation metadata."""

    __slots__ = ("_appendable", "_ends", "_segments", "_start", "_stop")

    def __init__(self) -> None:
        self._segments: tuple[_RevlogSegment, ...] = (_RevlogSegment(bytearray(), 0),)
        self._ends = (0,)
        self._start = 0
        self._stop = 0
        self._appendable = True

    @classmethod
    def _from_parts(
        cls,
        segments: tuple[_RevlogSegment, ...],
        *,
        start: int,
        stop: int,
    ) -> PackedRevlogRows:
        rows = object.__new__(cls)
        rows._segments = segments
        total = 0
        ends: list[int] = []
        for segment in segments:
            total += int(segment.count)
            ends.append(total)
        rows._ends = tuple(ends)
        rows._start = max(0, min(total, int(start)))
        rows._stop = max(rows._start, min(total, int(stop)))
        rows._appendable = False
        return rows

    @classmethod
    def concatenate(cls, *parts: PackedRevlogRows) -> PackedRevlogRows:
        segments: list[_RevlogSegment] = []
        count = 0
        for part in parts:
            if part._start != 0 or part._stop != part._total_segment_count:
                raise ValueError("Only complete packed revlog histories can be concatenated.")
            part.seal()
            segments.extend(part._segments)
            count += len(part)
        return cls._from_parts(
            tuple(segment for segment in segments if segment.count),
            start=0,
            stop=count,
        )

    @property
    def _total_segment_count(self) -> int:
        return self._ends[-1] if self._ends else 0

    @property
    def retained_bytes(self) -> int:
        return sum(len(segment.records) for segment in self._segments)

    def append(self, row: Mapping[str, Any]) -> None:
        self.append_values(
            review_id=int(row["review_id"]),
            card_id=int(row["card_id"]),
            note_id=_optional_int(row.get("note_id")),
            deck_id=_optional_int(row.get("deck_id")),
            preset_id=_optional_int(row.get("preset_id")),
            rating=int(row.get("rating", row.get("button_chosen", 0))),
            duration=float(row.get("duration", 0.0)),
            taken_millis=int(row.get("taken_millis", 0)),
            review_kind=int(row.get("review_kind", row.get("state", 0))),
            interval=int(row.get("interval", 0)),
            last_interval=int(row.get("last_interval", 0)),
            ease_factor=int(row.get("ease_factor", 0)),
            days_elapsed=int(row.get("days_elapsed", 0)),
        )

    def append_values(
        self,
        *,
        review_id: int,
        card_id: int,
        note_id: int | None,
        deck_id: int | None,
        preset_id: int | None,
        rating: int,
        duration: float,
        taken_millis: int,
        review_kind: int,
        interval: int,
        last_interval: int,
        ease_factor: int,
        days_elapsed: int,
    ) -> None:
        if not self._appendable or len(self._segments) != 1:
            raise TypeError("Packed revlog history is sealed and cannot be appended.")
        segment = self._segments[0]
        segment.records.extend(
            _REVLOG_RECORD.pack(
                int(review_id),
                int(card_id),
                int(note_id is not None),
                0 if note_id is None else note_id,
                int(deck_id is not None),
                0 if deck_id is None else deck_id,
                int(preset_id is not None),
                0 if preset_id is None else preset_id,
                int(rating),
                float(duration),
                int(taken_millis),
                int(review_kind),
                int(interval),
                int(last_interval),
                int(ease_factor),
                int(days_elapsed),
            )
        )
        segment.count += 1
        self._ends = (segment.count,)
        self._stop = segment.count

    def seal(self) -> PackedRevlogRows:
        self._appendable = False
        return self

    def clear(self) -> None:
        self._segments = ()
        self._ends = ()
        self._start = 0
        self._stop = 0
        self._appendable = False

    def __len__(self) -> int:
        return self._stop - self._start

    def __getitem__(
        self,
        index: int | slice,
    ) -> Mapping[str, Any] | PackedRevlogRows | list[Mapping[str, Any]]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return [self[position] for position in range(start, stop, step)]
            return self._from_parts(
                self._segments,
                start=self._start + start,
                stop=self._start + stop,
            )
        normalized = _normalized_index(index, len(self))
        absolute = self._start + normalized
        segment_index, local_index = self._locate(absolute)
        return _PackedRevlogRow(self, segment_index, local_index)

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        for absolute in range(self._start, self._stop):
            segment_index, local_index = self._locate(absolute)
            yield _PackedRevlogRow(self, segment_index, local_index)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Sequence) or len(self) != len(other):
            return False
        return all(left == right for left, right in zip(self, other, strict=True))

    def __repr__(self) -> str:
        return f"PackedRevlogRows(count={len(self):,}, bytes={self.retained_bytes:,})"

    def _locate(self, absolute_index: int) -> tuple[int, int]:
        segment_index = bisect_right(self._ends, absolute_index)
        segment_start = 0 if segment_index == 0 else self._ends[segment_index - 1]
        return segment_index, absolute_index - segment_start


class _PackedRevlogRow(Mapping[str, Any]):
    __slots__ = ("_local", "_segment", "_table", "_values")

    def __init__(self, table: PackedRevlogRows, segment: int, local: int) -> None:
        self._table = table
        self._segment = segment
        self._local = local
        self._values: tuple[Any, ...] | None = None

    def __getitem__(self, key: str) -> Any:
        values = self._record_values
        if key == "review_id":
            return values[0]
        if key == "card_id":
            return values[1]
        if key == "note_id":
            return values[3] if values[2] else None
        if key == "deck_id":
            return values[5] if values[4] else None
        if key == "preset_id":
            return values[7] if values[6] else None
        if key in {"rating", "button_chosen"}:
            return values[8]
        if key == "duration":
            return values[9]
        if key == "taken_millis":
            return values[10]
        if key in {"review_kind", "state"}:
            return values[11]
        if key == "interval":
            return values[12]
        if key == "last_interval":
            return values[13]
        if key == "ease_factor":
            return values[14]
        if key == "days_elapsed":
            return values[15]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(REVLOG_KEYS)

    def __len__(self) -> int:
        return len(REVLOG_KEYS)

    @property
    def _record_values(self) -> tuple[Any, ...]:
        values = self._values
        if values is None:
            segment = self._table._segments[self._segment]
            values = _REVLOG_RECORD.unpack_from(
                segment.records,
                self._local * _REVLOG_RECORD.size,
            )
            self._values = values
        return values


def concatenate_process_review_rows(
    base: Sequence[Mapping[str, Any]],
    tail: Sequence[Mapping[str, Any]],
) -> Sequence[Mapping[str, Any]]:
    return PackedProcessReviewRows.concatenate(
        _complete_packed_process_rows(base),
        _complete_packed_process_rows(tail),
    )


def concatenate_revlog_rows(
    base: Sequence[Mapping[str, Any]],
    tail: Sequence[Mapping[str, Any]],
) -> Sequence[Mapping[str, Any]]:
    return PackedRevlogRows.concatenate(
        _complete_packed_revlog_rows(base),
        _complete_packed_revlog_rows(tail),
    )


def native_review_batch_for_rows(review_batch_type: Any, rows: object) -> Any:
    build = getattr(rows, "to_native_review_batch", None)
    if callable(build):
        return build(review_batch_type)
    return review_batch_type(rows)


def set_first_raw_day_offset(rows: object, value: int) -> bool:
    setter = getattr(rows, "set_first_raw_day_offset", None)
    if not callable(setter):
        return False
    setter(int(value))
    return True


def _complete_packed_process_rows(
    rows: Sequence[Mapping[str, Any]],
) -> PackedProcessReviewRows:
    if (
        isinstance(rows, PackedProcessReviewRows)
        and rows._start == 0
        and rows._stop == rows._total_segment_count
    ):
        return rows
    packed = PackedProcessReviewRows()
    for row in rows:
        packed.append(row)
    return packed.seal()


def _complete_packed_revlog_rows(
    rows: Sequence[Mapping[str, Any]],
) -> PackedRevlogRows:
    if (
        isinstance(rows, PackedRevlogRows)
        and rows._start == 0
        and rows._stop == rows._total_segment_count
    ):
        return rows
    packed = PackedRevlogRows()
    for row in rows:
        packed.append(row)
    return packed.seal()


def _normalized_index(index: int, length: int) -> int:
    normalized = int(index)
    if normalized < 0:
        normalized += length
    if normalized < 0 or normalized >= length:
        raise IndexError("review row index out of range")
    return normalized


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _float_or_int(value: float) -> float | int:
    number = float(value)
    return int(number) if number.is_integer() else number
