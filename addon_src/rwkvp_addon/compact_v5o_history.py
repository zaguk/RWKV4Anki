"""Packed V5 provenance beside the shared scheduler/evaluation row storage.

The shared buffer is *not* a V5 native batch. V5 transport always uses its own
answer columns, retaining exact clocks, source ordinals and first-review flags.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from .compact_review_data import (
    PROCESS_REVIEW_KEYS,
    PackedProcessReviewRows,
    concatenate_process_review_rows,
)

V5O_SOURCE_RECORD = struct.Struct("<q?")
_SOURCE_KEYS = (
    "is_original_first_review", "source_ordinal", "review_finish_ms",
    "duration_ms", "review_start_ms",
)
_V5O_REVIEW_KEYS = PROCESS_REVIEW_KEYS + _SOURCE_KEYS


class PackedV5oReviewRows(Sequence[Mapping[str, Any]]):
    """Immutable mapping-shaped history, with nine provenance bytes per row.

    Contiguous slices and incremental concatenation share the sealed buffers.
    Exact finish/duration/start values are derived from the shared integer
    review ID and taken-millisecond fields, never from their float aliases.
    """

    __slots__ = ("_rows", "_sources")

    def __init__(self, rows: PackedProcessReviewRows, sources: memoryview) -> None:
        if len(sources) != len(rows) * V5O_SOURCE_RECORD.size:
            raise ValueError("V5 provenance and process history lengths differ")
        self._rows = rows.seal()
        self._sources = (sources.toreadonly(),)

    @classmethod
    def concatenate(
        cls, base: PackedV5oReviewRows, tail: PackedV5oReviewRows,
    ) -> PackedV5oReviewRows:
        result = object.__new__(cls)
        result._rows = concatenate_process_review_rows(base._rows, tail._rows)
        result._sources = tuple(
            source for part in (base, tail) for source in part._sources if source
        )
        return result

    @property
    def retained_bytes(self) -> int:
        return self._rows.retained_bytes + sum(len(source) for source in self._sources)

    def clear(self) -> None:
        self._rows = PackedProcessReviewRows().seal()
        self._sources = ()

    def set_first_raw_day_offset(self, value: int) -> None:
        self._rows.set_first_raw_day_offset(value)

    def iter_native_review_buffers(self) -> Iterator[memoryview]:
        # The evaluation cache hashes the shared compatibility schema, exactly
        # as it does for ordinary mappings. This is not V5 native transport.
        return self._rows.iter_native_review_buffers()

    def to_native_review_batch(self, batch_type: Any) -> Any:
        # Explicit V1 consumers may use the compatibility projection, as they
        # could with the former dictionaries. V5 runtime dispatch never takes
        # this path: it constructs V5oReviewBatch from exact answer columns.
        return self._rows.to_native_review_batch(batch_type)

    def iter_cache_record_values(self) -> Iterator[tuple[Any, ...]]:
        return self._rows.iter_cache_record_values()

    def iter_review_identity_values(self) -> Iterator[tuple[int, int]]:
        return self._rows.iter_review_identity_values()

    def iter_last_review_values(self) -> Iterator[tuple[int, int, int, float, int, int]]:
        return self._rows.iter_last_review_values()

    def iter_context_values(self) -> Iterator[tuple[int, int, int, float, int, int, int, bool]]:
        return self._rows.iter_context_values()

    def iter_history_fingerprint_values(self) -> Iterator[tuple[Any, ...]]:
        return self._rows.iter_history_fingerprint_values()

    def iter_answer_values(self) -> Iterator[tuple[int, int, int, int, int, bool]]:
        for (review_id, card_id, taken_millis, rating), (ordinal, first) in zip(
            self._rows.iter_answer_scalar_values(), self._iter_sources(), strict=True,
        ):
            yield review_id, card_id, taken_millis, rating, ordinal, first

    def _iter_sources(self) -> Iterator[tuple[int, bool]]:
        for source in self._sources:
            yield from V5O_SOURCE_RECORD.iter_unpack(source)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        for row, (ordinal, first) in zip(self._rows, self._iter_sources(), strict=True):
            yield _PackedV5oReviewRow(row, ordinal, first)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return [self[position] for position in range(start, stop, step)]
            result = object.__new__(type(self))
            result._rows = self._rows[index]
            sources = []
            offset = 0
            for source in self._sources:
                count = len(source) // V5O_SOURCE_RECORD.size
                left, right = max(0, start - offset), min(count, stop - offset)
                if left < right:
                    sources.append(
                        source[left * V5O_SOURCE_RECORD.size:right * V5O_SOURCE_RECORD.size]
                    )
                offset += count
            result._sources = tuple(sources)
            return result
        row = self._rows[index]  # Also validates negative/out-of-range indexes.
        position = int(index)
        if position < 0:
            position += len(self)
        for source in self._sources:
            count = len(source) // V5O_SOURCE_RECORD.size
            if position < count:
                ordinal, first = V5O_SOURCE_RECORD.unpack_from(
                    source, position * V5O_SOURCE_RECORD.size,
                )
                return _PackedV5oReviewRow(row, ordinal, first)
            position -= count
        raise IndexError(index)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        return (
            isinstance(other, Sequence) and len(self) == len(other)
            and all(left == right for left, right in zip(self, other, strict=True))
        )


class _PackedV5oReviewRow(Mapping[str, Any]):
    __slots__ = ("_row", "_ordinal", "_first")

    def __init__(self, row: Mapping[str, Any], ordinal: int, first: bool) -> None:
        self._row, self._ordinal, self._first = row, ordinal, first

    def __getitem__(self, key: str) -> Any:
        if key == "source_ordinal":
            return self._ordinal
        if key == "is_original_first_review":
            return self._first
        if key == "review_finish_ms":
            return self._row["review_id"]
        if key == "duration_ms":
            return self._row["taken_millis"]
        if key == "review_start_ms":
            return self._row["review_id"] - self._row["taken_millis"]
        return self._row[key]

    def __iter__(self) -> Iterator[str]:
        return iter(_V5O_REVIEW_KEYS)

    def __len__(self) -> int:
        return len(_V5O_REVIEW_KEYS)
