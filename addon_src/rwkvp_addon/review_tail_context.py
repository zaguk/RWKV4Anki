from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .review_rows import (
    LastReviewInfo,
    ReviewData,
    ReviewRowBuildContext,
    build_last_review_map,
    review_row_build_context_from_rows,
)
from .review_type_normalization import FilteredReviewNormalizationPolicy

REVIEW_TAIL_CONTEXT_VERSION = 1
_MAGIC = b"RWKVRTC1"
_PREFIX = struct.Struct("<8sII")
_CARD_RECORD = struct.Struct("<10q")
_MAX_HEADER_BYTES = 1024 * 1024


class ReviewTailContextError(ValueError):
    """Raised when a persisted review-tail context is malformed or stale."""


@dataclass(frozen=True)
class CollectionRevision:
    """Conservative snapshot used to reject non-append-only collection changes."""

    modified: int
    revlog_count: int
    latest_review_id: int | None
    created: int = 0
    schema_modified: int = 0

    def as_dict(self) -> dict[str, int | None]:
        return {
            "modified": int(self.modified),
            "revlog_count": int(self.revlog_count),
            "latest_review_id": (
                None if self.latest_review_id is None else int(self.latest_review_id)
            ),
            "created": int(self.created),
            "schema_modified": int(self.schema_modified),
        }

    @classmethod
    def from_dict(cls, value: object) -> CollectionRevision:
        if not isinstance(value, Mapping):
            raise ReviewTailContextError("Collection revision is missing.")
        try:
            latest = value.get("latest_review_id")
            revision = cls(
                modified=int(value["modified"]),
                revlog_count=int(value["revlog_count"]),
                latest_review_id=None if latest is None else int(latest),
                created=int(value.get("created", 0)),
                schema_modified=int(value.get("schema_modified", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewTailContextError("Collection revision is malformed.") from exc
        if revision.revlog_count < 0:
            raise ReviewTailContextError("Collection review count is invalid.")
        return revision


@dataclass(frozen=True)
class PersistedReviewTailContext:
    model_id: str
    exclude_deleted_card_revlogs: bool
    filtered_review_normalization_signature: dict[str, Any]
    checkpoint_history_fingerprint: dict[str, Any]
    collection_revision: CollectionRevision
    processed_review_count: int
    first_review_id: int
    first_card_id: int
    last_review_id: int
    last_card_id: int
    next_day_at: int
    day_offset_origin: int
    context: ReviewRowBuildContext
    last_by_card: dict[int, LastReviewInfo]
    latest_revlog_id_by_card: dict[int, int]


class CheckpointTailRows(Sequence[Mapping[str, Any]]):
    """A durable virtual prefix followed by a materialized review tail.

    Only boundary access and slices wholly inside the materialized tail are
    supported.  Callers must never scan or serialize the virtual prefix: its
    bytes live in the checkpoint/evaluation cache, not in this acceleration
    object.
    """

    __slots__ = ("_durable_count", "_first", "_last", "_tail")

    def __init__(
        self,
        *,
        durable_count: int,
        first_review_id: int,
        first_card_id: int,
        last_review_id: int,
        last_card_id: int,
        day_offset_origin: int,
        tail: Sequence[Mapping[str, Any]],
    ) -> None:
        durable_count = int(durable_count)
        if durable_count <= 0:
            raise ValueError("A virtual checkpoint prefix must contain reviews.")
        self._durable_count = durable_count
        self._first = {
            "review_id": int(first_review_id),
            "card_id": int(first_card_id),
            "raw_day_offset": int(day_offset_origin),
            "day_offset": 0,
        }
        self._last = {
            "review_id": int(last_review_id),
            "card_id": int(last_card_id),
        }
        self._tail = tail

    @property
    def durable_count(self) -> int:
        return self._durable_count

    @property
    def materialized_tail(self) -> Sequence[Mapping[str, Any]]:
        return self._tail

    def __len__(self) -> int:
        return self._durable_count + len(self._tail)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if start < self._durable_count:
                raise ReviewTailContextError(
                    "The durable checkpoint prefix is not materialized in Python."
                )
            tail_start = start - self._durable_count
            tail_stop = stop - self._durable_count
            return self._tail[slice(tail_start, tail_stop, step)]

        resolved = int(index)
        if resolved < 0:
            resolved += len(self)
        if resolved < 0 or resolved >= len(self):
            raise IndexError(index)
        if resolved == 0:
            return self._first
        if resolved == self._durable_count - 1:
            return self._last
        if resolved >= self._durable_count:
            return self._tail[resolved - self._durable_count]
        raise ReviewTailContextError("The durable checkpoint prefix is not materialized in Python.")

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        raise ReviewTailContextError("The durable checkpoint prefix is not materialized in Python.")


def read_collection_revision(col: object) -> CollectionRevision | None:
    """Read a stable, cheap collection/history invalidation snapshot."""

    try:
        # One SQLite statement gives all values from the same read snapshot, so
        # an answer cannot land between the identity/mod/count/max reads.
        row = col.db.first(
            "select (select crt from col), (select scm from col), "
            "(select mod from col), count(*), max(id) from revlog"
        )
        if row is None:
            return None
        created, schema_modified, modified, count, latest = row
        return CollectionRevision(
            modified=int(modified),
            revlog_count=int(count),
            latest_review_id=None if latest is None else int(latest),
            created=int(created),
            schema_modified=int(schema_modified),
        )
    except Exception:
        # Small test doubles and older development adapters may not expose the
        # complete schema through ``first``. Keep a conservative fallback; the
        # production Anki collection uses the atomic statement above.
        try:
            modified_value = col.mod
            modified = int(modified_value() if callable(modified_value) else modified_value)
            row = col.db.first("select count(*), max(id) from revlog")
            if row is None:
                return None
            count, latest = row
            return CollectionRevision(
                modified=modified,
                revlog_count=int(count),
                latest_review_id=None if latest is None else int(latest),
                created=int(getattr(col, "crt", 0)),
                schema_modified=0,
            )
        except Exception:
            return None


def write_review_tail_context(
    path: Path,
    review_data: ReviewData,
    *,
    durable_processed_review_count: int,
    model_id: str,
    exclude_deleted_card_revlogs: bool,
    filtered_review_normalization_policy: FilteredReviewNormalizationPolicy,
    checkpoint_history_fingerprint: Mapping[str, Any],
    collection_revision: CollectionRevision,
) -> None:
    """Persist context for the checkpoint's exact durable process-row prefix."""

    durable_count = int(durable_processed_review_count)
    if durable_count <= 0 or durable_count > len(review_data.rows):
        raise ReviewTailContextError("Durable review prefix is unavailable.")
    prefix = review_data.rows[:durable_count]
    first = prefix[0]
    last = prefix[-1]
    last_review_id = int(last["review_id"])
    context = review_row_build_context_from_rows(
        prefix,
        day_offset_origin=int(review_data.day_offset_origin),
        filtered_review_normalization_policy=filtered_review_normalization_policy,
    )
    last_by_card = build_last_review_map(prefix)
    latest_revlog_id_by_card: dict[int, int] = {}
    for row in review_data.revlogs:
        review_id = int(row["review_id"])
        if review_id > last_review_id:
            break
        latest_revlog_id_by_card[int(row["card_id"])] = review_id

    card_ids = sorted(context.previous_by_card)
    records = bytearray(len(card_ids) * _CARD_RECORD.size)
    for index, card_id in enumerate(card_ids):
        previous_day, previous_review_id = context.previous_by_card[card_id]
        last_review = last_by_card.get(card_id)
        if last_review is None:
            raise ReviewTailContextError("Latest-review context is incomplete.")
        _CARD_RECORD.pack_into(
            records,
            index * _CARD_RECORD.size,
            int(card_id),
            int(previous_day),
            int(previous_review_id),
            int(context.previous_review_kind_by_card.get(card_id, 0)),
            int(context.positive_day_counts_by_card.get(card_id, 0)),
            int(context.prior_lapses_by_card.get(card_id, 0)),
            int(context.filtered_review_phase_by_card.get(card_id, 0)),
            int(last_review.interval),
            int(last_review.lapse_count),
            int(latest_revlog_id_by_card.get(card_id, previous_review_id)),
        )

    normalized_fingerprint = _normalized_fingerprint(checkpoint_history_fingerprint)
    if normalized_fingerprint["processed_review_count"] != durable_count:
        raise ReviewTailContextError("Checkpoint history count does not match the prefix.")
    if normalized_fingerprint["last_review_id"] != last_review_id:
        raise ReviewTailContextError("Checkpoint last review does not match the prefix.")
    header = {
        "version": REVIEW_TAIL_CONTEXT_VERSION,
        "model_id": str(model_id),
        "exclude_deleted_card_revlogs": bool(exclude_deleted_card_revlogs),
        "filtered_review_normalization_signature": (
            filtered_review_normalization_policy.semantic_signature()
        ),
        "checkpoint_history_fingerprint": normalized_fingerprint,
        "collection_revision": collection_revision.as_dict(),
        "processed_review_count": durable_count,
        "first_review_id": int(first["review_id"]),
        "first_card_id": int(first["card_id"]),
        "last_review_id": last_review_id,
        "last_card_id": int(last["card_id"]),
        "next_day_at": int(review_data.next_day_at),
        "day_offset_origin": int(review_data.day_offset_origin),
        "card_record_count": len(card_ids),
        "card_record_size": _CARD_RECORD.size,
        "card_records_sha256": hashlib.sha256(records).hexdigest(),
    }
    header_bytes = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise ReviewTailContextError("Review-tail context header is too large.")
    payload = _PREFIX.pack(_MAGIC, REVIEW_TAIL_CONTEXT_VERSION, len(header_bytes))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.write(header_bytes)
            handle.write(records)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def load_review_tail_context(
    path: Path,
    *,
    model_id: str,
    exclude_deleted_card_revlogs: bool,
    filtered_review_normalization_policy: FilteredReviewNormalizationPolicy,
    checkpoint_history_fingerprint: Mapping[str, Any],
    collection_revision: CollectionRevision,
) -> PersistedReviewTailContext | None:
    """Load a matching context, returning ``None`` for any stale/corrupt cache."""

    try:
        return _load_review_tail_context(
            path,
            model_id=model_id,
            exclude_deleted_card_revlogs=exclude_deleted_card_revlogs,
            filtered_review_normalization_policy=filtered_review_normalization_policy,
            checkpoint_history_fingerprint=checkpoint_history_fingerprint,
            collection_revision=collection_revision,
        )
    except (OSError, json.JSONDecodeError, ReviewTailContextError, struct.error):
        return None


def _load_review_tail_context(
    path: Path,
    *,
    model_id: str,
    exclude_deleted_card_revlogs: bool,
    filtered_review_normalization_policy: FilteredReviewNormalizationPolicy,
    checkpoint_history_fingerprint: Mapping[str, Any],
    collection_revision: CollectionRevision,
) -> PersistedReviewTailContext:
    raw = Path(path).read_bytes()
    if len(raw) < _PREFIX.size:
        raise ReviewTailContextError("Review-tail context is truncated.")
    magic, version, header_size = _PREFIX.unpack_from(raw)
    if magic != _MAGIC or version != REVIEW_TAIL_CONTEXT_VERSION:
        raise ReviewTailContextError("Review-tail context version is unsupported.")
    if header_size <= 0 or header_size > _MAX_HEADER_BYTES:
        raise ReviewTailContextError("Review-tail context header is invalid.")
    header_end = _PREFIX.size + header_size
    if header_end > len(raw):
        raise ReviewTailContextError("Review-tail context header is truncated.")
    header = json.loads(raw[_PREFIX.size : header_end].decode("utf-8"))
    if not isinstance(header, Mapping):
        raise ReviewTailContextError("Review-tail context header is malformed.")
    if int(header.get("version", -1)) != REVIEW_TAIL_CONTEXT_VERSION:
        raise ReviewTailContextError("Review-tail context version is unsupported.")
    if str(header.get("model_id")) != str(model_id):
        raise ReviewTailContextError("Review-tail context model changed.")
    if bool(header.get("exclude_deleted_card_revlogs")) != bool(exclude_deleted_card_revlogs):
        raise ReviewTailContextError("Deleted-card review policy changed.")
    expected_policy = filtered_review_normalization_policy.semantic_signature()
    if header.get("filtered_review_normalization_signature") != expected_policy:
        raise ReviewTailContextError("Filtered-review policy changed.")
    expected_fingerprint = _normalized_fingerprint(checkpoint_history_fingerprint)
    stored_fingerprint = _normalized_fingerprint(header.get("checkpoint_history_fingerprint"))
    if stored_fingerprint != expected_fingerprint:
        raise ReviewTailContextError("Checkpoint history changed.")
    if CollectionRevision.from_dict(header.get("collection_revision")) != collection_revision:
        raise ReviewTailContextError("Collection revision changed.")

    try:
        processed_count = int(header["processed_review_count"])
        first_review_id = int(header["first_review_id"])
        first_card_id = int(header["first_card_id"])
        last_review_id = int(header["last_review_id"])
        last_card_id = int(header["last_card_id"])
        next_day_at = int(header["next_day_at"])
        day_offset_origin = int(header["day_offset_origin"])
        record_count = int(header["card_record_count"])
        record_size = int(header["card_record_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReviewTailContextError("Review-tail context metadata is malformed.") from exc
    if processed_count <= 0 or record_count < 0 or record_size != _CARD_RECORD.size:
        raise ReviewTailContextError("Review-tail context counts are invalid.")
    if processed_count != stored_fingerprint["processed_review_count"]:
        raise ReviewTailContextError("Review-tail context count changed.")
    if last_review_id != stored_fingerprint["last_review_id"]:
        raise ReviewTailContextError("Review-tail context last review changed.")
    records = raw[header_end:]
    if len(records) != record_count * _CARD_RECORD.size:
        raise ReviewTailContextError("Review-tail context card records are truncated.")
    if hashlib.sha256(records).hexdigest() != str(header.get("card_records_sha256")):
        raise ReviewTailContextError("Review-tail context card records are corrupt.")

    previous_by_card: dict[int, tuple[int, int]] = {}
    previous_kind_by_card: dict[int, int] = {}
    positive_day_counts_by_card: dict[int, int] = {}
    prior_lapses_by_card: dict[int, int] = {}
    filtered_phase_by_card: dict[int, int] = {}
    last_by_card: dict[int, LastReviewInfo] = {}
    latest_revlog_id_by_card: dict[int, int] = {}
    prior_card_id: int | None = None
    for values in _CARD_RECORD.iter_unpack(records):
        (
            card_id,
            previous_day,
            previous_review_id,
            previous_kind,
            positive_day_count,
            prior_lapses,
            filtered_phase,
            interval,
            lapse_count,
            latest_metadata_review_id,
        ) = values
        if prior_card_id is not None and card_id <= prior_card_id:
            raise ReviewTailContextError("Review-tail card records are not unique/sorted.")
        prior_card_id = card_id
        if positive_day_count < 0 or prior_lapses < 0 or lapse_count < 0:
            raise ReviewTailContextError("Review-tail card counters are invalid.")
        previous_by_card[card_id] = (previous_day, previous_review_id)
        previous_kind_by_card[card_id] = previous_kind
        if positive_day_count:
            positive_day_counts_by_card[card_id] = positive_day_count
        if prior_lapses:
            prior_lapses_by_card[card_id] = prior_lapses
        filtered_phase_by_card[card_id] = filtered_phase
        last_by_card[card_id] = LastReviewInfo(
            review_id=previous_review_id,
            day_offset=previous_day,
            timestamp_seconds=previous_review_id / 1000.0,
            interval=interval,
            lapse_count=lapse_count,
        )
        latest_revlog_id_by_card[card_id] = latest_metadata_review_id

    context = ReviewRowBuildContext(
        day_offset_origin=day_offset_origin,
        previous_by_card=previous_by_card,
        previous_review_kind_by_card=previous_kind_by_card,
        positive_day_counts_by_card=positive_day_counts_by_card,
        prior_lapses_by_card=prior_lapses_by_card,
        filtered_review_normalization_policy=filtered_review_normalization_policy,
        filtered_review_phase_by_card=filtered_phase_by_card,
    )
    return PersistedReviewTailContext(
        model_id=str(model_id),
        exclude_deleted_card_revlogs=bool(exclude_deleted_card_revlogs),
        filtered_review_normalization_signature=dict(expected_policy),
        checkpoint_history_fingerprint=stored_fingerprint,
        collection_revision=collection_revision,
        processed_review_count=processed_count,
        first_review_id=first_review_id,
        first_card_id=first_card_id,
        last_review_id=last_review_id,
        last_card_id=last_card_id,
        next_day_at=next_day_at,
        day_offset_origin=day_offset_origin,
        context=context,
        last_by_card=last_by_card,
        latest_revlog_id_by_card=latest_revlog_id_by_card,
    )


def _normalized_fingerprint(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReviewTailContextError("Checkpoint history fingerprint is missing.")
    try:
        digest = str(value["digest"]).strip().lower()
        fields = [str(field) for field in value["fields"]]
        last_review_id = value.get("last_review_id")
        result = {
            "version": int(value["version"]),
            "algorithm": str(value["algorithm"]),
            "canonicalization": str(value["canonicalization"]),
            "fields": fields,
            "processed_review_count": int(value["processed_review_count"]),
            "last_review_id": None if last_review_id is None else int(last_review_id),
            "digest": digest,
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ReviewTailContextError("Checkpoint history fingerprint is malformed.") from exc
    if result["processed_review_count"] < 0 or len(digest) != 64:
        raise ReviewTailContextError("Checkpoint history fingerprint is invalid.")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise ReviewTailContextError("Checkpoint history fingerprint is invalid.") from exc
    return result
