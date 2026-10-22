from __future__ import annotations

import hashlib
import json
import math
import numbers
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .review_schema import HISTORY_FINGERPRINT_FIELDS

RUST_CHECKPOINT_MAGIC = b"RWKVPCPUBINCHK1"
CURRENT_RUST_CHECKPOINT_STORAGE_VERSION = 2
RECOGNIZED_RUST_CHECKPOINT_STORAGE_VERSIONS = frozenset({1, 2, 3, 4, 5, 6})
HISTORY_FINGERPRINT_VERSION = 1
HISTORY_FINGERPRINT_ALGORITHM = "sha256-chain"
HISTORY_FINGERPRINT_CANONICALIZATION = "rwkv-p-review-v1"
V5O_HISTORY_FINGERPRINT_VERSION = 3
V5O_HISTORY_FINGERPRINT_CANONICALIZATION = "rwkv-p-v5o-review-le-v1"
V5O_HISTORY_FINGERPRINT_FIELDS = (
    "review_id",
    "card_id",
    "note_id",
    "deck_id",
    "preset_id",
    "card_creation_ms",
    "review_start_ms",
    "review_finish_ms",
    "duration_ms",
    "source_ordinal",
    "rollover_phase_ms",
    "rating",
    "is_original_first_review",
)
INITIAL_HISTORY_DIGEST = "0" * 64


_HISTORY_INTEGER_FIELDS = frozenset(
    {
        "review_id",
        "card_id",
        "note_id",
        "deck_id",
        "preset_id",
        "rating",
        "state",
    }
)
_HISTORY_FLOAT_FIELDS = frozenset(
    {
        "day_offset",
        "elapsed_days",
        "elapsed_seconds",
        "duration",
    }
)


class CheckpointMetadataError(ValueError):
    pass


@dataclass(frozen=True)
class RustCheckpointMetadata:
    storage_version: int
    values: dict[str, Any]

    @property
    def state_precision(self) -> str:
        return "fp16" if self.storage_version in {5, 6} else "fp32"

    @property
    def processed_review_count(self) -> int:
        return int(self.values.get("processed_review_count", 0))

    @property
    def last_review_id(self) -> int | None:
        value = self.values.get("last_review_id")
        return None if value is None else int(value)

    @property
    def history_fingerprint(self) -> Mapping[str, Any] | None:
        value = self.values.get("history_fingerprint")
        return value if isinstance(value, Mapping) else None

    @property
    def model_version(self) -> str:
        identity = self.values.get("model_identity")
        if not isinstance(identity, Mapping):
            return "v1"
        value = identity.get("model_version", "v1")
        return str(value)


def read_rust_checkpoint_metadata(path: str | Path) -> RustCheckpointMetadata:
    checkpoint_path = Path(path)
    try:
        with checkpoint_path.open("rb") as handle:
            magic = handle.read(len(RUST_CHECKPOINT_MAGIC))
            if magic != RUST_CHECKPOINT_MAGIC:
                raise CheckpointMetadataError("Unsupported Rust checkpoint magic.")
            version_bytes = handle.read(4)
            if len(version_bytes) != 4:
                raise CheckpointMetadataError("Rust checkpoint version is truncated.")
            storage_version = int.from_bytes(version_bytes, "little")
            if storage_version not in RECOGNIZED_RUST_CHECKPOINT_STORAGE_VERSIONS:
                raise CheckpointMetadataError(
                    f"Unsupported Rust checkpoint storage version {storage_version}."
                )
            metadata_length_bytes = handle.read(8)
            if len(metadata_length_bytes) != 8:
                raise CheckpointMetadataError("Rust checkpoint metadata length is truncated.")
            metadata_length = int.from_bytes(metadata_length_bytes, "little")
            metadata_bytes = handle.read(metadata_length)
            if len(metadata_bytes) != metadata_length:
                raise CheckpointMetadataError("Rust checkpoint metadata is truncated.")
    except OSError as exc:
        raise CheckpointMetadataError(f"Could not read Rust checkpoint metadata: {exc}") from exc

    try:
        values = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointMetadataError("Rust checkpoint metadata is not valid JSON.") from exc
    if not isinstance(values, dict):
        raise CheckpointMetadataError("Rust checkpoint metadata must be an object.")
    if values.get("format") not in {None, "rwkv-p-rust-checkpoint-v1"}:
        raise CheckpointMetadataError(
            f"Unexpected Rust checkpoint format: {values.get('format')!r}."
        )
    expected_storage_format = f"rwkv-p-rust-checkpoint-bin-v{storage_version}"
    if values.get("storage_format") != expected_storage_format:
        raise CheckpointMetadataError(
            f"Unexpected Rust checkpoint storage format: {values.get('storage_format')!r}."
        )
    expected_precision = "fp16" if storage_version in {5, 6} else "fp32"
    if values.get("state_precision", "fp32") != expected_precision:
        raise CheckpointMetadataError("Checkpoint precision differs from its binary container.")
    return RustCheckpointMetadata(storage_version=storage_version, values=values)


def checkpoint_history_is_consistent(
    metadata: RustCheckpointMetadata,
    rows: Iterable[Mapping[str, Any]],
) -> bool:
    expected = metadata.history_fingerprint
    if expected is None:
        raise CheckpointMetadataError(
            "The checkpoint does not contain a review-history fingerprint."
        )
    model_version = metadata.model_version
    _validate_history_fingerprint(expected, model_version=model_version)
    actual = history_fingerprint(
        rows,
        limit=int(expected["processed_review_count"]),
        model_version=model_version,
    )
    return all(
        actual[key] == expected.get(key)
        for key in ("digest", "processed_review_count", "last_review_id")
    )


def history_fingerprint(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int,
    model_version: str = "v1",
) -> dict[str, Any]:
    version, canonicalization, fields, _integers, _floats = _history_schema(model_version)
    digest = INITIAL_HISTORY_DIGEST
    count = 0
    last_review_id: int | None = None
    for row in rows:
        if count >= int(limit):
            break
        digest = _chain_digest(digest, row, model_version=model_version)
        count += 1
        last_review_id = _canonical_int(row["review_id"], "review_id")
    return {
        "version": version,
        "algorithm": HISTORY_FINGERPRINT_ALGORITHM,
        "canonicalization": canonicalization,
        "fields": list(fields),
        "last_review_id": last_review_id,
        "processed_review_count": count,
        "digest": digest,
    }


def _validate_history_fingerprint(
    value: Mapping[str, Any],
    *,
    model_version: str = "v1",
) -> None:
    version, canonicalization, fields, _integers, _floats = _history_schema(model_version)
    expected = {
        "version": version,
        "algorithm": HISTORY_FINGERPRINT_ALGORITHM,
        "canonicalization": canonicalization,
        "fields": list(fields),
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise CheckpointMetadataError(
                f"Unsupported history fingerprint {key}: "
                f"expected {expected_value!r}, got {value.get(key)!r}."
            )
    try:
        processed_count = int(value["processed_review_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointMetadataError(
            "History fingerprint has an invalid processed-review count."
        ) from exc
    if processed_count < 0:
        raise CheckpointMetadataError(
            "History fingerprint processed-review count must not be negative."
        )
    digest = value.get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise CheckpointMetadataError("History fingerprint digest is invalid.")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise CheckpointMetadataError("History fingerprint digest is invalid.") from exc


def _chain_digest(
    previous_digest: str,
    row: Mapping[str, Any],
    *,
    model_version: str = "v1",
) -> str:
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(previous_digest))
    digest.update(_canonical_review_bytes(row, model_version=model_version))
    return digest.hexdigest()


def _canonical_review_bytes(
    row: Mapping[str, Any],
    *,
    model_version: str = "v1",
) -> bytes:
    if model_version == "v5o":
        return _v5o_review_bytes(row)
    _version, _canonicalization, fields, integers, floats = _history_schema(model_version)
    missing = [field for field in fields if field not in row]
    if missing:
        raise CheckpointMetadataError(
            f"Review row is missing fingerprint fields: {', '.join(missing)}."
        )
    record = {
        field: _canonical_review_value(
            field,
            row[field],
            integer_fields=integers,
            float_fields=floats,
        )
        for field in fields
    }
    return json.dumps(
        record,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")


def _v5o_review_bytes(row: Mapping[str, Any]) -> bytes:
    """Match native v5o/history.rs, including optional presence and domain bytes."""

    from .v5o_review_rows import V5O_OPTIONAL_FIELDS, v5o_answer_input_row

    try:
        answer = v5o_answer_input_row(row)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise CheckpointMetadataError(f"Invalid V5o fingerprint row: {exc}") from exc
    result = bytearray(V5O_HISTORY_FINGERPRINT_CANONICALIZATION.encode("ascii"))
    result.extend(struct.pack("<qq", answer["review_id"], answer["card_id"]))
    for field in V5O_OPTIONAL_FIELDS:
        value = answer[field]
        result.extend(struct.pack("<Bq", value is not None, 0 if value is None else value))
    result.extend(
        struct.pack(
            "<qqqqqBB",
            answer["review_start_ms"],
            answer["review_finish_ms"],
            answer["duration_ms"],
            answer["source_ordinal"],
            answer["rollover_phase_ms"],
            answer["rating"],
            answer["is_original_first_review"],
        )
    )
    return bytes(result)


def _canonical_review_value(
    field: str,
    value: Any,
    *,
    integer_fields: frozenset[str] = _HISTORY_INTEGER_FIELDS,
    float_fields: frozenset[str] = _HISTORY_FLOAT_FIELDS,
) -> Any:
    value = value.item() if hasattr(value, "item") else value
    if _is_missing(value):
        return None
    if field in integer_fields:
        return _canonical_int(value, field)
    if field in float_fields:
        return _canonical_float(value, field)
    return value


def _history_schema(
    model_version: str,
) -> tuple[int, str, tuple[str, ...], frozenset[str], frozenset[str]]:
    if model_version == "v5o":
        return (
            V5O_HISTORY_FINGERPRINT_VERSION,
            V5O_HISTORY_FINGERPRINT_CANONICALIZATION,
            V5O_HISTORY_FINGERPRINT_FIELDS,
            frozenset(),
            frozenset(),
        )
    if model_version == "v1":
        return (
            HISTORY_FINGERPRINT_VERSION,
            HISTORY_FINGERPRINT_CANONICALIZATION,
            HISTORY_FINGERPRINT_FIELDS,
            _HISTORY_INTEGER_FIELDS,
            _HISTORY_FLOAT_FIELDS,
        )
    raise CheckpointMetadataError(f"Unsupported RWKV model version: {model_version!r}.")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    return isinstance(value, numbers.Real) and math.isnan(float(value))


def _canonical_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        parsed = float(value)
        if math.isfinite(parsed) and parsed.is_integer():
            return int(parsed)
    raise CheckpointMetadataError(
        f"Review fingerprint field {field!r} must be an integer or missing."
    )


def _canonical_float(value: Any, field: str) -> float:
    if isinstance(value, numbers.Real):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    raise CheckpointMetadataError(
        f"Review fingerprint field {field!r} must be a finite number or missing."
    )
