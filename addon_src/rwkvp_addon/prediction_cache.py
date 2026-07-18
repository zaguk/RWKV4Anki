from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import closing, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compact_review_data import NATIVE_PROCESS_REVIEW_RECORD
from .review_schema import (
    HISTORY_FINGERPRINT_FIELDS,
    cached_prediction_row,
    optional_int,
    required_int,
)

SCHEMA_VERSION = 5
SUPPORTED_SCHEMA_VERSIONS = frozenset((SCHEMA_VERSION,))
EVALUATION_CACHE_KIND = "rwkvp-evaluation-cache"
CACHE_KIND = "rwkvp-review-predictions"
PREDICT_AHEAD_CACHE_KIND = "rwkvp-predict-ahead-review-predictions"
EVALUATION_CACHE_MANIFEST_KEY = "evaluation_cache_path"
CARD_ID_QUERY_CHUNK_SIZE = 900
NATIVE_REVIEW_HISTORY_DIGEST_FORMAT = "sha256-rwkv-srs-review-batch-v1"

BASE_RECORD_COLUMNS = (
    "review_id",
    "card_id",
    "deck_id",
    "preset_id",
    "rating",
    "elapsed_days",
    "elapsed_seconds",
    "review_count",
    "i",
    "prior_lapses",
    "rmse_bins_lapse",
)


class PredictionCacheError(RuntimeError):
    pass


class MissingPredictionCacheError(PredictionCacheError):
    pass


class StalePredictionCacheError(PredictionCacheError):
    pass


@dataclass(frozen=True)
class PredictionCache:
    metadata: dict[str, Any]
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class PredictionValueCache:
    """One prediction column aligned with the caller-owned history rows."""

    metadata: dict[str, Any]
    predictions: list[float | None]
    validation: EvaluationCacheValidation


@dataclass(frozen=True)
class CurveCache:
    metadata: dict[str, Any]
    latest_curves_by_card: dict[int, Any]
    validation: EvaluationCacheValidation | None = None


@dataclass(frozen=True)
class EvaluationCacheValidation:
    """Proof that one immutable evaluation cache matched a review prefix.

    SQLite caches are replaced atomically when a checkpoint is saved.  The file
    identity therefore lets short-lived scoped runtimes reuse the expensive
    review-history digest while still forcing a complete validation after any
    normal cache replacement.
    """

    path: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    schema_version: int
    kind: str
    model_id: str
    processed_review_count: int
    first_review_id: int | None
    last_review_id: int | None
    history_digest: str
    history_digest_format: str
    enabled_cache_kinds: tuple[str, ...]


@dataclass(frozen=True)
class PredictionCacheSpec:
    cache_kind: str
    label: str
    manifest_key: str

    @property
    def prediction_column(self) -> str:
        return prediction_column_for_cache_kind(self.cache_kind)

    def path(self, store: Any, *, partial: bool = False) -> Path:
        return store.partial_evaluation_cache_path if partial else store.evaluation_cache_path

    def manifest_path(self, manifest: dict[str, Any]) -> Path | None:
        value = manifest.get(EVALUATION_CACHE_MANIFEST_KEY)
        if value:
            return Path(value)

        # Legacy manifests stored one path per cache mode. Keep the two active
        # mode fallbacks so stale cache errors remain understandable while users
        # rebuild into the SQLite cache.
        value = manifest.get(self.manifest_key)
        return Path(value) if value else None


@dataclass
class PredictionRecordSet:
    """Predictions aligned by index with the caller-owned chronological rows."""

    immediate_predictions: list[float | None]
    predict_ahead_predictions: list[float | None]

    @classmethod
    def empty(cls) -> PredictionRecordSet:
        return cls(
            immediate_predictions=[],
            predict_ahead_predictions=[],
        )

    def slice(self, count: int) -> PredictionRecordSet:
        return PredictionRecordSet(
            immediate_predictions=list(self.immediate_predictions[:count]),
            predict_ahead_predictions=list(self.predict_ahead_predictions[:count]),
        )

    def slice_from(self, start: int) -> PredictionRecordSet:
        return PredictionRecordSet(
            immediate_predictions=list(self.immediate_predictions[start:]),
            predict_ahead_predictions=list(self.predict_ahead_predictions[start:]),
        )

    def extend(self, other: PredictionRecordSet) -> None:
        self.immediate_predictions.extend(other.immediate_predictions)
        self.predict_ahead_predictions.extend(other.predict_ahead_predictions)

    @classmethod
    def combine(cls, *record_sets: PredictionRecordSet) -> PredictionRecordSet:
        combined = cls.empty()
        for record_set in record_sets:
            combined.extend(record_set)
        return combined


@dataclass(frozen=True)
class PredictionTailSnapshot:
    """Immutable prediction values newer than one durable cache prefix.

    Scoped checkpoint readiness may replay a deliberately unsaved review tail.
    The mutable runtime bookkeeping is cleared when that scope closes, so
    evaluation callers take this compact copy first and align it with the
    durable SQLite prediction prefix by ``start_index``.
    """

    start_index: int
    immediate_predictions: tuple[float | None, ...]
    predict_ahead_predictions: tuple[float | None, ...]

    @classmethod
    def empty(cls, start_index: int) -> PredictionTailSnapshot:
        return cls(
            start_index=max(0, int(start_index)),
            immediate_predictions=(),
            predict_ahead_predictions=(),
        )

    @property
    def processed_review_count(self) -> int:
        return self.start_index + len(self.immediate_predictions)

    def predictions_for(self, spec: PredictionCacheSpec) -> tuple[float | None, ...]:
        if spec.cache_kind == CACHE_KIND:
            return self.immediate_predictions
        if spec.cache_kind == PREDICT_AHEAD_CACHE_KIND:
            return self.predict_ahead_predictions
        raise ValueError(f"Unsupported RWKV prediction cache spec: {spec}")


PER_REVIEW_CACHE_SPEC = PredictionCacheSpec(
    cache_kind=CACHE_KIND,
    label="cached RWKV review predictions",
    manifest_key="prediction_cache_path",
)
PREDICT_AHEAD_CACHE_SPEC = PredictionCacheSpec(
    cache_kind=PREDICT_AHEAD_CACHE_KIND,
    label="cached RWKV predict-ahead review predictions",
    manifest_key="predict_ahead_prediction_cache_path",
)


def prediction_record(row: dict[str, Any], prediction: float | None) -> dict[str, Any]:
    if prediction is None:
        record = cached_prediction_row(row, 0.0)
        record["prediction"] = None
        return record
    return cached_prediction_row(row, prediction)


def prediction_cache_specs() -> tuple[PredictionCacheSpec, ...]:
    return (PER_REVIEW_CACHE_SPEC, PREDICT_AHEAD_CACHE_SPEC)


def prediction_column_for_cache_kind(cache_kind: str) -> str:
    if cache_kind == CACHE_KIND:
        return "immediate_prediction"
    if cache_kind == PREDICT_AHEAD_CACHE_KIND:
        return "predict_ahead_prediction"
    raise ValueError(f"Unsupported RWKV prediction cache kind: {cache_kind}")


def write_prediction_cache(
    path: Path,
    rows: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    *,
    model_id: str,
    cache_kind: str = CACHE_KIND,
) -> None:
    # Preserve compact/packed history sequences.  Converting them to a list
    # would retain one row-view object (and its unpack cache) per review for the
    # complete checkpoint write.
    if not isinstance(rows, Sequence):
        rows = tuple(rows)
    records = list(records)
    spec = prediction_cache_spec_for_cache_kind(cache_kind)
    _validate_record_count(rows, records, spec.label)
    _validate_records_match_rows(records, rows)
    record_set = PredictionRecordSet.empty()
    _set_predictions_for_cache_kind(
        record_set,
        cache_kind,
        [_prediction_from_record(record, cache_kind=cache_kind) for record in records],
    )
    write_evaluation_cache(
        path,
        rows,
        record_set,
        (spec,),
        model_id=model_id,
    )


def write_evaluation_cache(
    path: Path,
    rows: Sequence[dict[str, Any]],
    records: PredictionRecordSet,
    specs: Sequence[PredictionCacheSpec],
    *,
    model_id: str,
    latest_curves_by_card: dict[int, Any] | None = None,
) -> None:
    # Keep packed history columnar throughout the durable cache write.  A list
    # of row views would cache unpacked tuples for every review until the write
    # completed, recreating much of the memory pressure this representation is
    # intended to remove.
    if not isinstance(rows, Sequence):
        rows = tuple(rows)
    specs = _dedupe_specs(specs)
    predictions_by_column = {
        spec.prediction_column: predictions_for_cache_spec(records, spec) for spec in specs
    }
    for spec, predictions in zip(specs, predictions_by_column.values(), strict=True):
        _validate_prediction_count(rows, predictions, spec.label)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(tmp)) as db, db:
            _create_schema(db)
            _write_metadata(
                db,
                rows,
                specs,
                model_id=model_id,
            )
            latest_review_ids_by_card = _write_review_rows(
                db,
                rows,
                predictions_by_column,
            )
            _write_latest_card_curves(
                db,
                latest_review_ids_by_card,
                latest_curves_by_card=latest_curves_by_card,
            )
        os.replace(tmp, path)
    except BaseException:
        with suppress(FileNotFoundError):
            tmp.unlink()
        raise


def load_prediction_cache(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    model_id: str,
    cache_kind: str = CACHE_KIND,
    card_ids: Iterable[int] | None = None,
    validation: EvaluationCacheValidation | None = None,
) -> PredictionCache:
    spec = prediction_cache_spec_for_cache_kind(cache_kind)
    cache = _load_evaluation_cache(
        path,
        rows,
        model_id=model_id,
        validation=validation,
    )
    try:
        _require_enabled_cache_kind(cache.metadata, cache_kind)
        processed_rows = rows[: cache.processed_count]
        if card_ids is None:
            records = _read_prediction_records(cache.db, spec.prediction_column)
            _validate_records_match_rows(records, processed_rows)
        else:
            selected_card_ids = tuple(sorted({int(card_id) for card_id in card_ids}))
            records = _read_prediction_records_for_card_ids(
                cache.db,
                spec.prediction_column,
                selected_card_ids,
            )
            _validate_filtered_records_match_rows(
                records,
                processed_rows,
                selected_card_ids,
            )
        return PredictionCache(metadata=cache.metadata, records=records)
    finally:
        cache.db.close()


def load_prediction_value_cache(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    model_id: str,
    cache_kind: str = CACHE_KIND,
    validation: EvaluationCacheValidation | None = None,
) -> PredictionValueCache:
    """Load only one aligned prediction stream for evaluation.

    The complete review metadata already lives in ``rows``.  Reading it into a
    second list of dictionaries used to account for most of Evaluate's memory
    overhead, so this path verifies the cache's review/card alignment while
    retaining only the selected scalar prediction column.
    """

    spec = prediction_cache_spec_for_cache_kind(cache_kind)
    cache = _load_evaluation_cache(
        path,
        rows,
        model_id=model_id,
        validation=validation,
    )
    try:
        _require_enabled_cache_kind(cache.metadata, cache_kind)
        processed_rows = rows[: cache.processed_count]
        predictions = _read_aligned_prediction_columns(
            cache.db,
            (spec.prediction_column,),
            processed_rows,
        )[spec.prediction_column]
        return PredictionValueCache(
            metadata=cache.metadata,
            predictions=predictions,
            validation=cache.validation,
        )
    finally:
        cache.db.close()


def load_prediction_record_set(
    path: Path,
    rows: Sequence[dict[str, Any]],
    specs: Sequence[PredictionCacheSpec],
    *,
    model_id: str,
    validation: EvaluationCacheValidation | None = None,
) -> PredictionRecordSet:
    specs = _dedupe_specs(specs)
    cache = _load_evaluation_cache(
        path,
        rows,
        model_id=model_id,
        validation=validation,
    )
    try:
        records = PredictionRecordSet.empty()
        processed_rows = rows[: cache.processed_count]
        for spec in specs:
            _require_enabled_cache_kind(cache.metadata, spec.cache_kind)
        prediction_columns = tuple(spec.prediction_column for spec in specs)
        predictions_by_column = _read_aligned_prediction_columns(
            cache.db,
            prediction_columns,
            processed_rows,
        )
        for spec in specs:
            set_predictions_for_cache_spec(
                records,
                spec,
                predictions_by_column[spec.prediction_column],
            )
        return records
    finally:
        cache.db.close()


def load_latest_curves_from_evaluation_cache(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    model_id: str,
    validation: EvaluationCacheValidation | None = None,
) -> CurveCache:
    cache = _load_evaluation_cache(
        path,
        rows,
        model_id=model_id,
        validation=validation,
    )
    try:
        latest_curves: dict[int, Any] = {}
        for card_id, curve_blob in cache.db.execute(
            """
            SELECT card_id, forgetting_curve
            FROM latest_card_curves
            ORDER BY card_id
            """
        ):
            latest_curves[int(card_id)] = _deserialize_curve(curve_blob)
        expected_cards = {
            int(card_id)
            for (card_id,) in cache.db.execute(
                "SELECT DISTINCT card_id FROM review_predictions ORDER BY card_id"
            )
        }
        if set(latest_curves) != expected_cards:
            raise StalePredictionCacheError(
                "RWKV evaluation cache is missing predict-ahead curve data."
            )
        return CurveCache(
            metadata=cache.metadata,
            latest_curves_by_card=latest_curves,
            validation=cache.validation,
        )
    finally:
        cache.db.close()


def load_latest_curves_for_cards_from_evaluation_cache(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    model_id: str,
    card_ids: Iterable[int],
    validation: EvaluationCacheValidation | None = None,
) -> CurveCache:
    selected_ids = tuple(sorted({int(card_id) for card_id in card_ids}))
    cache = _load_evaluation_cache(
        path,
        rows,
        model_id=model_id,
        validation=validation,
    )
    try:
        latest_curves: dict[int, Any] = {}
        expected_cards: set[int] = set()
        for chunk in _chunks(selected_ids, CARD_ID_QUERY_CHUNK_SIZE):
            placeholders = ", ".join("?" for _card_id in chunk)
            for card_id, curve_blob in cache.db.execute(
                f"""
                SELECT card_id, forgetting_curve
                FROM latest_card_curves
                WHERE card_id IN ({placeholders})
                ORDER BY card_id
                """,
                tuple(chunk),
            ):
                latest_curves[int(card_id)] = _deserialize_curve(curve_blob)
            expected_cards.update(
                int(card_id)
                for (card_id,) in cache.db.execute(
                    f"""
                    SELECT DISTINCT card_id
                    FROM review_predictions
                    WHERE card_id IN ({placeholders})
                    """,
                    tuple(chunk),
                )
            )
        if set(latest_curves) != expected_cards:
            raise StalePredictionCacheError(
                "RWKV evaluation cache is missing selected predict-ahead curve data."
            )
        return CurveCache(
            metadata=cache.metadata,
            latest_curves_by_card=latest_curves,
            validation=cache.validation,
        )
    finally:
        cache.db.close()


def evaluation_cache_has_specs(
    path: Path,
    specs: Sequence[PredictionCacheSpec],
    *,
    processed_review_count: int | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        with closing(sqlite3.connect(path)) as db:
            metadata = _read_metadata(db)
            if not _metadata_uses_current_cache_format(metadata):
                return False
            if processed_review_count is not None:
                if int(metadata["processed_review_count"]) != int(processed_review_count):
                    return False
                review_count = int(
                    db.execute("SELECT COUNT(*) FROM review_predictions").fetchone()[0]
                )
                if review_count != int(processed_review_count):
                    return False
    except Exception:
        return False
    enabled = set(_metadata_enabled_cache_kinds(metadata))
    return all(spec.cache_kind in enabled for spec in specs)


def evaluation_cache_has_latest_curves(
    path: Path,
    *,
    processed_review_count: int | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        with closing(sqlite3.connect(path)) as db:
            metadata = _read_metadata(db)
            if not _metadata_uses_current_cache_format(metadata):
                return False
            if processed_review_count is not None and int(
                metadata["processed_review_count"]
            ) != int(processed_review_count):
                return False
            review_count = int(db.execute("SELECT COUNT(*) FROM review_predictions").fetchone()[0])
            if processed_review_count is not None and review_count != int(processed_review_count):
                return False
            expected_card_count = int(
                db.execute("SELECT COUNT(DISTINCT card_id) FROM review_predictions").fetchone()[0]
            )
            curve_card_count = int(
                db.execute("SELECT COUNT(*) FROM latest_card_curves").fetchone()[0]
            )
    except Exception:
        return False
    return curve_card_count == expected_card_count


def validate_evaluation_cache_against_history(
    path: Path,
    rows: Sequence[dict[str, Any]],
    specs: Sequence[PredictionCacheSpec],
    *,
    model_id: str,
) -> EvaluationCacheValidation:
    """Fully validate an unbound evaluation cache before trusting its bytes.

    Checkpoint-bound caches can be verified cheaply by hashing the immutable
    SQLite file.  A legacy cache has no such provenance, so its metadata,
    review rows and enabled prediction streams must first be checked against
    the current canonical history. Latest-card curves are also required and
    validated when the predict-ahead stream is enabled. Only the returned file
    identity is safe to persist in a new cryptographic checkpoint binding.
    """

    specs = _dedupe_specs(specs)
    cache = _load_evaluation_cache(path, rows, model_id=model_id)
    try:
        for spec in specs:
            _require_enabled_cache_kind(cache.metadata, spec.cache_kind)
        declared_specs = _declared_prediction_cache_specs(cache.metadata)
        processed_rows = rows[: cache.processed_count]
        # A successful result may be used to cryptographically bind the entire
        # immutable SQLite file to its checkpoint.  Validate every prediction
        # stream the file claims to contain, not merely the subset enabled by
        # the caller's current configuration, before granting that trust.
        validated_specs = _dedupe_specs((*specs, *declared_specs))
        _validate_evaluation_review_rows(cache.db, processed_rows, validated_specs)
        if PREDICT_AHEAD_CACHE_SPEC in validated_specs:
            _validate_evaluation_latest_curves(cache.db, processed_rows)
        return cache.validation
    finally:
        cache.db.close()


def load_complete_prediction_records(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    model_id: str,
    cache_kind: str = CACHE_KIND,
) -> list[dict[str, Any]]:
    cache = load_prediction_cache(path, rows, model_id=model_id, cache_kind=cache_kind)
    if int(cache.metadata["processed_review_count"]) != len(rows):
        raise StalePredictionCacheError(
            "RWKV prediction cache does not cover the full current review history."
        )
    return cache.records


def prediction_cache_metadata(
    rows: Sequence[dict[str, Any]],
    *,
    model_id: str,
    cache_kind: str = EVALUATION_CACHE_KIND,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": cache_kind,
        "model_id": model_id,
        "processed_review_count": len(rows),
        "first_review_id": None if not rows else required_int(rows[0], "review_id"),
        "last_review_id": None if not rows else required_int(rows[-1], "review_id"),
        "history_digest": native_review_history_digest(rows),
        "history_digest_format": NATIVE_REVIEW_HISTORY_DIGEST_FORMAT,
    }


def review_history_digest(rows: Iterable[dict[str, Any]]) -> str:
    """Return the historical schema-3/4 digest for parity benchmarks only."""

    digest = hashlib.sha256()
    packed_values = getattr(rows, "iter_history_fingerprint_values", None)
    values = packed_values() if callable(packed_values) else None
    encoded_rows = (
        (_canonical_packed_history_values_bytes(row_values) for row_values in values)
        if values is not None
        else (_canonical_row_bytes(row) for row in rows)
    )
    for encoded in encoded_rows:
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def native_review_history_digest(rows: Iterable[dict[str, Any]]) -> str:
    """Hash canonical RWKV-SRS ReviewBatch records without Python row assembly."""

    digest = hashlib.sha256()
    packed_buffers = getattr(rows, "iter_native_review_buffers", None)
    if callable(packed_buffers):
        for buffer in packed_buffers():
            digest.update(buffer)
        return digest.hexdigest()

    for row in rows:
        digest.update(_native_review_record_bytes(row))
    return digest.hexdigest()


def predictions_for_cache_spec(
    records: PredictionRecordSet,
    spec: PredictionCacheSpec,
) -> list[float | None]:
    if spec.cache_kind == CACHE_KIND:
        return records.immediate_predictions
    if spec.cache_kind == PREDICT_AHEAD_CACHE_KIND:
        return records.predict_ahead_predictions
    raise ValueError(f"Unsupported RWKV prediction cache spec: {spec}")


def set_predictions_for_cache_spec(
    records: PredictionRecordSet,
    spec: PredictionCacheSpec,
    predictions: Sequence[float | None],
) -> None:
    _set_predictions_for_cache_kind(records, spec.cache_kind, predictions)


def prediction_cache_spec_for_cache_kind(cache_kind: str) -> PredictionCacheSpec:
    if cache_kind == CACHE_KIND:
        return PER_REVIEW_CACHE_SPEC
    if cache_kind == PREDICT_AHEAD_CACHE_KIND:
        return PREDICT_AHEAD_CACHE_SPEC
    raise ValueError(f"Unsupported RWKV prediction cache kind: {cache_kind}")


@dataclass(frozen=True)
class _LoadedEvaluationCache:
    db: sqlite3.Connection
    metadata: dict[str, Any]
    processed_count: int
    validation: EvaluationCacheValidation


def _load_evaluation_cache(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    model_id: str,
    validation: EvaluationCacheValidation | None = None,
) -> _LoadedEvaluationCache:
    path = Path(path)
    if not path.exists():
        raise MissingPredictionCacheError(f"RWKV evaluation cache is missing: {path}")

    try:
        initial_file_identity = _evaluation_cache_file_identity(path)
        db = sqlite3.connect(path)
        metadata = _validate_metadata(_read_metadata(db), model_id=model_id)
        processed_count = int(metadata["processed_review_count"])
        if processed_count > len(rows):
            raise StalePredictionCacheError(
                "RWKV evaluation cache contains more reviews than the current history."
            )
        reusable_validation = (
            validation
            if _evaluation_cache_validation_matches(
                validation,
                file_identity=initial_file_identity,
                metadata=metadata,
                rows=rows,
            )
            else None
        )
        expected = (
            _history_metadata_from_validation(reusable_validation)
            if reusable_validation is not None
            else prediction_cache_metadata(
                rows[:processed_count],
                model_id=model_id,
                cache_kind=EVALUATION_CACHE_KIND,
            )
        )
        for key in (
            "first_review_id",
            "last_review_id",
            "processed_review_count",
            "history_digest",
            "history_digest_format",
        ):
            if metadata.get(key) != expected.get(key):
                raise StalePredictionCacheError(
                    f"RWKV evaluation cache does not match current history ({key})."
                )
        if reusable_validation is None:
            row_count = int(db.execute("SELECT COUNT(*) FROM review_predictions").fetchone()[0])
            if row_count != processed_count:
                raise StalePredictionCacheError(
                    "RWKV evaluation cache row count does not match its metadata."
                )
        final_file_identity = _evaluation_cache_file_identity(path)
        if final_file_identity != initial_file_identity:
            raise StalePredictionCacheError(
                "RWKV evaluation cache changed while it was being read."
            )
        resolved_validation = reusable_validation or _evaluation_cache_validation(
            file_identity=final_file_identity,
            metadata=metadata,
        )
        return _LoadedEvaluationCache(
            db=db,
            metadata=metadata,
            processed_count=processed_count,
            validation=resolved_validation,
        )
    except PredictionCacheError:
        with _suppress_close_error(locals().get("db")):
            pass
        raise
    except Exception as exc:
        with _suppress_close_error(locals().get("db")):
            pass
        raise StalePredictionCacheError(f"RWKV evaluation cache is unreadable: {exc}") from exc


def evaluation_cache_validation_is_current(
    validation: EvaluationCacheValidation | None,
) -> bool:
    if validation is None:
        return False
    try:
        return _evaluation_cache_file_identity(Path(validation.path)) == (
            validation.path,
            validation.device,
            validation.inode,
            validation.size,
            validation.mtime_ns,
        )
    except PredictionCacheError:
        return False


def evaluation_cache_file_digest(path: Path) -> tuple[str, int]:
    """Return a stable SHA-256 and byte size for an immutable cache file."""

    digest, identity = _stable_file_sha256(Path(path))
    return digest, identity[3]


def validate_evaluation_cache_file_binding(
    path: Path,
    *,
    model_id: str,
    expected_sha256: str,
    expected_size: int,
    expected_processed_review_count: int,
) -> EvaluationCacheValidation:
    """Trust cache history metadata after verifying its bound file digest.

    The caller separately verifies that the manifest's checkpoint history
    fingerprint belongs to the active, history-checked checkpoint. Hashing the
    immutable SQLite file then proves that its metadata, predictions, and curves
    are the exact cache that manifest bound to that checkpoint. This avoids
    rebuilding the cache's review-history digest from Python dictionaries.
    """

    expected_digest = _normalized_sha256(expected_sha256)
    try:
        expected_bytes = int(expected_size)
        expected_count = int(expected_processed_review_count)
    except (TypeError, ValueError) as exc:
        raise StalePredictionCacheError(
            "RWKV evaluation cache binding metadata is invalid."
        ) from exc
    if expected_bytes < 0 or expected_count < 0:
        raise StalePredictionCacheError("RWKV evaluation cache binding metadata is invalid.")

    path = Path(path)
    actual_digest, initial_identity = _stable_file_sha256(path)
    if initial_identity[3] != expected_bytes or actual_digest != expected_digest:
        raise StalePredictionCacheError(
            "RWKV evaluation cache does not match its checkpoint binding."
        )

    try:
        db = sqlite3.connect(path)
        metadata = _validate_metadata(_read_metadata(db), model_id=model_id)
        processed_count = int(metadata["processed_review_count"])
        if processed_count != expected_count:
            raise StalePredictionCacheError(
                "RWKV evaluation cache processed-review count does not match "
                "its checkpoint binding."
            )
        row_count = int(db.execute("SELECT COUNT(*) FROM review_predictions").fetchone()[0])
        if row_count != processed_count:
            raise StalePredictionCacheError(
                "RWKV evaluation cache row count does not match its metadata."
            )
        final_identity = _evaluation_cache_file_identity(path)
        if final_identity != initial_identity:
            raise StalePredictionCacheError(
                "RWKV evaluation cache changed while its checkpoint binding was checked."
            )
        return _evaluation_cache_validation(
            file_identity=final_identity,
            metadata=metadata,
        )
    except PredictionCacheError:
        raise
    except Exception as exc:
        raise StalePredictionCacheError(f"RWKV evaluation cache is unreadable: {exc}") from exc
    finally:
        with _suppress_close_error(locals().get("db")):
            pass


def _evaluation_cache_file_identity(path: Path) -> tuple[str, int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise MissingPredictionCacheError(f"RWKV evaluation cache is missing: {path}") from exc
    return (
        str(path.resolve()),
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _stable_file_sha256(path: Path) -> tuple[str, tuple[str, int, int, int, int]]:
    initial_identity = _evaluation_cache_file_identity(path)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise MissingPredictionCacheError(
            f"RWKV evaluation cache could not be read: {path}"
        ) from exc
    final_identity = _evaluation_cache_file_identity(path)
    if final_identity != initial_identity:
        raise StalePredictionCacheError(
            "RWKV evaluation cache changed while its digest was calculated."
        )
    return digest.hexdigest(), final_identity


def _normalized_sha256(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64:
        raise StalePredictionCacheError("RWKV evaluation cache binding SHA-256 is invalid.")
    try:
        bytes.fromhex(normalized)
    except ValueError as exc:
        raise StalePredictionCacheError(
            "RWKV evaluation cache binding SHA-256 is invalid."
        ) from exc
    return normalized


def _evaluation_cache_validation_matches(
    validation: EvaluationCacheValidation | None,
    *,
    file_identity: tuple[str, int, int, int, int],
    metadata: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> bool:
    if validation is None:
        return False
    if file_identity != (
        validation.path,
        validation.device,
        validation.inode,
        validation.size,
        validation.mtime_ns,
    ):
        return False
    if _evaluation_cache_metadata_key(metadata) != _evaluation_cache_validation_key(validation):
        return False
    count = int(validation.processed_review_count)
    if count > len(rows):
        return False
    if count == 0:
        return validation.first_review_id is None and validation.last_review_id is None
    return (
        required_int(rows[0], "review_id") == validation.first_review_id
        and required_int(rows[count - 1], "review_id") == validation.last_review_id
    )


def _evaluation_cache_validation(
    *,
    file_identity: tuple[str, int, int, int, int],
    metadata: dict[str, Any],
) -> EvaluationCacheValidation:
    return EvaluationCacheValidation(
        path=file_identity[0],
        device=file_identity[1],
        inode=file_identity[2],
        size=file_identity[3],
        mtime_ns=file_identity[4],
        schema_version=int(metadata["schema_version"]),
        kind=str(metadata["kind"]),
        model_id=str(metadata["model_id"]),
        processed_review_count=int(metadata["processed_review_count"]),
        first_review_id=optional_int(metadata.get("first_review_id")),
        last_review_id=optional_int(metadata.get("last_review_id")),
        history_digest=str(metadata["history_digest"]),
        history_digest_format=str(metadata["history_digest_format"]),
        enabled_cache_kinds=tuple(_metadata_enabled_cache_kinds(metadata)),
    )


def _evaluation_cache_metadata_key(metadata: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(metadata["schema_version"]),
        str(metadata["kind"]),
        str(metadata["model_id"]),
        int(metadata["processed_review_count"]),
        optional_int(metadata.get("first_review_id")),
        optional_int(metadata.get("last_review_id")),
        str(metadata["history_digest"]),
        str(metadata["history_digest_format"]),
        tuple(_metadata_enabled_cache_kinds(metadata)),
    )


def _evaluation_cache_validation_key(
    validation: EvaluationCacheValidation,
) -> tuple[Any, ...]:
    return (
        validation.schema_version,
        validation.kind,
        validation.model_id,
        validation.processed_review_count,
        validation.first_review_id,
        validation.last_review_id,
        validation.history_digest,
        validation.history_digest_format,
        validation.enabled_cache_kinds,
    )


def _history_metadata_from_validation(
    validation: EvaluationCacheValidation,
) -> dict[str, Any]:
    return {
        "first_review_id": validation.first_review_id,
        "last_review_id": validation.last_review_id,
        "processed_review_count": validation.processed_review_count,
        "history_digest": validation.history_digest,
        "history_digest_format": validation.history_digest_format,
    }


def _create_schema(db: sqlite3.Connection) -> None:
    db.execute(
        """
        CREATE TABLE cache_metadata (
            schema_version INTEGER NOT NULL,
            kind TEXT NOT NULL,
            model_id TEXT NOT NULL,
            processed_review_count INTEGER NOT NULL,
            first_review_id INTEGER,
            last_review_id INTEGER,
            history_digest TEXT NOT NULL,
            history_digest_format TEXT NOT NULL,
            enabled_cache_kinds TEXT NOT NULL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE review_predictions (
            review_id INTEGER NOT NULL,
            card_id INTEGER NOT NULL,
            deck_id INTEGER,
            preset_id INTEGER,
            rating INTEGER NOT NULL,
            elapsed_days REAL NOT NULL,
            elapsed_seconds REAL NOT NULL,
            review_count INTEGER NOT NULL,
            i INTEGER NOT NULL,
            prior_lapses INTEGER NOT NULL,
            rmse_bins_lapse INTEGER NOT NULL,
            immediate_prediction REAL,
            predict_ahead_prediction REAL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE latest_card_curves (
            card_id INTEGER NOT NULL,
            review_id INTEGER NOT NULL,
            forgetting_curve BLOB NOT NULL
        )
        """
    )
    db.execute(
        "CREATE UNIQUE INDEX review_predictions_review_id_idx ON review_predictions(review_id)"
    )
    db.execute("CREATE INDEX review_predictions_card_id_idx ON review_predictions(card_id)")
    db.execute("CREATE UNIQUE INDEX latest_card_curves_card_id_idx ON latest_card_curves(card_id)")


def _write_metadata(
    db: sqlite3.Connection,
    rows: Sequence[dict[str, Any]],
    specs: Sequence[PredictionCacheSpec],
    *,
    model_id: str,
) -> None:
    metadata = prediction_cache_metadata(
        rows,
        model_id=model_id,
        cache_kind=EVALUATION_CACHE_KIND,
    )
    db.execute(
        """
        INSERT INTO cache_metadata (
            schema_version,
            kind,
            model_id,
            processed_review_count,
            first_review_id,
            last_review_id,
            history_digest,
            history_digest_format,
            enabled_cache_kinds
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metadata["schema_version"],
            metadata["kind"],
            metadata["model_id"],
            metadata["processed_review_count"],
            metadata["first_review_id"],
            metadata["last_review_id"],
            metadata["history_digest"],
            metadata["history_digest_format"],
            json.dumps([spec.cache_kind for spec in specs], separators=(",", ":")),
        ),
    )


def _write_review_rows(
    db: sqlite3.Connection,
    rows: Sequence[dict[str, Any]],
    predictions_by_column: dict[str, Sequence[float | None]],
) -> dict[int, int]:
    insert_columns = (
        *BASE_RECORD_COLUMNS,
        "immediate_prediction",
        "predict_ahead_prediction",
    )
    sql = (
        f"INSERT INTO review_predictions ({', '.join(insert_columns)}) "
        f"VALUES ({', '.join('?' for _ in insert_columns)})"
    )
    packed_values = getattr(rows, "iter_cache_record_values", None)
    base_values = (
        packed_values() if callable(packed_values) else (_base_record_values(row) for row in rows)
    )
    latest_review_ids_by_card: dict[int, int] = {}

    def insert_values():
        for index, values in enumerate(base_values):
            latest_review_ids_by_card[int(values[1])] = int(values[0])
            yield (
                *values,
                _prediction_value_at(
                    predictions_by_column,
                    "immediate_prediction",
                    index,
                ),
                _prediction_value_at(
                    predictions_by_column,
                    "predict_ahead_prediction",
                    index,
                ),
            )

    db.executemany(
        sql,
        insert_values(),
    )
    return latest_review_ids_by_card


def _base_record_values(row: dict[str, Any]) -> tuple[Any, ...]:
    review_count = optional_int(row.get("review_count", row.get("i")))
    prior_lapses = optional_int(row.get("prior_lapses", row.get("rmse_bins_lapse")))
    review_count = 1 if review_count is None else review_count
    prior_lapses = 0 if prior_lapses is None else prior_lapses
    return (
        required_int(row, "review_id"),
        required_int(row, "card_id"),
        optional_int(row.get("deck_id")),
        optional_int(row.get("preset_id")),
        required_int(row, "rating"),
        float(row["elapsed_days"]),
        float(row["elapsed_seconds"]),
        review_count,
        review_count,
        prior_lapses,
        prior_lapses,
    )


def _prediction_value_at(
    predictions_by_column: dict[str, Sequence[float | None]],
    column: str,
    index: int,
) -> float | None:
    predictions = predictions_by_column.get(column)
    if predictions is None:
        return None
    return _sqlite_prediction_value(predictions[index])


def _write_latest_card_curves(
    db: sqlite3.Connection,
    latest_review_ids_by_card: dict[int, int],
    *,
    latest_curves_by_card: dict[int, Any] | None,
) -> None:
    if latest_curves_by_card is None:
        return
    latest = {
        int(card_id): (latest_review_ids_by_card[int(card_id)], curve)
        for card_id, curve in latest_curves_by_card.items()
        if int(card_id) in latest_review_ids_by_card
    }
    if not latest:
        return
    db.executemany(
        """
        INSERT INTO latest_card_curves (card_id, review_id, forgetting_curve)
        VALUES (?, ?, ?)
        """,
        (
            (card_id, review_id, _serialize_curve(curve))
            for card_id, (review_id, curve) in sorted(latest.items())
        ),
    )


def _latest_review_ids_by_card(rows: Sequence[dict[str, Any]]) -> dict[int, int]:
    packed_values = getattr(rows, "iter_review_identity_values", None)
    values = (
        packed_values()
        if callable(packed_values)
        else ((required_int(row, "review_id"), required_int(row, "card_id")) for row in rows)
    )
    latest: dict[int, int] = {}
    for review_id, card_id in values:
        latest[card_id] = review_id
    return latest


def _read_metadata(db: sqlite3.Connection) -> dict[str, Any]:
    cursor = db.execute("SELECT * FROM cache_metadata")
    row = cursor.fetchone()
    if row is None:
        raise StalePredictionCacheError("RWKV evaluation cache metadata is missing.")
    values = {
        str(description[0]): value
        for description, value in zip(cursor.description or (), row, strict=True)
    }
    return {
        "schema_version": int(values["schema_version"]),
        "kind": values["kind"],
        "model_id": values["model_id"],
        "processed_review_count": int(values["processed_review_count"]),
        "first_review_id": values["first_review_id"],
        "last_review_id": values["last_review_id"],
        "history_digest": values["history_digest"],
        "history_digest_format": values.get("history_digest_format"),
        "enabled_cache_kinds": json.loads(values["enabled_cache_kinds"]),
    }


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    model_id: str,
) -> dict[str, Any]:
    schema_version = metadata.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise StalePredictionCacheError("Unsupported RWKV evaluation cache schema.")
    if metadata.get("kind") != EVALUATION_CACHE_KIND:
        raise StalePredictionCacheError("Unexpected RWKV evaluation cache kind.")
    if metadata.get("model_id") != model_id:
        raise StalePredictionCacheError("RWKV evaluation cache was built with another model.")
    if not isinstance(metadata.get("enabled_cache_kinds"), list):
        raise StalePredictionCacheError("RWKV evaluation cache enabled cache metadata is missing.")
    if metadata.get("history_digest_format") != NATIVE_REVIEW_HISTORY_DIGEST_FORMAT:
        raise StalePredictionCacheError(
            "RWKV evaluation cache history digest format is unsupported."
        )
    return metadata


def _metadata_uses_current_cache_format(metadata: dict[str, Any]) -> bool:
    return (
        metadata.get("schema_version") == SCHEMA_VERSION
        and metadata.get("history_digest_format") == NATIVE_REVIEW_HISTORY_DIGEST_FORMAT
    )


def _read_prediction_records(
    db: sqlite3.Connection,
    prediction_column: str,
) -> list[dict[str, Any]]:
    columns = (*BASE_RECORD_COLUMNS, prediction_column)
    sql = f"SELECT {', '.join(columns)} FROM review_predictions ORDER BY review_id"
    records: list[dict[str, Any]] = []
    for row in db.execute(sql):
        records.append(_prediction_record_from_sql_row(columns, row, prediction_column))
    return records


def _read_prediction_records_for_card_ids(
    db: sqlite3.Connection,
    prediction_column: str,
    card_ids: Sequence[int],
) -> list[dict[str, Any]]:
    if not card_ids:
        return []

    columns = (*BASE_RECORD_COLUMNS, prediction_column)
    records: list[dict[str, Any]] = []
    for chunk in _chunks(card_ids, CARD_ID_QUERY_CHUNK_SIZE):
        placeholders = ", ".join("?" for _card_id in chunk)
        sql = (
            f"SELECT {', '.join(columns)} "
            "FROM review_predictions "
            f"WHERE card_id IN ({placeholders}) "
            "ORDER BY review_id"
        )
        for row in db.execute(sql, tuple(chunk)):
            records.append(_prediction_record_from_sql_row(columns, row, prediction_column))
    records.sort(key=lambda record: int(record["review_id"]))
    return records


def _read_aligned_prediction_columns(
    db: sqlite3.Connection,
    prediction_columns: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> dict[str, list[float | None]]:
    if not prediction_columns:
        return {}
    sql = (
        f"SELECT review_id, card_id, {', '.join(prediction_columns)} "
        "FROM review_predictions ORDER BY review_id"
    )
    predictions_by_column: dict[str, list[float | None]] = {
        column: [] for column in prediction_columns
    }
    row_count = 0
    for index, cached_row in enumerate(db.execute(sql)):
        if index >= len(rows):
            raise StalePredictionCacheError(
                "RWKV prediction cache contains more aligned rows than current history."
            )
        review_id, card_id = cached_row[:2]
        row = rows[index]
        if int(review_id) != required_int(row, "review_id"):
            raise StalePredictionCacheError(
                f"RWKV prediction cache review id mismatch at row {index + 1}."
            )
        if int(card_id) != required_int(row, "card_id"):
            raise StalePredictionCacheError(
                f"RWKV prediction cache card id mismatch at row {index + 1}."
            )
        for column, prediction in zip(prediction_columns, cached_row[2:], strict=True):
            predictions_by_column[column].append(None if prediction is None else float(prediction))
        row_count += 1
    if row_count != len(rows):
        raise StalePredictionCacheError(
            "RWKV prediction cache aligned row count does not match current history."
        )
    return predictions_by_column


def _validate_evaluation_review_rows(
    db: sqlite3.Connection,
    rows: Sequence[dict[str, Any]],
    specs: Sequence[PredictionCacheSpec],
) -> None:
    prediction_columns = tuple(spec.prediction_column for spec in specs)
    columns = (*BASE_RECORD_COLUMNS, *prediction_columns)
    sql = f"SELECT {', '.join(columns)} FROM review_predictions ORDER BY review_id"
    row_count = 0
    for index, cached_row in enumerate(db.execute(sql)):
        if index >= len(rows):
            raise StalePredictionCacheError(
                "RWKV evaluation cache contains more review rows than current history."
            )
        expected = _base_record_values(rows[index])
        actual = tuple(cached_row[: len(BASE_RECORD_COLUMNS)])
        if actual != expected:
            raise StalePredictionCacheError(
                f"RWKV evaluation cache review data mismatch at row {index + 1}."
            )
        for prediction in cached_row[len(BASE_RECORD_COLUMNS) :]:
            if prediction is not None:
                float(prediction)
        row_count += 1
    if row_count != len(rows):
        raise StalePredictionCacheError(
            "RWKV evaluation cache review row count does not match current history."
        )


def _validate_evaluation_latest_curves(
    db: sqlite3.Connection,
    rows: Sequence[dict[str, Any]],
) -> None:
    expected_review_ids = _latest_review_ids_by_card(rows)
    actual_card_ids: set[int] = set()
    for card_id, review_id, curve_blob in db.execute(
        """
        SELECT card_id, review_id, forgetting_curve
        FROM latest_card_curves
        ORDER BY card_id
        """
    ):
        resolved_card_id = int(card_id)
        if resolved_card_id in actual_card_ids:
            raise StalePredictionCacheError(
                "RWKV evaluation cache contains duplicate latest-card curve data."
            )
        actual_card_ids.add(resolved_card_id)
        if expected_review_ids.get(resolved_card_id) != int(review_id):
            raise StalePredictionCacheError(
                "RWKV evaluation cache latest-card curve does not match current history."
            )
        _deserialize_curve(curve_blob)
    if actual_card_ids != set(expected_review_ids):
        raise StalePredictionCacheError("RWKV evaluation cache is missing latest-card curve data.")


def _prediction_record_from_sql_row(
    columns: Sequence[str],
    row: Sequence[Any],
    prediction_column: str,
) -> dict[str, Any]:
    record = {column: row[index] for index, column in enumerate(columns)}
    record["prediction"] = record.pop(prediction_column)
    return record


def _validate_record_count(
    rows: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    label: str,
) -> None:
    if len(rows) != len(records):
        raise ValueError(
            f"Prediction cache row count mismatch for {label}: "
            f"{len(rows)} review rows but {len(records)} prediction records."
        )


def _validate_prediction_count(
    rows: Sequence[dict[str, Any]],
    predictions: Sequence[float | None],
    label: str,
) -> None:
    if len(rows) != len(predictions):
        raise ValueError(
            f"Prediction cache row count mismatch for {label}: "
            f"{len(rows)} review rows but {len(predictions)} predictions."
        )


def _validate_records_match_rows(
    records: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
) -> None:
    for index, (record, row) in enumerate(zip(records, rows, strict=True), start=1):
        if int(record["review_id"]) != required_int(row, "review_id"):
            raise StalePredictionCacheError(
                f"RWKV prediction cache review id mismatch at row {index}."
            )
        if int(record["card_id"]) != required_int(row, "card_id"):
            raise StalePredictionCacheError(
                f"RWKV prediction cache card id mismatch at row {index}."
            )
        _backfill_metric_fields(record, row, index=index)


def _validate_filtered_records_match_rows(
    records: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    card_ids: Sequence[int],
) -> None:
    allowed = set(card_ids)
    expected_by_review_id = {
        required_int(row, "review_id"): row
        for row in rows
        if required_int(row, "card_id") in allowed
    }
    if len(records) != len(expected_by_review_id):
        raise StalePredictionCacheError(
            "RWKV prediction cache filtered row count does not match current history."
        )

    seen_review_ids: set[int] = set()
    for index, record in enumerate(records, start=1):
        review_id = int(record["review_id"])
        if review_id in seen_review_ids:
            raise StalePredictionCacheError(
                f"RWKV prediction cache duplicate review id at filtered row {index}."
            )
        seen_review_ids.add(review_id)

        row = expected_by_review_id.get(review_id)
        if row is None:
            raise StalePredictionCacheError(
                f"RWKV prediction cache unexpected review id at filtered row {index}."
            )
        if int(record["card_id"]) != required_int(row, "card_id"):
            raise StalePredictionCacheError(
                f"RWKV prediction cache card id mismatch at filtered row {index}."
            )
        _backfill_metric_fields(record, row, index=index)


def _backfill_metric_fields(record: dict[str, Any], row: dict[str, Any], *, index: int) -> None:
    review_count = optional_int(row.get("review_count", row.get("i")))
    prior_lapses = optional_int(row.get("prior_lapses", row.get("rmse_bins_lapse")))
    if review_count is None:
        review_count = 1
    if prior_lapses is None:
        prior_lapses = 0

    _validate_or_set_int(record, "review_count", review_count, index=index)
    _validate_or_set_int(record, "i", review_count, index=index)
    _validate_or_set_int(record, "prior_lapses", prior_lapses, index=index)
    _validate_or_set_int(record, "rmse_bins_lapse", prior_lapses, index=index)


def _validate_or_set_int(
    record: dict[str, Any],
    key: str,
    expected: int,
    *,
    index: int,
) -> None:
    if key not in record:
        record[key] = expected
        return
    if int(record[key]) != expected:
        raise StalePredictionCacheError(f"RWKV prediction cache {key} mismatch at row {index}.")


def _require_enabled_cache_kind(metadata: dict[str, Any], cache_kind: str) -> None:
    if cache_kind not in set(_metadata_enabled_cache_kinds(metadata)):
        raise MissingPredictionCacheError(
            "RWKV evaluation cache does not include the requested prediction mode. "
            "Run Update Checkpoint or Rebuild Checkpoint to refresh it."
        )


def _metadata_enabled_cache_kinds(metadata: dict[str, Any]) -> list[str]:
    enabled = metadata.get("enabled_cache_kinds", [])
    if not isinstance(enabled, list):
        return []
    return [str(cache_kind) for cache_kind in enabled]


def _declared_prediction_cache_specs(
    metadata: dict[str, Any],
) -> tuple[PredictionCacheSpec, ...]:
    specs: list[PredictionCacheSpec] = []
    for cache_kind in _metadata_enabled_cache_kinds(metadata):
        try:
            spec = prediction_cache_spec_for_cache_kind(cache_kind)
        except ValueError as exc:
            raise StalePredictionCacheError(
                "RWKV evaluation cache declares an unsupported prediction mode."
            ) from exc
        if spec not in specs:
            specs.append(spec)
    return tuple(specs)


def _set_predictions_for_cache_kind(
    records: PredictionRecordSet,
    cache_kind: str,
    predictions: Sequence[float | None],
) -> None:
    if cache_kind == CACHE_KIND:
        records.immediate_predictions = [
            None if prediction is None else float(prediction) for prediction in predictions
        ]
    elif cache_kind == PREDICT_AHEAD_CACHE_KIND:
        records.predict_ahead_predictions = [
            None if prediction is None else float(prediction) for prediction in predictions
        ]
    else:
        raise ValueError(f"Unsupported RWKV prediction cache kind: {cache_kind}")


def _dedupe_specs(specs: Sequence[PredictionCacheSpec]) -> tuple[PredictionCacheSpec, ...]:
    unique: list[PredictionCacheSpec] = []
    for spec in specs:
        if spec not in unique:
            unique.append(spec)
    return tuple(unique)


def _chunks(values: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    if size <= 0:
        raise ValueError("chunk size must be positive.")
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _canonical_row_bytes(row: dict[str, Any]) -> bytes:
    record = {field: _canonical_value(row.get(field)) for field in HISTORY_FINGERPRINT_FIELDS}
    return json.dumps(
        record,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _native_review_record_bytes(row: dict[str, Any]) -> bytes:
    note_id = optional_int(row.get("note_id"))
    deck_id = optional_int(row.get("deck_id"))
    preset_id = optional_int(row.get("preset_id"))
    return NATIVE_PROCESS_REVIEW_RECORD.pack(
        required_int(row, "review_id"),
        required_int(row, "card_id"),
        int(note_id is not None),
        0 if note_id is None else note_id,
        int(deck_id is not None),
        0 if deck_id is None else deck_id,
        int(preset_id is not None),
        0 if preset_id is None else preset_id,
        _finite_history_float(row["day_offset"], "day_offset"),
        _finite_history_float(row["elapsed_days"], "elapsed_days"),
        _finite_history_float(row["elapsed_seconds"], "elapsed_seconds"),
        required_int(row, "rating"),
        _finite_history_float(row["duration"], "duration"),
        float(required_int(row, "state")),
    )


def _finite_history_float(value: Any, field: str) -> float:
    number = float(value.item() if hasattr(value, "item") else value)
    if not math.isfinite(number):
        raise ValueError(f"Review history field {field!r} must be finite.")
    return number


def _canonical_packed_history_values_bytes(values: Sequence[Any]) -> bytes:
    """Encode known packed scalars exactly like the legacy sorted JSON row.

    Packed history values are already plain ``int``/``float``/``None`` values
    in ``HISTORY_FINGERPRINT_FIELDS`` order.  Avoid constructing a dictionary
    and invoking the general JSON encoder for every historical review while
    retaining the existing digest bytes exactly.
    """

    if len(values) != 11 or tuple(HISTORY_FINGERPRINT_FIELDS) != (
        "review_id",
        "card_id",
        "note_id",
        "deck_id",
        "preset_id",
        "day_offset",
        "elapsed_days",
        "elapsed_seconds",
        "rating",
        "duration",
        "state",
    ):
        raise RuntimeError(
            "Packed review history does not match the evaluation-cache digest schema."
        )
    scalar = _canonical_packed_scalar
    return (
        '{"card_id":'
        + scalar(values[1])
        + ',"day_offset":'
        + scalar(values[5])
        + ',"deck_id":'
        + scalar(values[3])
        + ',"duration":'
        + scalar(values[9])
        + ',"elapsed_days":'
        + scalar(values[6])
        + ',"elapsed_seconds":'
        + scalar(values[7])
        + ',"note_id":'
        + scalar(values[2])
        + ',"preset_id":'
        + scalar(values[4])
        + ',"rating":'
        + scalar(values[8])
        + ',"review_id":'
        + scalar(values[0])
        + ',"state":'
        + scalar(values[10])
        + "}"
    ).encode("ascii")


def _canonical_packed_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Review history fields must be finite.")
    return str(int(number)) if number.is_integer() else repr(number)


def _canonical_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Review history fields must be finite.")
        return int(value) if value.is_integer() else value
    return value


def _sqlite_prediction_value(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _prediction_from_record(
    record: dict[str, Any],
    *,
    cache_kind: str,
) -> float | None:
    prediction = record.get("prediction")
    if prediction is None:
        return None
    prediction = float(prediction)
    if not math.isfinite(prediction):
        return None
    return prediction


def _serialize_curve(curve: Any) -> bytes:
    return pickle.dumps(curve, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize_curve(blob: bytes) -> Any:
    return pickle.loads(blob)


class _suppress_close_error:
    def __init__(self, db: sqlite3.Connection | None) -> None:
        self.db = db

    def __enter__(self):
        return None

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        if self.db is not None:
            with suppress(Exception):
                self.db.close()
        return False
