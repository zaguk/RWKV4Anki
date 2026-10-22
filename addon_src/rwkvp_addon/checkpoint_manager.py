from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .checkpoint_history import (
    CURRENT_RUST_CHECKPOINT_STORAGE_VERSION,
    CheckpointMetadataError,
    RustCheckpointMetadata,
    checkpoint_history_is_consistent,
    read_rust_checkpoint_metadata,
)
from .checkpoint_progress import (
    update_checkpoint_consistency,
    update_checkpoint_curve_data,
    update_checkpoint_load,
)
from .checkpoint_storage import (
    InsufficientCheckpointDiskSpaceError as StorageDiskSpaceError,
)
from .checkpoint_storage import (
    RustCheckpointIdentityCounts,
    RustCheckpointStorageEstimate,
    ensure_rust_checkpoint_disk_space_for_estimate,
    estimate_rust_checkpoint_storage_from_counts,
    rust_checkpoint_identity_counts,
)
from .compact_review_data import native_review_batch_for_rows
from .constants import (
    CHECKPOINT_SAVE_INTERVAL,
    DEFAULT_MODEL_ID,
    LIVE_REVIEW_RUST_UNDO_LIMIT,
)
from .model_generation import (
    DEFAULT_MODEL_GENERATION,
    model_generation_uses_exact_timestamps,
    model_generation_uses_v5o,
    normalize_model_generation,
)
from .model_storage import bundled_model_path, model_storage_subdirectory
from .prediction_cache import (
    EVALUATION_CACHE_MANIFEST_KEY,
    PER_REVIEW_CACHE_SPEC,
    PREDICT_AHEAD_CACHE_SPEC,
    EvaluationCacheValidation,
    PredictionCacheError,
    PredictionCacheSpec,
    PredictionRecordSet,
    PredictionTailSnapshot,
    evaluation_cache_file_digest,
    evaluation_cache_has_latest_curves,
    evaluation_cache_has_specs,
    evaluation_cache_validation_is_current,
    load_latest_curves_for_cards_from_evaluation_cache,
    load_latest_curves_from_evaluation_cache,
    load_prediction_record_set,
    predictions_for_cache_spec,
    validate_evaluation_cache_against_history,
    validate_evaluation_cache_file_binding,
    write_evaluation_cache,
)
from .profile_store import ProfileStore
from .progress import CancelledError, ProgressReporter
from .review_rows import rebased_day_offset_origin
from .review_tail_context import (
    CheckpointTailRows,
    CollectionRevision,
    ReviewTailContextError,
    write_review_tail_context,
)
from .review_type_normalization import (
    FILTERED_REVIEW_NORMALIZATION_MANIFEST_KEY,
    FilteredReviewNormalizationPolicy,
    checkpoint_policy_matches,
)
from .runtime_inputs import RuntimeInputContext, input_context_for_generation
from .rwkv_backend import configured_rwkv_backend, rwkv_checkpoint_suffix
from .rwkv_performance_modes import (
    DEFAULT_PREDICT_MANY_MODE,
    DEFAULT_PROCESS_MANY_MODE,
    PREDICT_MANY_FAST_MODE,
    PREDICT_MANY_MODES,
    PREDICT_MANY_ORACLE_MODE,
    normalize_predict_many_mode,
    normalize_process_many_mode,
    predict_many_uses_fast,
    predict_many_uses_gpu,
    prediction_progress_chunk_size,
    process_many_uses_gpu,
)
from .rwkv_processing import (
    _runtime_constructor_kwargs,
    new_rwkvp_runtime,
    process_review_rows_with_predictions,
    runtime_live_candidate_row_input,
    runtime_predict_many_input,
    runtime_process_many_input,
    runtime_process_row_input,
    runtime_uses_exact_timestamps,
)
from .rwkv_runtime_resources import release_runtime_resources, runtime_is_rust
from .threading_utils import detach_exception_from_thread
from .timestamp_context import (
    MILLISECONDS_PER_DAY,
)
from .v5o_review_rows import V5oInputContext
from .vendor_bootstrap import (
    DependencyError,
    require_rwkv_checkpoint_history_consistency,
    require_rwkv_review_batch,
    require_rwkv_srs,
    require_rwkv_v5o_review_batch,
)

if TYPE_CHECKING:
    from .review_rows import ReviewData


EVALUATION_CACHE_BINDING_MANIFEST_KEY = "evaluation_cache_binding"
EVALUATION_CACHE_BINDING_VERSION = 1
V5O_ROLLOVER_PHASE_MS_MANIFEST_KEY = "v5o_rollover_phase_ms"
V5O_CATALOG_FINGERPRINT_MANIFEST_KEY = "v5o_catalog_fingerprint"


class CheckpointError(RuntimeError):
    pass


class InsufficientCheckpointDiskSpaceError(CheckpointError):
    pass


class MissingCheckpointError(CheckpointError):
    pass


class LegacyCheckpointError(CheckpointError):
    pass


class InconsistentCheckpointError(CheckpointError):
    pass


class StaleCheckpointDataError(InconsistentCheckpointError):
    pass


class CheckpointCacheBindingError(StaleCheckpointDataError):
    """A bound cache changed or no longer belongs to its checkpoint."""


class CheckpointBusyError(CheckpointError):
    pass


@dataclass(frozen=True)
class CheckpointResult:
    status: str
    processed_review_count: int
    last_review_id: Any
    checkpoint_path: str


class ScopedLivePredictionSession:
    """Scope-checked facade over RWKV-SRS's Rust-owned live index.

    The upstream facade remains attached to the loaded runtime.  Every call is
    routed through the checkpoint manager so a stale/closed lease cannot keep
    mutating a runtime that has already been replaced.
    """

    def __init__(
        self,
        lease: ScopedRuntimeLease,
        native_session: Any,
    ) -> None:
        self._lease = lease
        self._native_session = native_session
        self._closed = False
        self._close_lock = threading.Lock()

    @property
    def closed(self) -> bool:
        with self._close_lock:
            return self._closed

    @property
    def generation(self) -> int:
        return int(self._call(lambda live: live.generation))

    @property
    def initial_result(self):
        return self._call(lambda live: live.initial_result)

    def current_selection(self, **kwargs):
        return self._call(lambda live: live.current_selection(**kwargs))

    def mixed_selection(self, **kwargs):
        return self._call(lambda live: live.mixed_selection(**kwargs))

    def refresh(self, **kwargs):
        return self._call(
            lambda live: live.refresh(**self._lease._manager._runtime_live_kwargs(kwargs))
        )

    def reconcile_candidates(self, candidates: Iterable[Any], **kwargs):
        seeds = tuple(candidates)
        return self._call(
            lambda live: live.reconcile_candidates(
                self._lease._manager._runtime_live_candidates(seeds),
                **self._lease._manager._runtime_live_kwargs(kwargs),
            )
        )

    def reconcile_membership(
        self,
        card_ids: Iterable[int],
        changed_candidates: Iterable[Any] = (),
        **kwargs,
    ):
        desired_card_ids = tuple(int(card_id) for card_id in card_ids)
        seeds = tuple(changed_candidates)
        return self._call(
            lambda live: live.reconcile_membership(
                desired_card_ids,
                self._lease._manager._runtime_live_candidates(seeds),
                **self._lease._manager._runtime_live_kwargs(kwargs),
            )
        )

    def process_answer(
        self,
        review_row: dict[str, Any],
        *,
        requeue_after_prediction: bool = False,
    ) -> tuple[float | None, Any]:
        calculate_curves = self._lease.calculate_curves
        result = self._call(
            lambda live: self._process_answer_on_runtime(
                live,
                review_row,
                requeue_after_prediction=bool(requeue_after_prediction),
                return_curves=calculate_curves,
            )
        )
        prediction, curve = _normalized_scalar_process_result(
            result,
            return_curves=calculate_curves,
        )
        if calculate_curves:
            self._lease._remember_undoable_curve(int(review_row["card_id"]), curve)
        return prediction, curve

    def benchmark_process_answer(self, review_row: dict[str, Any]) -> float:
        """Process one curve-free benchmark answer without production fallback."""

        result = self._call(
            lambda live: self._process_answer_on_runtime(
                live,
                review_row,
                requeue_after_prediction=True,
                return_curves=False,
            )
        )
        prediction, _curve = _normalized_scalar_process_result(
            result,
            return_curves=False,
        )
        return prediction

    def process_answer_and_exclude(
        self,
        review_row: dict[str, Any],
    ) -> tuple[float | None, Any]:
        """Process an answer and quarantine its candidate in one worker call."""

        calculate_curves = self._lease.calculate_curves

        def process_and_exclude(live):
            result = self._process_answer_on_runtime(
                live,
                review_row,
                requeue_after_prediction=True,
                return_curves=calculate_curves,
            )
            live.exclude_card(int(review_row["card_id"]))
            return result

        result = self._call(process_and_exclude)
        prediction, curve = _normalized_scalar_process_result(
            result,
            return_curves=calculate_curves,
        )
        if calculate_curves:
            self._lease._remember_undoable_curve(int(review_row["card_id"]), curve)
        return prediction, curve

    def _process_answer_on_runtime(self, live, row, **kwargs):
        manager = self._lease._manager
        native_row = manager._runtime_live_process_row(row)
        return live.process_answer(native_row, **kwargs)


    def undo_last_process(self) -> int:
        def undo(live):
            remaining = live.undo_last_process()
            return remaining

        remaining = int(self._call(undo))
        self._lease._restore_undoable_curve()
        return remaining

    def exclude_card(self, card_id: int) -> int:
        return int(self._call(lambda live: live.exclude_card(int(card_id))))

    def include_card(self, card_id: int) -> int:
        return int(self._call(lambda live: live.include_card(int(card_id))))

    def remove_candidate(self, card_id: int) -> int:
        return int(self._call(lambda live: live.remove_candidate(int(card_id))))

    def upsert_candidates(self, candidates: Iterable[Any]) -> int:
        seeds = tuple(candidates)
        return int(
            self._call(
                lambda live: live.upsert_candidates(
                    self._lease._manager._runtime_live_candidates(seeds)
                )
            )
        )

    def upsert_and_include_candidates(
        self,
        candidates: Iterable[Any],
        card_ids: Iterable[int],
    ) -> int:
        """Refresh quarantined seeds and include them in one worker call."""

        seeds = tuple(candidates)
        normalized_card_ids = tuple(int(card_id) for card_id in card_ids)

        def upsert_and_include(live):
            generation = int(
                live.upsert_candidates(self._lease._manager._runtime_live_candidates(seeds))
            )
            for card_id in normalized_card_ids:
                generation = int(live.include_card(card_id))
            return generation

        return int(self._call(upsert_and_include))

    def replace_candidates(self, candidates: Iterable[Any]) -> int:
        seeds = tuple(candidates)
        generation = int(
            self._call(
                lambda live: live.replace_candidates(
                    self._lease._manager._runtime_live_candidates(seeds)
                )
            )
        )
        self._lease._clear_undoable_curves()
        return generation

    def candidate(self, card_id: int):
        return self._call(lambda live: live.candidate(int(card_id)))

    def snapshot(self):
        return self._call(lambda live: live.snapshot())

    def set_retention_extra(self, value: float) -> int:
        return int(self._call(lambda live: live.set_retention_extra(float(value))))

    def profile(self) -> dict[str, Any]:
        return dict(self._call(lambda live: live.profile()))

    def last_refresh_debug(self) -> dict[str, tuple[int, ...]]:
        return dict(self._call(lambda live: live.last_refresh_debug()))

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._lease._call_live_session(
                self,
                lambda live: live.close(),
                allow_closed_session=True,
            )
        finally:
            self._lease._forget_live_session(self)

    def _call(self, operation):
        with self._close_lock:
            if self._closed:
                raise RuntimeError("live prediction session is closed")
        return self._lease._call_live_session(self, operation)

    def __enter__(self) -> ScopedLivePredictionSession:
        if self.closed:
            raise RuntimeError("live prediction session is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close()


class RuntimeSlotLease:
    """Exclusive manager slot for a caller-owned disposable RWKV runtime.

    The lease reserves the same slot used by partial checkpoint runtimes but
    deliberately does not load or mutate the canonical checkpoint. Callers own
    every runtime they create while holding it and must close those runtimes
    before releasing the slot.
    """

    def __init__(self, manager: RWKVCheckpointManager, token: object) -> None:
        self._manager = manager
        self._token = token
        self._closed = False
        self._close_lock = threading.Lock()

    @property
    def closed(self) -> bool:
        with self._close_lock:
            return self._closed

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._manager._clear_scope_reservation(self._token)

    def __enter__(self) -> RuntimeSlotLease:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close()


class ScopedRuntimeLease:
    """Exclusive, bounded access to one selectively loaded RWKV runtime."""

    def __init__(self, manager: RWKVCheckpointManager, token: object) -> None:
        self._manager = manager
        self._token = token
        self._closed = False
        self._close_lock = threading.Lock()
        self._latest_curve_overrides: dict[int, Any] = {}
        self._undo_curve_snapshots: list[tuple[int, bool, Any]] = []
        self._live_prediction_session: ScopedLivePredictionSession | None = None

    @property
    def model_id(self) -> str:
        return self._manager.model_id

    @property
    def model_generation(self) -> str:
        return self._manager.model_generation

    @property
    def state_precision(self) -> str:
        return self._manager.state_precision

    @property
    def input_context(self) -> RuntimeInputContext | None:
        """Expose the checkpoint-bound exact-timestamp context (V5)."""

        return self._manager._input_context

    @property
    def closed(self) -> bool:
        with self._close_lock:
            return self._closed

    @property
    def predict_many_progress_chunk_size(self) -> int:
        return self._manager.predict_many_progress_chunk_size

    def contains_card(self, card_id: int) -> bool:
        """Return whether this lease loaded the card's entity scope.

        This is a metadata-only check. It deliberately does not dispatch to the
        runtime worker or ask the native runtime to synthesize missing state.
        """

        with self._close_lock:
            if self._closed:
                return False
            token = self._token
        return self._manager._scoped_runtime_contains_card(token, int(card_id))

    def contained_card_ids(self, card_ids: Iterable[int]) -> set[int]:
        """Return the requested card IDs present in this loaded entity scope."""

        requested = {int(card_id) for card_id in card_ids}
        with self._close_lock:
            if self._closed:
                return set()
            token = self._token
        return self._manager._scoped_runtime_contained_card_ids(token, requested)

    def predict_many(
        self,
        rows: list[dict[str, Any]],
        *,
        batch_size: int | None = None,
        allow_gpu: bool = True,
    ) -> list[float]:
        return self._manager._scoped_predict_many(
            self._token,
            rows,
            batch_size=batch_size,
            allow_gpu=allow_gpu,
        )

    def benchmark_predict_many(
        self,
        rows: list[dict[str, Any]],
        *,
        mode: str,
        batch_size: int | None = None,
    ) -> list[float]:
        """Run one explicitly selected predict_many mode without fallback.

        Normal add-on prediction deliberately falls back from GPU to Fast when
        necessary. A speed test must fail instead, or it could label Fast CPU
        timings as GPU results.
        """

        return self._manager._scoped_benchmark_predict_many(
            self._token,
            rows,
            mode=mode,
            batch_size=batch_size,
        )

    def predict_many_live_session(
        self,
        candidates: Iterable[Any],
        *,
        initial_target_timestamp_seconds: float | None = None,
        initial_target_timestamp_ms: int | None = None,
        initial_rollover_phase_ms: int | None = None,
        initial_target_day_offset: float | None = None,
        order: str,
        mode: str | None = None,
        batch_size: int | None = None,
        refresh_limit: int,
        profiling: bool = False,
        initial_select_limit: int = 2,
    ) -> ScopedLivePredictionSession:
        with self._close_lock:
            if self._closed:
                raise MissingCheckpointError("This RWKV runtime scope has been closed.")
            existing = self._live_prediction_session
            token = self._token
            if existing is not None and not existing.closed:
                return existing
            # Keep creation and installation under one lease lock. Native
            # construction can block, and otherwise two callers can both see
            # an empty slot and orphan the first native session.
            native = self._manager._scoped_predict_many_live_session(
                token,
                tuple(candidates),
                initial_target_timestamp_seconds=(
                    None
                    if initial_target_timestamp_seconds is None
                    else float(initial_target_timestamp_seconds)
                ),
                initial_target_timestamp_ms=initial_target_timestamp_ms,
                initial_rollover_phase_ms=initial_rollover_phase_ms,
                initial_target_day_offset=initial_target_day_offset,
                order=str(order),
                mode=mode,
                batch_size=batch_size,
                refresh_limit=int(refresh_limit),
                profiling=bool(profiling),
                initial_select_limit=int(initial_select_limit),
            )
            live = ScopedLivePredictionSession(self, native)
            self._live_prediction_session = live
        return live

    def benchmark_predict_many_live_session(
        self,
        candidates: Iterable[Any],
        *,
        initial_target_timestamp_seconds: float | None = None,
        initial_target_timestamp_ms: int | None = None,
        initial_rollover_phase_ms: int | None = None,
        initial_target_day_offset: float | None = None,
        order: str,
        mode: str,
        batch_size: int | None = None,
        refresh_limit: int,
        profiling: bool = False,
        initial_select_limit: int = 2,
    ) -> ScopedLivePredictionSession:
        """Open one strict native session for warmed prediction measurements.

        Unlike production Live Session startup, a benchmark must surface GPU
        preflight or execution failure instead of silently timing Fast CPU under
        the GPU label.
        """

        with self._close_lock:
            if self._closed:
                raise MissingCheckpointError("This RWKV runtime scope has been closed.")
            existing = self._live_prediction_session
            token = self._token
            if existing is not None and not existing.closed:
                raise RuntimeError("This RWKV runtime scope already owns a live session.")
            native = self._manager._scoped_benchmark_predict_many_live_session(
                token,
                tuple(candidates),
                initial_target_timestamp_seconds=(
                    None
                    if initial_target_timestamp_seconds is None
                    else float(initial_target_timestamp_seconds)
                ),
                initial_target_timestamp_ms=initial_target_timestamp_ms,
                initial_rollover_phase_ms=initial_rollover_phase_ms,
                initial_target_day_offset=initial_target_day_offset,
                order=str(order),
                mode=str(mode),
                batch_size=batch_size,
                refresh_limit=int(refresh_limit),
                profiling=bool(profiling),
                initial_select_limit=int(initial_select_limit),
            )
            live = ScopedLivePredictionSession(self, native)
            self._live_prediction_session = live
        return live

    def process_one(self, row: dict[str, Any]) -> tuple[float | None, Any]:
        return self._manager._scoped_process_one(self._token, row)

    def process_simulation_one(
        self,
        row: dict[str, Any],
        *,
        return_curves: bool,
    ) -> tuple[float | None, Any | None]:
        """Mutate only this disposable scope with an explicit curve contract.

        Behavior Lab branches never merge their state into the durable checkpoint,
        so their transient curve choice is deliberately independent of the
        checkpoint evaluation-cache setting.
        """

        return self._manager._scoped_process_simulation_one(
            self._token,
            row,
            return_curves=bool(return_curves),
        )

    def process_simulation_many(
        self,
        rows: list[dict[str, Any]],
    ) -> list[float]:
        """Process a curve-less simulation block on this disposable scope."""

        return self._manager._scoped_process_simulation_many(self._token, rows)

    @property
    def calculate_curves(self) -> bool:
        return self._manager.calculate_curves

    def evaluation_prediction_tail(self) -> PredictionTailSnapshot:
        """Copy predictions replayed beyond the durable evaluation cache."""

        return self._manager._scoped_evaluation_prediction_tail(self._token)

    def undoable_process_one(self, row: dict[str, Any]) -> tuple[float | None, Any]:
        result = self._manager._scoped_undoable_process_one(self._token, row)
        if self.calculate_curves:
            self._remember_undoable_curve(int(row["card_id"]), result[1])
            return result
        return (None if result[0] is None else float(result[0])), None

    def undo_last_process(self) -> int:
        result = self._manager._scoped_undo_last_process(self._token)
        self._restore_undoable_curve()
        return result

    def _remember_undoable_curve(self, card_id: int, curve: Any) -> None:
        with self._close_lock:
            normalized = int(card_id)
            had_override = normalized in self._latest_curve_overrides
            previous_curve = self._latest_curve_overrides.get(normalized)
            self._undo_curve_snapshots.append(
                (normalized, had_override, previous_curve),
            )
            excess = len(self._undo_curve_snapshots) - LIVE_REVIEW_RUST_UNDO_LIMIT
            if excess > 0:
                del self._undo_curve_snapshots[:excess]
            self._latest_curve_overrides[normalized] = curve

    def _restore_undoable_curve(self) -> None:
        with self._close_lock:
            if not self._undo_curve_snapshots:
                return
            card_id, had_override, previous_curve = self._undo_curve_snapshots.pop()
            if had_override:
                self._latest_curve_overrides[card_id] = previous_curve
            else:
                self._latest_curve_overrides.pop(card_id, None)

    def _clear_undoable_curves(self) -> None:
        with self._close_lock:
            self._undo_curve_snapshots.clear()

    def _call_live_session(
        self,
        session: ScopedLivePredictionSession,
        operation,
        *,
        allow_closed_session: bool = False,
    ):
        with self._close_lock:
            if self._closed:
                raise MissingCheckpointError("This RWKV runtime scope has been closed.")
            if self._live_prediction_session is not session:
                raise RuntimeError("live prediction session is no longer active")
            native = session._native_session
            token = self._token
        if not allow_closed_session and session.closed:
            raise RuntimeError("live prediction session is closed")
        return self._manager._run_scoped_call(token, lambda: operation(native))

    def _forget_live_session(self, session: ScopedLivePredictionSession) -> None:
        with self._close_lock:
            if self._live_prediction_session is session:
                self._live_prediction_session = None

    def latest_curve_for_card(self, card_id: int) -> Any | None:
        normalized_card_id = int(card_id)
        with self._close_lock:
            if normalized_card_id in self._latest_curve_overrides:
                return self._latest_curve_overrides[normalized_card_id]
        return self._manager.latest_curve_for_card(normalized_card_id)

    def latest_curves_for_cards(self, card_ids: Iterable[int]) -> dict[int, Any]:
        ids = {int(card_id) for card_id in card_ids}
        curves = self._manager.latest_curves_for_cards(ids)
        with self._close_lock:
            for card_id in ids:
                if card_id in self._latest_curve_overrides:
                    curves[card_id] = self._latest_curve_overrides[card_id]
        return curves

    def current_undo_depth(self) -> int:
        return self._manager._scoped_current_undo_depth(self._token)

    def close(self) -> None:
        with self._close_lock:
            live_session = self._live_prediction_session
        if live_session is not None:
            with suppress(Exception):
                live_session.close()
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._latest_curve_overrides.clear()
            self._undo_curve_snapshots.clear()
        self._manager._close_scoped_runtime(self._token)

    def __enter__(self) -> ScopedRuntimeLease:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close()


class _RuntimeWorker:
    def __init__(self) -> None:
        self._queue: queue.Queue[object] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._lock = threading.Lock()
        self._stopping = False
        self._stop_result: dict[str, object] = {}

    def is_worker_thread(self) -> bool:
        return self._thread_id == threading.get_ident()

    def call(self, op):
        if self.is_worker_thread():
            return op()

        done = threading.Event()
        result: dict[str, object] = {}
        with self._lock:
            if self._stopping:
                raise CheckpointError("The RWKV runtime worker is shutting down.")
            self._ensure_started()
            # Submit under the same lock as the stop sentinel. A task queued
            # behind that sentinel would otherwise wait forever for its result.
            self._queue.put((op, done, result))
        done.wait()
        if "exception" in result:
            raise result["exception"]
        return result.get("value")

    def stop(self, *, finalize=None) -> None:
        if self.is_worker_thread():
            return
        with self._lock:
            if not self._stopping:
                self._stopping = True
                if finalize is not None:
                    self._ensure_started()
                    self._queue.put((finalize, threading.Event(), self._stop_result))
                thread = self._thread
                if thread is not None:
                    self._queue.put(None)
            else:
                thread = self._thread
        if thread is None:
            return
        thread.join()
        with self._lock:
            if self._thread is thread:
                self._thread = None
                self._thread_id = None
        if "exception" in self._stop_result:
            raise self._stop_result["exception"]

    def _ensure_started(self) -> None:
        # The caller holds _lock through startup and submission.
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name="RWKV runtime",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        self._thread_id = threading.get_ident()
        while True:
            item = self._queue.get()
            if item is None:
                return
            op, done, result = item
            try:
                result["value"] = op()
            except BaseException as exc:
                result["exception"] = detach_exception_from_thread(exc)
            finally:
                done.set()
                del op, done, result, item


@dataclass(frozen=True)
class _DurableCheckpointWrite:
    runtime: Any
    rows: list[dict[str, Any]]
    prediction_records: PredictionRecordSet
    prediction_cache_specs: tuple[PredictionCacheSpec, ...]
    latest_curves_by_card: dict[int, Any] | None
    status: str
    checkpoint_path: Path
    storage_estimate: RustCheckpointStorageEstimate | None
    rollover_phase_ms: int | None
    v5o_catalog_fingerprint: str | None = None
    snapshot_store: ProfileStore | None = None


class RWKVCheckpointManager:


    def __init__(
        self,
        store: ProfileStore,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        model_generation: str = DEFAULT_MODEL_GENERATION,
        state_precision: str = "fp32",
        gpu_prediction_math: str = "fp32",
        async_saves: bool = True,
        prediction_cache_specs: tuple[PredictionCacheSpec, ...] | None = None,
        predict_many_mode: str = DEFAULT_PREDICT_MANY_MODE,
        predict_many_batch_size: int | None = None,
        process_many_mode: str = DEFAULT_PROCESS_MANY_MODE,
        checkpoint_save_interval: int | None = None,
        exclude_deleted_card_revlogs: bool = True,
        filtered_review_normalization_policy: (FilteredReviewNormalizationPolicy | None) = None,
    ) -> None:
        self.model_id = model_id
        self.model_generation = normalize_model_generation(model_generation)
        if state_precision not in {"fp16", "fp32"}:
            raise ValueError("State precision must be fp16 or fp32.")
        self.state_precision = state_precision
        if gpu_prediction_math not in {"fp16", "fp32"}:
            raise ValueError("GPU prediction math must be fp16 or fp32.")
        self.gpu_prediction_math = gpu_prediction_math
        self._scope_gpu_prediction_math: str | None = None
        # Explicit unpublished stores must stay isolated throughout a comparison
        # rebuild; switching generations still selects the canonical slot.
        if normalize_model_generation(store.model_generation) != self.model_generation:
            store = store.for_generation(self.model_generation)
        self.store = store
        store.ensure()
        self.async_saves = async_saves
        self.predict_many_mode = normalize_predict_many_mode(predict_many_mode)
        self.predict_many_batch_size = _positive_optional_int(predict_many_batch_size)
        self.process_many_mode = normalize_process_many_mode(process_many_mode)
        self.checkpoint_save_interval = _positive_int_or_default(
            checkpoint_save_interval,
            CHECKPOINT_SAVE_INTERVAL,
        )
        self.exclude_deleted_card_revlogs = bool(exclude_deleted_card_revlogs)
        self.filtered_review_normalization_policy = (
            filtered_review_normalization_policy or FilteredReviewNormalizationPolicy.disabled()
        )
        self._prediction_cache_specs = self._generation_prediction_cache_specs(
            prediction_cache_specs
        )
        self._durable_writer = _DurableCheckpointWriter(
            store,
            model_id,
            self.model_generation,
            self._prediction_cache_specs,
            self.filtered_review_normalization_policy,
            self.exclude_deleted_card_revlogs,
            state_precision=self.state_precision,
        )
        self._runtime = None
        self._loaded_path: str | None = None
        self._loaded_scope_cards: list[dict[str, Any]] | None = None
        self._loaded_scope_key: tuple[tuple[int | None, ...], ...] | None = None
        self._active_scope_token: object | None = None
        self._active_scope_requester_thread_id: int | None = None
        self._scope_condition = threading.Condition()
        self._runtime_processed_review_count: int | None = None
        self._runtime_last_review_id: int | None = None
        self._runtime_dirty = False
        self._runtime_gpu_ready: bool | None = None
        self._runtime_gpu_failure: str | None = None
        self._unsaved_prediction_records = PredictionRecordSet.empty()
        self._latest_curves_by_card: dict[int, Any] | None = None
        self._latest_curves_complete = False
        self._review_data_cache: ReviewData | None = None
        self._input_context: RuntimeInputContext | None = None
        self._review_data_cache_last_review_id: int | None = None
        self._review_data_cache_latest_collection_review_id: int | None = None
        self._review_data_cache_collection_revision: CollectionRevision | None = None
        self._pending_review_tail_context: tuple[ReviewData, CollectionRevision] | None = None
        self._verified_for_incremental_updates = False
        self._verified_history_rows: list[dict[str, Any]] | None = None
        self._verified_checkpoint_signature: tuple[str, int, int, int, int] | None = None
        self._evaluation_cache_validation: EvaluationCacheValidation | None = None
        self._durable_status_cache_key: tuple[Any, ...] | None = None
        self._durable_status_cache_value: str | None = None
        self._save_lock = threading.RLock()
        self._save_thread: threading.Thread | None = None
        self._save_token: object | None = None
        self._save_status: str | None = None
        self._save_error: BaseException | None = None
        self._runtime_worker: _RuntimeWorker | None = None
        self._runtime_worker_condition = threading.Condition()
        self._runtime_worker_unloading = False

    def __del__(self) -> None:
        with suppress(BaseException):
            self.unload()

    def set_prediction_cache_specs(
        self,
        specs: tuple[PredictionCacheSpec, ...],
    ) -> None:
        normalized_specs = self._generation_prediction_cache_specs(specs)
        if normalized_specs == self._prediction_cache_specs:
            return
        if self.runtime_scope_active:
            raise CheckpointBusyError(
                "Stop the active RWKV operation or Live Session before changing curve calculation."
            )
        if self.save_in_progress:
            raise CheckpointBusyError(
                "Wait for the pending RWKV checkpoint save before changing curve calculation."
            )
        if self._runtime_dirty:
            # The unsaved prediction arrays were produced under the old cache
            # contract and cannot safely be combined with rows produced under
            # the new one.  The durable checkpoint remains authoritative, so
            # discard the transient runtime/tail and replay it from review
            # history on the next operation.
            self.release_runtime(preserve_review_data=True)
        curves_were_enabled = self.calculate_curves
        self._prediction_cache_specs = normalized_specs
        self._durable_writer.prediction_cache_specs = normalized_specs
        self._evaluation_cache_validation = None
        self._durable_status_cache_key = None
        self._durable_status_cache_value = None
        if curves_were_enabled != self.calculate_curves:
            self._latest_curves_by_card = None
            self._latest_curves_complete = not self.calculate_curves

    def _generation_prediction_cache_specs(
        self,
        specs: tuple[PredictionCacheSpec, ...] | None,
    ) -> tuple[PredictionCacheSpec, ...]:
        normalized = _normalize_prediction_cache_specs(specs)
        if model_generation_uses_exact_timestamps(self.model_generation):
            return (PER_REVIEW_CACHE_SPEC,)
        return normalized

    @property
    def calculate_curves(self) -> bool:
        return PREDICT_AHEAD_CACHE_SPEC in self._prediction_cache_specs

    def set_predict_many_batch_size(self, batch_size: int | None) -> None:
        self.predict_many_batch_size = _positive_optional_int(batch_size)

    def set_predict_many_mode(self, mode: str) -> None:
        self.predict_many_mode = normalize_predict_many_mode(mode)

    def set_process_many_mode(self, mode: str) -> None:
        """Apply a Fast/GPU review-history mode without replacing the manager."""

        normalized = normalize_process_many_mode(mode)
        if normalized == self.process_many_mode:
            return
        self.process_many_mode = normalized

    def set_exclude_deleted_card_revlogs(self, exclude: bool) -> None:
        """Change the history policy and discard data loaded under the old policy."""

        normalized = bool(exclude) or model_generation_uses_v5o(self.model_generation)
        if normalized == self.exclude_deleted_card_revlogs:
            return
        if self.runtime_scope_active:
            raise CheckpointBusyError(
                "Stop the active RWKV operation or Live Session before changing "
                "deleted-card history."
            )
        if self.save_in_progress:
            raise CheckpointBusyError(
                "Wait for the pending RWKV checkpoint save before changing deleted-card history."
            )
        self.release_runtime(preserve_review_data=False)
        self.exclude_deleted_card_revlogs = normalized
        self._durable_writer.exclude_deleted_card_revlogs = normalized
        self._review_data_cache_collection_revision = None
        self._pending_review_tail_context = None
        self._durable_status_cache_key = None
        self._durable_status_cache_value = None

    @property
    def _effective_predict_many_mode(self) -> str:
        if configured_rwkv_backend() == "rust":
            return self.predict_many_mode
        return PREDICT_MANY_ORACLE_MODE

    @property
    def predict_many_progress_chunk_size(self) -> int:
        return prediction_progress_chunk_size(
            self._effective_predict_many_mode,
            self.predict_many_batch_size,
        )

    @property
    def gpu_fallback_reason(self) -> str | None:
        return self._runtime_gpu_failure

    def set_checkpoint_save_interval(self, interval: int) -> None:
        self.checkpoint_save_interval = _positive_int_or_default(
            interval,
            CHECKPOINT_SAVE_INTERVAL,
        )

    @property
    def has_checkpoint(self) -> bool:
        return self._runtime is not None or self.store.active_checkpoint_path() is not None

    @property
    def runtime_loaded(self) -> bool:
        return self._runtime is not None

    @property
    def runtime_scope_active(self) -> bool:
        """Return whether an exclusive bounded runtime lease is reserved."""

        with self._scope_condition:
            return self._active_scope_token is not None

    @property
    def save_in_progress(self) -> bool:
        with self._save_lock:
            return self._save_thread is not None and self._save_thread.is_alive()

    def durable_processed_review_count(self) -> int | None:
        if self._runtime is not None:
            processed = self._runtime_processed_count()
            if self._runtime_dirty:
                processed -= len(self._unsaved_prediction_records.immediate_predictions)
            return max(0, processed)
        manifest_count = self.store.manifest().get("processed_review_count")
        return None if manifest_count is None else int(manifest_count)

    def checkpoint_history_fingerprint(self) -> dict[str, Any] | None:
        """Return the normalized fingerprint of the active durable checkpoint."""

        return _checkpoint_history_fingerprint(self.store.active_checkpoint_path())

    def _manifest_processed_review_count(self) -> int:
        value = self.store.manifest().get("processed_review_count")
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def current_last_review_id(self) -> int | None:
        if self._runtime is not None:
            return _optional_int(self._runtime_last_review_id_value())
        if (
            self._review_data_cache is not None
            and self._history_verification_is_current(self._review_data_cache.rows)
            and self._review_data_cache_last_review_id is not None
        ):
            return int(self._review_data_cache_last_review_id)
        return _optional_int(self.store.manifest().get("last_review_id"))

    def latest_curves_by_card(self) -> dict[int, Any]:
        if not self.calculate_curves:
            return {}
        return _copy_curve_map(self._latest_curves_by_card) or {}

    def latest_curve_for_card(self, card_id: int) -> Any | None:
        return self.latest_curves_for_cards((int(card_id),)).get(int(card_id))

    def evaluation_cache_validation(self) -> EvaluationCacheValidation | None:
        """Return a current checkpoint-bound proof for evaluation-cache reads.

        The immutable cache is already cryptographically bound to the durable
        checkpoint during readiness checks.  Consumers may pass this proof back
        to the cache loader to avoid rebuilding the complete Python history
        digest for every read.  File identity and checkpoint lineage are checked
        again before the proof is exposed.
        """

        if not self._checkpoint_lineage_is_current():
            return None
        if (
            not evaluation_cache_validation_is_current(self._evaluation_cache_validation)
            and not self._restore_evaluation_cache_validation_from_binding()
        ):
            return None
        return self._evaluation_cache_validation

    def latest_curves_for_cards(self, card_ids: Iterable[int]) -> dict[int, Any]:
        if not self.calculate_curves:
            return {}
        ids = {int(card_id) for card_id in card_ids}
        if not ids:
            return {}
        self._discard_stale_selected_curves()
        cached_rows = self._review_data_cache
        loaded_ids = set(self._latest_curves_by_card or {})
        if cached_rows is not None and not ids <= loaded_ids:
            self._ensure_latest_curves_loaded(cached_rows.rows, card_ids=ids)
        if self._latest_curves_by_card is None:
            return {}
        return {
            card_id: self._latest_curves_by_card[card_id]
            for card_id in ids
            if card_id in self._latest_curves_by_card
        }

    def cached_review_data_if_current(
        self,
        latest_collection_review_id: int | None,
    ) -> ReviewData | None:
        if latest_collection_review_id is None:
            return None
        current_last_review_id = self.current_last_review_id()
        if current_last_review_id is None:
            return None
        if self._review_data_cache is None:
            return None
        if not self._review_data_cache.complete_history:
            return None
        if self._review_data_cache_latest_collection_review_id != int(latest_collection_review_id):
            return None
        if self._review_data_cache_last_review_id != int(current_last_review_id):
            return None
        return self._review_data_cache

    def cached_review_data_for_revision(
        self,
        collection_revision: CollectionRevision,
    ) -> ReviewData | None:
        if self._review_data_cache_collection_revision != collection_revision:
            return None
        return self.cached_review_data_if_current(collection_revision.latest_review_id)

    def cached_day_offset_origin(self, next_day_at: int | None = None) -> int | None:
        """Return the loaded review history's global day-offset origin.

        The normalized history coordinate is stable across ordinary reviews.
        When Anki's day cutoff advances, callers can pass the current cutoff to
        rebase the raw origin without requiring the full cached ReviewData to
        still match the latest revlog.
        """

        if self._review_data_cache is not None:
            review_data = self._review_data_cache
            if next_day_at is not None:
                return rebased_day_offset_origin(
                    review_data.rows,
                    current_origin=review_data.day_offset_origin,
                    next_day_at=int(next_day_at),
                    previous_next_day_at=review_data.next_day_at,
                )
            return int(review_data.day_offset_origin)
        value = self.store.manifest().get("day_offset_origin")
        return None if value is None else int(value)

    def incremental_review_data_source(
        self,
        latest_collection_review_id: int | None,
    ) -> tuple[ReviewData, int] | None:
        if latest_collection_review_id is None:
            return None
        if self._review_data_cache is None:
            return None
        if not self._review_data_cache.complete_history:
            return None
        if not self._history_verification_is_current(self._review_data_cache.rows):
            return None
        cache_latest = self._review_data_cache_latest_collection_review_id
        if cache_latest is None or int(latest_collection_review_id) <= int(cache_latest):
            return None
        current_last_review_id = self.current_last_review_id()
        if current_last_review_id != self._review_data_cache_last_review_id:
            return None
        return self._review_data_cache, int(cache_latest)

    def incremental_review_data_source_for_revision(
        self,
        collection_revision: CollectionRevision,
    ) -> tuple[ReviewData, int] | None:
        # A changed collection revision does not prove that only a suffix was
        # appended.  Reuse is safe only when the complete snapshot is unchanged;
        # in that case cached_review_data_for_revision() handles it already.
        if self._review_data_cache_collection_revision != collection_revision:
            return None
        return self.incremental_review_data_source(collection_revision.latest_review_id)

    def _history_verification_is_current(
        self,
        rows: list[dict[str, Any]],
    ) -> bool:
        return (
            self._verified_for_incremental_updates
            and self._verified_history_rows is rows
            and self._verified_checkpoint_signature is not None
            and self._verified_checkpoint_signature
            == _checkpoint_file_signature(self.store.active_checkpoint_path())
        )

    def _checkpoint_lineage_is_current(self) -> bool:
        return (
            self._checkpoint_model_matches_config()
            and self._checkpoint_policy_matches_config()
            and self._verified_for_incremental_updates
            and self._verified_checkpoint_signature is not None
            and self._verified_checkpoint_signature
            == _checkpoint_file_signature(self.store.active_checkpoint_path())
        )

    def _mark_history_verified(
        self,
        rows: list[dict[str, Any]],
        *,
        incremental: bool = False,
    ) -> None:
        checkpoint_signature = _checkpoint_file_signature(self.store.active_checkpoint_path())
        lineage_changed = (
            self._verified_history_rows is not rows
            or self._verified_checkpoint_signature != checkpoint_signature
        )
        if lineage_changed and not incremental:
            self._evaluation_cache_validation = None
            if not self._latest_curves_complete:
                self._latest_curves_by_card = None
        self._verified_for_incremental_updates = True
        self._verified_history_rows = rows
        self._verified_checkpoint_signature = checkpoint_signature

    def _invalidate_history_verification(self) -> None:
        self._verified_for_incremental_updates = False
        self._verified_history_rows = None
        self._verified_checkpoint_signature = None
        self._evaluation_cache_validation = None

    def _discard_stale_selected_curves(self) -> None:
        if self._latest_curves_complete or self._latest_curves_by_card is None:
            return
        if not self._checkpoint_lineage_is_current() or not evaluation_cache_validation_is_current(
            self._evaluation_cache_validation
        ):
            self._latest_curves_by_card = None
            self._evaluation_cache_validation = None

    def remember_review_data(
        self,
        review_data: ReviewData,
        *,
        latest_collection_review_id: int | None = None,
    ) -> None:
        self._review_data_cache = review_data
        if model_generation_uses_exact_timestamps(self.model_generation):
            current_context = input_context_for_generation(review_data, self.model_generation)
            durable_phase = (
                _rollover_phase_from_manifest(self.store.manifest(), self.model_generation)
                if self.store.active_checkpoint_path() is not None
                else None
            )
            self._input_context = replace(
                current_context,
                rollover_phase_ms=(
                    current_context.rollover_phase_ms if durable_phase is None else durable_phase
                ),
            )
        else:
            self._input_context = None
        self._review_data_cache_collection_revision = None
        self._review_data_cache_last_review_id = _last_review_id_from_rows(review_data.rows)
        self._review_data_cache_latest_collection_review_id = (
            int(latest_collection_review_id)
            if latest_collection_review_id is not None
            else self._review_data_cache_last_review_id
        )

    def remember_review_data_collection_revision(
        self,
        review_data: ReviewData,
        collection_revision: CollectionRevision,
    ) -> None:
        if self._review_data_cache is review_data:
            self._review_data_cache_collection_revision = collection_revision

    def accept_persisted_review_tail(
        self,
        rows: CheckpointTailRows,
        checkpoint_history_fingerprint: Mapping[str, Any],
    ) -> bool:
        """Trust a virtual checkpoint prefix after its durable bindings match.

        This is deliberately stricter than ordinary consistency checking.  The
        persisted context is only an acceleration cache, so any uncertainty
        returns ``False`` and lets the caller rebuild complete Python history.
        """

        if model_generation_uses_v5o(self.model_generation):
            # V1's virtual-prefix cache has no retained-history provenance.
            return False
        manifest = self.store.manifest()
        if (
            model_generation_uses_exact_timestamps(self.model_generation)
            and _rollover_phase_from_manifest(manifest, self.model_generation) is None
        ):
            # A virtual prefix deliberately skips replaying durable history.
            # Without the checkpoint's canonical phase, it cannot reconstruct
            # the omitted retained-history rows. Fall back to complete history and rebuild the
            # unpublished predecessor contract instead of guessing.
            return False
        if (
            self.store.checkpoint_status() != "valid"
            or self.save_in_progress
            or self._runtime_dirty
            or not self._checkpoint_model_matches_config()
            or not self._checkpoint_policy_matches_config()
        ):
            return False
        current_fingerprint = _checkpoint_history_fingerprint(self.store.active_checkpoint_path())
        expected_fingerprint = _normalized_history_fingerprint(checkpoint_history_fingerprint)
        if current_fingerprint is None or current_fingerprint != expected_fingerprint:
            return False
        durable_count = int(current_fingerprint["processed_review_count"])
        if durable_count != rows.durable_count:
            return False
        # A save needs the complete durable prediction prefix and process rows.
        # Keep the virtual-prefix path strictly below that boundary.
        if len(rows.materialized_tail) >= self.checkpoint_save_interval:
            return False

        self._mark_history_verified(rows, incremental=True)
        try:
            if not self._restore_evaluation_cache_validation_from_binding():
                self._invalidate_history_verification()
                return False
        except CheckpointError:
            self._invalidate_history_verification()
            return False
        return True

    def persist_review_tail_context(
        self,
        review_data: ReviewData,
        collection_revision: CollectionRevision,
    ) -> None:
        """Persist V1's disposable cold-tail context for verified complete rows."""

        if model_generation_uses_v5o(self.model_generation):
            # This V1 cache omits V5 provenance/catalog bindings and is never
            # accepted by its loader. Skip both writing and deferred-save
            # queuing; in-memory revision/history proofs remain independent.
            return
        if not review_data.complete_history:
            return
        with self._save_lock:
            if self._save_thread is not None and self._save_thread.is_alive():
                # The durable fingerprint/file binding changes when the writer
                # finishes. Retain the already-loaded compact history and write
                # its context only after the new checkpoint is atomically exposed.
                self._pending_review_tail_context = (review_data, collection_revision)
                return
        if (
            not self._history_verification_is_current(review_data.rows)
            or self.store.checkpoint_status() != "valid"
        ):
            return
        fingerprint = _checkpoint_history_fingerprint(self.store.active_checkpoint_path())
        if fingerprint is None:
            return
        durable_count = int(fingerprint["processed_review_count"])
        try:
            write_review_tail_context(
                self.store.review_tail_context_path,
                review_data,
                durable_processed_review_count=durable_count,
                model_id=self.model_id,
                exclude_deleted_card_revlogs=self.exclude_deleted_card_revlogs,
                filtered_review_normalization_policy=(self.filtered_review_normalization_policy),
                checkpoint_history_fingerprint=fingerprint,
                collection_revision=collection_revision,
            )
        except (OSError, ReviewTailContextError, TypeError, ValueError):
            # This file is optional acceleration data.  The complete-history
            # path remains authoritative and will simply try again later.
            return

    def status(self) -> str:
        with self._save_lock:
            if self._runtime is not None and self._save_thread is not None and self._save_status:
                return self._save_status
        status = self.store.checkpoint_status()
        manifest = self.store.manifest()
        cache_key = self._durable_status_key(status, manifest)
        if cache_key == self._durable_status_cache_key:
            return self._durable_status_cache_value or status
        resolved = status
        if status != "missing" and self._legacy_rust_checkpoint_requires_rebuild():
            resolved = "legacy"
        elif status in {"valid", "partial"} and (
            not self._checkpoint_model_matches_config(manifest)
            or not self._checkpoint_policy_matches_config(manifest)
        ):
            resolved = "invalid"
        elif status in {"valid", "partial"} and not self._durable_cache_files_available(
            status,
            manifest,
        ):
            resolved = "stale_cache"
        self._durable_status_cache_key = cache_key
        self._durable_status_cache_value = resolved
        return resolved

    def _durable_status_key(
        self,
        status: str,
        manifest: dict[str, Any],
    ) -> tuple[Any, ...]:
        partial = status == "partial"
        cache_path = _evaluation_cache_path_from_manifest_or_store(
            manifest,
            self.store,
            partial=partial,
        )
        return (
            status,
            manifest.get("model_id"),
            manifest.get("model_generation"),
            manifest.get("state_precision", "fp32"),
            self.state_precision,
            manifest.get("exclude_deleted_card_revlogs"),
            self.exclude_deleted_card_revlogs,
            manifest.get(V5O_ROLLOVER_PHASE_MS_MANIFEST_KEY),
            manifest.get(FILTERED_REVIEW_NORMALIZATION_MANIFEST_KEY),
            tuple(sorted(self.filtered_review_normalization_policy.semantic_signature().items())),
            manifest.get("processed_review_count"),
            manifest.get("checkpoint_path"),
            _checkpoint_file_signature(self.store.active_checkpoint_path()),
            manifest.get(EVALUATION_CACHE_MANIFEST_KEY),
            _checkpoint_file_signature(cache_path),
            tuple(spec.cache_kind for spec in self._prediction_cache_specs),
        )

    def wait_for_pending_save(self, timeout: float | None = None) -> bool:
        with self._save_lock:
            thread = self._save_thread
        if thread is None:
            self._raise_background_save_error()
            return True
        if thread is threading.current_thread():
            return True
        thread.join(timeout)
        if thread.is_alive():
            return False
        self._raise_background_save_error()
        return True

    def initialize_or_rebuild(
        self,
        rows: list[dict[str, Any]],
        progress: ProgressReporter,
        *,
        partial_on_cancel: bool = True,
    ) -> CheckpointResult:
        with self._reserve_maintenance_slot("rebuilding the checkpoint"):
            if self._should_use_runtime_worker():
                return self._run_on_runtime_worker(
                    lambda: self._initialize_or_rebuild_impl(
                        rows,
                        progress,
                        partial_on_cancel=partial_on_cancel,
                    )
                )
            return self._initialize_or_rebuild_impl(
                rows,
                progress,
                partial_on_cancel=partial_on_cancel,
            )

    def expected_checkpoint_storage(
        self,
        rows: list[dict[str, Any]],
    ) -> RustCheckpointStorageEstimate:
        """Plan the next full Rust checkpoint size without building its state."""

        with self._reserve_maintenance_slot("estimating checkpoint storage", require_unloaded=True):
            counts = rust_checkpoint_identity_counts(rows)
            if configured_rwkv_backend() != "rust":
                return estimate_rust_checkpoint_storage_from_counts(counts)
            if self._should_use_runtime_worker():
                return self._run_on_runtime_worker(
                    lambda: self._expected_checkpoint_storage_impl(counts)
                )
            return self._expected_checkpoint_storage_impl(counts)

    def _expected_checkpoint_storage_impl(
        self,
        counts: RustCheckpointIdentityCounts,
    ) -> RustCheckpointStorageEstimate:
        runtime = new_rwkvp_runtime(
            model_id=self.model_id,
            model_generation=self.model_generation,
            process_many_mode=self.process_many_mode,
            state_precision=self.state_precision,
            gpu_prediction_math=self.gpu_prediction_math,
        )
        try:
            expected_bytes = _runtime_expected_checkpoint_size(runtime, counts)
        finally:
            _release_runtime_resources(runtime)
        return estimate_rust_checkpoint_storage_from_counts(
            counts,
            expected_checkpoint_bytes=expected_bytes,
        )

    def open_scoped_runtime(
        self,
        rows: list[dict[str, Any]],
        scope_cards: Iterable[dict[str, Any]] | None,
        progress: ProgressReporter,
        *,
        allow_inconsistent: bool = False,
        force_save: bool = False,
        check_consistency: bool = True,
        gpu_prediction_math: str | None = None,
    ) -> tuple[ScopedRuntimeLease, CheckpointResult]:
        """Load requested identities, or the complete state, for one operation.

        The returned lease is exclusive and must be closed. Closing always drops
        recurrent state; reviews newer than the durable checkpoint are replayed
        from ``rows`` by the next lease. Passing ``scope_cards=None`` requests the
        complete checkpoint state; an iterable requests only those identities plus
        identities required to replay the pending review tail.
        ``gpu_prediction_math`` overrides arithmetic for this lease only (for
        strict comparisons); it never changes stored state or manager settings.
        """

        if gpu_prediction_math is not None and (
            gpu_prediction_math not in {"fp16", "fp32"}
        ):
            raise ValueError("Unsupported scoped GPU prediction math.")
        requested_cards = None if scope_cards is None else _dedupe_scope_cards(scope_cards)
        token = object()
        self._reserve_scoped_runtime(token, progress)

        def open_runtime() -> CheckpointResult:
            # Exclusive lease-local constructor choice. Never alter the manager's
            # configured math or checkpoint identity for a disposable benchmark.
            self._scope_gpu_prediction_math = gpu_prediction_math
            self._runtime_operation_rows = rows
            try:
                result = self._ensure_ready_for_prediction_impl(
                    rows,
                    progress,
                    allow_inconsistent=allow_inconsistent,
                    force_save=force_save,
                    check_consistency=check_consistency,
                    scope_cards=requested_cards,
                )
                return result
            except BaseException:
                try:
                    self._release_scoped_runtime_state(preserve_curves=False)
                finally:
                    self._clear_scope_reservation(token)
                raise
            finally:
                self._runtime_operation_rows = None

        try:
            if self._should_use_runtime_worker():
                result = self._run_on_runtime_worker(open_runtime)
            else:
                result = open_runtime()
        except BaseException:
            # Submission itself can fail during worker shutdown, before the
            # operation's runtime cleanup has had a chance to run.
            self._clear_scope_reservation(token)
            raise
        return ScopedRuntimeLease(self, token), result

    def reserve_runtime_slot(self, progress: ProgressReporter) -> RuntimeSlotLease:
        """Reserve exclusive capacity for independently constructed transient states.

        Settings comparisons rebuild several states without reading or writing
        the user's checkpoint. Reserving the manager slot prevents a Browser
        load, Live Session, or checkpoint rebuild from overlapping those large
        caller-owned runtimes.
        """

        token = object()
        self._reserve_scoped_runtime(token, progress)
        try:
            while self.save_in_progress:
                progress.update(0, 1, "Waiting for the current RWKV checkpoint write")
                progress.check_cancelled()
                if self.wait_for_pending_save(timeout=0.1):
                    break
            if self._runtime is not None:
                raise CheckpointBusyError(
                    "Stop the active RWKV operation or Live Session before starting "
                    "a state comparison."
                )
            return RuntimeSlotLease(self, token)
        except BaseException:
            self._clear_scope_reservation(token)
            raise

    def _reserve_scoped_runtime(
        self,
        token: object,
        progress: ProgressReporter,
    ) -> None:
        requester_thread_id = threading.get_ident()
        with self._scope_condition:
            while self._active_scope_token is not None:
                if self._active_scope_requester_thread_id == requester_thread_id:
                    raise CheckpointBusyError(
                        "This task already owns an RWKV runtime scope. Close it "
                        "before opening another one."
                    )
                check_cancelled = getattr(progress, "check_cancelled", None)
                if check_cancelled is not None:
                    check_cancelled()
                self._scope_condition.wait(timeout=0.1)
            self._active_scope_token = token
            self._active_scope_requester_thread_id = requester_thread_id

    def _reserve_maintenance_slot(
        self,
        action: str,
        *,
        require_unloaded: bool = False,
    ) -> RuntimeSlotLease:
        """Claim native operation capacity before submitting any background work.

        Rebuilds and size probes initially own only a local runtime, so neither
        ``runtime_loaded`` nor ``save_in_progress`` identifies their lifetime.
        Use the same atomic reservation as scoped loads and comparisons to keep
        settings changes fail-fast and prevent overlapping large states.
        """
        token = object()
        with self._scope_condition:
            if self._active_scope_token is not None or (
                require_unloaded and self._runtime is not None
            ):
                raise CheckpointBusyError(
                    f"Stop the active RWKV operation or Live Session before {action}."
                )
            self._active_scope_token = token
            self._active_scope_requester_thread_id = threading.get_ident()
        return RuntimeSlotLease(self, token)

    def _clear_scope_reservation(self, token: object | None = None) -> bool:
        with self._scope_condition:
            if token is not None and token is not self._active_scope_token:
                return False
            self._active_scope_token = None
            self._active_scope_requester_thread_id = None
            self._scope_condition.notify_all()
            return True

    def _initialize_or_rebuild_impl(
        self,
        rows: list[dict[str, Any]],
        progress: ProgressReporter,
        *,
        partial_on_cancel: bool = True,
    ) -> CheckpointResult:
        self.wait_for_pending_save()
        if model_generation_uses_v5o(self.model_generation) and self.store.active_checkpoint_path():
            # Retained-catalog replacement must leave the old durable state
            # authoritative until the complete replacement can be published.
            partial_on_cancel = False
        self._release_loaded_runtime()
        previous_input_context = self._input_context
        if model_generation_uses_exact_timestamps(self.model_generation):
            # A deliberate rebuild is the explicit boundary at which a changed
            # timezone/rollover setting may establish a new canonical phase.
            if self._review_data_cache is None:
                raise CheckpointError("This generation requires review context before rebuilding.")
            self._input_context = input_context_for_generation(
                self._review_data_cache, self.model_generation
            )
        runtime = None
        try:
            runtime = new_rwkvp_runtime(
                model_id=self.model_id,
                model_generation=self.model_generation,
                process_many_mode=self.process_many_mode,
                state_precision=self.state_precision,
                gpu_prediction_math=self.gpu_prediction_math,
            )
            if isinstance(self._input_context, V5oInputContext):
                # Persist the complete inventory, including unreviewed cards,
                # without allocating their causal identities before an answer.
                runtime.rebuild_retained_history(
                    [], cards=self._input_context.catalog_rows(),
                    rollover_phase_ms=self._input_context.rollover_phase_ms,
                    mode="fast",
                )
            self._remember_runtime_metadata(runtime)
        except BaseException:
            if runtime is not None:
                _release_runtime_resources(runtime)
            self._input_context = previous_input_context
            raise
        prediction_records = PredictionRecordSet.empty()
        latest_curves_by_card: dict[int, Any] | None = {} if self.calculate_curves else None
        try:
            prediction_records = process_review_rows_with_predictions(
                runtime,
                rows,
                progress,
                label="Processing review history...",
                record_set=prediction_records,
                latest_curves_by_card=latest_curves_by_card,
                process_many_mode=self.process_many_mode,
                calculate_curves=self.calculate_curves,
                input_context=self._input_context,
            )
            check_cancelled = getattr(progress, "check_cancelled", None)
            if check_cancelled is not None:
                check_cancelled()
        except CancelledError:
            partial_saved = False
            try:
                if partial_on_cancel:
                    self._latest_curves_by_card = latest_curves_by_card
                    self._latest_curves_complete = True
                    processed_count = len(prediction_records.immediate_predictions)
                    self._save_checkpoint_and_cache(
                        runtime,
                        rows[:processed_count],
                        prediction_records.slice(processed_count),
                        status="partial",
                        checkpoint_path=self.store.partial_checkpoint_path,
                        progress=ProgressReporter(),
                    )
                    partial_saved = True
            finally:
                self._release_rebuild_runtime(runtime)
                if not partial_saved:
                    self._input_context = previous_input_context
            raise
        except BaseException:
            self._release_rebuild_runtime(runtime)
            self._input_context = previous_input_context
            raise

        self._latest_curves_by_card = latest_curves_by_card
        self._latest_curves_complete = True
        try:
            return self._save_checkpoint_and_cache(
                runtime,
                rows,
                prediction_records,
                status="valid",
                checkpoint_path=self.store.latest_checkpoint_path,
                progress=progress,
            )
        except BaseException:
            self._release_rebuild_runtime(runtime)
            self._input_context = previous_input_context
            raise

    def _release_rebuild_runtime(
        self,
        runtime: Any,
    ) -> None:
        """Release a newly built runtime on cancellation or failed handoff."""

        if self._runtime is runtime:
            # A non-Rust background save may have accepted ownership before the
            # caller was cancelled. Let it stop touching the runtime first.
            with suppress(CheckpointError):
                self.wait_for_pending_save()
            self._release_loaded_runtime()
            self._clear_runtime_state(preserve_review_data=True)
            return
        _release_runtime_resources(runtime)
        self._clear_runtime_state(preserve_review_data=True)

    def ensure_ready_for_prediction(
        self,
        rows: list[dict[str, Any]],
        progress: ProgressReporter,
        *,
        allow_inconsistent: bool = False,
        force_save: bool = False,
        check_consistency: bool = True,
    ) -> CheckpointResult:
        with self._reserve_maintenance_slot("replacing its checkpoint state"):
            if self._should_use_runtime_worker():
                return self._run_on_runtime_worker(
                    lambda: self._ensure_ready_for_prediction_impl(
                        rows,
                        progress,
                        allow_inconsistent=allow_inconsistent,
                        force_save=force_save,
                        check_consistency=check_consistency,
                    )
                )
            return self._ensure_ready_for_prediction_impl(
                rows,
                progress,
                allow_inconsistent=allow_inconsistent,
                force_save=force_save,
                check_consistency=check_consistency,
            )

    def check_integrity(
        self,
        rows: list[dict[str, Any]],
        progress: ProgressReporter,
    ) -> CheckpointResult:
        with self._reserve_maintenance_slot("checking checkpoint history"):
            if self._should_use_runtime_worker():
                return self._run_on_runtime_worker(
                    lambda: self._check_integrity_impl(rows, progress)
                )
            return self._check_integrity_impl(rows, progress)

    def _check_integrity_impl(
        self,
        rows: list[dict[str, Any]],
        progress: ProgressReporter,
    ) -> CheckpointResult:
        self._raise_background_save_error()
        if configured_rwkv_backend() == "rust":
            checkpoint_path = self.store.active_checkpoint_path()
            if checkpoint_path is not None:
                try:
                    metadata = read_rust_checkpoint_metadata(checkpoint_path)
                except CheckpointMetadataError:
                    # Test doubles and checkpoints from unsupported future
                    # backends still get the legacy runtime validation path.
                    pass
                else:
                    self._raise_if_legacy_rust_checkpoint(metadata)
                    return self._check_rust_history_integrity_without_runtime(
                        rows,
                        progress,
                    )
        runtime = self._load_runtime(progress, cards=[])
        try:
            durable_checkpoint_path = self.store.active_checkpoint_path()
            update_checkpoint_consistency(progress, "Checking checkpoint integrity")
            consistent = self._is_consistent(
                runtime,
                rows,
                input_context=self._input_context,
                model_generation=self.model_generation,
            )
            if not consistent:
                reloaded_runtime = self._reload_runtime_from_durable_checkpoint(
                    progress,
                )
                if reloaded_runtime is not runtime:
                    runtime = reloaded_runtime
                    update_checkpoint_consistency(
                        progress,
                        "Rechecking durable checkpoint integrity",
                    )
                    consistent = self._is_consistent(
                        runtime,
                        rows,
                        input_context=self._input_context,
                        model_generation=self.model_generation,
                    )
            if not consistent:
                self._invalidate_history_verification()
                self._durable_writer.write_manifest(
                    runtime,
                    "invalid",
                    durable_checkpoint_path,
                )
                raise InconsistentCheckpointError(
                    "The loaded RWKV checkpoint is inconsistent with the current review "
                    "history. Rebuild it before relying on predictions."
                )
            self._mark_history_verified(rows)
            return CheckpointResult(
                status=self._current_durable_status(),
                processed_review_count=int(getattr(runtime, "processed_review_count", 0)),
                last_review_id=getattr(runtime, "last_review_id", None),
                checkpoint_path=str(durable_checkpoint_path or ""),
            )
        finally:
            self._release_loaded_runtime()
            self._clear_runtime_state(preserve_review_data=True)

    def _check_rust_history_integrity_without_runtime(
        self,
        rows: list[dict[str, Any]],
        progress: ProgressReporter,
    ) -> CheckpointResult:
        self._require_v5o_catalog_matches()
        checkpoint_path = self.store.active_checkpoint_path()
        if checkpoint_path is None:
            raise MissingCheckpointError("Initialize an RWKV checkpoint first.")
        update_checkpoint_consistency(progress, "Reading checkpoint history metadata")
        try:
            metadata = read_rust_checkpoint_metadata(checkpoint_path)
            try:
                check_checkpoint_history = require_rwkv_checkpoint_history_consistency(
                    backend="rust"
                )
            except DependencyError:
                # Preserve compatibility with test doubles and an older staged
                # dependency. Current release bundles always take the native path.
                consistency_rows = self._checkpoint_consistency_rows(metadata, rows)
                consistent = checkpoint_history_is_consistent(metadata, consistency_rows)
            else:
                if metadata.model_version in {"v5o"}:
                    context = self._require_input_context()
                    batch_type = require_rwkv_v5o_review_batch(backend="rust")
                    consistency_input = batch_type.from_columns(
                        kind="answer",
                        columns=context.process_columns(rows),
                    )
                else:
                    ReviewBatch = require_rwkv_review_batch(backend="rust")
                    consistency_input = native_review_batch_for_rows(ReviewBatch, rows)
                consistent = bool(
                    check_checkpoint_history(
                        checkpoint_path,
                        consistency_input,
                    )
                )
        except CheckpointMetadataError as exc:
            raise CheckpointError(str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise CheckpointError(str(exc)) from exc
        if not consistent:
            self._invalidate_history_verification()
            manifest = self.store.manifest()
            manifest["status"] = "invalid"
            self.store.write_manifest(manifest)
            raise InconsistentCheckpointError(
                "The durable RWKV checkpoint history fingerprint is inconsistent "
                "with the current review history. Rebuild it before relying on "
                "predictions."
            )
        self._mark_history_verified(rows)
        update_checkpoint_consistency(progress, "Checkpoint review history verified")
        return CheckpointResult(
            status=self._current_durable_status(),
            processed_review_count=metadata.processed_review_count,
            last_review_id=metadata.last_review_id,
            checkpoint_path=str(checkpoint_path),
        )

    def _ensure_ready_for_prediction_impl(
        self,
        rows: list[dict[str, Any]],
        progress: ProgressReporter,
        *,
        allow_inconsistent: bool = False,
        force_save: bool = False,
        check_consistency: bool = True,
        scope_cards: list[dict[str, Any]] | None = None,
    ) -> CheckpointResult:
        if isinstance(rows, CheckpointTailRows) and force_save:
            raise CheckpointError(
                "A virtual checkpoint history cannot be used for a durable save; "
                "reload complete review history first."
            )
        self._raise_background_save_error()
        runtime_cards = _runtime_scope_cards(
            rows,
            scope_cards,
            durable_processed_count=self._manifest_processed_review_count(),
        )
        runtime = self._load_runtime(
            progress,
            cards=runtime_cards,
        )
        if scope_cards is not None:
            actual_scope_cards = _runtime_scope_cards(
                rows,
                scope_cards,
                durable_processed_count=int(getattr(runtime, "processed_review_count", 0)),
            )
            if _scope_key(actual_scope_cards) != _scope_key(runtime_cards):
                runtime_cards = actual_scope_cards
                runtime = self._load_runtime(
                    progress,
                    cards=runtime_cards,
                )
        durable_checkpoint_path = self.store.active_checkpoint_path()
        if check_consistency:
            if self._history_verification_is_current(rows):
                update_checkpoint_consistency(
                    progress,
                    "Checkpoint consistency is already verified",
                )
                consistent = True
            else:
                update_checkpoint_consistency(progress, "Checking checkpoint consistency")
                consistent = self._is_consistent(
                    runtime,
                    rows,
                    input_context=self._input_context,
                    model_generation=self.model_generation,
                )
                if not consistent:
                    reloaded_runtime = self._reload_runtime_from_durable_checkpoint(
                        progress,
                    )
                    if reloaded_runtime is not runtime:
                        runtime = reloaded_runtime
                        update_checkpoint_consistency(
                            progress,
                            "Rechecking durable checkpoint consistency",
                        )
                        consistent = self._is_consistent(
                            runtime,
                            rows,
                            input_context=self._input_context,
                            model_generation=self.model_generation,
                        )
        else:
            if not self._checkpoint_lineage_is_current():
                raise CheckpointError(
                    "RWKV checkpoint must be verified before incremental updates."
                )
            consistent = True

        if not consistent:
            self._invalidate_history_verification()
            manifest = self.store.manifest()
            acknowledged = bool(manifest.get("inconsistent_acknowledged"))
            self._durable_writer.write_manifest(runtime, "invalid", durable_checkpoint_path)
            if not (allow_inconsistent or acknowledged):
                raise InconsistentCheckpointError(
                    "The loaded RWKV checkpoint is inconsistent with the current "
                    "review history. Rebuild it, or acknowledge the inconsistency "
                    "before using predictions."
                )
        self._mark_history_verified(rows, incremental=not check_consistency)
        cache_binding_verified = self._restore_evaluation_cache_validation_from_binding()
        if cache_binding_verified:
            update_checkpoint_curve_data(
                progress,
                (
                    "Verified checkpoint-bound RWKV curve data"
                    if self.calculate_curves
                    else "Verified checkpoint-bound RWKV evaluation data"
                ),
            )
        pending = rows[int(getattr(runtime, "processed_review_count", 0)) :]
        pending_card_ids = {int(row["card_id"]) for row in pending}
        if self.calculate_curves and pending_card_ids:
            update_checkpoint_curve_data(progress, "Loading cached RWKV curve data")
            self._ensure_latest_curves_loaded(rows, card_ids=pending_card_ids)
        elif self.status() == "stale_cache":
            if self.calculate_curves:
                update_checkpoint_curve_data(progress, "Checking cached RWKV curve data")
                self._ensure_latest_curves_loaded(rows)
            else:
                raise StaleCheckpointDataError(
                    "RWKV evaluation cache is missing or stale. Rebuild the checkpoint "
                    "to recreate its immediate-prediction data."
                )
        else:
            if not cache_binding_verified:
                self._validate_unbound_evaluation_cache(rows)
            update_checkpoint_curve_data(progress, "Scoped checkpoint state ready")

        was_partial = self.store.checkpoint_status() == "partial"
        if pending or (force_save and self._runtime_dirty) or was_partial:
            progress.update(0, 1, "Preparing checkpoint update")
            self.wait_for_pending_save()
            runtime = self._load_runtime(
                progress,
                cards=runtime_cards,
            )
            durable_checkpoint_path = self.store.active_checkpoint_path()
            was_partial = self.store.checkpoint_status() == "partial"
            pending = rows[int(getattr(runtime, "processed_review_count", 0)) :]
            pending_card_ids = {int(row["card_id"]) for row in pending}
            if self.calculate_curves and pending_card_ids:
                update_checkpoint_curve_data(progress, "Loading cached RWKV curve data")
                self._ensure_latest_curves_loaded(rows, card_ids=pending_card_ids)

        if pending:
            start_count = int(getattr(runtime, "processed_review_count", 0))
            unsaved_count = len(self._unsaved_prediction_records.immediate_predictions)
            durable_prefix_count = start_count - unsaved_count
            processed_since_save = unsaved_count + len(pending)
            should_save = (
                force_save or processed_since_save >= self.checkpoint_save_interval or was_partial
            )
            if should_save and self.calculate_curves:
                progress.update(0, 1, "Preparing cached evaluation data for save")
                self._ensure_latest_curves_loaded(rows)
            prefix_records = (
                self._prediction_record_prefixes_for_durable_update(rows, durable_prefix_count)
                if should_save
                else PredictionRecordSet.empty()
            )

            prediction_records = self._process_pending(runtime, pending, progress)
            self._remember_runtime_metadata(runtime)
            if should_save:
                full_prediction_records = PredictionRecordSet.combine(
                    prefix_records,
                    self._unsaved_prediction_records,
                    prediction_records,
                )
                return self._save_checkpoint_and_cache(
                    runtime,
                    rows[: len(full_prediction_records.immediate_predictions)],
                    full_prediction_records,
                    status=self._current_durable_status(),
                    checkpoint_path=self.store.latest_checkpoint_path,
                    progress=progress,
                    durable_prefix_count=durable_prefix_count,
                )
            else:
                self._runtime_dirty = True
                self._unsaved_prediction_records.extend(prediction_records)

        if force_save and self._runtime_dirty:
            start_count = int(getattr(runtime, "processed_review_count", 0))
            unsaved_count = len(self._unsaved_prediction_records.immediate_predictions)
            durable_prefix_count = start_count - unsaved_count
            progress.update(0, 1, "Preparing cached evaluation data for save")
            prefix_records = self._prediction_record_prefixes_for_durable_update(
                rows,
                durable_prefix_count,
            )
            full_prediction_records = PredictionRecordSet.combine(
                prefix_records,
                self._unsaved_prediction_records,
            )
            return self._save_checkpoint_and_cache(
                runtime,
                rows[: len(full_prediction_records.immediate_predictions)],
                full_prediction_records,
                status=self._current_durable_status(),
                checkpoint_path=self.store.latest_checkpoint_path,
                progress=progress,
                durable_prefix_count=durable_prefix_count,
            )

        if was_partial and int(getattr(runtime, "processed_review_count", 0)) == len(rows):
            progress.update(0, 1, "Preparing checkpoint save")
            cache_records = self._load_available_prediction_records(rows)
            if cache_records is not None and _record_set_covers(
                cache_records,
                self._prediction_cache_specs,
                len(rows),
            ):
                return self._save_checkpoint_and_cache(
                    runtime,
                    rows,
                    cache_records,
                    status=self._current_durable_status(),
                    checkpoint_path=self.store.latest_checkpoint_path,
                    progress=progress,
                )

        status = self._current_durable_status()
        if not pending and not self._runtime_dirty:
            progress.update(1, 1, "Checkpoint is already saved")
            return self._durable_writer.write_manifest(
                runtime,
                status,
                self.store.active_checkpoint_path(),
            )

        return CheckpointResult(
            status=status,
            processed_review_count=self._runtime_processed_count(),
            last_review_id=self._runtime_last_review_id_value(),
            checkpoint_path=str(durable_checkpoint_path or ""),
        )

    def acknowledge_inconsistent_checkpoint(self) -> None:
        manifest = self.store.manifest()
        manifest["inconsistent_acknowledged"] = True
        manifest["status"] = "invalid"
        self.store.write_manifest(manifest)

    def predict_many(
        self,
        rows: list[dict[str, Any]],
        *,
        batch_size: int | None = None,
        allow_gpu: bool = True,
    ) -> list[float]:
        effective_batch_size = self._effective_predict_many_batch_size(batch_size)
        if self._should_use_runtime_worker():
            return self._run_on_runtime_worker(
                lambda: self._predict_many_impl(
                    rows,
                    batch_size=effective_batch_size,
                    allow_gpu=allow_gpu,
                )
            )
        return self._predict_many_impl(
            rows,
            batch_size=effective_batch_size,
            allow_gpu=allow_gpu,
        )

    def process_one(self, row: dict[str, Any]) -> tuple[float | None, Any]:
        """Process one review on the loaded runtime without durable cache bookkeeping.

        This is intended for live-review transient state. Durable checkpoint and
        evaluation-cache updates should continue to use the normal checkpoint
        update path, which can rebuild all required prediction cache columns.
        """
        if self._should_use_runtime_worker():
            return self._run_on_runtime_worker(lambda: self._process_one_impl(row))
        return self._process_one_impl(row)

    def _process_one_impl(self, row: dict[str, Any]) -> tuple[float | None, Any]:
        if self._runtime is None:
            raise MissingCheckpointError("No RWKV runtime is loaded.")
        process_row = runtime_process_row_input(
            self._runtime,
            row,
            input_context=self._input_context,
            model_generation=self.model_generation,
        )

        result = _normalized_scalar_process_result(
            self._runtime.process(
                process_row,
                return_curves=self.calculate_curves,
            ),
            return_curves=self.calculate_curves,
        )
        return result

    def _process_simulation_one_impl(
        self,
        row: dict[str, Any],
        *,
        return_curves: bool,
    ) -> tuple[float | None, Any | None]:
        if model_generation_uses_v5o(self.model_generation):
            raise ValueError("Behavior Lab is unavailable for V5.")
        if self._runtime is None:
            raise MissingCheckpointError("No RWKV runtime is loaded.")
        process_row = runtime_process_row_input(
            self._runtime,
            row,
            input_context=self._input_context,
            model_generation=self.model_generation,
        )
        effective_return_curves = bool(return_curves) and not runtime_uses_exact_timestamps(
            self._runtime,
            model_generation=self.model_generation,
        )
        result = self._runtime.process(
            process_row,
            return_curves=effective_return_curves,
        )
        return _normalized_scalar_process_result(
            result,
            return_curves=effective_return_curves,
        )

    def _process_simulation_many_impl(
        self,
        rows: list[dict[str, Any]],
    ) -> list[float]:
        if model_generation_uses_v5o(self.model_generation):
            raise ValueError("Behavior Lab is unavailable for V5.")
        if self._runtime is None:
            raise MissingCheckpointError("No RWKV runtime is loaded.")
        if not rows:
            return []
        process_many = getattr(self._runtime, "process_many", None)
        if callable(process_many):
            process_input = runtime_process_many_input(
                self._runtime,
                rows,
                input_context=self._input_context,
                model_generation=self.model_generation,
            )
            return [
                float(value)
                for value in process_many(
                    process_input,
                    return_curves=False,
                )
            ]
        return [
            float(
                self._runtime.process(
                    runtime_process_row_input(
                        self._runtime,
                        row,
                        input_context=self._input_context,
                        model_generation=self.model_generation,
                    ),
                    return_curves=False,
                )
            )
            for row in rows
        ]

    def undoable_process_one(self, row: dict[str, Any]) -> tuple[float | None, Any]:
        """Process one live-review row and keep Rust-side state undoable."""
        if self._should_use_runtime_worker():
            return self._run_on_runtime_worker(lambda: self._undoable_process_one_impl(row))
        return self._undoable_process_one_impl(row)

    def _undoable_process_one_impl(self, row: dict[str, Any]) -> tuple[float | None, Any]:
        if self._runtime is None:
            raise MissingCheckpointError("No RWKV runtime is loaded.")
        process_row = runtime_process_row_input(
            self._runtime,
            row,
            input_context=self._input_context,
            model_generation=self.model_generation,
        )

        result = _normalized_scalar_process_result(
            self._runtime.undoable_process(
                process_row,
                return_curves=self.calculate_curves,
            ),
            return_curves=self.calculate_curves,
        )

        return result

    def undo_last_process(self) -> int:
        """Undo the latest live-review runtime process and return remaining depth."""
        if self._should_use_runtime_worker():
            return self._run_on_runtime_worker(self._undo_last_process_impl)
        return self._undo_last_process_impl()

    def _undo_last_process_impl(self) -> int:
        if self._runtime is None:
            raise MissingCheckpointError("No RWKV runtime is loaded.")
        depth = int(self._runtime.undo_last_process())
        return depth

    def current_undo_depth(self) -> int:
        if self._should_use_runtime_worker():
            return self._run_on_runtime_worker(self._current_undo_depth_impl)
        return self._current_undo_depth_impl()

    def _current_undo_depth_impl(self) -> int:
        if self._runtime is None:
            raise MissingCheckpointError("No RWKV runtime is loaded.")
        return int(self._runtime.current_undo_depth)

    def _predict_many_impl(
        self,
        rows: list[dict[str, Any]],
        *,
        batch_size: int | None,
        allow_gpu: bool = True,
    ) -> list[float]:
        if self._runtime is None:
            raise MissingCheckpointError("No RWKV runtime is loaded.")
        if not rows:
            return []
        mode = self._effective_predict_many_mode
        if predict_many_uses_gpu(mode):
            if allow_gpu and self._loaded_runtime_gpu_is_ready():
                # RWKV-SRS owns the recoverability decision. In particular, it
                # retries only GpuError instances explicitly marked safe for a
                # CPU retry; caller errors and non-recoverable device failures
                # must continue to surface instead of being hidden here.
                return _call_runtime_predict_many(
                    self._runtime,
                    rows,
                    batch_size=batch_size,
                    mode="gpu",
                    fallback_mode=PREDICT_MANY_FAST_MODE,
                    input_context=self._input_context,
                    model_generation=self.model_generation,
                )
            # Small interactive callers may deliberately skip GPU dispatch,
            # and a failed availability probe has not mutated prediction state.
            mode = PREDICT_MANY_FAST_MODE
        return _call_runtime_predict_many(
            self._runtime,
            rows,
            batch_size=batch_size,
            mode="fast" if predict_many_uses_fast(mode) else "oracle",
            input_context=self._input_context,
            model_generation=self.model_generation,
        )

    def _runtime_live_process_row(
        self,
        row: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self._runtime is None:
            raise MissingCheckpointError("No RWKV runtime is loaded.")
        return runtime_process_row_input(
            self._runtime,
            row,
            input_context=self._input_context,
            model_generation=self.model_generation,
        )

    def _runtime_live_candidates(self, candidates: Iterable[Any]) -> tuple[Any, ...]:
        if self._runtime is None:
            raise MissingCheckpointError("No RWKV runtime is loaded.")
        materialized = tuple(candidates)
        if not runtime_uses_exact_timestamps(
            self._runtime,
            model_generation=self.model_generation,
        ):
            return materialized
        converted: list[Any] = []
        for seed in materialized:
            row = getattr(seed, "row", None)
            if row is None:
                raise TypeError("RWKV Live Session candidates must expose a row.")
            converted.append(
                replace(
                    seed,
                    row=runtime_live_candidate_row_input(
                        self._runtime,
                        row,
                        input_context=self._input_context,
                        model_generation=self.model_generation,
                    ),
                )
            )
        return tuple(converted)

    def _runtime_live_kwargs(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        values = dict(kwargs)
        runtime = self._runtime
        if runtime is None:
            raise MissingCheckpointError("No RWKV runtime is loaded.")
        if runtime_uses_exact_timestamps(runtime, model_generation=self.model_generation):
            if values.get("target_timestamp_seconds") is not None:
                raise ValueError(
                    "V5o Live prediction accepts target_timestamp_ms, not target_timestamp_seconds."
                )
            values.pop("target_timestamp_seconds", None)
            if values.get("target_timestamp_ms") is None:
                raise ValueError("V5o Live prediction requires target_timestamp_ms.")
            values["rollover_phase_ms"] = self._require_input_context().rollover_phase_ms
        else:
            if values.get("target_timestamp_ms") is not None:
                raise ValueError(
                    "V1 Live prediction accepts target_timestamp_seconds, not target_timestamp_ms."
                )
            values.pop("target_timestamp_ms", None)
            if values.get("target_timestamp_seconds") is None:
                raise ValueError("V1 Live prediction requires target_timestamp_seconds.")
        return values

    def _effective_predict_many_batch_size(self, batch_size: int | None) -> int | None:
        if batch_size is not None:
            return _positive_optional_int(batch_size) or self.predict_many_batch_size
        return self.predict_many_batch_size

    def _loaded_runtime_gpu_is_ready(self) -> bool:
        if self._runtime_gpu_ready is not None:
            return self._runtime_gpu_ready
        runtime = self._runtime
        check = getattr(runtime, "gpu_available", None)
        if not callable(check):
            self._disable_gpu_for_loaded_runtime(
                RuntimeError("The loaded rwkv-srs runtime has no GPU availability API.")
            )
            return False
        try:
            ready = bool(check("predict"))
        except (RuntimeError, ValueError) as exc:
            self._disable_gpu_for_loaded_runtime(exc)
            return False
        if not ready:
            self._disable_gpu_for_loaded_runtime(
                RuntimeError("The selected checkpoint scope could not initialize on the GPU.")
            )
            return False
        self._runtime_gpu_ready = True
        self._runtime_gpu_failure = None
        return True

    def _disable_gpu_for_loaded_runtime(self, error: BaseException) -> None:
        self._runtime_gpu_ready = False
        self._runtime_gpu_failure = str(error)
        release = getattr(self._runtime, "release_gpu", None)
        if callable(release):
            with suppress(Exception):
                release()

    def _scoped_predict_many(
        self,
        token: object,
        rows: list[dict[str, Any]],
        *,
        batch_size: int | None,
        allow_gpu: bool,
    ) -> list[float]:
        effective_batch_size = self._effective_predict_many_batch_size(batch_size)
        return self._run_scoped_call(
            token,
            lambda: self._predict_many_impl(
                rows,
                batch_size=effective_batch_size,
                allow_gpu=allow_gpu,
            ),
        )

    def _scoped_benchmark_predict_many(
        self,
        token: object,
        rows: list[dict[str, Any]],
        *,
        mode: str,
        batch_size: int | None,
    ) -> list[float]:
        return self._run_scoped_call(
            token,
            lambda: self._benchmark_predict_many_impl(
                rows,
                mode=mode,
                batch_size=batch_size,
            ),
        )

    def _scoped_predict_many_live_session(
        self,
        token: object,
        candidates: tuple[Any, ...],
        *,
        initial_target_timestamp_seconds: float | None,
        initial_target_timestamp_ms: int | None,
        initial_rollover_phase_ms: int | None,
        initial_target_day_offset: float | None = None,
        order: str,
        mode: str | None,
        batch_size: int | None,
        refresh_limit: int,
        profiling: bool,
        initial_select_limit: int,
    ):
        return self._run_scoped_call(
            token,
            lambda: self._predict_many_live_session_impl(
                candidates,
                initial_target_timestamp_seconds=initial_target_timestamp_seconds,
                initial_target_timestamp_ms=initial_target_timestamp_ms,
                initial_rollover_phase_ms=initial_rollover_phase_ms,
                initial_target_day_offset=initial_target_day_offset,
                order=order,
                mode=mode,
                batch_size=batch_size,
                refresh_limit=refresh_limit,
                profiling=profiling,
                initial_select_limit=initial_select_limit,
                strict_gpu=False,
            ),
        )

    def _scoped_benchmark_predict_many_live_session(
        self,
        token: object,
        candidates: tuple[Any, ...],
        *,
        initial_target_timestamp_seconds: float | None,
        initial_target_timestamp_ms: int | None,
        initial_rollover_phase_ms: int | None,
        initial_target_day_offset: float | None = None,
        order: str,
        mode: str,
        batch_size: int | None,
        refresh_limit: int,
        profiling: bool,
        initial_select_limit: int,
    ):
        return self._run_scoped_call(
            token,
            lambda: self._predict_many_live_session_impl(
                candidates,
                initial_target_timestamp_seconds=initial_target_timestamp_seconds,
                initial_target_timestamp_ms=initial_target_timestamp_ms,
                initial_rollover_phase_ms=initial_rollover_phase_ms,
                initial_target_day_offset=initial_target_day_offset,
                order=order,
                mode=mode,
                batch_size=batch_size,
                refresh_limit=refresh_limit,
                profiling=profiling,
                initial_select_limit=initial_select_limit,
                strict_gpu=True,
            ),
        )

    def _predict_many_live_session_impl(
        self,
        candidates: tuple[Any, ...],
        *,
        initial_target_timestamp_seconds: float | None,
        initial_target_timestamp_ms: int | None,
        initial_rollover_phase_ms: int | None,
        initial_target_day_offset: float | None = None,
        order: str,
        mode: str | None,
        batch_size: int | None,
        refresh_limit: int,
        profiling: bool,
        initial_select_limit: int,
        strict_gpu: bool,
    ):
        runtime = self._runtime
        if runtime is None:
            raise MissingCheckpointError("No RWKV runtime is loaded.")
        factory = getattr(runtime, "predict_many_live_session", None)
        if not callable(factory):
            raise RuntimeError(
                "The bundled RWKV-SRS runtime does not provide predict_many_live_session()."
            )
        if runtime_uses_exact_timestamps(runtime, model_generation=self.model_generation):
            if initial_target_timestamp_seconds is not None:
                raise ValueError(
                    "V5o Live prediction accepts initial_target_timestamp_ms, "
                    "not initial_target_timestamp_seconds."
                )
            if initial_target_timestamp_ms is None:
                raise ValueError("V5o Live prediction requires initial_target_timestamp_ms.")
            target_kwargs: dict[str, Any] = {
                "initial_target_timestamp_ms": initial_target_timestamp_ms,
                "initial_rollover_phase_ms": (self._require_input_context().rollover_phase_ms),
            }
        else:
            if initial_target_timestamp_ms is not None:
                raise ValueError(
                    "V1 Live prediction accepts initial_target_timestamp_seconds, "
                    "not initial_target_timestamp_ms."
                )
            if initial_target_timestamp_seconds is None:
                raise ValueError("V1 Live prediction requires initial_target_timestamp_seconds.")
            if initial_rollover_phase_ms is not None:
                raise ValueError("V1 Live prediction does not accept initial_rollover_phase_ms.")
            target_kwargs = {
                "initial_target_timestamp_seconds": float(initial_target_timestamp_seconds),
            }

        selected_mode = (
            normalize_predict_many_mode(mode)
            if mode is not None
            else self._effective_predict_many_mode
        )
        if predict_many_uses_gpu(selected_mode) and not self._loaded_runtime_gpu_is_ready():
            reason = self._runtime_gpu_failure or "GPU prediction is unavailable."
            if strict_gpu:
                raise RuntimeError(
                    f"GPU Live Session prediction speed test is unavailable: {reason}"
                )
            # A failed production preflight should not prevent review startup.
            selected_mode = PREDICT_MANY_FAST_MODE
        if strict_gpu and predict_many_uses_gpu(selected_mode):
            self._require_benchmark_gpu_math()
        resolved_batch_size = self._effective_predict_many_batch_size(batch_size)
        kwargs: dict[str, Any] = {
            "order": str(order),
            "mode": str(selected_mode),
            "batch_size": resolved_batch_size,
            "refresh_limit": int(refresh_limit),
            "profiling": bool(profiling),
            "initial_select_limit": int(initial_select_limit),
            **target_kwargs,
        }
        if not model_generation_uses_v5o(self.model_generation):
            kwargs["initial_target_day_offset"] = float(initial_target_day_offset)
        if predict_many_uses_gpu(selected_mode) and not strict_gpu:
            # The native session applies this to construction and every later
            # refresh/reconciliation. Its switch to Fast is sticky, preserving
            # candidate rank and paired undo state.
            kwargs["fallback_mode"] = PREDICT_MANY_FAST_MODE
        seeds = self._runtime_live_candidates(candidates)
        return factory(seeds, **kwargs)

    def _benchmark_predict_many_impl(
        self,
        rows: list[dict[str, Any]],
        *,
        mode: str,
        batch_size: int | None,
    ) -> list[float]:
        if self._runtime is None:
            raise MissingCheckpointError("No RWKV runtime is loaded.")
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in PREDICT_MANY_MODES:
            raise ValueError(f"Unsupported predict_many speed-test mode: {mode!r}.")
        if normalized_mode == "gpu" and not self._loaded_runtime_gpu_is_ready():
            reason = self._runtime_gpu_failure or "GPU prediction is unavailable."
            raise RuntimeError(f"GPU predict_many speed test is unavailable: {reason}")
        if normalized_mode == "gpu":
            self._require_benchmark_gpu_math()
        return _call_runtime_predict_many(
            self._runtime,
            rows,
            batch_size=_positive_optional_int(batch_size),
            mode=normalized_mode,
            input_context=self._input_context,
            model_generation=self.model_generation,
        )

    def _require_benchmark_gpu_math(self) -> None:
        requested = self._scope_gpu_prediction_math or self.gpu_prediction_math
        if getattr(self._runtime, "gpu_prediction_math", "fp32") != requested:
            raise RuntimeError("The benchmark runtime did not honor its GPU arithmetic choice.")
        initialize = getattr(self._runtime, "initialize_gpu", None)
        if not callable(initialize):
            raise RuntimeError("GPU benchmarks require native GPU profile validation.")
        profile = initialize("predict")
        if profile.get("prediction_math") != requested:
            label = "GPU (Full FP16)" if requested == "fp16" else "GPU"
            raise RuntimeError(
                f"{label} is unavailable; refusing to benchmark "
                "a different kernel under the requested label. "
                + str(profile.get("prediction_math_fallback_reason") or "Kernel unavailable.")
            )

    def _scoped_process_one(
        self,
        token: object,
        row: dict[str, Any],
    ) -> tuple[float | None, Any]:
        return self._run_scoped_call(token, lambda: self._process_one_impl(row))

    def _scoped_process_simulation_one(
        self,
        token: object,
        row: dict[str, Any],
        *,
        return_curves: bool,
    ) -> tuple[float | None, Any | None]:
        return self._run_scoped_call(
            token,
            lambda: self._process_simulation_one_impl(
                row,
                return_curves=return_curves,
            ),
        )

    def _scoped_process_simulation_many(
        self,
        token: object,
        rows: list[dict[str, Any]],
    ) -> list[float]:
        return self._run_scoped_call(
            token,
            lambda: self._process_simulation_many_impl(rows),
        )

    def _scoped_undoable_process_one(
        self,
        token: object,
        row: dict[str, Any],
    ) -> tuple[float | None, Any]:
        return self._run_scoped_call(
            token,
            lambda: self._undoable_process_one_impl(row),
        )

    def _scoped_undo_last_process(self, token: object) -> int:
        return self._run_scoped_call(token, self._undo_last_process_impl)

    def _scoped_current_undo_depth(self, token: object) -> int:
        return self._run_scoped_call(token, self._current_undo_depth_impl)

    def _scoped_evaluation_prediction_tail(
        self,
        token: object,
    ) -> PredictionTailSnapshot:
        return self._run_scoped_call(token, self._evaluation_prediction_tail_impl)

    def _evaluation_prediction_tail_impl(self) -> PredictionTailSnapshot:
        immediate = tuple(self._unsaved_prediction_records.immediate_predictions)
        predict_ahead = tuple(self._unsaved_prediction_records.predict_ahead_predictions)
        if self.calculate_curves and len(predict_ahead) != len(immediate):
            raise PredictionCacheError(
                "Transient RWKV prediction columns are not aligned with each other."
            )
        if not self.calculate_curves and predict_ahead:
            raise PredictionCacheError(
                "Transient RWKV curve predictions exist while curve calculation is disabled."
            )
        processed_count = self._runtime_processed_count()
        start_index = processed_count - len(immediate)
        if start_index < 0:
            raise PredictionCacheError(
                "Transient RWKV predictions exceed the loaded runtime history."
            )
        return PredictionTailSnapshot(
            start_index=start_index,
            immediate_predictions=immediate,
            predict_ahead_predictions=predict_ahead,
        )

    def _run_scoped_call(self, token: object, op):
        def call():
            self._require_active_scope_token(token)
            return op()

        if self._should_use_runtime_worker():
            return self._run_on_runtime_worker(call)
        return call()

    def _require_active_scope_token(self, token: object) -> None:
        if token is not self._active_scope_token:
            raise MissingCheckpointError(
                "This RWKV runtime scope has already been closed or replaced."
            )

    def _scoped_runtime_contains_card(self, token: object, card_id: int) -> bool:
        with self._scope_condition:
            if token is not self._active_scope_token or self._runtime is None:
                return False
            scope_cards = self._loaded_scope_cards
            if scope_cards is None:
                return True
            return any(int(card.get("card_id", -1)) == int(card_id) for card in scope_cards)

    def _scoped_runtime_contained_card_ids(
        self,
        token: object,
        card_ids: set[int],
    ) -> set[int]:
        with self._scope_condition:
            if token is not self._active_scope_token or self._runtime is None:
                return set()
            scope_cards = self._loaded_scope_cards
            if scope_cards is None:
                return set(card_ids)
            loaded = {
                int(card["card_id"]) for card in scope_cards if card.get("card_id") is not None
            }
            return loaded.intersection(card_ids)

    def _close_scoped_runtime(self, token: object) -> None:
        def close() -> None:
            with self._scope_condition:
                if token is not self._active_scope_token:
                    return
            try:
                self._release_scoped_runtime_state()
            finally:
                # Keep the reservation in place until recurrent state is gone.
                # Otherwise a non-Rust requester could start loading the next
                # scope concurrently with this release.
                self._clear_scope_reservation(token)

        if self._should_use_runtime_worker():
            self._run_on_runtime_worker(close)
        else:
            close()

    def _release_scoped_runtime_state(self, *, preserve_curves: bool = True) -> None:
        with suppress(CheckpointError):
            self.wait_for_pending_save()
        preserve_selected_curves = preserve_curves and not self._latest_curves_complete
        try:
            self._release_loaded_runtime()
            self._clear_runtime_state(
                preserve_review_data=True,
                preserve_curves=preserve_selected_curves,
            )
        finally:
            self._scope_gpu_prediction_math = None

    def unload(self) -> None:
        with self._runtime_worker_condition:
            worker = self._runtime_worker
            on_worker = worker is not None and worker.is_worker_thread()
            if not on_worker:
                if self._runtime_worker_unloading:
                    # A concurrent caller owns teardown. Wait for that teardown
                    # without touching any replacement worker opened afterward.
                    while self._runtime_worker_unloading:
                        self._runtime_worker_condition.wait()
                    return
                self._runtime_worker_unloading = True
        if on_worker:
            self._unload_impl()
            return
        try:
            if worker is None:
                self._unload_impl()
            else:
                # Reject new work before queuing cleanup; previously work could
                # reopen native state between cleanup and the stop sentinel.
                worker.stop(finalize=self._unload_impl)
        finally:
            with self._runtime_worker_condition:
                if self._runtime_worker is worker:
                    self._runtime_worker = None
                self._runtime_worker_unloading = False
                self._runtime_worker_condition.notify_all()

    def _unload_impl(self) -> None:
        with suppress(CheckpointError):
            self.wait_for_pending_save()
        try:
            self._release_loaded_runtime()
            self._clear_runtime_state(preserve_review_data=False)
        finally:
            self._scope_gpu_prediction_math = None
            self._clear_scope_reservation()

    def release_runtime(self, *, preserve_review_data: bool = True) -> None:
        """Release any non-leased runtime after maintenance work."""

        if self._active_scope_token is not None:
            raise CheckpointBusyError("An RWKV runtime scope is still active.")

        def release() -> None:
            with suppress(CheckpointError):
                self.wait_for_pending_save()
            self._release_loaded_runtime()
            self._clear_runtime_state(preserve_review_data=preserve_review_data)

        if self._should_use_runtime_worker():
            self._run_on_runtime_worker(release)
        else:
            release()

    def _clear_runtime_state(
        self,
        *,
        preserve_review_data: bool,
        preserve_curves: bool = False,
    ) -> None:
        self._loaded_path = None
        self._loaded_scope_cards = None
        self._loaded_scope_key = None
        self._runtime_processed_review_count = None
        self._runtime_last_review_id = None
        self._runtime_dirty = False
        self._runtime_gpu_ready = None
        self._runtime_gpu_failure = None
        self._clear_unsaved_prediction_records()
        if not preserve_curves:
            self._latest_curves_by_card = None
            self._latest_curves_complete = False
        if not preserve_review_data:
            self._invalidate_history_verification()
            self._review_data_cache = None
            self._input_context = None
            self._review_data_cache_last_review_id = None
            self._review_data_cache_latest_collection_review_id = None

    def _load_runtime(
        self,
        progress: ProgressReporter,
        *,
        cards: list[dict[str, Any]] | None = None,
    ):
        checkpoint_path = self.store.active_checkpoint_path()
        self._require_checkpoint_selection_matches()
        self._require_v5o_catalog_matches()
        if cards is not None and isinstance(self._input_context, V5oInputContext):
            catalog = self._input_context.cards
            # Native selective scopes need the retained hierarchy as well as
            # card IDs, even though V5 query and Live seed rows are ID-only.
            cards = [
                {
                    "card_id": int(row["card_id"]),
                    **{
                        field: getattr(catalog[int(row["card_id"])], field)
                        for field in ("note_id", "deck_id", "preset_id", "card_creation_ms")
                    },
                }
                for row in cards
            ]
        scope_key = _scope_key(cards)
        if (
            self._runtime is not None
            and (checkpoint_path is None or self._loaded_path == str(checkpoint_path))
            and self._loaded_scope_key == scope_key
            and getattr(self._runtime, "gpu_prediction_math", "fp32")
            == (self._scope_gpu_prediction_math or self.gpu_prediction_math)
        ):
            return self._runtime
        if checkpoint_path is None:
            raise MissingCheckpointError("Initialize an RWKV checkpoint first.")
        self._raise_if_legacy_rust_checkpoint_path(checkpoint_path)
        update_checkpoint_load(progress, "Loading RWKV checkpoint")
        RWKV_SRS = require_rwkv_srs()
        checkpoint_kwargs: dict[str, Any] = {"checkpoint": checkpoint_path}
        if cards is not None:
            checkpoint_kwargs["cards"] = cards
        runtime = RWKV_SRS(
            **checkpoint_kwargs,
            **_runtime_constructor_kwargs(
                configured_rwkv_backend(),
                process_many_mode=self.process_many_mode,
                state_precision=self.state_precision,
                gpu_prediction_math=self._scope_gpu_prediction_math or self.gpu_prediction_math,
            ),
        )
        self._release_loaded_runtime()
        self._runtime = runtime
        self._runtime_gpu_ready = None
        self._runtime_gpu_failure = None
        self._loaded_path = str(checkpoint_path)
        self._loaded_scope_cards = None if cards is None else list(cards)
        self._loaded_scope_key = scope_key
        self._remember_runtime_metadata(runtime)
        self._runtime_dirty = False
        self._clear_unsaved_prediction_records()
        update_checkpoint_load(progress, "Loaded RWKV checkpoint")
        return runtime

    def _reload_runtime_from_durable_checkpoint(
        self,
        progress: ProgressReporter,
    ):
        checkpoint_path = self.store.active_checkpoint_path()
        self._require_checkpoint_selection_matches()
        if checkpoint_path is None:
            return self._runtime
        self._raise_if_legacy_rust_checkpoint_path(checkpoint_path)
        update_checkpoint_consistency(progress, "Reloading durable RWKV checkpoint")
        RWKV_SRS = require_rwkv_srs()
        checkpoint_kwargs: dict[str, Any] = {"checkpoint": checkpoint_path}
        if self._loaded_scope_cards is not None:
            checkpoint_kwargs["cards"] = list(self._loaded_scope_cards)
        runtime = RWKV_SRS(
            **checkpoint_kwargs,
            **_runtime_constructor_kwargs(
                configured_rwkv_backend(),
                process_many_mode=self.process_many_mode,
                state_precision=self.state_precision,
                gpu_prediction_math=self._scope_gpu_prediction_math or self.gpu_prediction_math,
            ),
        )
        self._release_loaded_runtime()
        self._runtime = runtime
        self._loaded_path = str(checkpoint_path)
        self._remember_runtime_metadata(runtime)
        self._runtime_dirty = False
        self._clear_unsaved_prediction_records()
        self._latest_curves_by_card = None
        self._latest_curves_complete = False
        self._invalidate_history_verification()
        update_checkpoint_consistency(progress, "Reloaded durable RWKV checkpoint")
        return runtime

    def _legacy_rust_checkpoint_requires_rebuild(self) -> bool:
        if configured_rwkv_backend() != "rust":
            return False
        checkpoint_path = self.store.active_checkpoint_path()
        if checkpoint_path is None:
            return False
        try:
            metadata = read_rust_checkpoint_metadata(checkpoint_path)
        except CheckpointMetadataError:
            return False
        return self._rust_checkpoint_rebuild_reason(metadata) is not None

    def _require_checkpoint_selection_matches(self) -> None:
        # Never silently quantize a pre-existing FP32 trajectory, even though
        # native loading supports that conversion. Both toggle directions must
        # replay the complete history using the selected recurrent precision.
        if (
            self.store.active_checkpoint_path() is not None
            and self.store.manifest().get("state_precision", "fp32") != self.state_precision
        ):
            raise LegacyCheckpointError(
                "Card/note state precision changed. Rebuild the checkpoint from Anki's "
                "review history before using RWKV."
            )
        if (
            self.store.active_checkpoint_path() is not None
            and not self._checkpoint_model_matches_config()
        ):
            raise LegacyCheckpointError(
                "The selected RWKV model or generation changed. Rebuild the checkpoint "
                "from Anki's review history before using RWKV."
            )

    def _raise_if_legacy_rust_checkpoint_path(self, checkpoint_path: Path) -> None:
        if configured_rwkv_backend() != "rust":
            return
        try:
            metadata = read_rust_checkpoint_metadata(checkpoint_path)
        except CheckpointMetadataError:
            return
        self._raise_if_legacy_rust_checkpoint(metadata)

    def _raise_if_legacy_rust_checkpoint(
        self,
        metadata: RustCheckpointMetadata,
    ) -> None:
        reason = self._rust_checkpoint_rebuild_reason(metadata)
        if reason is not None:
            raise LegacyCheckpointError(reason)

    def _rust_checkpoint_rebuild_reason(
        self,
        metadata: RustCheckpointMetadata,
    ) -> str | None:
        if metadata.state_precision != self.state_precision:
            return (
                "This RWKV checkpoint uses a different card/note state precision. "
                "Rebuild it from Anki's review history after changing FP16/FP32."
            )
        if metadata.storage_version < CURRENT_RUST_CHECKPOINT_STORAGE_VERSION:
            return (
                "This RWKV checkpoint uses the obsolete Rust binary-v1 storage "
                "format. Rebuild it from Anki's review history before using RWKV."
            )
        if (
            metadata.model_version in {"v5o"}
            and _rollover_phase_from_manifest(self.store.manifest(), self.model_generation) is None
        ):
            return (
                "This checkpoint predates its generation's canonical rollover-phase contract. "
                "Rebuild it from Anki's review history before using RWKV."
            )
        if self._checkpoint_model_reference_requires_rebuild(metadata):
            return (
                "This RWKV checkpoint's saved model reference is unavailable to "
                "the bundled Rust runtime. This can happen after the v1/v2 model "
                "layout migration. Rebuild it from Anki's review history before "
                "using RWKV."
            )
        return None

    def _checkpoint_model_reference_requires_rebuild(
        self,
        metadata: RustCheckpointMetadata,
    ) -> bool:
        """Detect checkpoint references the bundled runtime cannot relocate.

        The runtime resolves a checkpoint ``model_id`` only in the old
        flat ``pretrained/`` directory. New runtimes are constructed with an
        explicit generation-scoped path, so only checkpoints created before
        that switch retain a name-only reference. Rebuilding from authoritative
        review history is safer than rewriting the native checkpoint format.
        """

        configured_model_path = bundled_model_path(
            self.model_id,
            model_generation=self.model_generation,
        )
        if configured_model_path is None:
            return False

        model_id = metadata.values.get("model_id")
        if model_id is not None:
            if not isinstance(model_id, str) or model_id != self.model_id:
                return False
            generation_directory = model_storage_subdirectory(self.model_generation)
            if configured_model_path.parent.name != generation_directory:
                # A legacy flat bundle remains resolvable by the frozen runtime.
                return False
            legacy_flat_path = configured_model_path.parent.parent / f"{model_id}.safetensors"
            return not legacy_flat_path.is_file()

        model_path = metadata.values.get("model_path")
        if model_path is None:
            return False
        try:
            return not Path(model_path).is_file()
        except (OSError, TypeError, ValueError):
            return True

    def _process_pending(
        self,
        runtime,
        pending: list[dict[str, Any]],
        progress: ProgressReporter,
    ) -> PredictionRecordSet:
        prediction_records = PredictionRecordSet.empty()
        try:
            return process_review_rows_with_predictions(
                runtime,
                pending,
                progress,
                label="Updating checkpoint with new reviews",
                record_set=prediction_records,
                latest_curves_by_card=self._latest_curves_by_card,
                process_many_mode=self.process_many_mode,
                calculate_curves=self.calculate_curves,
                input_context=self._input_context,
            )
        except CancelledError:
            if prediction_records.immediate_predictions:
                # Cancellation is observed only at an outer chunk boundary,
                # after the corresponding predictions and curves are aligned.
                # Keep that reconstructible progress in memory for the next
                # checkpoint update instead of replaying it immediately.
                self._remember_runtime_metadata(runtime)
                self._runtime_dirty = True
                self._unsaved_prediction_records.extend(prediction_records)
            raise
        except BaseException:
            # A mutating GPU dispatch can commit an internal prefix without
            # returning the prediction/curve outputs needed by our evaluation
            # cache. Discard the in-memory runtime and reconstruct it from the
            # durable checkpoint on the next attempt instead of replaying rows
            # and risking a double mutation.
            self._release_loaded_runtime()
            self._clear_runtime_state(preserve_review_data=True)
            raise
        finally:
            if process_many_uses_gpu(self.process_many_mode):
                # GPU processing owns a different cache from GPU prediction.
                # The bulk helper releases it after materializing CPU state, so
                # any previous prediction readiness result is now stale.
                self._runtime_gpu_ready = None

    def _prediction_record_prefixes_for_durable_update(
        self,
        rows: list[dict[str, Any]],
        start_count: int,
    ) -> PredictionRecordSet:
        if not start_count:
            return PredictionRecordSet.empty()

        records = self._load_available_prediction_records(rows)
        if records is None or not _record_set_covers(
            records,
            self._prediction_cache_specs,
            start_count,
        ):
            raise PredictionCacheError(
                "RWKV prediction cache is missing or stale. Rebuild the "
                "checkpoint to recreate a cache aligned with the durable checkpoint."
            )
        return records.slice(start_count)

    def _load_available_prediction_records(
        self,
        rows: list[dict[str, Any]],
    ) -> PredictionRecordSet | None:
        for candidate in _evaluation_cache_candidates(self.store):
            try:
                return load_prediction_record_set(
                    candidate,
                    rows,
                    self._prediction_cache_specs,
                    model_id=self.model_id,
                    validation=self.evaluation_cache_validation(),
                )
            except PredictionCacheError:
                continue
        return None

    def _ensure_latest_curves_loaded(
        self,
        rows: list[dict[str, Any]],
        *,
        card_ids: Iterable[int] | None = None,
    ) -> None:
        if not self.calculate_curves:
            return
        self._discard_stale_selected_curves()
        requested_ids = None if card_ids is None else {int(card_id) for card_id in card_ids}
        if requested_ids is not None and not requested_ids:
            return
        if self._evaluation_cache_validation is None:
            self._restore_evaluation_cache_validation_from_binding()
        if self._latest_curves_complete:
            return
        if (
            requested_ids is not None
            and self._latest_curves_by_card is not None
            and requested_ids <= set(self._latest_curves_by_card)
        ):
            return
        # Evaluation caches describe the durable checkpoint. A scoped runtime
        # may already have replayed a newer, deliberately unsaved review tail,
        # whose updated curves are overlaid below. Comparing the cache with the
        # transient runtime count would incorrectly mark valid durable curve
        # data stale when another card is requested during that lease.
        processed_count = self._manifest_processed_review_count()
        if processed_count == 0:
            if self._latest_curves_by_card is None:
                self._latest_curves_by_card = {}
            if requested_ids is None:
                self._latest_curves_complete = True
            return

        cache = self._load_available_latest_curves(rows, card_ids=requested_ids)
        if cache is None or int(cache.metadata["processed_review_count"]) != processed_count:
            if self._invalid_checkpoint_acknowledged():
                self._latest_curves_by_card = {}
                return
            self._mark_loaded_checkpoint_invalid()
            raise StaleCheckpointDataError(
                "RWKV predict-ahead curve cache is missing or stale. Rebuild the "
                "checkpoint to recreate curve state aligned with the durable checkpoint."
            )
        overlay = dict(cache.latest_curves_by_card)
        overlay.update(self._latest_curves_by_card or {})
        self._latest_curves_by_card = overlay
        self._evaluation_cache_validation = cache.validation
        if requested_ids is None:
            self._latest_curves_complete = True

    def _restore_evaluation_cache_validation_from_binding(self) -> bool:
        if evaluation_cache_validation_is_current(self._evaluation_cache_validation):
            return True
        if not self._checkpoint_lineage_is_current():
            return False

        manifest = self.store.manifest()
        binding = manifest.get(EVALUATION_CACHE_BINDING_MANIFEST_KEY)
        if binding is None:
            return False
        if not isinstance(binding, Mapping):
            raise CheckpointCacheBindingError(
                "RWKV evaluation cache checkpoint binding is malformed. Rebuild "
                "the checkpoint to recreate it."
            )

        try:
            version = int(binding.get("version", -1))
        except (TypeError, ValueError) as exc:
            raise CheckpointCacheBindingError(
                "RWKV evaluation cache checkpoint binding is malformed. Rebuild "
                "the checkpoint to recreate it."
            ) from exc
        if version != EVALUATION_CACHE_BINDING_VERSION or binding.get("algorithm") != "sha256":
            raise CheckpointCacheBindingError(
                "RWKV evaluation cache checkpoint binding is unsupported. Rebuild "
                "the checkpoint to recreate it."
            )

        if binding.get("state_precision", "fp32") != self.state_precision:
            raise CheckpointCacheBindingError(
                "RWKV evaluation cache belongs to a different state precision. "
                "Rebuild the checkpoint to recreate aligned predictions."
            )

        checkpoint_path = self.store.active_checkpoint_path()
        checkpoint_fingerprint = _checkpoint_history_fingerprint(checkpoint_path)
        bound_fingerprint = _normalized_history_fingerprint(
            binding.get("checkpoint_history_fingerprint")
        )
        if checkpoint_fingerprint is None or bound_fingerprint != checkpoint_fingerprint:
            raise CheckpointCacheBindingError(
                "RWKV evaluation cache belongs to a different checkpoint history. "
                "Rebuild the checkpoint to recreate aligned curve data."
            )

        cache_path_value = manifest.get(EVALUATION_CACHE_MANIFEST_KEY)
        if not cache_path_value:
            raise CheckpointCacheBindingError(
                "RWKV evaluation cache checkpoint binding has no cache file. Rebuild "
                "the checkpoint to recreate it."
            )
        processed_count = int(checkpoint_fingerprint["processed_review_count"])
        if processed_count != self._manifest_processed_review_count():
            raise CheckpointCacheBindingError(
                "RWKV evaluation cache checkpoint binding has inconsistent history "
                "metadata. Rebuild the checkpoint to recreate it."
            )
        try:
            validation = validate_evaluation_cache_file_binding(
                Path(cache_path_value),
                model_id=self.model_id,
                expected_sha256=str(binding.get("cache_sha256") or ""),
                expected_size=int(binding.get("cache_size", -1)),
                expected_processed_review_count=processed_count,
            )
        except (PredictionCacheError, TypeError, ValueError) as exc:
            raise CheckpointCacheBindingError(
                "RWKV evaluation cache no longer matches its checkpoint binding. "
                "Rebuild the checkpoint to recreate aligned curve data."
            ) from exc
        self._evaluation_cache_validation = validation
        return True

    def _validate_unbound_evaluation_cache(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        """Validate and bind a legacy cache, or leave an acknowledged one unbound."""

        manifest = self.store.manifest()
        if manifest.get(EVALUATION_CACHE_BINDING_MANIFEST_KEY) is not None:
            return
        partial = self.store.checkpoint_status() == "partial"
        cache_path = _evaluation_cache_path_from_manifest_or_store(
            manifest,
            self.store,
            partial=partial,
        )
        try:
            validation = validate_evaluation_cache_against_history(
                cache_path,
                rows,
                self._prediction_cache_specs,
                model_id=self.model_id,
            )
        except PredictionCacheError as exc:
            if self._invalid_checkpoint_acknowledged():
                self._latest_curves_by_card = {}
                return
            self._mark_loaded_checkpoint_invalid()
            raise StaleCheckpointDataError(
                "RWKV evaluation cache is not aligned with the durable checkpoint. "
                "Rebuild the checkpoint to recreate its prediction and curve data."
            ) from exc
        self._evaluation_cache_validation = validation
        self._persist_legacy_evaluation_cache_binding()

    def _persist_legacy_evaluation_cache_binding(self) -> None:
        """Upgrade a fully history-validated legacy cache for future starts."""

        validation = self._evaluation_cache_validation
        if (
            validation is None
            or not evaluation_cache_validation_is_current(validation)
            or not self._checkpoint_lineage_is_current()
        ):
            return
        manifest = self.store.manifest()
        if manifest.get(EVALUATION_CACHE_BINDING_MANIFEST_KEY) is not None:
            return
        cache_path_value = manifest.get(EVALUATION_CACHE_MANIFEST_KEY)
        if not cache_path_value:
            return
        cache_path = Path(cache_path_value)
        try:
            if cache_path.resolve() != Path(validation.path):
                return
        except OSError:
            return
        binding = _evaluation_cache_binding(
            checkpoint_path=self.store.active_checkpoint_path(),
            cache_path=cache_path,
        )
        if binding is None or not evaluation_cache_validation_is_current(validation):
            return
        manifest[EVALUATION_CACHE_BINDING_MANIFEST_KEY] = binding
        self.store.write_manifest(manifest)

    def _load_available_latest_curves(
        self,
        rows: list[dict[str, Any]],
        *,
        card_ids: set[int] | None = None,
    ):
        for candidate in _evaluation_cache_candidates(self.store):
            try:
                if card_ids is None:
                    return load_latest_curves_from_evaluation_cache(
                        candidate,
                        rows,
                        model_id=self.model_id,
                        validation=self._evaluation_cache_validation,
                    )
                return load_latest_curves_for_cards_from_evaluation_cache(
                    candidate,
                    rows,
                    model_id=self.model_id,
                    card_ids=card_ids,
                    validation=self._evaluation_cache_validation,
                )
            except PredictionCacheError:
                continue
        return None

    def _save_checkpoint_and_cache(
        self,
        runtime,
        rows: list[dict[str, Any]],
        prediction_records: PredictionRecordSet,
        *,
        status: str,
        checkpoint_path: Path,
        progress: ProgressReporter,
        durable_prefix_count: int = 0,
    ) -> CheckpointResult:
        if self.calculate_curves and not self._latest_curves_complete:
            update_checkpoint_curve_data(progress, "Loading complete RWKV curve data")
            self._ensure_latest_curves_loaded(rows)
        storage_estimate = _rust_checkpoint_storage_estimate(runtime, rows)
        self._durable_writer.cleanup_snapshots(runtime)
        _remove_stale_checkpoint_tmp(checkpoint_path)
        _ensure_checkpoint_write_has_disk_space(
            runtime,
            checkpoint_path,
            rows,
            storage_estimate=storage_estimate,
        )
        staged = self.store.create_checkpoint_snapshot()
        checkpoint_path = (
            staged.partial_checkpoint_path if status == "partial" else staged.latest_checkpoint_path
        )
        write = _DurableCheckpointWrite(
            runtime=runtime,
            rows=rows,
            prediction_records=prediction_records,
            prediction_cache_specs=self._prediction_cache_specs,
            latest_curves_by_card=_copy_curve_map(self._latest_curves_by_card),
            status=status,
            checkpoint_path=checkpoint_path,
            storage_estimate=storage_estimate,
            rollover_phase_ms=(
                self._require_input_context().rollover_phase_ms
                if model_generation_uses_exact_timestamps(self.model_generation)
                else None
            ),
            v5o_catalog_fingerprint=(
                self._input_context.catalog_fingerprint()
                if isinstance(self._input_context, V5oInputContext) else None
            ),
            snapshot_store=staged,
        )
        result = self._durable_writer.result_for(runtime, status, checkpoint_path)
        self._runtime = runtime
        # Until publication this runtime still represents an unsaved tail on
        # the old durable lineage, even if native save advances its backing path.
        self._loaded_path = str(self.store.active_checkpoint_path() or checkpoint_path)
        self._remember_runtime_metadata(runtime)
        self._runtime_dirty = True
        self._unsaved_prediction_records = prediction_records.slice_from(durable_prefix_count)
        progress.update(0, 1, "Writing RWKV checkpoint and evaluation cache")
        if self.async_saves and _runtime_supports_background_checkpoint_save(runtime):
            self._start_background_save(write)
            progress.update(1, 1, "RWKV checkpoint write is running in the background")
            return result

        result = self._durable_writer.write(write)
        self._loaded_path = result.checkpoint_path
        self._runtime_dirty = False
        self._clear_unsaved_prediction_records()
        self._remember_runtime_metadata(runtime)
        self._evaluation_cache_validation = None
        self._mark_history_verified(rows)
        progress.update(1, 1, "Wrote RWKV checkpoint and evaluation cache")
        return result

    def _start_background_save(
        self,
        write: _DurableCheckpointWrite,
    ) -> None:
        self.wait_for_pending_save()
        token = object()

        def save() -> None:
            error: BaseException | None = None
            pending_review_tail_context: tuple[ReviewData, CollectionRevision] | None = None
            try:
                result = self._durable_writer.write(write)
            except BaseException as exc:
                error = detach_exception_from_thread(exc)
            with self._save_lock:
                if self._save_token is token:
                    self._save_thread = None
                    self._save_token = None
                    self._save_status = None
                    self._save_error = error
                    if error is None:
                        self._runtime_dirty = False
                        self._clear_unsaved_prediction_records()
                        self._loaded_path = result.checkpoint_path
                        self._evaluation_cache_validation = None
                        self._mark_history_verified(write.rows)
                        pending_review_tail_context = self._pending_review_tail_context
                    self._pending_review_tail_context = None
            if pending_review_tail_context is not None:
                self.persist_review_tail_context(*pending_review_tail_context)

        thread = threading.Thread(target=save, name="RWKV checkpoint save", daemon=True)
        with self._save_lock:
            self._save_thread = thread
            self._save_token = token
            self._save_status = write.status
            self._save_error = None
        thread.start()

    def _current_durable_status(self) -> str:
        return "invalid" if self.store.manifest().get("status") == "invalid" else "valid"

    def _should_use_runtime_worker(self) -> bool:
        return configured_rwkv_backend() == "rust" and not self._on_runtime_worker()

    def _run_on_runtime_worker(self, op):
        with self._runtime_worker_condition:
            worker = self._runtime_worker
            if self._runtime_worker_unloading and (
                worker is None or not worker.is_worker_thread()
            ):
                raise CheckpointError("The RWKV runtime worker is shutting down.")
            if worker is None:
                worker = _RuntimeWorker()
                self._runtime_worker = worker
        return worker.call(op)

    def _on_runtime_worker(self) -> bool:
        worker = self._runtime_worker
        return worker is not None and worker.is_worker_thread()

    def _remember_runtime_metadata(self, runtime) -> None:
        self._runtime_processed_review_count = int(getattr(runtime, "processed_review_count", 0))
        self._runtime_last_review_id = _optional_int(getattr(runtime, "last_review_id", None))

    def _runtime_processed_count(self) -> int:
        if self._on_runtime_worker():
            self._remember_runtime_metadata(self._runtime)
        return int(self._runtime_processed_review_count or 0)

    def _runtime_last_review_id_value(self) -> int | None:
        if self._on_runtime_worker():
            self._remember_runtime_metadata(self._runtime)
        return self._runtime_last_review_id

    def _raise_background_save_error(self) -> None:
        with self._save_lock:
            error = self._save_error
            self._save_error = None
        if error is not None:
            if isinstance(error, (CheckpointError, InsufficientCheckpointDiskSpaceError)):
                raise error
            raise CheckpointError(f"RWKV checkpoint save failed: {error}") from error

    def _release_loaded_runtime(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        self._runtime = None
        _release_runtime_resources(runtime)

    def _clear_unsaved_prediction_records(self) -> None:
        self._unsaved_prediction_records = PredictionRecordSet.empty()

    def _durable_cache_files_available(
        self,
        status: str,
        manifest: dict[str, Any] | None = None,
    ) -> bool:
        manifest = self.store.manifest() if manifest is None else manifest
        partial = status == "partial"
        path = _evaluation_cache_path_from_manifest_or_store(
            manifest,
            self.store,
            partial=partial,
        )
        processed_review_count = _optional_int(manifest.get("processed_review_count"))
        if not evaluation_cache_has_specs(
            path,
            self._prediction_cache_specs,
            processed_review_count=processed_review_count,
        ):
            return False
        return not self.calculate_curves or evaluation_cache_has_latest_curves(
            path,
            processed_review_count=processed_review_count,
        )

    def _checkpoint_model_matches_config(
        self,
        manifest: dict[str, Any] | None = None,
    ) -> bool:
        manifest = self.store.manifest() if manifest is None else manifest
        manifest_model = manifest.get("model_id")
        manifest_generation = manifest.get("model_generation")
        generation_matches = (
            self.model_generation == DEFAULT_MODEL_GENERATION
            if manifest_generation is None
            else manifest_generation == self.model_generation
        )
        return (
            manifest.get("state_precision", "fp32") == self.state_precision
            and generation_matches
            and (manifest_model is None or str(manifest_model) == self.model_id)
        )

    def _checkpoint_policy_matches_config(
        self,
        manifest: Mapping[str, Any] | None = None,
    ) -> bool:
        manifest = self.store.manifest() if manifest is None else manifest
        deleted_policy = manifest.get("exclude_deleted_card_revlogs")
        if deleted_policy is not None and deleted_policy != self.exclude_deleted_card_revlogs:
            return False
        if isinstance(self._input_context, V5oInputContext) and (
            manifest.get(V5O_CATALOG_FINGERPRINT_MANIFEST_KEY)
            != self._input_context.catalog_fingerprint()
        ):
            return False
        return checkpoint_policy_matches(
            manifest,
            self.filtered_review_normalization_policy,
        )

    def _require_v5o_catalog_matches(self) -> None:
        if not model_generation_uses_v5o(self.model_generation):
            return
        context = self._require_input_context()
        if not isinstance(context, V5oInputContext):
            raise CheckpointError("V5 requires its retained-history catalog before loading state.")
        manifest = self.store.manifest()
        if self.store.active_checkpoint_path() is not None and (
            manifest.get(V5O_CATALOG_FINGERPRINT_MANIFEST_KEY) != context.catalog_fingerprint()
        ):
            raise InconsistentCheckpointError(
                "V5's retained card catalog changed or is missing. Rebuild the complete "
                "checkpoint from current Anki history before using predictions."
            )

    def _invalid_checkpoint_acknowledged(self) -> bool:
        manifest = self.store.manifest()
        return manifest.get("status") == "invalid" and bool(
            manifest.get("inconsistent_acknowledged")
        )

    def _mark_loaded_checkpoint_invalid(self) -> None:
        runtime = self._runtime
        if runtime is not None:
            self._durable_writer.write_manifest(
                runtime,
                "invalid",
                self.store.active_checkpoint_path(),
            )
            return

        manifest = self.store.manifest()
        manifest["status"] = "invalid"
        self.store.write_manifest(manifest)

    @staticmethod
    def _is_consistent(
        runtime,
        rows: list[dict[str, Any]],
        *,
        input_context: RuntimeInputContext | None = None,
        model_generation: str = DEFAULT_MODEL_GENERATION,
    ) -> bool:
        try:
            if not runtime_uses_exact_timestamps(
                runtime,
                model_generation=model_generation,
            ) and bool(getattr(runtime, "supports_native_review_batch_consistency", False)):
                ReviewBatch = require_rwkv_review_batch(backend="rust")
                consistency_input = native_review_batch_for_rows(ReviewBatch, rows)
            else:
                consistency_input = runtime_process_many_input(
                    runtime,
                    rows,
                    input_context=input_context,
                    model_generation=model_generation,
                )
            return bool(runtime.check_history_consistency(consistency_input))
        except Exception:
            return False

    def _checkpoint_consistency_rows(
        self,
        metadata: RustCheckpointMetadata,
        rows: list[dict[str, Any]],
    ) -> Iterable[Mapping[str, Any]]:
        if metadata.model_version not in {"v5o"}:
            return rows
        return tuple(self._require_input_context().process_row(row) for row in rows)

    def _require_input_context(self) -> RuntimeInputContext:
        if self._input_context is None:
            raise CheckpointError(
                "This generation requires review timestamp context. Reload review data before "
                "using this checkpoint."
            )
        return self._input_context


def write_comparison_checkpoint(
    store: ProfileStore,
    runtime: Any,
    rows,
    records: PredictionRecordSet,
    *,
    model_id: str,
    model_generation: str,
    context: RuntimeInputContext | None,
    normalization_policy: FilteredReviewNormalizationPolicy,
    exclude_deleted_card_revlogs: bool = True,
) -> None:
    """Save an already-built complete comparison state in an unpublished slot.

    Use the normal durable writer so cache digests, native history, catalog and
    rollover contracts remain identical to a normal checkpoint build.
    """
    if store.checkpoint_directory is None:
        raise ValueError("Comparison checkpoints must be staged before publication")
    store.ensure()
    specs = (PER_REVIEW_CACHE_SPEC,)
    writer = _DurableCheckpointWriter(
        store, model_id, model_generation, specs, normalization_policy,
        exclude_deleted_card_revlogs,
        state_precision=getattr(runtime, "state_precision", "fp32"),
    )
    estimate = _rust_checkpoint_storage_estimate(runtime, rows)
    writer.cleanup_snapshots(runtime)
    _ensure_checkpoint_write_has_disk_space(
        runtime, store.latest_checkpoint_path, rows, storage_estimate=estimate
    )
    writer.write(_DurableCheckpointWrite(
        runtime=runtime,
        rows=rows,
        prediction_records=records,
        prediction_cache_specs=specs,
        latest_curves_by_card=None,
        status="valid",
        checkpoint_path=store.latest_checkpoint_path,
        storage_estimate=estimate,
        rollover_phase_ms=(
            context.rollover_phase_ms
            if model_generation_uses_exact_timestamps(model_generation) and context is not None
            else None
        ),
        v5o_catalog_fingerprint=(
            context.catalog_fingerprint() if isinstance(context, V5oInputContext) else None
        ),
    ))


class _DurableCheckpointWriter:
    def __init__(
        self,
        store: ProfileStore,
        model_id: str,
        model_generation: str,
        prediction_cache_specs: tuple[PredictionCacheSpec, ...],
        filtered_review_normalization_policy: FilteredReviewNormalizationPolicy,
        exclude_deleted_card_revlogs: bool | None = None,
        *,
        state_precision: str = "fp32",
    ) -> None:
        self.store = store
        self.model_id = model_id
        self.model_generation = normalize_model_generation(model_generation)
        self.prediction_cache_specs = prediction_cache_specs
        self.filtered_review_normalization_policy = filtered_review_normalization_policy
        self.exclude_deleted_card_revlogs = exclude_deleted_card_revlogs
        self.state_precision = state_precision

    def result_for(self, runtime, status: str, checkpoint_path) -> CheckpointResult:
        return CheckpointResult(
            status=status,
            processed_review_count=int(getattr(runtime, "processed_review_count", 0)),
            last_review_id=getattr(runtime, "last_review_id", None),
            checkpoint_path=str(checkpoint_path or self.store.active_checkpoint_path() or ""),
        )

    def cleanup_snapshots(self, runtime, *, keep_paths: Iterable[Path] = ()) -> None:
        """Reclaim abandoned writes only while preserving live native backing."""
        protected = list(keep_paths)
        backing_path = getattr(runtime, "_backing_checkpoint_path", None)
        if backing_path is not None:
            protected.append(Path(backing_path))
        with suppress(OSError):
            self.store.cleanup_checkpoint_snapshots(keep_paths=protected)

    def write(self, write: _DurableCheckpointWrite) -> CheckpointResult:
        if int(getattr(write.runtime, "processed_review_count", 0)) != len(write.rows):
            raise CheckpointError("Checkpoint state and evaluation rows have different lengths.")
        # The manifest is the commit point for the complete checkpoint/cache
        # pair. Native scoped saves remember their destination as a backing
        # path: allocate its final name now and never rename it after saving.
        # A failed publication may still back a live runtime, so leave that
        # unpublished snapshot until the next safe write/cleanup boundary.
        self.cleanup_snapshots(write.runtime, keep_paths=(write.checkpoint_path,))
        staged = write.snapshot_store or self.store.create_checkpoint_snapshot()
        writer = _DurableCheckpointWriter(
            staged, self.model_id, self.model_generation, self.prediction_cache_specs,
            self.filtered_review_normalization_policy, self.exclude_deleted_card_revlogs,
            state_precision=self.state_precision,
        )
        partial = write.status == "partial"
        checkpoint_path = (
            staged.partial_checkpoint_path if partial else staged.latest_checkpoint_path
        )
        # Materialize the cache before the native disk-space check. The guard
        # then measures actual remaining free space while both the old snapshot
        # and new cache exist, rather than guessing at SQLite/curve sizes.
        write_evaluation_cache(
            _evaluation_cache_path(staged, partial=partial),
            write.rows,
            write.prediction_records,
            write.prediction_cache_specs,
            model_id=self.model_id,
            latest_curves_by_card=write.latest_curves_by_card,
        )
        _write_checkpoint_atomic(
            write.runtime,
            checkpoint_path,
            write.rows,
            storage_estimate=write.storage_estimate,
        )
        fingerprint = _checkpoint_history_fingerprint(checkpoint_path)
        if fingerprint is not None and (
            fingerprint["processed_review_count"] != len(write.rows)
            or fingerprint["last_review_id"] != getattr(write.runtime, "last_review_id", None)
        ):
            raise CheckpointError("Saved checkpoint history does not match its evaluation rows.")
        result = writer.write_manifest(
            write.runtime,
            write.status,
            checkpoint_path,
            day_offset_origin=_day_offset_origin_from_rows(write.rows),
            evaluation_cache_written=True,
            rollover_phase_ms=write.rollover_phase_ms,
            v5o_catalog_fingerprint=write.v5o_catalog_fingerprint,
            previous_manifest=self.store.manifest(),
        )
        if self.store.publish_checkpoint_snapshot(staged):
            # Publication succeeded. Cleanup is only garbage collection: failure
            # must not turn a committed save into a reported save failure.
            with suppress(OSError):
                self.store.cleanup_checkpoint_snapshots()
            self._remove_legacy_durable_files()
        return result

    def write_manifest(
        self,
        runtime,
        status: str,
        checkpoint_path,
        *,
        day_offset_origin: int | None = None,
        evaluation_cache_written: bool = False,
        rollover_phase_ms: int | None = None,
        v5o_catalog_fingerprint: str | None = None,
        previous_manifest: dict[str, Any] | None = None,
    ) -> CheckpointResult:
        result = self.result_for(runtime, status, checkpoint_path)
        old_manifest = self.store.manifest() if previous_manifest is None else previous_manifest
        partial = status == "partial"
        evaluation_cache_path = _existing_evaluation_cache_path_for_manifest(
            self.store,
            partial=partial,
        )
        if not evaluation_cache_written and old_manifest.get(EVALUATION_CACHE_MANIFEST_KEY):
            # A status-only invalidation may describe a partial snapshot. Its
            # files do not change names when status changes to "invalid".
            evaluation_cache_path = old_manifest[EVALUATION_CACHE_MANIFEST_KEY]
        evaluation_cache_binding = old_manifest.get(EVALUATION_CACHE_BINDING_MANIFEST_KEY)
        if evaluation_cache_written:
            evaluation_cache_binding = _evaluation_cache_binding(
                checkpoint_path=(Path(result.checkpoint_path) if result.checkpoint_path else None),
                cache_path=Path(evaluation_cache_path) if evaluation_cache_path else None,
            )
        manifest = {
            "status": status,
            "model_id": self.model_id,
            "model_generation": self.model_generation,
            "state_precision": self.state_precision,
            "exclude_deleted_card_revlogs": self.exclude_deleted_card_revlogs,
            FILTERED_REVIEW_NORMALIZATION_MANIFEST_KEY: (
                self.filtered_review_normalization_policy.semantic_signature()
            ),
            "checkpoint_path": result.checkpoint_path,
            EVALUATION_CACHE_MANIFEST_KEY: evaluation_cache_path,
            EVALUATION_CACHE_BINDING_MANIFEST_KEY: evaluation_cache_binding,
            "prediction_cache_path": None,
            "predict_ahead_prediction_cache_path": None,
            "predict_ahead_curve_cache_path": None,
            "processed_review_count": result.processed_review_count,
            "last_review_id": result.last_review_id,
            "day_offset_origin": (
                int(day_offset_origin)
                if day_offset_origin is not None
                else old_manifest.get("day_offset_origin")
            ),
            "updated_at": int(time.time()),
            "inconsistent_acknowledged": (
                bool(old_manifest.get("inconsistent_acknowledged"))
                if status == "invalid"
                else False
            ),
        }
        if model_generation_uses_exact_timestamps(self.model_generation):
            rollover_phase_ms = (
                rollover_phase_ms
                if rollover_phase_ms is not None
                else _rollover_phase_from_manifest(old_manifest, self.model_generation)
            )
            if rollover_phase_ms is not None:
                manifest[V5O_ROLLOVER_PHASE_MS_MANIFEST_KEY] = int(
                    rollover_phase_ms
                )
        if model_generation_uses_v5o(self.model_generation):
            manifest[V5O_CATALOG_FINGERPRINT_MANIFEST_KEY] = (
                v5o_catalog_fingerprint or old_manifest.get(V5O_CATALOG_FINGERPRINT_MANIFEST_KEY)
            )
        # Status-only updates must keep pointing at the same immutable snapshot.
        # A newly staged manifest gets its own ID at publication, never the old
        # snapshot's ID.
        if previous_manifest is None and "snapshot_id" in old_manifest:
            manifest["snapshot_id"] = old_manifest["snapshot_id"]
        self.store.write_manifest(manifest)
        return result

    def _remove_legacy_durable_files(self) -> None:
        # Snapshot-aware store properties now select the newly committed files.
        # Clean only the old, explicitly known flat-layout files, not arbitrary
        # files in the profile or the selected snapshot.
        directory = self.store.checkpoints_dir
        cache_directory = directory / "evaluation_cache"
        cache_is_link = cache_directory.is_symlink() or (
            hasattr(cache_directory, "is_junction") and cache_directory.is_junction()
        )
        for path in (
            directory / f"latest{rwkv_checkpoint_suffix()}",
            directory / f"partial{rwkv_checkpoint_suffix()}",
            *(cache_directory / name for name in (
                "rwkv_evaluation_cache.sqlite",
                "rwkv_evaluation_cache.partial.sqlite",
                "rwkv_review_predictions.jsonl.gz",
                "rwkv_review_predictions.partial.jsonl.gz",
                "rwkv_predict_ahead_predictions.jsonl.gz",
                "rwkv_predict_ahead_predictions.partial.jsonl.gz",
                "rwkv_predict_ahead_curves.pkl",
                "rwkv_predict_ahead_curves.partial.pkl",
            ) if not cache_is_link),
        ):
            with suppress(Exception):
                path.unlink(missing_ok=True)


def _write_checkpoint_atomic(
    runtime,
    path: Path,
    rows: list[dict[str, Any]],
    *,
    storage_estimate: RustCheckpointStorageEstimate | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_checkpoint_write_has_disk_space(
        runtime,
        path,
        rows,
        storage_estimate=storage_estimate,
    )
    if getattr(runtime, "_state_scope", None) is not None:
        # Current RWKV-SRS scoped saves already merge through their own adjacent
        # temporary file and atomically replace ``path``. Saving to our secondary
        # temporary name would make the runtime remember a backing path that we
        # immediately rename away.
        runtime.save_checkpoint(path)
        return

    tmp = _remove_stale_checkpoint_tmp(path)
    try:
        runtime.save_checkpoint(tmp)
        os.replace(tmp, path)
    except BaseException:
        with suppress(Exception):
            tmp.unlink(missing_ok=True)
        raise


def _runtime_supports_background_checkpoint_save(runtime) -> bool:
    return not _runtime_is_rust(runtime)


def _remove_stale_checkpoint_tmp(path: Path) -> Path:
    tmp = path.with_name(f"{path.name}.tmp{path.suffix}")
    with suppress(FileNotFoundError):
        tmp.unlink()
    return tmp


def _ensure_checkpoint_write_has_disk_space(
    runtime,
    path: Path,
    rows: list[dict[str, Any]],
    *,
    storage_estimate: RustCheckpointStorageEstimate | None = None,
) -> None:
    if configured_rwkv_backend() == "rust" or _runtime_is_rust(runtime):
        try:
            estimate = storage_estimate or _rust_checkpoint_storage_estimate(runtime, rows)
            if estimate is None:  # pragma: no cover - guarded by the Rust branch
                return
            ensure_rust_checkpoint_disk_space_for_estimate(estimate, path)
        except StorageDiskSpaceError as exc:
            raise InsufficientCheckpointDiskSpaceError(str(exc)) from exc


def _rust_checkpoint_storage_estimate(
    runtime,
    rows: list[dict[str, Any]],
) -> RustCheckpointStorageEstimate | None:
    if configured_rwkv_backend() != "rust" and not _runtime_is_rust(runtime):
        return None
    counts = rust_checkpoint_identity_counts(rows)
    expected_bytes = _runtime_expected_checkpoint_size(runtime, counts)
    return estimate_rust_checkpoint_storage_from_counts(
        counts,
        expected_checkpoint_bytes=expected_bytes,
    )


def _runtime_expected_checkpoint_size(
    runtime,
    counts: RustCheckpointIdentityCounts,
) -> int | None:
    if getattr(runtime, "model_version", None) == "v5o":
        # API 76's public facade describes a model-specific helper, but its
        # native dispatch still rejects V5. Retain the conservative FP32-sized
        # fallback even for FP16 until upstream exposes a working V5 estimate.
        return None
    expected_checkpoint_size = getattr(runtime, "expected_checkpoint_size", None)
    if not callable(expected_checkpoint_size):
        return None
    return int(expected_checkpoint_size(**counts.as_kwargs()))


def _runtime_is_rust(runtime) -> bool:
    return runtime_is_rust(runtime)


def _release_runtime_resources(runtime) -> None:
    release_runtime_resources(runtime)


def _existing_evaluation_cache_path_for_manifest(
    store: ProfileStore,
    *,
    partial: bool,
) -> str | None:
    path = _evaluation_cache_path(store, partial=partial)
    return str(path) if path.exists() else None


def _evaluation_cache_path(store: ProfileStore, *, partial: bool) -> Path:
    return store.partial_evaluation_cache_path if partial else store.evaluation_cache_path


def _evaluation_cache_path_from_manifest_or_store(
    manifest: dict[str, Any],
    store: ProfileStore,
    *,
    partial: bool,
) -> Path:
    value = manifest.get(EVALUATION_CACHE_MANIFEST_KEY)
    if value:
        path = Path(value)
        if path.exists():
            return path
    return _evaluation_cache_path(store, partial=partial)


def _evaluation_cache_binding(
    *,
    checkpoint_path: Path | None,
    cache_path: Path | None,
) -> dict[str, Any] | None:
    fingerprint = _checkpoint_history_fingerprint(checkpoint_path)
    if fingerprint is None or cache_path is None:
        return None
    try:
        precision = read_rust_checkpoint_metadata(checkpoint_path).state_precision
    except CheckpointMetadataError:
        return None
    try:
        cache_sha256, cache_size = evaluation_cache_file_digest(cache_path)
    except PredictionCacheError:
        return None
    return {
        "version": EVALUATION_CACHE_BINDING_VERSION,
        "algorithm": "sha256",
        "state_precision": precision,
        "cache_sha256": cache_sha256,
        "cache_size": cache_size,
        "checkpoint_history_fingerprint": fingerprint,
    }


def _checkpoint_history_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        metadata = read_rust_checkpoint_metadata(path)
    except CheckpointMetadataError:
        return None
    return _normalized_history_fingerprint(metadata.history_fingerprint)


def _normalized_history_fingerprint(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        version = int(value["version"])
        algorithm = str(value["algorithm"])
        canonicalization = str(value["canonicalization"])
        fields = [str(field) for field in value["fields"]]
        processed_count = int(value["processed_review_count"])
        last_review_id = _optional_int(value.get("last_review_id"))
        digest = str(value["digest"]).strip().lower()
    except (KeyError, TypeError, ValueError):
        return None
    if processed_count < 0 or len(digest) != 64:
        return None
    try:
        bytes.fromhex(digest)
    except ValueError:
        return None
    return {
        "version": version,
        "algorithm": algorithm,
        "canonicalization": canonicalization,
        "fields": fields,
        "last_review_id": last_review_id,
        "processed_review_count": processed_count,
        "digest": digest,
    }


def _evaluation_cache_candidates(store: ProfileStore) -> list[Path]:
    seen: set[str] = set()
    candidates: list[Path] = []
    manifest_path = store.manifest().get(EVALUATION_CACHE_MANIFEST_KEY)
    for candidate in (
        Path(manifest_path) if manifest_path else None,
        store.evaluation_cache_path,
        store.partial_evaluation_cache_path,
    ):
        if candidate is None:
            continue
        path = str(candidate)
        if path in seen:
            continue
        seen.add(path)
        candidates.append(candidate)
    return candidates


def _legacy_evaluation_cache_paths(store: ProfileStore) -> tuple[Path, ...]:
    return (
        store.prediction_cache_path,
        store.predict_ahead_prediction_cache_path,
        store.predict_ahead_curve_cache_path,
        store.partial_prediction_cache_path,
        store.partial_predict_ahead_prediction_cache_path,
        store.partial_predict_ahead_curve_cache_path,
    )


def _copy_curve_map(curves: dict[int, Any] | None) -> dict[int, Any] | None:
    return None if curves is None else dict(curves)


def _record_set_covers(
    records: PredictionRecordSet,
    specs: tuple[PredictionCacheSpec, ...],
    count: int,
) -> bool:
    return all(len(predictions_for_cache_spec(records, spec)) >= count for spec in specs)


def _normalize_prediction_cache_specs(
    specs: tuple[PredictionCacheSpec, ...] | None,
) -> tuple[PredictionCacheSpec, ...]:
    if specs is None:
        return (PER_REVIEW_CACHE_SPEC, PREDICT_AHEAD_CACHE_SPEC)
    unique: list[PredictionCacheSpec] = []
    for spec in specs:
        if spec not in (PER_REVIEW_CACHE_SPEC, PREDICT_AHEAD_CACHE_SPEC):
            raise ValueError(f"Unsupported RWKV prediction cache spec: {spec}")
        if spec not in unique:
            unique.append(spec)
    if PER_REVIEW_CACHE_SPEC not in unique:
        raise ValueError("RWKV prediction caching requires immediate review predictions.")
    return tuple(
        spec for spec in (PER_REVIEW_CACHE_SPEC, PREDICT_AHEAD_CACHE_SPEC) if spec in unique
    )


def _normalized_scalar_process_result(
    result: Any,
    *,
    return_curves: bool,
) -> tuple[float | None, Any | None]:
    """Keep every add-on live mutation on one stable result contract.

    Current RWKV-SRS returns a float when curves are disabled and a
    ``(prediction, curve)`` pair otherwise. Accepting a pair in no-curve mode
    also makes an in-place add-on upgrade tolerant of an already-loaded native
    adapter that still materializes the curve; the add-on discards it instead
    of failing after the state mutation has already committed.
    """

    if return_curves:
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("RWKV scalar processing did not return a prediction/curve pair.")
        prediction, curve = result
        return float(prediction), curve
    prediction = result[0] if isinstance(result, tuple) and result else result
    return None if prediction is None else float(prediction), None


def _positive_int_or_default(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _positive_optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _call_runtime_predict_many(
    runtime,
    rows: list[dict[str, Any]],
    *,
    batch_size: int | None,
    mode: str,
    fallback_mode: str | None = None,
    input_context: RuntimeInputContext | None = None,
    model_generation: str | None = None,
) -> list[float]:
    kwargs: dict[str, Any] = {
        "mode": str(mode),
        "batch_size": None if batch_size is None else int(batch_size),
    }
    if fallback_mode is not None:
        kwargs["fallback_mode"] = str(fallback_mode)
    prediction_input = runtime_predict_many_input(
        runtime,
        rows,
        input_context=input_context,
        model_generation=model_generation,
    )

    predictions = list(runtime.predict_many(prediction_input, **kwargs))
    if len(predictions) != len(rows):
        raise RuntimeError(
            f"RWKV-SRS predict_many() returned {len(predictions)} results for {len(rows)} rows."
        )
    return predictions


def _last_review_id_from_rows(rows: list[dict[str, Any]]) -> int | None:
    if not rows:
        return None
    return int(rows[-1]["review_id"])


def _day_offset_origin_from_rows(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    first = rows[0]
    return int(first.get("raw_day_offset", first.get("day_offset", 0)))


def _optional_int(value) -> int | None:
    if value is None:
        return None
    return int(value)






def _rollover_phase_from_manifest(manifest: Mapping[str, Any], generation: str) -> int | None:
    value = manifest.get(V5O_ROLLOVER_PHASE_MS_MANIFEST_KEY)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        phase = value
    elif isinstance(value, float) and value.is_integer():
        phase = int(value)
    else:
        return None
    return phase if 0 <= phase < MILLISECONDS_PER_DAY else None


def _checkpoint_file_signature(
    path: Path | None,
) -> tuple[str, int, int, int, int] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return (
        str(path.resolve()),
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


_SCOPE_ID_FIELDS = ("card_id", "note_id", "deck_id", "preset_id")


def _runtime_scope_cards(
    rows: list[dict[str, Any]],
    requested_cards: list[dict[str, Any]] | None,
    *,
    durable_processed_count: int,
) -> list[dict[str, Any]] | None:
    if requested_cards is None:
        return None
    pending_start = min(max(0, int(durable_processed_count)), len(rows))
    return _dedupe_scope_cards((*requested_cards, *rows[pending_start:]))


def _dedupe_scope_cards(
    cards: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[int | None, ...]] = set()
    for card in cards:
        try:
            card_id = int(card["card_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(
                "RWKV checkpoint scope cards require an integer card_id."
            ) from exc
        values = (
            card_id,
            _scope_optional_int(card.get("note_id")),
            _scope_optional_int(card.get("deck_id")),
            _scope_optional_int(card.get("preset_id")),
        )
        if values in seen:
            continue
        seen.add(values)
        unique.append(dict(zip(_SCOPE_ID_FIELDS, values, strict=True)))
    return unique


def _scope_key(
    cards: list[dict[str, Any]] | None,
) -> tuple[tuple[int | None, ...], ...] | None:
    if cards is None:
        return None
    return tuple(
        tuple(_scope_optional_int(card.get(field)) for field in _SCOPE_ID_FIELDS) for card in cards
    )


def _scope_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed_float = float(value)
    except (TypeError, ValueError):
        return int(value)
    if parsed_float != parsed_float:
        return None
    return int(value)
