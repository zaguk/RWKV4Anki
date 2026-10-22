"""Disposable, paired card/note storage-precision measurements.

No Anki imports, checkpoint publication, or config writes belong in this module.
The caller holds the exclusive runtime reservation and runs this on a worker.
"""

from __future__ import annotations

import math
import time
from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Any

from .checkpoint_storage import (
    ensure_rust_checkpoint_disk_space,
    estimate_rust_checkpoint_storage,
    rust_checkpoint_identity_counts,
)
from .metrics import EvaluationScope, MetricResult, calculate_aligned_scoped_metrics
from .model_generation import MODEL_GENERATIONS, model_generation_uses_v5o
from .progress import ProgressReporter
from .review_rows import ReviewData, prediction_rows_for_card_ids
from .runtime_inputs import RuntimeInputContext, input_context_for_generation
from .rwkv_performance_modes import (
    PREDICT_MANY_MODES,
    PROCESS_MANY_MODES,
    prediction_progress_chunk_size,
)
from .rwkv_processing import (
    PROCESS_MANY_CHUNK_SIZE,
    PROCESS_MANY_THREADS,
    new_rwkvp_runtime,
    runtime_predict_many_input,
    runtime_process_many_input,
)
from .rwkv_runtime_resources import release_runtime_resources
from .v5o_review_rows import V5oInputContext, exact_i64

PRECISION_COMPARISON_REPETITIONS = 5
_PRECISIONS = ("fp32", "fp16")


@dataclass(frozen=True)
class StatePrecisionComparisonPlan:
    rows: Sequence[Mapping[str, Any]]
    query_rows: tuple[Mapping[str, Any], ...]
    model_id: str
    model_generation: str
    process_many_mode: str
    predict_many_mode: str
    temporary_directory: Path
    input_context: RuntimeInputContext | None = None
    calculate_curves: bool = False
    predict_batch_size: int | None = None
    repetitions: int = PRECISION_COMPARISON_REPETITIONS


@dataclass(frozen=True)
class StatePrecisionMeasurement:
    state_precision: str
    state_seconds: float
    checkpoint_bytes: int
    canonical_state_bytes: int
    card_note_state_bytes: int
    prediction_durations_seconds: tuple[float, ...]
    query_card_count: int
    all_metrics: MetricResult = MetricResult(None, None)
    all_review_count: int = 0
    recent_metrics: MetricResult = MetricResult(None, None)
    recent_review_count: int = 0

    @property
    def average_prediction_seconds(self) -> float:
        return sum(self.prediction_durations_seconds) / len(self.prediction_durations_seconds)

    @property
    def predictions_per_second(self) -> float:
        seconds = self.average_prediction_seconds
        return self.query_card_count / seconds if seconds else math.inf


@dataclass(frozen=True)
class StatePrecisionComparisonResult:
    model_id: str
    model_generation: str
    process_many_mode: str
    predict_many_mode: str
    calculate_curves: bool
    processed_review_count: int
    query_card_count: int
    repetitions: int
    recent_start_index: int
    measurements: tuple[StatePrecisionMeasurement, ...]
    predict_batch_size: int | None = None

    def measurement(self, state_precision: str) -> StatePrecisionMeasurement:
        return next(m for m in self.measurements if m.state_precision == state_precision)


def state_precision_comparison_plan(
    review_data: ReviewData,
    *,
    model_id: str,
    model_generation: str,
    process_many_mode: str,
    predict_many_mode: str,
    target_timestamp_ms: int,
    temporary_directory: Path,
    calculate_curves: bool = False,
    predict_batch_size: int | None = None,
) -> StatePrecisionComparisonPlan:
    """Capture one query clock and one history for both independent builds.

    Only current cards with processed state are queried; new/unprocessed cards
    would measure the native scalar fallback instead of the requested mode.
    Metrics cover the selected history, including deleted history if supplied.
    """

    if not review_data.complete_history:
        raise ValueError("Precision comparison requires complete chronological review history.")
    if model_generation not in MODEL_GENERATIONS:
        raise ValueError("Unsupported RWKV generation for precision comparison.")
    if review_data.history_generation != model_generation:
        raise ValueError("Precision comparison history belongs to a different generation.")
    target = exact_i64(target_timestamp_ms, "target_timestamp_ms")
    if review_data.rows:
        target = max(target, int(review_data.rows[-1]["review_id"]))
    current_processed = sorted(set(review_data.cards).intersection(review_data.last_by_card))
    context = input_context_for_generation(review_data, model_generation)
    if isinstance(context, V5oInputContext):
        queries = [
            {"card_id": card_id, "target_timestamp_ms": target,
             "rollover_phase_ms": context.rollover_phase_ms}
            for card_id in current_processed
        ]
    else:
        queries = prediction_rows_for_card_ids(
            current_processed, review_data, target_timestamp_seconds=target / 1000.0,
        )
    plan = StatePrecisionComparisonPlan(
        rows=review_data.rows,
        query_rows=tuple(MappingProxyType(dict(row)) for row in queries),
        model_id=str(model_id), model_generation=model_generation,
        process_many_mode=process_many_mode, predict_many_mode=predict_many_mode,
        temporary_directory=Path(temporary_directory), input_context=context,
        calculate_curves=bool(calculate_curves), predict_batch_size=predict_batch_size,
    )
    _validate_plan(plan)
    return plan


def run_state_precision_comparison(
    plan: StatePrecisionComparisonPlan,
    progress: ProgressReporter,
    *,
    runtime_factory: Callable[..., Any] | None = None,
    release_runtime: Callable[[Any], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> StatePrecisionComparisonResult:
    """Build FP32 then FP16, retaining only scalar predictions between owners.

    Timing includes chronological process_many, output collection, and final
    GPU synchronization/release, but excludes model construction, metrics and
    checkpoint serialization. Prediction timing excludes one full warm-up and
    input preparation, and includes the same bounded native calls in each run.
    GPU execution is strict: errors abort rather than relabel CPU measurements.
    """

    _validate_plan(plan)
    progress.check_cancelled()
    _validate_history(plan.rows, progress)
    plan.temporary_directory.mkdir(parents=True, exist_ok=True)
    builds: list[tuple[StatePrecisionMeasurement, array]] = []
    for precision in _PRECISIONS:
        progress.check_cancelled()
        # Each file is removed only after its owner has closed, including on
        # Windows and error paths; two complete checkpoint files never coexist.
        with TemporaryDirectory(prefix=".precision-", dir=plan.temporary_directory) as directory:
            runtime = None
            try:
                progress.update(0, len(plan.rows), f"{precision.upper()}: loading model")
                runtime = (runtime_factory or new_rwkvp_runtime)(
                    model_id=plan.model_id, model_generation=plan.model_generation,
                    process_many_mode=plan.process_many_mode, state_precision=precision,
                )
                if getattr(runtime, "state_precision", None) != precision:
                    raise RuntimeError("RWKV runtime did not honor the requested state precision.")
                _require_gpu_mode(runtime, plan.process_many_mode, "process")
                progress.check_cancelled()
                if isinstance(plan.input_context, V5oInputContext):
                    runtime.rebuild_retained_history(
                        [], cards=plan.input_context.catalog_rows(),
                        rollover_phase_ms=plan.input_context.rollover_phase_ms,
                        mode=plan.process_many_mode,
                    )
                started = clock()
                predictions = _build_state(runtime, plan, progress, precision)
                # Materialize deferred process state before stopping the timer.
                # This is also the boundary between process and predict caches.
                runtime.release_gpu()
                state_seconds = max(0.0, clock() - started)
                progress.check_cancelled()
                storage = runtime.state_storage_info()
                canonical_bytes, card_note_bytes = _storage_bytes(storage, precision)
                progress.update(len(plan.rows), len(plan.rows),
                                f"{precision.upper()}: measuring checkpoint size")
                path = Path(directory) / "state.bin"
                _check_checkpoint_space(runtime, plan.rows, path, canonical_bytes=canonical_bytes)
                runtime.save_checkpoint(path)
                checkpoint_bytes = path.stat().st_size
                progress.check_cancelled()
                durations = _measure_predictions(runtime, plan, progress, clock, precision)
                builds.append((StatePrecisionMeasurement(
                    state_precision=precision, state_seconds=state_seconds,
                    checkpoint_bytes=checkpoint_bytes, canonical_state_bytes=canonical_bytes,
                    card_note_state_bytes=card_note_bytes,
                    prediction_durations_seconds=durations, query_card_count=len(plan.query_rows),
                ), predictions))
            finally:
                if runtime is not None:
                    (release_runtime or release_runtime_resources)(runtime)

    progress.check_cancelled()
    measurements = _score_paired(plan, builds, progress)
    return StatePrecisionComparisonResult(
        model_id=plan.model_id, model_generation=plan.model_generation,
        process_many_mode=plan.process_many_mode, predict_many_mode=plan.predict_many_mode,
        calculate_curves=plan.calculate_curves, processed_review_count=len(plan.rows),
        query_card_count=len(plan.query_rows), repetitions=plan.repetitions,
        recent_start_index=3 * len(plan.rows) // 4, measurements=measurements,
        predict_batch_size=plan.predict_batch_size,
    )


def _build_state(runtime, plan, progress, precision) -> array:
    predictions = array("d")
    for start in range(0, len(plan.rows), PROCESS_MANY_CHUNK_SIZE):
        progress.check_cancelled()
        chunk = plan.rows[start:start + PROCESS_MANY_CHUNK_SIZE]
        batch = runtime_process_many_input(runtime, chunk, input_context=plan.input_context)
        values = runtime.process_many(
            batch, return_curves=plan.calculate_curves, mode=plan.process_many_mode,
            batch_size=PROCESS_MANY_CHUNK_SIZE, num_threads=PROCESS_MANY_THREADS,
        )
        count_before = len(predictions)
        for value in values:
            prediction = value[0] if plan.calculate_curves else value
            predictions.append(math.nan if prediction is None else float(prediction))
        if len(predictions) - count_before != len(chunk):
            raise RuntimeError("RWKV process_many() returned an unaligned prediction count.")
        progress.update(len(predictions), len(plan.rows), f"{precision.upper()}: building state")
    return predictions


def _measure_predictions(runtime, plan, progress, clock, precision) -> tuple[float, ...]:
    _require_gpu_mode(runtime, plan.predict_many_mode, "predict")
    chunk_size = prediction_progress_chunk_size(plan.predict_many_mode, plan.predict_batch_size)
    batches = []
    for start in range(0, len(plan.query_rows), chunk_size):
        progress.check_cancelled()
        chunk = plan.query_rows[start:start + chunk_size]
        batches.append((runtime_predict_many_input(
            runtime, chunk, input_context=plan.input_context,
        ), len(chunk)))
    durations = []
    for repetition in range(plan.repetitions + 1):
        progress.check_cancelled()
        label = "warming predict_many()" if repetition == 0 else (
            f"predict_many() run {repetition}/{plan.repetitions}"
        )
        progress.update(0, len(plan.query_rows), f"{precision.upper()}: {label}")
        total_seconds = 0.0
        count = 0
        for batch, expected in batches:
            progress.check_cancelled()
            started = clock()
            values = runtime.predict_many(
                batch, mode=plan.predict_many_mode, batch_size=plan.predict_batch_size,
            )
            total_seconds += max(0.0, clock() - started)
            if len(values) != expected:
                raise RuntimeError("RWKV predict_many() returned an unaligned prediction count.")
            if any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for p in values):
                raise RuntimeError("RWKV predict_many() returned an invalid probability.")
            count += expected
            progress.update(count, len(plan.query_rows), f"{precision.upper()}: {label}")
        if repetition:
            durations.append(total_seconds)
    return tuple(durations)


def _score_paired(plan, builds, progress) -> tuple[StatePrecisionMeasurement, ...]:
    recent_start = 3 * len(plan.rows) // 4
    eligible: set[int] = set()
    recent: set[int] = set()
    for index, row in enumerate(plan.rows):
        if index % PROCESS_MANY_CHUNK_SIZE == 0:
            progress.check_cancelled()
        if int(row["rating"]) not in {1, 2, 3, 4} or float(row["elapsed_days"]) <= 0:
            continue
        pair = tuple(predictions[index] for _measurement, predictions in builds)
        if not all(math.isfinite(value) for value in pair):
            continue
        if not all(0 <= value <= 1 for value in pair):
            raise ValueError("RWKV historical prediction is outside [0, 1].")
        review_id = int(row["review_id"])
        eligible.add(review_id)
        if index >= recent_start:
            recent.add(review_id)
    scopes = tuple(EvaluationScope(key, key, "collection", "") for key in ("all", "recent"))
    measurements = []
    for measurement, predictions in builds:
        progress.update(0, len(plan.rows),
                        f"{measurement.state_precision.upper()}: scoring paired review cohorts")
        scores = calculate_aligned_scoped_metrics(
            plan.rows, predictions, scopes,
            target_review_ids_by_scope={"all": eligible, "recent": recent}, progress=progress,
        )
        measurements.append(replace(
            measurement, all_metrics=scores.metrics["all"], all_review_count=scores.counts["all"],
            recent_metrics=scores.metrics["recent"], recent_review_count=scores.counts["recent"],
        ))
    return tuple(measurements)


def _require_gpu_mode(runtime, mode: str, operation: str) -> None:
    if mode == "gpu" and not runtime.gpu_available(operation):
        raise RuntimeError(
            f"GPU {operation} is unavailable; precision comparison was not run on CPU."
        )


def _storage_bytes(storage: Mapping[str, Any], precision: str) -> tuple[int, int]:
    if storage.get("state_precision") != precision:
        raise RuntimeError("RWKV storage metadata disagrees with the requested precision.")
    # Native stream order is card, deck, note, preset, collection. These are
    # canonical payloads only, not process RSS or prediction GPU mirror sizes.
    payload = tuple(storage["payload_bytes"])
    if len(payload) != 5 or any(type(value) is not int or value < 0 for value in payload):
        raise RuntimeError("RWKV returned invalid canonical state storage sizes.")
    return sum(payload), payload[0] + payload[2]


def _check_checkpoint_space(runtime, rows, path: Path, *, canonical_bytes: int) -> None:
    expected = getattr(runtime, "expected_checkpoint_size", None)
    expected_bytes = None
    if getattr(runtime, "model_version", None) == "v5o":
        # Native API 76's facade exposes this method for V5, but the native
        # estimator remains V1-only. Preserve the conservative full-FP32
        # estimate, never below the actually measured tensor payload. The
        # standard free-space margin is applied by the guard below; actual
        # serialized bytes (including V5 compact metadata) are measured later.
        estimate = estimate_rust_checkpoint_storage(rows)
        expected_bytes = max(estimate.estimated_checkpoint_bytes, canonical_bytes)
    elif callable(expected):
        counts = rust_checkpoint_identity_counts(rows)
        expected_bytes = int(expected(**counts.as_kwargs()))
    ensure_rust_checkpoint_disk_space(rows, path, expected_checkpoint_bytes=expected_bytes)


def _validate_history(rows, progress) -> None:
    previous_id: int | None = None
    for index, row in enumerate(rows):
        if index % PROCESS_MANY_CHUNK_SIZE == 0:
            progress.check_cancelled()
        review_id = int(row["review_id"])
        if previous_id is not None and review_id <= previous_id:
            raise ValueError("Precision comparison requires strictly chronological review IDs.")
        previous_id = review_id
        # A truncated/reprocessed tail must never silently reset metric bins.
        # Canonical ReviewData supplies these full-history source fields.
        if ("review_count" not in row or "prior_lapses" not in row
                or int(row["review_count"]) < 1 or int(row["prior_lapses"]) < 0):
            raise ValueError("Precision comparison requires full-history review/lapse counts.")
        if not math.isfinite(float(row["elapsed_days"])):
            raise ValueError("Precision comparison requires finite elapsed days.")


def _validate_plan(plan: StatePrecisionComparisonPlan) -> None:
    if not plan.rows or not plan.query_rows:
        raise ValueError("Precision comparison requires reviews and current cards with state.")
    if plan.model_generation not in MODEL_GENERATIONS:
        raise ValueError("Unsupported RWKV model generation.")
    if plan.process_many_mode not in PROCESS_MANY_MODES:
        raise ValueError("Unsupported precision comparison processing mode.")
    if plan.predict_many_mode not in PREDICT_MANY_MODES:
        raise ValueError("Unsupported precision comparison prediction mode.")
    if type(plan.repetitions) is not int or plan.repetitions <= 0:
        raise ValueError("Precision comparison repetitions must be positive.")
    if plan.predict_batch_size is not None and (
        type(plan.predict_batch_size) is not int or plan.predict_batch_size <= 0
    ):
        raise ValueError("Prediction batch size must be a positive integer or Automatic.")
    if model_generation_uses_v5o(plan.model_generation) and (
        plan.calculate_curves or not isinstance(plan.input_context, V5oInputContext)
    ):
        raise ValueError("V5 precision comparison requires retained history and no curves.")
