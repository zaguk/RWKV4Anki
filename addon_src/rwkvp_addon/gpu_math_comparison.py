"""Paired GPU arithmetic prediction comparison, independent of Anki."""

from __future__ import annotations

import math
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .model_generation import MODEL_GENERATIONS
from .precision_comparison import (
    StatePrecisionComparisonPlan,
    _check_checkpoint_space,
    state_precision_comparison_plan,
)
from .progress import ProgressReporter
from .rwkv_performance_modes import prediction_progress_chunk_size
from .rwkv_processing import (
    PROCESS_MANY_CHUNK_SIZE,
    PROCESS_MANY_THREADS,
    _runtime_constructor_kwargs,
    new_rwkvp_runtime,
    runtime_predict_many_input,
    runtime_process_many_input,
)
from .rwkv_runtime_resources import release_runtime_resources
from .v5o_review_rows import V5oInputContext
from .vendor_bootstrap import require_rwkv_srs


@dataclass(frozen=True)
class GPUMathMeasurement:
    math: str
    kernel: str
    fallback_reason: str | None = None


@dataclass(frozen=True)
class PredictionDifferences:
    count: int
    mean_pp: float
    median_pp: float
    p90_pp: float
    p99_pp: float
    max_pp: float


@dataclass(frozen=True)
class GPUMathComparisonResult:
    model_id: str
    processed_review_count: int
    measurements: tuple[GPUMathMeasurement, ...]
    differences: PredictionDifferences
    predict_batch_size: int | None = None
    model_generation: str = "v1"


def gpu_math_comparison_plan(review_data, **kwargs) -> StatePrecisionComparisonPlan:
    return state_precision_comparison_plan(
        review_data,
        predict_many_mode="gpu",
        calculate_curves=False,
        **kwargs,
    )


def prediction_differences(
    baseline: Sequence[float],
    candidate: Sequence[float],
) -> PredictionDifferences:
    """Absolute probability differences, *100 (pp), not relative percent errors.

    Percentiles use linear interpolation at (n-1)*q, including singleton cohorts.
    """
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Prediction difference vectors must be nonempty and aligned.")
    differences = []
    for first, second in zip(baseline, candidate, strict=True):
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in (first, second)):
            raise ValueError("Prediction differences require finite probabilities in [0, 1].")
        differences.append(abs(second - first) * 100)
    differences.sort()

    def percentile(q: float) -> float:
        index = (len(differences) - 1) * q
        lower = math.floor(index)
        upper = math.ceil(index)
        return differences[lower] + (differences[upper] - differences[lower]) * (index - lower)

    return PredictionDifferences(
        len(differences),
        math.fsum(differences) / len(differences),
        percentile(0.5),
        percentile(0.9),
        percentile(0.99),
        differences[-1],
    )


def run_gpu_math_comparison(
    plan: StatePrecisionComparisonPlan,
    progress: ProgressReporter,
    *,
    runtime_factory: Callable[..., Any] = new_rwkvp_runtime,
    checkpoint_factory: Callable[..., Any] | None = None,
) -> GPUMathComparisonResult:
    """One FP16 build, sequential restores, strict GPU execution, no publication.

    Each mode predicts the frozen queries once in cancellable chunks; no speed
    measurements or CPU fallback are allowed. Kernel fallback to
    FP32 invalidates the FP16 comparison instead of producing a misleading label.
    """
    if plan.model_generation not in MODEL_GENERATIONS or plan.predict_many_mode != "gpu":
        raise ValueError("GPU math comparison requires a supported generation and GPU prediction.")
    if plan.model_generation == "v5o" and not isinstance(plan.input_context, V5oInputContext):
        raise ValueError("V5 GPU math comparison requires retained-history input context.")
    if not plan.rows or not plan.query_rows:
        raise ValueError(
            "GPU math comparison requires reviews and state-backed cards."
        )
    progress.check_cancelled()
    plan.temporary_directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".gpu-math-", dir=plan.temporary_directory) as directory:
        checkpoint = Path(directory) / "state.bin"
        runtime = None
        try:
            progress.update(0, len(plan.rows), "Building one disposable FP16 state")
            runtime = runtime_factory(
                model_id=plan.model_id,
                model_generation=plan.model_generation,
                process_many_mode=plan.process_many_mode,
                state_precision="fp16",
                gpu_prediction_math="fp32",
            )
            if runtime.state_precision != "fp16":
                raise RuntimeError("RWKV did not honor FP16 state storage.")
            if plan.process_many_mode == "gpu" and not runtime.gpu_available("process"):
                raise RuntimeError(
                    "GPU state building is unavailable; comparison was not run on CPU."
                )
            if isinstance(plan.input_context, V5oInputContext):
                runtime.rebuild_retained_history(
                    [], cards=plan.input_context.catalog_rows(),
                    rollover_phase_ms=plan.input_context.rollover_phase_ms,
                    mode=plan.process_many_mode,
                )
            for start in range(0, len(plan.rows), PROCESS_MANY_CHUNK_SIZE):
                progress.check_cancelled()
                chunk = plan.rows[start : start + PROCESS_MANY_CHUNK_SIZE]
                values = runtime.process_many(
                    runtime_process_many_input(runtime, chunk, input_context=plan.input_context),
                    mode=plan.process_many_mode,
                    return_curves=False,
                    batch_size=PROCESS_MANY_CHUNK_SIZE,
                    num_threads=PROCESS_MANY_THREADS,
                )
                if len(values) != len(chunk):
                    raise RuntimeError(
                        "RWKV process_many() returned an unaligned prediction count."
                    )
                progress.update(
                    start + len(chunk), len(plan.rows), "Building disposable FP16 state"
                )
            runtime.release_gpu()
            progress.check_cancelled()
            canonical_bytes = (
                sum(runtime.state_storage_info()["payload_bytes"])
                if isinstance(plan.input_context, V5oInputContext) else 0
            )
            _check_checkpoint_space(
                runtime, plan.rows, checkpoint, canonical_bytes=canonical_bytes,
            )
            runtime.save_checkpoint(checkpoint)
        finally:
            if runtime is not None:
                release_runtime_resources(runtime)

        measurements = []
        predictions = []
        scope_cards = plan.query_rows
        if isinstance(plan.input_context, V5oInputContext):
            queried_ids = {int(row["card_id"]) for row in plan.query_rows}
            scope_cards = tuple(
                row for row in plan.input_context.catalog_rows() if row["card_id"] in queried_ids
            )
        for math_mode in ("fp32", "fp16"):
            progress.check_cancelled()
            runtime = None
            try:
                progress.update(
                    0, len(plan.query_rows), f"{math_mode.upper()} math: loading identical state"
                )
                runtime = (checkpoint_factory or require_rwkv_srs(backend="rust"))(
                    checkpoint=checkpoint,
                    cards=scope_cards,
                    **_runtime_constructor_kwargs(
                        "rust",
                        state_precision="fp16",
                        gpu_prediction_math=math_mode,
                    ),
                )
                if runtime.state_precision != "fp16" or runtime.gpu_prediction_math != math_mode:
                    raise RuntimeError(
                        "RWKV did not honor the requested storage/math configuration."
                    )
                profile = runtime.initialize_gpu("predict")
                if profile.get("prediction_math") != math_mode:
                    reason = (
                        profile.get("prediction_math_fallback_reason")
                        or "requested kernel unavailable"
                    )
                    raise RuntimeError(
                        f"{math_mode.upper()} GPU math unavailable: {reason}. "
                        "Comparison stopped; fallback was not compared as FP16."
                    )
                measurement, values = _predict(runtime, plan, progress, math_mode, profile)
                measurements.append(measurement)
                predictions.append(values)
            finally:
                if runtime is not None:
                    release_runtime_resources(runtime)
        progress.check_cancelled()
        progress.update(0, 1, "Calculating paired percentage-point differences")
        differences = prediction_differences(*predictions)
        progress.check_cancelled()
        return GPUMathComparisonResult(
            plan.model_id,
            len(plan.rows),
            tuple(measurements),
            differences,
            plan.predict_batch_size,
            plan.model_generation,
        )


def _predict(runtime, plan, progress, math_mode, profile):
    chunk_size = prediction_progress_chunk_size("gpu", plan.predict_batch_size)
    snapshot = array("f")
    for start in range(0, len(plan.query_rows), chunk_size):
        progress.check_cancelled()
        chunk = plan.query_rows[start : start + chunk_size]
        values = runtime.predict_many(
            runtime_predict_many_input(runtime, chunk, input_context=plan.input_context),
            mode="gpu", batch_size=plan.predict_batch_size,
        )
        if len(values) != len(chunk):
            raise RuntimeError("RWKV predict_many() returned an unaligned prediction count.")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
            raise RuntimeError("RWKV predict_many() returned an invalid probability.")
        snapshot.extend(values)
        progress.update(
            len(snapshot), len(plan.query_rows), f"{math_mode.upper()} math: predicting"
        )
    return GPUMathMeasurement(
        math_mode,
        str(profile.get("prediction_math_kernel") or profile.get("precision") or math_mode),
        profile.get("prediction_math_fallback_reason"),
    ), snapshot
