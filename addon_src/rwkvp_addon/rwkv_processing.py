from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .compact_review_data import native_review_batch_for_rows
from .constants import (
    DEFAULT_MODEL_ID,
    LIVE_REVIEW_RUST_UNDO_LIMIT,
    PREDICT_BATCH_SIZE,
    RWKV_SRS_TORCH_SEED,
)
from .model_generation import (
    model_generation_uses_exact_timestamps,
    model_generation_uses_v5o,
    normalize_model_generation,
)
from .model_storage import bundled_model_path
from .prediction_cache import PredictionRecordSet, prediction_record
from .progress import ProgressReporter
from .runtime_inputs import RuntimeInputContext
from .rwkv_backend import normalize_rwkv_backend, rwkv_backend_scope
from .rwkv_performance_modes import (
    DEFAULT_PROCESS_MANY_MODE,
    cpu_mode_for_process_many_mode,
    process_many_uses_gpu,
)
from .rwkv_runtime_resources import release_runtime_resources
from .v5o_review_rows import (
    V5oInputContext,
    v5o_answer_input_row,
    v5o_live_candidate_input_row,
    v5o_prediction_input_columns,
    v5o_prediction_input_row,
    v5o_process_input_columns,
)
from .vendor_bootstrap import (
    require_rwkv_probability,
    require_rwkv_review_batch,
    require_rwkv_srs,
    require_rwkv_v5o_review_batch,
    seed_rwkv_srs_torch,
)

PROCESS_MANY_CHUNK_SIZE = 10_000
# V5's causal pipeline uses parallel preparation. Bound the shared policy
# for both supported generations to avoid oversubscription.
PROCESS_MANY_THREADS = min(8, os.cpu_count() or 1)


def runtime_uses_v5o(runtime: Any, *, model_generation: str | None = None) -> bool:
    return model_generation_uses_v5o(getattr(runtime, "model_version", None) or model_generation)


def runtime_uses_exact_timestamps(runtime: Any, *, model_generation: str | None = None) -> bool:
    return model_generation_uses_exact_timestamps(
        getattr(runtime, "model_version", None) or model_generation
    )


def runtime_process_row_input(
    runtime: Any,
    row: Mapping[str, Any],
    *,
    input_context: RuntimeInputContext | None = None,
    model_generation: str | None = None,
) -> Mapping[str, Any]:
    """Return one model-version-correct answer row at the native boundary."""
    if runtime_uses_v5o(runtime, model_generation=model_generation):
        context = _v5o_context(input_context)
        return context.process_row(row) if context is not None else v5o_answer_input_row(row)
    return row


def runtime_prediction_row_input(
    runtime: Any,
    row: Mapping[str, Any],
    *,
    input_context: RuntimeInputContext | None = None,
    model_generation: str | None = None,
) -> Mapping[str, Any]:
    """Return one model-version-correct immutable prediction query."""
    if runtime_uses_v5o(runtime, model_generation=model_generation):
        context = _v5o_context(input_context)
        return context.prediction_row(row) if context is not None else v5o_prediction_input_row(row)
    return row


def runtime_live_candidate_row_input(
    runtime: Any,
    row: Mapping[str, Any],
    *,
    input_context: RuntimeInputContext | None = None,
    model_generation: str | None = None,
) -> Mapping[str, Any]:
    """Return one model-version-correct native Live candidate seed row."""
    if runtime_uses_v5o(runtime, model_generation=model_generation):
        return v5o_live_candidate_input_row(row)
    return row


def runtime_process_many_input(
    runtime: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    input_context: RuntimeInputContext | None = None,
    model_generation: str | None = None,
) -> Any:
    """Build the versioned immutable native transport when one is available."""
    if runtime_uses_v5o(runtime, model_generation=model_generation):
        batch_type = require_rwkv_v5o_review_batch(backend="rust")
        return batch_type.from_columns(
            kind="answer",
            columns=v5o_process_input_columns(rows, context=_v5o_context(input_context)),
        )
    if bool(getattr(runtime, "supports_native_review_batch_consistency", False)):
        ReviewBatch = require_rwkv_review_batch(backend="rust")
        return native_review_batch_for_rows(ReviewBatch, rows)
    return rows


def runtime_predict_many_input(
    runtime: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    input_context: RuntimeInputContext | None = None,
    model_generation: str | None = None,
) -> Any:
    """Build a V5 query batch or preserve the V1 mappings."""
    if runtime_uses_v5o(runtime, model_generation=model_generation):
        batch_type = require_rwkv_v5o_review_batch(backend="rust")
        return batch_type.from_columns(
            kind="query",
            columns=v5o_prediction_input_columns(rows, context=_v5o_context(input_context)),
        )
    return rows


def _v5o_context(context: RuntimeInputContext | None) -> V5oInputContext | None:
    # V5 requires retained catalog and source timing facts from its own context.
    if context is not None and not isinstance(context, V5oInputContext):
        raise ValueError("V5o requires its own retained-history input context")
    return context


def new_rwkvp_runtime(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    model_generation: str | None = None,
    backend: str | None = None,
    process_many_mode: str = DEFAULT_PROCESS_MANY_MODE,
    state_precision: str = "fp32",
    gpu_prediction_math: str = "fp32",
):
    backend_name = normalize_rwkv_backend(backend)
    if model_generation_uses_v5o(model_generation) and backend_name != "rust":
        raise ValueError("V5 requires the Rust runtime.")
    with rwkv_backend_scope(backend_name):
        RWKV_SRS = require_rwkv_srs()
        seed_rwkv_srs_torch(
            RWKV_SRS_TORCH_SEED,
            required=backend_name == "python",
        )
        model: str | Path = model_id
        if backend_name == "rust":
            bundled = bundled_model_path(
                model_id,
                model_generation=model_generation,
            )
            if bundled is not None:
                # RWKV-SRS named lookup is intentionally flat. The add-on owns
                # a generation-separated layout, so pass its exact artifact.
                model = bundled
        runtime = RWKV_SRS(
            model=model,
            **_runtime_constructor_kwargs(
                backend_name,
                process_many_mode=process_many_mode,
                state_precision=state_precision,
                gpu_prediction_math=gpu_prediction_math,
            ),
        )
        version = getattr(runtime, "model_version", None)
        if (
            model_generation is not None
            and version is not None
            and (str(version) != normalize_model_generation(model_generation))
        ):
            release_runtime_resources(runtime)
            raise ValueError("The selected model artifact belongs to a different RWKV generation.")
        return runtime


def _runtime_constructor_kwargs(
    backend_name: str,
    *,
    process_many_mode: str = DEFAULT_PROCESS_MANY_MODE,
    state_precision: str = "fp32",
    gpu_prediction_math: str = "fp32",
) -> dict[str, int | bool | str]:
    if gpu_prediction_math not in {"fp16", "fp32"}:
        raise ValueError("GPU prediction math must be fp16 or fp32.")
    if gpu_prediction_math == "fp16" and normalize_rwkv_backend(backend_name) != "rust":
        raise ValueError("FP16 GPU prediction math requires the Rust runtime.")
    if state_precision not in {"fp16", "fp32"}:
        raise ValueError("State precision must be fp16 or fp32.")
    if state_precision == "fp16" and normalize_rwkv_backend(backend_name) != "rust":
        raise ValueError("FP16 card/note state requires the Rust runtime.")
    return (
        {
            "undo_limit": LIVE_REVIEW_RUST_UNDO_LIMIT,
            "runtime_owner_thread": True,
            "cpu_mode": cpu_mode_for_process_many_mode(process_many_mode),
            "state_precision": state_precision,
            "gpu_prediction_math": gpu_prediction_math,
        }
        if normalize_rwkv_backend(backend_name) == "rust"
        else {}
    )


def process_review_rows(
    runtime,
    rows: Sequence[Mapping[str, Any]],
    progress: ProgressReporter,
    *,
    label: str,
    records: list[dict[str, Any]] | None = None,
    input_context: RuntimeInputContext | None = None,
) -> list[dict[str, Any]]:
    records = records if records is not None else []
    total = len(rows)
    processed = 0
    for chunk in _process_many_chunks(rows):
        _check_progress_cancelled(progress)
        predictions = _runtime_process_many(
            runtime,
            chunk,
            return_curves=False,
            input_context=input_context,
        )
        records.extend(
            prediction_record(row, prediction)
            for row, prediction in zip(chunk, predictions, strict=True)
        )
        processed += len(chunk)
        progress.update(processed, total, label)
    return records


def process_review_rows_with_predictions(
    runtime,
    rows: Sequence[Mapping[str, Any]],
    progress: ProgressReporter,
    *,
    label: str,
    record_set: PredictionRecordSet | None = None,
    latest_curves_by_card: dict[int, Any] | None = None,
    process_many_mode: str = DEFAULT_PROCESS_MANY_MODE,
    calculate_curves: bool = True,
    input_context: RuntimeInputContext | None = None,
) -> PredictionRecordSet:
    if calculate_curves and runtime_uses_exact_timestamps(runtime):
        raise ValueError("This RWKV generation supports Immediate predictions only")
    record_set = record_set if record_set is not None else PredictionRecordSet.empty()
    if calculate_curves and latest_curves_by_card is None:
        latest_curves_by_card = {}
    _check_progress_cancelled(progress)
    use_gpu = _begin_bulk_gpu_process(
        runtime,
        process_many_mode=process_many_mode,
        row_count=len(rows),
    )
    try:
        _process_rows_and_record_predictions(
            runtime,
            rows,
            progress,
            label=label,
            record_set=record_set,
            latest_curves_by_card=latest_curves_by_card,
            use_gpu=use_gpu,
            calculate_curves=bool(calculate_curves),
            input_context=input_context,
        )
    finally:
        if use_gpu:
            _finish_bulk_gpu_process(runtime)
    return record_set


def _process_rows_and_record_predictions(
    runtime,
    rows: Sequence[Mapping[str, Any]],
    progress: ProgressReporter,
    *,
    label: str,
    record_set: PredictionRecordSet,
    latest_curves_by_card: dict[int, Any] | None,
    use_gpu: bool,
    calculate_curves: bool,
    input_context: RuntimeInputContext | None,
) -> None:
    processed = 0
    total = len(rows)
    for chunk in _process_many_chunks(rows):
        _check_progress_cancelled(progress)
        if not calculate_curves:
            predictions = _runtime_process_many(
                runtime,
                chunk,
                return_curves=False,
                use_gpu=use_gpu,
                input_context=input_context,
            )
            record_set.immediate_predictions.extend(
                None if value is None else float(value) for value in predictions
            )
            processed += len(chunk)
            progress.update(processed, total, label)
            continue

        if latest_curves_by_card is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("Curve calculation requires a latest-curve map.")
        results = _runtime_process_many(
            runtime,
            chunk,
            return_curves=True,
            use_gpu=use_gpu,
            input_context=input_context,
        )
        chunk_predict_ahead: list[float | None] = [None] * len(chunk)
        prior_curves: list[Any] = []
        prior_curve_elapsed_seconds: list[float] = []
        prior_curve_indexes: list[int] = []
        for index, (row, result) in enumerate(zip(chunk, results, strict=True)):
            prediction, curve = result
            previous_curve = latest_curves_by_card.get(int(row["card_id"]))
            if previous_curve is not None:
                prior_curves.append(previous_curve)
                prior_curve_elapsed_seconds.append(float(row["elapsed_seconds"]))
                prior_curve_indexes.append(index)
            if curve is not None:
                curve = _detach_curve(curve)
                latest_curves_by_card[int(row["card_id"])] = curve
            record_set.immediate_predictions.append(float(prediction))
        if prior_curves:
            probabilities = _predict_curves_many(
                runtime,
                prior_curves,
                prior_curve_elapsed_seconds,
            )
            for index, probability in zip(
                prior_curve_indexes,
                probabilities,
                strict=True,
            ):
                chunk_predict_ahead[index] = probability
        record_set.predict_ahead_predictions.extend(chunk_predict_ahead)
        processed += len(chunk)
        progress.update(processed, total, label)


def _runtime_process_many(
    runtime,
    rows: Sequence[Mapping[str, Any]],
    *,
    return_curves: bool,
    use_gpu: bool = False,
    input_context: RuntimeInputContext | None = None,
    model_generation: str | None = None,
) -> list[Any]:
    if return_curves and runtime_uses_exact_timestamps(runtime, model_generation=model_generation):
        raise ValueError("This RWKV generation supports Immediate predictions only")
    # Scalar process() always calculates a curve in RWKV-SRS. Keep it for the
    # curve-producing one-row path, but route no-curve tails through
    # process_many(return_curves=False) so the General setting remains a real
    # computation guarantee even when only one new review is pending.
    if len(rows) == 1 and return_curves and hasattr(runtime, "process"):
        process_row = runtime_process_row_input(
            runtime,
            rows[0],
            input_context=input_context,
            model_generation=model_generation,
        )
        prediction, curve = runtime.process(process_row)
        if return_curves:
            return [(prediction, curve)]
        return [prediction]
    if hasattr(runtime, "process_many"):
        kwargs: dict[str, Any] = {
            "return_curves": return_curves,
            "batch_size": PROCESS_MANY_CHUNK_SIZE,
            "num_threads": PROCESS_MANY_THREADS,
        }
        if use_gpu:
            kwargs["mode"] = "gpu"
            # RWKV-SRS can commit an ordered prefix before a recoverable GPU
            # processing failure. Let it materialize that prefix and process
            # only the untouched suffix with the CPU mode used to construct
            # this runtime. For add-on GPU builds that CPU mode is Fast.
            kwargs["fallback_mode"] = cpu_mode_for_process_many_mode("gpu")
        process_input = runtime_process_many_input(
            runtime,
            rows,
            input_context=input_context,
            model_generation=model_generation,
        )
        results = list(runtime.process_many(process_input, **kwargs))
        _require_result_count("process_many", results, len(rows))
        return results
    results = [
        runtime.process(
            runtime_process_row_input(
                runtime,
                row,
                input_context=input_context,
                model_generation=model_generation,
            )
        )
        for row in rows
    ]
    if return_curves:
        return results
    return [prediction for prediction, _curve in results]


def _begin_bulk_gpu_process(
    runtime,
    *,
    process_many_mode: str,
    row_count: int,
) -> bool:
    if row_count < 2 or not process_many_uses_gpu(process_many_mode):
        return False
    check = getattr(runtime, "gpu_available", None)
    release = getattr(runtime, "release_gpu", None)
    if not callable(check) or not callable(release):
        return False
    try:
        return bool(check("process"))
    except (RuntimeError, TypeError, ValueError):
        return False


def _finish_bulk_gpu_process(runtime) -> None:
    release = getattr(runtime, "release_gpu", None)
    if not callable(release):
        raise RuntimeError("RWKV-SRS initialized GPU processing without a release_gpu() API.")
    # RWKV-SRS release_gpu() first materializes deferred process state into the
    # canonical CPU runtime, then frees the derived GPU cache.
    release()


def _process_many_chunks(
    rows: Sequence[Mapping[str, Any]],
) -> Iterable[Sequence[Mapping[str, Any]]]:
    chunk_size = max(1, int(PROCESS_MANY_CHUNK_SIZE))
    for start in range(0, len(rows), chunk_size):
        yield rows[start : start + chunk_size]


def _sequence_chunks(
    rows: Sequence[Mapping[str, Any]],
    chunk_size: int,
) -> Iterable[Sequence[Mapping[str, Any]]]:
    for start in range(0, len(rows), chunk_size):
        yield rows[start : start + chunk_size]


def _predict_curve(runtime, curve: Any, elapsed_seconds: Any) -> float:
    if hasattr(runtime, "get_probability"):
        return float(runtime.get_probability(curve, elapsed_seconds))
    if hasattr(runtime, "predict_curve"):
        return float(runtime.predict_curve(curve, elapsed_seconds))
    return float(require_rwkv_probability()(curve, elapsed_seconds))


def _predict_curves_many(
    runtime,
    curves: list[Any],
    elapsed_seconds: list[float],
) -> list[float]:
    get_probability_many = getattr(runtime, "get_probability_many", None)
    if callable(get_probability_many):
        probabilities = [float(value) for value in get_probability_many(curves, elapsed_seconds)]
        if len(probabilities) != len(curves):
            raise RuntimeError(
                "RWKV-SRS get_probability_many() returned "
                f"{len(probabilities)} probabilities for {len(curves)} curves."
            )
        return probabilities
    return [
        _predict_curve(runtime, curve, elapsed)
        for curve, elapsed in zip(curves, elapsed_seconds, strict=True)
    ]


def _detach_curve(curve: Any) -> Any:
    if isinstance(curve, tuple):
        return tuple(_detach_curve(value) for value in curve)
    if hasattr(curve, "detach"):
        return curve.detach().cpu()
    return curve


def predict_many_batched(
    predict_many: Callable[..., Iterable[float]],
    rows: list[dict[str, Any]],
    progress: ProgressReporter,
    *,
    label: str,
    batch_size: int | None = None,
    chunk_size: int | None = None,
) -> list[float]:
    predictions: list[float] = []
    total = len(rows)
    effective_chunk_size = max(
        1,
        int(chunk_size or batch_size or PREDICT_BATCH_SIZE),
    )
    for start in range(0, total, effective_chunk_size):
        _check_progress_cancelled(progress)
        chunk = rows[start : start + effective_chunk_size]
        predictions.extend(
            _call_predict_many(
                predict_many,
                chunk,
                batch_size=batch_size,
            )
        )
        progress.update(min(start + len(chunk), total), total, label)
    return predictions


def _require_result_count(operation: str, results: Sequence[Any], expected: int) -> None:
    if len(results) != expected:
        raise RuntimeError(
            f"RWKV-SRS {operation}() returned {len(results)} results for {expected} input rows."
        )


def _check_progress_cancelled(progress: ProgressReporter) -> None:
    check_cancelled = getattr(progress, "check_cancelled", None)
    if callable(check_cancelled):
        check_cancelled()


def _call_predict_many(
    predict_many: Callable[..., Iterable[float]],
    rows: list[dict[str, Any]],
    *,
    batch_size: int | None,
) -> list[float]:
    effective_batch_size = None if batch_size is None else int(batch_size)
    results = list(predict_many(rows, batch_size=effective_batch_size))
    _require_result_count("predict_many", results, len(rows))
    return results
