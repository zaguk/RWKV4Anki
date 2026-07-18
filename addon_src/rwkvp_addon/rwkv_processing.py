from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .compact_review_data import native_review_batch_for_rows
from .constants import (
    DEFAULT_MODEL_ID,
    LIVE_REVIEW_RUST_UNDO_LIMIT,
    PREDICT_BATCH_SIZE,
    RWKV_SRS_TORCH_SEED,
)
from .prediction_cache import PredictionRecordSet, prediction_record
from .progress import ProgressReporter
from .rwkv_backend import normalize_rwkv_backend, rwkv_backend_scope
from .rwkv_performance_modes import (
    DEFAULT_PROCESS_MANY_MODE,
    cpu_mode_for_process_many_mode,
    process_many_uses_gpu,
)
from .vendor_bootstrap import (
    require_rwkv_probability,
    require_rwkv_review_batch,
    require_rwkv_srs,
    seed_rwkv_srs_torch,
)

PROCESS_MANY_CHUNK_SIZE = 10_000


def new_rwkvp_runtime(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    backend: str | None = None,
    process_many_mode: str = DEFAULT_PROCESS_MANY_MODE,
):
    backend_name = normalize_rwkv_backend(backend)
    with rwkv_backend_scope(backend_name):
        RWKV_SRS = require_rwkv_srs()
        seed_rwkv_srs_torch(
            RWKV_SRS_TORCH_SEED,
            required=backend_name == "python",
        )
        runtime = RWKV_SRS(
            model=model_id,
            **_runtime_constructor_kwargs(
                backend_name,
                process_many_mode=process_many_mode,
            ),
        )
        return runtime


def _runtime_constructor_kwargs(
    backend_name: str,
    *,
    process_many_mode: str = DEFAULT_PROCESS_MANY_MODE,
) -> dict[str, int | bool | str]:
    return (
        {
            "undo_limit": LIVE_REVIEW_RUST_UNDO_LIMIT,
            "runtime_owner_thread": True,
            "cpu_mode": cpu_mode_for_process_many_mode(process_many_mode),
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
) -> list[dict[str, Any]]:
    records = records if records is not None else []
    total = len(rows)
    processed = 0
    for chunk in _process_many_chunks(rows):
        predictions = _runtime_process_many(runtime, chunk, return_curves=False)
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
) -> PredictionRecordSet:
    record_set = record_set if record_set is not None else PredictionRecordSet.empty()
    if calculate_curves and latest_curves_by_card is None:
        latest_curves_by_card = {}
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
) -> None:
    processed = 0
    total = len(rows)
    for chunk in _process_many_chunks(rows):
        if not calculate_curves:
            predictions = _runtime_process_many(
                runtime,
                chunk,
                return_curves=False,
                use_gpu=use_gpu,
            )
            record_set.immediate_predictions.extend(float(value) for value in predictions)
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
) -> list[Any]:
    # Scalar process() always calculates a curve in RWKV-SRS. Keep it for the
    # curve-producing one-row path, but route no-curve tails through
    # process_many(return_curves=False) so the General setting remains a real
    # computation guarantee even when only one new review is pending.
    if len(rows) == 1 and return_curves and hasattr(runtime, "process"):
        prediction, curve = runtime.process(rows[0])
        if return_curves:
            return [(prediction, curve)]
        return [prediction]
    if hasattr(runtime, "process_many"):
        kwargs: dict[str, Any] = {
            "return_curves": return_curves,
            "batch_size": PROCESS_MANY_CHUNK_SIZE,
        }
        if use_gpu:
            kwargs["mode"] = "gpu"
            # RWKV-SRS can commit an ordered prefix before a recoverable GPU
            # processing failure. Let it materialize that prefix and process
            # only the untouched suffix with the CPU mode used to construct
            # this runtime. For add-on GPU builds that CPU mode is Fast.
            kwargs["fallback_mode"] = cpu_mode_for_process_many_mode("gpu")
        process_input: object = rows
        if bool(getattr(runtime, "supports_native_review_batch_consistency", False)):
            ReviewBatch = require_rwkv_review_batch(backend="rust")
            process_input = native_review_batch_for_rows(ReviewBatch, rows)
        return list(runtime.process_many(process_input, **kwargs))
    results = [runtime.process(row) for row in rows]
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


def _call_predict_many(
    predict_many: Callable[..., Iterable[float]],
    rows: list[dict[str, Any]],
    *,
    batch_size: int | None,
) -> list[float]:
    effective_batch_size = None if batch_size is None else int(batch_size)
    return list(predict_many(rows, batch_size=effective_batch_size))
