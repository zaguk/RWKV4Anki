from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from .rwkv_performance_modes import PROCESS_MANY_MODES
from .rwkv_processing import PROCESS_MANY_CHUNK_SIZE, new_rwkvp_runtime
from .rwkv_runtime_resources import release_runtime_resources


def benchmark_process_many_rows(
    rows: Sequence[dict[str, Any]],
    *,
    model_id: str,
    mode: str,
    return_curves: bool = True,
    warm_gpu_before_timing: bool = False,
    runtime_factory: Callable[..., Any] = new_rwkvp_runtime,
    release_runtime: Callable[[Any], None] = release_runtime_resources,
    clock: Callable[[], float] = time.perf_counter,
) -> float:
    """Time one fresh Fast or GPU state build inside the current process.

    Runtime/model construction remains outside the timer. GPU setup can also be
    moved outside the timer for Guided Setup, whose result estimates recurring
    checkpoint processing after hardware initialization. Every call owns and
    releases a fresh runtime so curve and mode comparisons do not share state.
    """

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in PROCESS_MANY_MODES:
        raise ValueError(f"Unsupported process_many speed-test mode: {mode!r}.")
    if not rows:
        raise ValueError("At least one review row is required for the process-many test.")

    runtime = runtime_factory(
        model_id=str(model_id),
        backend="rust",
        process_many_mode=normalized_mode,
    )
    try:
        warm_gpu_for_call = bool(warm_gpu_before_timing and normalized_mode == "gpu")
        if warm_gpu_for_call:
            _require_process_gpu(runtime)

        started = clock()
        kwargs: dict[str, Any] = {
            "batch_size": PROCESS_MANY_CHUNK_SIZE,
            "return_curves": bool(return_curves),
        }
        if normalized_mode == "gpu":
            if not warm_gpu_for_call:
                _require_process_gpu(runtime)
            kwargs["mode"] = "gpu"

        results = runtime.process_many(rows, **kwargs)
        if normalized_mode == "gpu":
            release_gpu = getattr(runtime, "release_gpu", None)
            if not callable(release_gpu):
                raise RuntimeError("GPU process_many() has no release_gpu() API.")
            release_gpu()
        elapsed = float(clock()) - float(started)
        if elapsed < 0:
            raise RuntimeError("The process-many speed-test clock moved backwards.")
        if len(results) != len(rows):
            raise RuntimeError(
                f"process_many() returned {len(results)} results for {len(rows)} reviews."
            )
        return elapsed
    finally:
        release_runtime(runtime)


def _require_process_gpu(runtime: Any) -> None:
    check = getattr(runtime, "gpu_available", None)
    if not callable(check) or not bool(check("process")):
        raise RuntimeError("GPU process_many() is unavailable in this runtime.")


__all__ = ["benchmark_process_many_rows"]
