from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from .model_generation import model_generation_uses_v5o
from .runtime_inputs import RuntimeInputContext
from .rwkv_performance_modes import (
    native_prediction_mode,
    prediction_mode_label,
    prediction_mode_math,
)
from .rwkv_processing import runtime_prediction_row_input
from .timestamp_context import timestamp_ms_from_seconds
from .v5o_review_rows import V5oInputContext, v5o_live_candidate_input_row
from .vendor_bootstrap import require_rwkv_live_candidate_seed

BENCHMARK_TARGET_RETRIEVABILITY = 0.9
BENCHMARK_LIVE_ORDER = "retrievability_ascending"


class LivePredictionBenchmarkSessionProtocol(Protocol):
    """Small steady-state surface consumed by prediction speed tests."""

    def run_cycle(self) -> int: ...

    def close(self) -> None: ...


class LivePredictionBenchmarkFactory(Protocol):
    def __call__(
        self,
        rows: Sequence[dict[str, Any]],
        mode: str,
        batch_size: int | None,
        refresh_limit: int,
    ) -> LivePredictionBenchmarkSessionProtocol: ...


def validate_live_prediction_refresh_count(
    refreshed_count: int,
    expected_count: int,
    mode: str,
    *,
    warmup: bool,
) -> None:
    actual = int(refreshed_count)
    expected = int(expected_count)
    if actual != expected:
        phase = " warm-up" if warmup else ""
        mode_label = prediction_mode_label(mode)
        raise RuntimeError(
            f"{mode_label} Live Session{phase} refreshed {actual} cards instead of {expected}."
        )


class LivePredictionBenchmarkSession:
    """Own one initialized native Live Session and run repeated review cycles.

    Construction performs RWKV-SRS's complete initial prediction and index build.
    Each later :meth:`run_cycle` processes one representative answer, requeues that
    candidate, and refreshes the native index. Closing the adapter undoes every
    synthetic answer before releasing the native session, so benchmark cells do
    not contaminate one another or the caller's in-memory checkpoint scope.
    """

    def __init__(
        self,
        native_session: Any,
        *,
        rows: Sequence[dict[str, Any]],
        target_timestamp_seconds: float,
        target_day_offset: float,
        v5o_context: V5oInputContext | None = None,
    ) -> None:
        self._native_session = native_session
        self._v5o_context = v5o_context
        self._rows_by_card_id = {int(row["card_id"]): dict(row) for row in rows}
        self._candidate_card_ids = tuple(self._rows_by_card_id)
        self._target_timestamp_seconds = float(target_timestamp_seconds)
        self._target_day_offset = float(target_day_offset)
        self._rollover_phase_ms = (
            int(rows[0]["rollover_phase_ms"]) if rows and "rollover_phase_ms" in rows[0] else None
        )
        self._initial_target_timestamp_ms = (
            int(rows[0]["target_timestamp_ms"]) if self._rollover_phase_ms is not None else None
        )
        self._next_card_id = self._selected_card_id(getattr(native_session, "initial_result", None))
        self._last_answer_timestamp_by_card: dict[int, int | float] = {}
        self._cycle_count = 0
        self._undo_count = 0
        self._closed = False

    def run_cycle(self) -> int:
        if self._closed:
            raise RuntimeError("live prediction benchmark session is closed")
        self._cycle_count += 1
        target_timestamp_ms = (
            self._initial_target_timestamp_ms + self._cycle_count * 30_000
            if self._initial_target_timestamp_ms is not None
            else None
        )
        target_timestamp = (
            target_timestamp_ms / 1_000.0
            if target_timestamp_ms is not None
            else self._target_timestamp_seconds + self._cycle_count * 30.0
        )
        card_id = int(self._next_card_id)
        row = self._answer_row(
            card_id,
            target_timestamp=target_timestamp,
            target_timestamp_ms=target_timestamp_ms,
        )
        process = getattr(self._native_session, "benchmark_process_answer", None)
        if callable(process):
            process(row)
        else:
            self._native_session.process_answer(
                row,
                requeue_after_prediction=True,
                return_curves=False,
            )
        self._undo_count += 1
        self._last_answer_timestamp_by_card[card_id] = (
            target_timestamp_ms if target_timestamp_ms is not None else target_timestamp
        )
        refresh_kwargs: dict[str, Any] = {}
        if self._v5o_context is None:
            refresh_kwargs["target_day_offset"] = self._target_day_offset
        if self._rollover_phase_ms is not None:
            refresh_kwargs["rollover_phase_ms"] = self._rollover_phase_ms
            refresh_kwargs["target_timestamp_ms"] = target_timestamp_ms
        else:
            refresh_kwargs["target_timestamp_seconds"] = target_timestamp
        result = self._native_session.refresh(
            select_limit=2,
            exclude_card_ids=(),
            retention_extra=0.0,
            **refresh_kwargs,
        )
        self._next_card_id = self._selected_card_id(result, fallback=card_id)
        return int(result.refreshed_count)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        cleanup_error: BaseException | None = None
        try:
            for _ in range(self._undo_count):
                try:
                    self._native_session.undo_last_process()
                except BaseException as exc:
                    cleanup_error = exc
                    break
        finally:
            self._native_session.close()
        if cleanup_error is not None:
            raise RuntimeError(
                "Could not restore the RWKV state after the Live Session speed test."
            ) from cleanup_error

    def _answer_row(
        self,
        card_id: int,
        *,
        target_timestamp: float,
        target_timestamp_ms: int | None,
    ) -> dict[str, Any]:
        row = dict(self._rows_by_card_id[int(card_id)])
        previous_timestamp = self._last_answer_timestamp_by_card.get(int(card_id))
        review_timestamp_ms = (
            int(target_timestamp_ms)
            if target_timestamp_ms is not None
            else int(target_timestamp * 1_000.0)
        )
        if self._v5o_context is not None:
            return self._v5o_context.process_row(
                {
                    "card_id": card_id,
                    "review_id": review_timestamp_ms,
                    "duration_ms": 1000,
                    "source_ordinal": review_timestamp_ms,
                    "is_original_first_review": False,
                    "rating": 3,
                }
            )
        if previous_timestamp is None:
            row["elapsed_seconds"] = max(
                0.0,
                float(row.get("elapsed_seconds", 0.0))
                + (target_timestamp - self._target_timestamp_seconds),
            )
        else:
            row["elapsed_seconds"] = max(0.0, target_timestamp - previous_timestamp)
            row["elapsed_days"] = 0
        duration_ms = 1_000
        row.update(
            review_id=review_timestamp_ms,
            day_offset=int(self._target_day_offset),
            rating=3,
            button_chosen=3,
            duration=duration_ms,
            taken_millis=duration_ms,
            state=2,
            review_kind=1,
            interval=1,
            last_interval=1,
            ease_factor=2_500,
        )
        return row

    def _selected_card_id(self, result: Any, *, fallback: int | None = None) -> int:
        selected = getattr(result, "selected", ()) if result is not None else ()
        if selected:
            card_id = int(selected[0].card_id)
            if card_id in self._rows_by_card_id:
                return card_id
        if fallback is not None and int(fallback) in self._rows_by_card_id:
            return int(fallback)
        return int(self._candidate_card_ids[0])

    def __enter__(self) -> LivePredictionBenchmarkSession:
        if self._closed:
            raise RuntimeError("live prediction benchmark session is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def open_live_prediction_benchmark_session(
    runtime_session: Any,
    rows: Sequence[dict[str, Any]],
    *,
    mode: str,
    batch_size: int | None,
    refresh_limit: int,
    target_timestamp_seconds: float,
    input_context: RuntimeInputContext | None = None,
    seed_factory: Callable[..., Any] | None = None,
) -> LivePredictionBenchmarkSession:
    """Initialize a strict native session for warmed refresh measurements.

    A checkpoint lease exposes ``benchmark_predict_many_live_session()`` so GPU
    preflight or execution failures remain visible. A caller-owned disposable
    RWKV-SRS runtime already has strict behavior when ``fallback_mode`` is omitted,
    so Guided Setup can use its public ``predict_many_live_session()`` directly.
    """

    generation = (
        "v5o"
        if isinstance(input_context, V5oInputContext)
        else getattr(
            runtime_session,
            "model_generation",
            getattr(runtime_session, "model_version", "v1"),
        )
    )
    uses_v5o = model_generation_uses_v5o(generation)
    if uses_v5o and not isinstance(input_context, V5oInputContext):
        input_context = getattr(runtime_session, "input_context", None)
        if not isinstance(input_context, V5oInputContext):
            raise ValueError("V5 Live benchmarks require the retained-card input context")
    materialized = tuple(
        dict(
            runtime_prediction_row_input(
                runtime_session,
                row,
                input_context=input_context,
                model_generation=generation,
            )
        )
        if uses_v5o
        else dict(row)
        for row in rows
    )
    if not materialized:
        raise ValueError("At least one prediction row is required for a live benchmark.")
    initial_target_timestamp_ms: int | None = None
    initial_rollover_phase_ms: int | None = None
    aligned_target_timestamp = float(target_timestamp_seconds)
    if uses_v5o:
        target_timestamps = {int(row["target_timestamp_ms"]) for row in materialized}
        if len(target_timestamps) != 1:
            raise ValueError("V5o live benchmark rows must share one exact target timestamp.")
        initial_target_timestamp_ms = target_timestamps.pop()
        if timestamp_ms_from_seconds(target_timestamp_seconds) != initial_target_timestamp_ms:
            raise ValueError("V5o live benchmark rows do not match the requested target timestamp.")
        rollover_phases = {int(row["rollover_phase_ms"]) for row in materialized}
        if len(rollover_phases) != 1:
            raise ValueError("V5o live benchmark rows must share one rollover phase.")
        initial_rollover_phase_ms = rollover_phases.pop()
        aligned_target_timestamp = initial_target_timestamp_ms / 1_000.0
    normalized_limit = int(refresh_limit)
    if normalized_limit < 1 or normalized_limit > len(materialized):
        raise ValueError("Live benchmark refresh_limit must fit the candidate rows.")

    target_day_offset = 0.0 if uses_v5o else float(materialized[0]["day_offset"])
    if not uses_v5o and any(
        float(row["day_offset"]) != target_day_offset for row in materialized[1:]
    ):
        raise ValueError("Live benchmark prediction rows must share one target day.")

    create_seed = seed_factory or require_rwkv_live_candidate_seed()
    seeds = tuple(
        create_seed(
            row=(
                v5o_live_candidate_input_row(row)
                if uses_v5o
                else row
            ),
            target_retrievability=BENCHMARK_TARGET_RETRIEVABILITY,
            intraday_target_retrievability=BENCHMARK_TARGET_RETRIEVABILITY,
            tie_breaker=index,
        )
        for index, row in enumerate(materialized)
    )
    factory = getattr(
        runtime_session,
        "benchmark_predict_many_live_session",
        None,
    )
    if not callable(factory):
        factory = getattr(runtime_session, "predict_many_live_session", None)
        # Disposable Setup runtimes bypass the checkpoint lease's strict GPU
        # preflight. Check actual arithmetic before any native session/timer so
        # a kernel fallback cannot be reported as GPU (Full FP16).
        if native_prediction_mode(mode) == "gpu":
            requested = prediction_mode_math(mode)
            if getattr(runtime_session, "gpu_prediction_math", None) != requested:
                raise RuntimeError("The benchmark runtime did not honor its GPU arithmetic choice.")
            profile = runtime_session.initialize_gpu("predict")
            if profile.get("prediction_math") != requested:
                raise RuntimeError(
                    f"{prediction_mode_label(mode)} is unavailable; refusing to benchmark "
                    "a different kernel under the requested label. "
                    + str(profile.get("prediction_math_fallback_reason") or "Kernel unavailable.")
                )
    if not callable(factory):
        raise RuntimeError(
            "The bundled RWKV-SRS runtime does not provide predict_many_live_session()."
        )

    target_kwargs: dict[str, Any] = (
        {
            "initial_target_timestamp_ms": initial_target_timestamp_ms,
            "initial_rollover_phase_ms": initial_rollover_phase_ms,
        }
        if uses_v5o
        else {"initial_target_timestamp_seconds": aligned_target_timestamp}
    )
    if not uses_v5o:
        target_kwargs["initial_target_day_offset"] = target_day_offset
    native_session = factory(
        seeds,
        order=BENCHMARK_LIVE_ORDER,
        mode=native_prediction_mode(mode),
        batch_size=batch_size,
        refresh_limit=normalized_limit,
        profiling=False,
        initial_select_limit=2,
        **target_kwargs,
    )
    return LivePredictionBenchmarkSession(
        native_session,
        rows=materialized,
        target_timestamp_seconds=aligned_target_timestamp,
        target_day_offset=target_day_offset,
        v5o_context=input_context if uses_v5o else None,
    )
