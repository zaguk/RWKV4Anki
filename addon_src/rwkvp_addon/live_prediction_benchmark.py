from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

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
        mode_label = "GPU" if str(mode) == "gpu" else str(mode).title()
        raise RuntimeError(
            f"{mode_label} Live Session{phase} refreshed "
            f"{actual} cards instead of {expected}."
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
    ) -> None:
        self._native_session = native_session
        self._rows_by_card_id = {int(row["card_id"]): dict(row) for row in rows}
        self._candidate_card_ids = tuple(self._rows_by_card_id)
        self._target_timestamp_seconds = float(target_timestamp_seconds)
        self._target_day_offset = float(target_day_offset)
        self._next_card_id = self._selected_card_id(
            getattr(native_session, "initial_result", None)
        )
        self._last_answer_timestamp_by_card: dict[int, float] = {}
        self._cycle_count = 0
        self._undo_count = 0
        self._closed = False

    def run_cycle(self) -> int:
        if self._closed:
            raise RuntimeError("live prediction benchmark session is closed")
        self._cycle_count += 1
        target_timestamp = self._target_timestamp_seconds + self._cycle_count * 30.0
        card_id = int(self._next_card_id)
        row = self._answer_row(card_id, target_timestamp=target_timestamp)
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
        self._last_answer_timestamp_by_card[card_id] = target_timestamp
        result = self._native_session.refresh(
            target_timestamp_seconds=target_timestamp,
            target_day_offset=self._target_day_offset,
            select_limit=2,
            exclude_card_ids=(),
            retention_extra=0.0,
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
    ) -> dict[str, Any]:
        row = dict(self._rows_by_card_id[int(card_id)])
        previous_timestamp = self._last_answer_timestamp_by_card.get(int(card_id))
        if previous_timestamp is None:
            row["elapsed_seconds"] = max(
                0.0,
                float(row.get("elapsed_seconds", 0.0))
                + (target_timestamp - self._target_timestamp_seconds),
            )
        else:
            row["elapsed_seconds"] = max(0.0, target_timestamp - previous_timestamp)
            row["elapsed_days"] = 0
        row.update(
            review_id=int(target_timestamp * 1000.0),
            day_offset=int(self._target_day_offset),
            rating=3,
            button_chosen=3,
            duration=1_000.0,
            taken_millis=1_000,
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
    seed_factory: Callable[..., Any] | None = None,
) -> LivePredictionBenchmarkSession:
    """Initialize a strict native session for warmed refresh measurements.

    A checkpoint lease exposes ``benchmark_predict_many_live_session()`` so GPU
    preflight or execution failures remain visible. A caller-owned disposable
    RWKV-SRS runtime already has strict behavior when ``fallback_mode`` is omitted,
    so Guided Setup can use its public ``predict_many_live_session()`` directly.
    """

    materialized = tuple(dict(row) for row in rows)
    if not materialized:
        raise ValueError("At least one prediction row is required for a live benchmark.")
    normalized_limit = int(refresh_limit)
    if normalized_limit < 1 or normalized_limit > len(materialized):
        raise ValueError("Live benchmark refresh_limit must fit the candidate rows.")

    target_day_offset = float(materialized[0]["day_offset"])
    if any(float(row["day_offset"]) != target_day_offset for row in materialized[1:]):
        raise ValueError("Live benchmark prediction rows must share one target day.")

    create_seed = seed_factory or require_rwkv_live_candidate_seed()
    seeds = tuple(
        create_seed(
            row=row,
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
    if not callable(factory):
        raise RuntimeError(
            "The bundled RWKV-SRS runtime does not provide "
            "predict_many_live_session()."
        )

    native_session = factory(
        seeds,
        initial_target_timestamp_seconds=float(target_timestamp_seconds),
        initial_target_day_offset=target_day_offset,
        order=BENCHMARK_LIVE_ORDER,
        mode=str(mode),
        batch_size=batch_size,
        refresh_limit=normalized_limit,
        profiling=False,
        initial_select_limit=2,
    )
    return LivePredictionBenchmarkSession(
        native_session,
        rows=materialized,
        target_timestamp_seconds=float(target_timestamp_seconds),
        target_day_offset=target_day_offset,
    )
