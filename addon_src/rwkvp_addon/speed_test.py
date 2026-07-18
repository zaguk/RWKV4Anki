from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .live_prediction_benchmark import (
    LivePredictionBenchmarkFactory,
    validate_live_prediction_refresh_count,
)
from .rwkv_performance_modes import PREDICT_MANY_MODES, PROCESS_MANY_MODES

PREDICT_MANY_SPEED_TEST_SIZES = (256, 512, 1024, 2048, 4096, 8192)
PREDICT_MANY_SPEED_TEST_REPETITIONS = 5
PROCESS_MANY_SPEED_TEST_MAX_REVIEWS = 60_000
CURVE_SPEED_TEST_MAX_REVIEWS = 10_000
CURVE_SPEED_TEST_REPETITIONS = 3
LIVE_PREDICTION_SPEED_TEST_REPETITIONS = 5
USABLE_SPEED_TEST_CHECKPOINT_STATUSES = frozenset({"valid", "partial"})


class SpeedTestProgress(Protocol):
    def update(self, current: int, total: int, text: str = "") -> None: ...

    def check_cancelled(self) -> None: ...


class SpeedTestCheckpointManager(Protocol):
    @property
    def has_checkpoint(self) -> bool: ...

    def status(self) -> str: ...


@dataclass(frozen=True)
class PredictManySpeedMeasurement:
    mode: str
    card_count: int
    durations_seconds: tuple[float, ...]
    batch_size: int | None

    @property
    def average_seconds(self) -> float:
        return sum(self.durations_seconds) / len(self.durations_seconds)

    @property
    def cards_per_second(self) -> float:
        average = self.average_seconds
        return math.inf if average == 0 else self.card_count / average


@dataclass(frozen=True)
class PredictManySpeedTestResult:
    model_id: str
    collection_card_count: int
    eligible_card_count: int
    repetitions: int
    measurements: tuple[PredictManySpeedMeasurement, ...]

    @property
    def card_counts(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(item.card_count for item in self.measurements))

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.mode for item in self.measurements))

    def measurement(self, card_count: int, mode: str) -> PredictManySpeedMeasurement:
        for item in self.measurements:
            if item.card_count == int(card_count) and item.mode == str(mode):
                return item
        raise KeyError((card_count, mode))


@dataclass(frozen=True)
class ProcessManySpeedMeasurement:
    mode: str
    review_count: int
    duration_seconds: float

    @property
    def reviews_per_minute(self) -> float:
        return (
            math.inf
            if self.duration_seconds == 0
            else self.review_count * 60.0 / self.duration_seconds
        )

    def estimated_seconds_for(self, review_count: int) -> float:
        if self.review_count <= 0:
            return 0.0
        return self.duration_seconds * max(0, int(review_count)) / self.review_count


@dataclass(frozen=True)
class ProcessManySpeedTestResult:
    model_id: str
    available_review_count: int
    review_count: int
    return_curves: bool
    measurements: tuple[ProcessManySpeedMeasurement, ...]

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(item.mode for item in self.measurements)

    def measurement(self, mode: str) -> ProcessManySpeedMeasurement:
        for item in self.measurements:
            if item.mode == str(mode):
                return item
        raise KeyError(mode)


@dataclass(frozen=True)
class ProcessManyCurveSpeedMeasurement:
    return_curves: bool
    review_count: int
    durations_seconds: tuple[float, ...]

    @property
    def average_seconds(self) -> float:
        return sum(self.durations_seconds) / len(self.durations_seconds)

    @property
    def reviews_per_minute(self) -> float:
        return (
            math.inf
            if self.average_seconds == 0
            else self.review_count * 60.0 / self.average_seconds
        )


@dataclass(frozen=True)
class ProcessManyCurveSpeedTestResult:
    model_id: str
    mode: str
    available_review_count: int
    review_count: int
    repetitions: int
    measurements: tuple[ProcessManyCurveSpeedMeasurement, ...]

    def measurement(self, return_curves: bool) -> ProcessManyCurveSpeedMeasurement:
        for item in self.measurements:
            if item.return_curves is bool(return_curves):
                return item
        raise KeyError(return_curves)


@dataclass(frozen=True)
class LivePredictionSpeedTestResult:
    model_id: str
    mode: str
    requested_card_count: int
    eligible_card_count: int
    card_count: int
    repetitions: int
    durations_seconds: tuple[float, ...]
    batch_size: int | None

    @property
    def average_seconds(self) -> float:
        return sum(self.durations_seconds) / len(self.durations_seconds)

    @property
    def cards_per_second(self) -> float:
        return (
            math.inf
            if self.average_seconds == 0
            else self.card_count / self.average_seconds
        )


def speed_test_checkpoint_is_usable(manager: SpeedTestCheckpointManager) -> bool:
    try:
        has_checkpoint = bool(manager.has_checkpoint)
        status = str(manager.status())
        runtime_loaded = bool(getattr(manager, "runtime_loaded", False))
    except Exception:
        return False
    return (
        has_checkpoint
        and status in USABLE_SPEED_TEST_CHECKPOINT_STATUSES
        and not runtime_loaded
    )


def capped_prediction_speed_test_sizes(eligible_card_count: int) -> tuple[int, ...]:
    eligible = max(0, int(eligible_card_count))
    if eligible == 0:
        return ()
    sizes: list[int] = []
    for requested in PREDICT_MANY_SPEED_TEST_SIZES:
        effective = min(requested, eligible)
        if effective not in sizes:
            sizes.append(effective)
        if effective == eligible:
            break
    return tuple(sizes)


def state_backed_prediction_card_ids(
    collection_card_ids: Iterable[int],
    processed_card_ids: Iterable[int],
) -> tuple[int, ...]:
    """Return current collection cards that have processed RWKV history.

    RWKV-SRS intentionally routes prediction rows without recurrent card state
    through its scalar Oracle fallback. Including those rows in an accelerated
    mode comparison would therefore benchmark the fallback instead of Fast or
    GPU inference.
    """

    processed = {int(card_id) for card_id in processed_card_ids}
    return tuple(
        card_id
        for card_id in sorted({int(card_id) for card_id in collection_card_ids})
        if card_id in processed
    )


def capped_process_speed_test_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return list(rows[:PROCESS_MANY_SPEED_TEST_MAX_REVIEWS])


def capped_curve_speed_test_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return list(rows[:CURVE_SPEED_TEST_MAX_REVIEWS])


def speed_test_modes(
    *,
    gpu_available: bool,
    process: bool = False,
) -> tuple[str, ...]:
    supported = PROCESS_MANY_MODES if process else PREDICT_MANY_MODES
    preferred_order = ("fast", "gpu") if process else ("oracle", "fast", "gpu")
    return tuple(
        mode
        for mode in preferred_order
        if mode in supported and (mode != "gpu" or gpu_available)
    )


def run_predict_many_speed_test(
    rows: Sequence[dict[str, Any]],
    *,
    collection_card_count: int,
    eligible_card_count: int,
    model_id: str,
    modes: Sequence[str],
    batch_sizes: Mapping[str, int | None],
    open_session: LivePredictionBenchmarkFactory,
    progress: SpeedTestProgress,
    repetitions: int = PREDICT_MANY_SPEED_TEST_REPETITIONS,
    clock: Callable[[], float] = time.perf_counter,
) -> PredictManySpeedTestResult:
    sizes = capped_prediction_speed_test_sizes(eligible_card_count)
    if not sizes:
        raise ValueError(
            "The collection has no cards with processed RWKV state to benchmark."
        )
    if len(rows) < max(sizes):
        raise ValueError(
            f"Only {len(rows)} prediction rows were prepared for a {max(sizes)}-card test."
        )
    repetitions = int(repetitions)
    if repetitions <= 0:
        raise ValueError("Predict-many speed-test repetitions must be positive.")
    normalized_modes = _validated_modes(modes, PREDICT_MANY_MODES)
    total = len(normalized_modes) * len(sizes) * (repetitions + 1)
    completed = 0
    measurements: list[PredictManySpeedMeasurement] = []

    for card_count in sizes:
        batch = list(rows[:card_count])
        for mode in normalized_modes:
            batch_size = _positive_optional_int(batch_sizes.get(mode))
            durations: list[float] = []
            label = (
                f"Live predictions - {_mode_label(mode)} - preparing and warming "
                f"{card_count:,} cards (untimed)"
            )
            progress.check_cancelled()
            progress.update(completed, total, label)
            session = open_session(batch, mode, batch_size, card_count)
            try:
                validate_live_prediction_refresh_count(
                    session.run_cycle(),
                    card_count,
                    mode,
                    warmup=True,
                )
                completed += 1
                progress.update(completed, total, label)
                for repetition in range(1, repetitions + 1):
                    progress.check_cancelled()
                    label = (
                        f"Live predictions - {_mode_label(mode)} - "
                        f"{card_count:,} cards (run {repetition}/{repetitions})"
                    )
                    progress.update(completed, total, label)
                    started = clock()
                    refreshed_count = session.run_cycle()
                    elapsed = _elapsed_seconds(started, clock())
                    validate_live_prediction_refresh_count(
                        refreshed_count,
                        card_count,
                        mode,
                        warmup=False,
                    )
                    durations.append(elapsed)
                    completed += 1
                    progress.update(completed, total, label)
            finally:
                session.close()
            measurements.append(
                PredictManySpeedMeasurement(
                    mode=mode,
                    card_count=card_count,
                    durations_seconds=tuple(durations),
                    batch_size=batch_size,
                )
            )

    return PredictManySpeedTestResult(
        model_id=str(model_id),
        collection_card_count=max(0, int(collection_card_count)),
        eligible_card_count=max(0, int(eligible_card_count)),
        repetitions=repetitions,
        measurements=tuple(measurements),
    )


def run_process_many_speed_test(
    *,
    review_count: int,
    available_review_count: int,
    model_id: str,
    return_curves: bool,
    modes: Sequence[str],
    run_mode: Callable[[str, int], float],
    progress: SpeedTestProgress,
) -> ProcessManySpeedTestResult:
    review_count = int(review_count)
    if review_count <= 0:
        raise ValueError("The collection has no processable reviews to benchmark.")
    normalized_modes = _validated_modes(modes, PROCESS_MANY_MODES)
    measurements: list[ProcessManySpeedMeasurement] = []
    total = len(normalized_modes)
    for index, mode in enumerate(normalized_modes):
        mode_review_count = review_count
        progress.check_cancelled()
        label = (
            f"Process many - {_mode_label(mode)} - "
            f"{mode_review_count:,} reviews - "
            f"{'with curves' if return_curves else 'without curves'} (one run)"
        )
        progress.update(index, total, label)
        elapsed = float(run_mode(mode, mode_review_count))
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError(
                f"{_mode_label(mode)} process_many() reported an invalid duration."
            )
        measurements.append(
            ProcessManySpeedMeasurement(
                mode=mode,
                review_count=mode_review_count,
                duration_seconds=elapsed,
            )
        )
        progress.update(index + 1, total, label)

    return ProcessManySpeedTestResult(
        model_id=str(model_id),
        available_review_count=max(0, int(available_review_count)),
        review_count=review_count,
        return_curves=bool(return_curves),
        measurements=tuple(measurements),
    )


def run_process_many_curve_speed_test(
    *,
    review_count: int,
    available_review_count: int,
    model_id: str,
    mode: str,
    run_once: Callable[[bool], float],
    progress: SpeedTestProgress,
    repetitions: int | None = None,
) -> ProcessManyCurveSpeedTestResult:
    review_count = int(review_count)
    if review_count <= 0:
        raise ValueError("The collection has no processable reviews to benchmark.")
    normalized_mode = _validated_modes((mode,), PROCESS_MANY_MODES)[0]
    effective_repetitions = (
        CURVE_SPEED_TEST_REPETITIONS if repetitions is None else int(repetitions)
    )
    if effective_repetitions <= 0:
        raise ValueError("Curve speed-test repetitions must be positive.")

    durations: dict[bool, list[float]] = {True: [], False: []}
    total = effective_repetitions * 2
    completed = 0
    for repetition in range(effective_repetitions):
        # Alternate order to reduce thermal and cache-order bias.
        order = (True, False) if repetition % 2 == 0 else (False, True)
        for return_curves in order:
            progress.check_cancelled()
            label = (
                f"State building - {_mode_label(normalized_mode)} - "
                f"{'with curves' if return_curves else 'without curves'} - "
                f"{review_count:,} reviews (run {repetition + 1}/{effective_repetitions})"
            )
            progress.update(completed, total, label)
            elapsed = float(run_once(return_curves))
            if not math.isfinite(elapsed) or elapsed < 0:
                raise RuntimeError("process_many() reported an invalid duration.")
            durations[return_curves].append(elapsed)
            completed += 1
            progress.update(completed, total, label)

    return ProcessManyCurveSpeedTestResult(
        model_id=str(model_id),
        mode=normalized_mode,
        available_review_count=max(0, int(available_review_count)),
        review_count=review_count,
        repetitions=effective_repetitions,
        measurements=tuple(
            ProcessManyCurveSpeedMeasurement(
                return_curves=return_curves,
                review_count=review_count,
                durations_seconds=tuple(durations[return_curves]),
            )
            for return_curves in (True, False)
        ),
    )


def run_live_prediction_speed_test(
    rows: Sequence[dict[str, Any]],
    *,
    requested_card_count: int,
    eligible_card_count: int,
    model_id: str,
    mode: str,
    batch_size: int | None,
    open_session: LivePredictionBenchmarkFactory,
    progress: SpeedTestProgress,
    repetitions: int = LIVE_PREDICTION_SPEED_TEST_REPETITIONS,
    clock: Callable[[], float] = time.perf_counter,
) -> LivePredictionSpeedTestResult:
    requested = max(1, int(requested_card_count))
    eligible = max(0, int(eligible_card_count))
    card_count = min(requested, eligible)
    if card_count <= 0:
        raise ValueError(
            "The collection has no cards with processed RWKV state to benchmark."
        )
    if len(rows) < card_count:
        raise ValueError(
            f"Only {len(rows)} prediction rows were prepared for a {card_count}-card test."
        )
    normalized_mode = _validated_modes((mode,), PREDICT_MANY_MODES)[0]
    repetitions = int(repetitions)
    if repetitions <= 0:
        raise ValueError("Live prediction speed-test repetitions must be positive.")
    selected_batch_size = _positive_optional_int(batch_size)
    batch = list(rows[:card_count])
    total = repetitions + 1
    label = (
        f"Live predictions - {_mode_label(normalized_mode)} - warming "
        f"{card_count:,} cards (untimed)"
    )
    progress.check_cancelled()
    progress.update(0, total, label)
    session = open_session(batch, normalized_mode, selected_batch_size, card_count)
    try:
        validate_live_prediction_refresh_count(
            session.run_cycle(),
            card_count,
            normalized_mode,
            warmup=True,
        )
        progress.update(1, total, label)

        durations: list[float] = []
        for repetition in range(1, repetitions + 1):
            progress.check_cancelled()
            label = (
                f"Live predictions - {_mode_label(normalized_mode)} - "
                f"{card_count:,} cards (run {repetition}/{repetitions})"
            )
            progress.update(repetition, total, label)
            started = clock()
            refreshed_count = session.run_cycle()
            elapsed = _elapsed_seconds(started, clock())
            validate_live_prediction_refresh_count(
                refreshed_count,
                card_count,
                normalized_mode,
                warmup=False,
            )
            durations.append(elapsed)
            progress.update(repetition + 1, total, label)
    finally:
        session.close()

    return LivePredictionSpeedTestResult(
        model_id=str(model_id),
        mode=normalized_mode,
        requested_card_count=requested,
        eligible_card_count=eligible,
        card_count=card_count,
        repetitions=repetitions,
        durations_seconds=tuple(durations),
        batch_size=selected_batch_size,
    )


def _validated_modes(modes: Sequence[str], supported: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(mode).strip().lower() for mode in modes)
    if not normalized:
        raise ValueError("At least one speed-test mode is required.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Speed-test modes must be unique.")
    invalid = tuple(mode for mode in normalized if mode not in supported)
    if invalid:
        raise ValueError(f"Unsupported speed-test mode: {invalid[0]!r}.")
    return normalized


def _elapsed_seconds(started: float, finished: float) -> float:
    elapsed = float(finished) - float(started)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("The speed-test clock returned an invalid duration.")
    return elapsed


def _positive_optional_int(value: int | None) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _mode_label(mode: str) -> str:
    return "GPU" if mode == "gpu" else mode.title()
