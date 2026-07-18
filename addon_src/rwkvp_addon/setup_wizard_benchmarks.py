from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .live_prediction_benchmark import (
    LivePredictionBenchmarkFactory,
    validate_live_prediction_refresh_count,
)
from .progress import CancelledError

SETUP_PROCESS_REVIEW_LIMIT = 10_000
SETUP_PREDICTION_CARD_LIMIT = 1_000
SETUP_PROCESS_REPETITIONS = 1
SETUP_MODE_REPETITIONS = 3
SETUP_PREDICTION_TARGET_SECONDS = 0.100
SETUP_PREDICTION_COUNT_LIMIT = 99_999
SETUP_PREDICTION_COUNT_STEP = 100
SETUP_STATE_PROCESS_CHUNK_SIZE = 10_000

_FAST_MODE = "fast"
_GPU_MODE = "gpu"


class NoProcessableReviewsError(ValueError):
    """Raised when a state-building benchmark has no usable review sample."""


class SetupBenchmarkProgress(Protocol):
    def update(self, current: int, total: int, text: str = "") -> None: ...

    def check_cancelled(self) -> None: ...


class ProcessOnce(Protocol):
    def __call__(
        self,
        mode: str,
        rows: Sequence[dict[str, Any]],
        *,
        return_curves: bool,
    ) -> float: ...


@dataclass(frozen=True)
class SetupModeMeasurement:
    mode: str
    item_count: int
    durations_seconds: tuple[float, ...]
    error: str | None = None
    expected_repetitions: int = SETUP_MODE_REPETITIONS

    @property
    def succeeded(self) -> bool:
        return (
            self.error is None
            and len(self.durations_seconds) == self.expected_repetitions
        )

    @property
    def average_seconds(self) -> float | None:
        if not self.succeeded:
            return None
        return sum(self.durations_seconds) / len(self.durations_seconds)

    @property
    def items_per_second(self) -> float | None:
        average = self.average_seconds
        if average is None:
            return None
        return math.inf if average == 0.0 else self.item_count / average


@dataclass(frozen=True)
class SetupProcessModeBenchmarkResult:
    review_count: int
    measurements: tuple[SetupModeMeasurement, ...]
    selected_mode: str

    def measurement(self, mode: str) -> SetupModeMeasurement:
        return _measurement_for_mode(self.measurements, mode)


@dataclass(frozen=True)
class SetupPredictionModeBenchmarkResult:
    eligible_card_count: int
    card_count: int
    measurements: tuple[SetupModeMeasurement, ...]
    selected_mode: str

    def measurement(self, mode: str) -> SetupModeMeasurement:
        return _measurement_for_mode(self.measurements, mode)


@dataclass(frozen=True)
class PredictionCountMeasurement:
    card_count: int
    duration_seconds: float | None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.duration_seconds is not None


@dataclass(frozen=True)
class PredictionRefreshSearchResult:
    mode: str
    eligible_card_count: int
    maximum_card_count: int
    target_seconds: float
    selected_card_count: int
    measurements: tuple[PredictionCountMeasurement, ...]

    def measurement(self, card_count: int) -> PredictionCountMeasurement:
        wanted = int(card_count)
        for measurement in self.measurements:
            if measurement.card_count == wanted:
                return measurement
        raise KeyError(card_count)


def run_process_mode_benchmark(
    rows: Sequence[dict[str, Any]],
    *,
    gpu_available: bool,
    run_once: ProcessOnce,
    progress: SetupBenchmarkProgress,
    return_curves: bool = False,
    repetitions: int = SETUP_PROCESS_REPETITIONS,
) -> SetupProcessModeBenchmarkResult:
    """Compare Fast and GPU with one substantial state-building sample.

    ``run_once`` owns construction of a fresh in-process runtime for every call.
    Immediate performance tuning passes ``return_curves=False``; a curve-only
    setup passes ``True`` so its hardware decision measures the workload the
    user actually selected.
    """

    batch = list(rows[:SETUP_PROCESS_REVIEW_LIMIT])
    if not batch:
        raise NoProcessableReviewsError(
            "The collection has no processable reviews to benchmark."
        )
    repetitions = _validated_repetitions(
        repetitions,
        expected=SETUP_PROCESS_REPETITIONS,
    )
    modes = (_FAST_MODE, _GPU_MODE) if gpu_available else (_FAST_MODE,)
    total = len(modes) * repetitions
    completed = 0
    measurements: list[SetupModeMeasurement] = []

    for mode in modes:
        durations: list[float] = []
        error: str | None = None
        for repetition in range(1, repetitions + 1):
            progress.check_cancelled()
            label = (
                f"Testing {_mode_label(mode)} state building - "
                f"{len(batch):,} reviews (run {repetition}/{repetitions})"
            )
            progress.update(completed, total, label)
            try:
                elapsed = float(
                    run_once(
                        mode,
                        batch,
                        return_curves=bool(return_curves),
                    )
                )
                _validate_duration(elapsed)
            except CancelledError:
                raise
            except Exception as exc:
                error = _error_text(exc)
                # A strict GPU/process failure makes this mode unusable. Do not
                # retry it through Fast or present partial timings as a result.
                completed += repetitions - repetition + 1
                progress.update(completed, total, f"{label} - failed")
                break
            durations.append(elapsed)
            completed += 1
            progress.update(completed, total, label)
        measurements.append(
            SetupModeMeasurement(
                mode=mode,
                item_count=len(batch),
                durations_seconds=tuple(durations),
                error=error,
                expected_repetitions=repetitions,
            )
        )

    return SetupProcessModeBenchmarkResult(
        review_count=len(batch),
        measurements=tuple(measurements),
        selected_mode=_select_mode(measurements),
    )


def run_prediction_mode_benchmark(
    rows: Sequence[dict[str, Any]],
    *,
    eligible_card_count: int,
    gpu_available: bool,
    open_session: LivePredictionBenchmarkFactory,
    progress: SetupBenchmarkProgress,
    repetitions: int = SETUP_MODE_REPETITIONS,
    clock: Callable[[], float] = time.perf_counter,
) -> SetupPredictionModeBenchmarkResult:
    """Compare warmed native Live Session cycles on at most 1,000 cards."""

    eligible = max(0, int(eligible_card_count))
    card_count = min(SETUP_PREDICTION_CARD_LIMIT, eligible)
    if card_count <= 0:
        raise ValueError(
            "The collection has no cards with processed RWKV state to benchmark."
        )
    if len(rows) < card_count:
        raise ValueError(
            f"Only {len(rows)} prediction rows were prepared for a {card_count}-card setup test."
        )
    repetitions = _validated_repetitions(
        repetitions,
        expected=SETUP_MODE_REPETITIONS,
    )
    batch = list(rows[:card_count])
    modes = (_FAST_MODE, _GPU_MODE) if gpu_available else (_FAST_MODE,)
    total = len(modes) * (repetitions + 1)
    completed = 0
    measurements: list[SetupModeMeasurement] = []

    for mode in modes:
        durations: list[float] = []
        error: str | None = None
        session = None
        warmup_label = (
            f"Preparing and warming {_mode_label(mode)} Live Session predictions - "
            f"{card_count:,} cards (untimed)"
        )
        progress.check_cancelled()
        progress.update(completed, total, warmup_label)
        try:
            session = open_session(batch, mode, None, card_count)
            validate_live_prediction_refresh_count(
                session.run_cycle(),
                card_count,
                mode,
                warmup=True,
            )
        except CancelledError:
            raise
        except Exception as exc:
            error = _error_text(exc)
            completed += repetitions + 1
            progress.update(completed, total, f"{warmup_label} - failed")
        else:
            completed += 1
            progress.update(completed, total, warmup_label)
            for repetition in range(1, repetitions + 1):
                progress.check_cancelled()
                label = (
                    f"Testing {_mode_label(mode)} predictions - "
                    f"{card_count:,} cards (run {repetition}/{repetitions})"
                )
                progress.update(completed, total, label)
                try:
                    started = float(clock())
                    refreshed_count = session.run_cycle()
                    elapsed = float(clock()) - started
                    _validate_duration(elapsed)
                    validate_live_prediction_refresh_count(
                        refreshed_count,
                        card_count,
                        mode,
                        warmup=False,
                    )
                except CancelledError:
                    raise
                except Exception as exc:
                    error = _error_text(exc)
                    completed += repetitions - repetition + 1
                    progress.update(completed, total, f"{label} - failed")
                    break
                durations.append(elapsed)
                completed += 1
                progress.update(completed, total, label)
        finally:
            if session is not None:
                session.close()
        measurements.append(
            SetupModeMeasurement(
                mode=mode,
                item_count=card_count,
                durations_seconds=tuple(durations),
                error=error,
                expected_repetitions=repetitions,
            )
        )

    return SetupPredictionModeBenchmarkResult(
        eligible_card_count=eligible,
        card_count=card_count,
        measurements=tuple(measurements),
        selected_mode=_select_mode(measurements),
    )


def search_prediction_refresh_count(
    rows: Sequence[dict[str, Any]],
    *,
    eligible_card_count: int,
    mode: str,
    open_session: LivePredictionBenchmarkFactory,
    progress: SetupBenchmarkProgress,
    target_seconds: float = SETUP_PREDICTION_TARGET_SECONDS,
    maximum_card_count: int = SETUP_PREDICTION_COUNT_LIMIT,
    step: int = SETUP_PREDICTION_COUNT_STEP,
    initial_card_count: int = SETUP_PREDICTION_CARD_LIMIT,
    maximum_binary_probes: int = 12,
    clock: Callable[[], float] = time.perf_counter,
    seeded_durations: Mapping[int, float] | None = None,
) -> PredictionRefreshSearchResult:
    """Find the warmed native refresh count whose latency is closest to target.

    Counts of 100 or more are probed only on ``step`` boundaries. Collections
    with fewer than 100 eligible cards use their exact cap. The search first
    doubles or halves to bracket the target, then probes rounded midpoints. A
    failed probe is retained and treated as an unusable upper bound.
    """

    eligible = max(0, int(eligible_card_count))
    configured_cap = min(
        eligible,
        max(1, int(maximum_card_count)),
        SETUP_PREDICTION_COUNT_LIMIT,
    )
    if configured_cap <= 0:
        raise ValueError(
            "The collection has no cards with processed RWKV state to benchmark."
        )
    if len(rows) < configured_cap:
        raise ValueError(
            f"Only {len(rows)} prediction rows were prepared for a "
            f"{configured_cap}-card setup search."
        )
    target = float(target_seconds)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("The prediction latency target must be finite and positive.")
    step = max(SETUP_PREDICTION_COUNT_STEP, int(step))
    binary_limit = max(0, int(maximum_binary_probes))

    if configured_cap < step:
        search_cap = configured_cap
        start = configured_cap
    else:
        search_cap = (configured_cap // step) * step
        requested_start = min(max(step, int(initial_card_count)), search_cap)
        start = max(step, (requested_start // step) * step)

    cache: dict[int, PredictionCountMeasurement] = {}
    probe_order: list[int] = []
    max_expected_probes = (
        1 + math.ceil(math.log2(max(1, search_cap // step))) + binary_limit
    )

    for count, duration in (seeded_durations or {}).items():
        normalized_count = int(count)
        normalized_duration = float(duration)
        if (
            normalized_count <= 0
            or normalized_count > search_cap
            or (search_cap >= step and normalized_count % step)
        ):
            continue
        _validate_duration(normalized_duration)
        cache[normalized_count] = PredictionCountMeasurement(
            card_count=normalized_count,
            duration_seconds=normalized_duration,
        )
        probe_order.append(normalized_count)

    def measure(card_count: int) -> PredictionCountMeasurement:
        normalized_count = int(card_count)
        existing = cache.get(normalized_count)
        if existing is not None:
            return existing
        progress.check_cancelled()
        label = (
            f"Finding a 100 ms Live Session refresh size - {_mode_label(mode)} - "
            f"{normalized_count:,} cards"
        )
        progress.update(len(probe_order), max_expected_probes, label)
        session = None
        try:
            batch = list(rows[:normalized_count])
            session = open_session(batch, str(mode), None, normalized_count)
            validate_live_prediction_refresh_count(
                session.run_cycle(),
                normalized_count,
                str(mode),
                warmup=True,
            )
            progress.check_cancelled()
            started = float(clock())
            refreshed_count = session.run_cycle()
            elapsed = float(clock()) - started
            _validate_duration(elapsed)
            validate_live_prediction_refresh_count(
                refreshed_count,
                normalized_count,
                str(mode),
                warmup=False,
            )
            measurement = PredictionCountMeasurement(
                card_count=normalized_count,
                duration_seconds=elapsed,
            )
        except CancelledError:
            raise
        except Exception as exc:
            measurement = PredictionCountMeasurement(
                card_count=normalized_count,
                duration_seconds=None,
                error=_error_text(exc),
            )
        finally:
            if session is not None:
                session.close()
        cache[normalized_count] = measurement
        probe_order.append(normalized_count)
        progress.update(len(probe_order), max_expected_probes, label)
        return measurement

    first = measure(start)
    lower: int | None = None
    upper: int | None = None
    if _is_at_or_below_target(first, target):
        lower = start
        current = start
        while current < search_cap:
            candidate = min(search_cap, current * 2)
            candidate = _round_down_to_step(candidate, step)
            if candidate <= current:
                break
            measured = measure(candidate)
            if _is_at_or_below_target(measured, target):
                lower = candidate
                current = candidate
                continue
            upper = candidate
            break
    else:
        upper = start
        current = start
        while current > step:
            candidate = _round_down_to_step(max(step, current // 2), step)
            if candidate >= current:
                break
            measured = measure(candidate)
            if _is_at_or_below_target(measured, target):
                lower = candidate
                break
            upper = candidate
            current = candidate

    if lower is not None and upper is not None:
        probes = 0
        while upper - lower > step and probes < binary_limit:
            midpoint = _rounded_midpoint(lower, upper, step)
            if midpoint <= lower or midpoint >= upper:
                break
            measured = measure(midpoint)
            probes += 1
            if _is_at_or_below_target(measured, target):
                lower = midpoint
            else:
                upper = midpoint

    successful = tuple(
        measurement for measurement in cache.values() if measurement.succeeded
    )
    if not successful:
        details = next(
            (measurement.error for measurement in cache.values() if measurement.error),
            "Prediction timing failed.",
        )
        raise RuntimeError(details)
    selected = min(successful, key=lambda item: _prediction_count_rank(item, target))
    return PredictionRefreshSearchResult(
        mode=str(mode),
        eligible_card_count=eligible,
        maximum_card_count=search_cap,
        target_seconds=target,
        selected_card_count=selected.card_count,
        measurements=tuple(cache[count] for count in probe_order),
    )


def process_history_state_only(
    rows: Sequence[dict[str, Any]],
    *,
    process: Callable[[Sequence[dict[str, Any]]], object],
    progress: SetupBenchmarkProgress,
    label: str = "Building temporary RWKV state",
    chunk_size: int = SETUP_STATE_PROCESS_CHUNK_SIZE,
) -> int:
    """Process full history in bounded chunks and immediately discard outputs."""

    total = len(rows)
    processed = 0
    effective_chunk_size = max(1, int(chunk_size))
    for start in range(0, total, effective_chunk_size):
        progress.check_cancelled()
        chunk = rows[start : start + effective_chunk_size]
        result = process(chunk)
        _validate_optional_result_count(result, len(chunk))
        # Keep no prediction/curve collection alive across chunks. Native state
        # remains owned by the injected processor/runtime.
        del result
        processed += len(chunk)
        progress.update(processed, total, label)
        progress.check_cancelled()
    return processed


def _measurement_for_mode(
    measurements: Sequence[SetupModeMeasurement],
    mode: str,
) -> SetupModeMeasurement:
    wanted = str(mode)
    for measurement in measurements:
        if measurement.mode == wanted:
            return measurement
    raise KeyError(mode)


def _select_mode(measurements: Sequence[SetupModeMeasurement]) -> str:
    successful = [measurement for measurement in measurements if measurement.succeeded]
    if not successful:
        failures = "; ".join(
            f"{_mode_label(measurement.mode)}: "
            f"{measurement.error or 'benchmark did not complete'}"
            for measurement in measurements
        )
        detail = f" ({failures})" if failures else ""
        raise RuntimeError(
            "No performance mode completed the benchmark successfully" + detail + "."
        )
    selected = min(
        successful,
        key=lambda measurement: (
            float(measurement.average_seconds),
            measurement.mode != _FAST_MODE,
        ),
    )
    return selected.mode


def _prediction_count_rank(
    measurement: PredictionCountMeasurement,
    target_seconds: float,
) -> tuple[float, bool, int]:
    duration = float(measurement.duration_seconds)
    return (
        round(abs(duration - target_seconds), 12),
        duration > target_seconds,
        -measurement.card_count,
    )


def _is_at_or_below_target(
    measurement: PredictionCountMeasurement,
    target_seconds: float,
) -> bool:
    return bool(
        measurement.succeeded and float(measurement.duration_seconds) <= target_seconds
    )


def _rounded_midpoint(lower: int, upper: int, step: int) -> int:
    midpoint = (int(lower) + int(upper)) // 2
    return _round_down_to_step(midpoint, step)


def _round_down_to_step(value: int, step: int) -> int:
    return max(step, (int(value) // step) * step)


def _validate_optional_result_count(result: object, expected: int) -> None:
    if result is None:
        return
    try:
        actual = len(result)  # type: ignore[arg-type]
    except TypeError:
        return
    if actual != expected:
        raise RuntimeError(
            f"State processing returned {actual} results for {expected} reviews."
        )


def _validated_repetitions(value: int, *, expected: int) -> int:
    parsed = int(value)
    if parsed != expected:
        raise ValueError(
            f"This setup benchmark requires exactly {expected} timed run"
            f"{'s' if expected != 1 else ''}."
        )
    return parsed


def _validate_duration(value: float) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise RuntimeError("The setup benchmark reported an invalid duration.")


def _error_text(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__


def _mode_label(mode: str) -> str:
    return "GPU" if str(mode) == _GPU_MODE else str(mode).title()
