from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .checkpoint_storage import UNMEASURED_PROCESS_MANY_REVIEWS_PER_MINUTE
from .profile_store import atomic_write_json, read_json
from .rwkv_performance_modes import PROCESS_MANY_FAST_MODE, PROCESS_MANY_MODES
from .speed_test import ProcessManySpeedMeasurement, ProcessManySpeedTestResult

if TYPE_CHECKING:
    from .setup_wizard_benchmarks import SetupProcessModeBenchmarkResult
    from .speed_test import ProcessManyCurveSpeedTestResult

PROCESS_MANY_SPEED_CACHE_FILENAME = "process_many_speed_test.json"
PROCESS_MANY_SPEED_CACHE_SCHEMA_VERSION = 1
CHECKPOINT_CURVE_SPEED_FACTOR = 0.75

CHECKPOINT_SPEED_MATCHING_MEASUREMENT = "matching_measurement"
CHECKPOINT_SPEED_CPU_MEASUREMENT = "cpu_measurement"
CHECKPOINT_SPEED_WITHOUT_CURVES = "without_curves"
CHECKPOINT_SPEED_UNMEASURED = "unmeasured"


@dataclass(frozen=True)
class CachedProcessManySpeed:
    model_id: str
    return_curves: bool
    mode: str
    review_count: int
    reviews_per_minute: float
    measured_at: float = 0.0
    source: str = "speed_test"
    state_precision: str = "fp32"


@dataclass(frozen=True)
class CheckpointBuildSpeedEstimate:
    """Throughput and provenance used by the checkpoint confirmation.

    ``measurement`` is retained when the estimate derives from a cache entry.
    The no-curve fallback deliberately changes only ``reviews_per_minute``;
    callers can still explain exactly which real measurement it came from.
    """

    reviews_per_minute: float
    basis: str
    measurement: CachedProcessManySpeed | None = None


def process_many_speed_cache_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / PROCESS_MANY_SPEED_CACHE_FILENAME


def cache_process_many_speed_test(
    path: Path,
    result: ProcessManySpeedTestResult,
    *,
    measured_at: float | None = None,
    state_precision: str = "fp32",
) -> None:
    """Remember the largest completed process_many() measurement per mode."""

    measurement_time = _measurement_time(measured_at)
    new_entries: list[CachedProcessManySpeed] = []
    for measurement in _largest_measurement_by_mode(result.measurements):
        mode = str(measurement.mode)
        if mode not in PROCESS_MANY_MODES:
            continue
        rate = float(measurement.reviews_per_minute)
        if not math.isfinite(rate) or rate <= 0:
            continue
        new_entries.append(
            CachedProcessManySpeed(
                model_id=str(result.model_id),
                return_curves=bool(result.return_curves),
                mode=mode,
                review_count=int(measurement.review_count),
                reviews_per_minute=rate,
                measured_at=measurement_time,
                source="speed_test",
                state_precision=state_precision,
            )
        )

    _cache_entries(path, new_entries)


def cache_process_many_curve_speed_test(
    path: Path,
    result: ProcessManyCurveSpeedTestResult,
    *,
    measured_at: float | None = None,
    state_precision: str = "fp32",
) -> None:
    """Remember both exact rates from the curve on/off comparison."""

    mode = str(result.mode)
    if mode not in PROCESS_MANY_MODES:
        return
    measurement_time = _measurement_time(measured_at)
    new_entries: list[CachedProcessManySpeed] = []
    for measurement in result.measurements:
        review_count = int(measurement.review_count)
        rate = float(measurement.reviews_per_minute)
        if review_count <= 0 or not math.isfinite(rate) or rate <= 0:
            continue
        new_entries.append(
            CachedProcessManySpeed(
                model_id=str(result.model_id),
                return_curves=bool(measurement.return_curves),
                mode=mode,
                review_count=review_count,
                reviews_per_minute=rate,
                measured_at=measurement_time,
                source="curve_speed_test",
                state_precision=state_precision,
            )
        )

    _cache_entries(path, new_entries)


def cache_setup_process_mode_benchmark(
    path: Path,
    result: SetupProcessModeBenchmarkResult,
    *,
    model_id: str,
    return_curves: bool,
    measured_at: float | None = None,
    state_precision: str = "fp32",
) -> None:
    """Remember completed Guided Setup state-building measurements.

    Guided Setup uses a smaller result type than the full Settings speed test,
    but both measurements describe the same ``process_many()`` workload. Keep
    them in the same model/curve/mode-partitioned cache so a checkpoint build
    started directly from Setup can use the rate it just measured.
    """

    measurement_time = _measurement_time(measured_at)
    new_entries: list[CachedProcessManySpeed] = []
    for measurement in result.measurements:
        if not measurement.succeeded:
            continue
        mode = str(measurement.mode)
        review_count = int(measurement.item_count)
        items_per_second = measurement.items_per_second
        if (
            mode not in PROCESS_MANY_MODES
            or review_count <= 0
            or items_per_second is None
        ):
            continue
        rate = float(items_per_second) * 60.0
        if not math.isfinite(rate) or rate <= 0:
            continue
        new_entries.append(
            CachedProcessManySpeed(
                model_id=str(model_id),
                return_curves=bool(return_curves),
                mode=mode,
                review_count=review_count,
                reviews_per_minute=rate,
                measured_at=measurement_time,
                source="setup_wizard",
                state_precision=state_precision,
            )
        )

    _cache_entries(path, new_entries)


def cache_completed_checkpoint_build(
    path: Path,
    *,
    model_id: str,
    return_curves: bool,
    mode: str,
    review_count: int,
    duration_seconds: float,
    measured_at: float | None = None,
    state_precision: str = "fp32",
) -> CachedProcessManySpeed | None:
    """Remember the observed rate of a successfully completed full build."""

    count = int(review_count)
    duration = float(duration_seconds)
    if (
        not str(model_id)
        or str(mode) not in PROCESS_MANY_MODES
        or count <= 0
        or not math.isfinite(duration)
        or duration <= 0
    ):
        return None

    entry = CachedProcessManySpeed(
        model_id=str(model_id),
        return_curves=bool(return_curves),
        mode=str(mode),
        review_count=count,
        reviews_per_minute=count * 60.0 / duration,
        measured_at=_measurement_time(measured_at),
        source="checkpoint_build",
        state_precision=state_precision,
    )
    _cache_entries(path, [entry])
    return entry


def _cache_entries(path: Path, new_entries: list[CachedProcessManySpeed]) -> None:
    if any(entry.state_precision not in {"fp16", "fp32"} for entry in new_entries):
        raise ValueError("State precision must be fp16 or fp32.")
    entries = {_entry_key(entry): entry for entry in _read_entries(path)}
    for entry in new_entries:
        existing = entries.get(_entry_key(entry))
        if existing is None or entry.measured_at >= existing.measured_at:
            entries[_entry_key(entry)] = entry

    atomic_write_json(
        Path(path),
        {
            "schema_version": PROCESS_MANY_SPEED_CACHE_SCHEMA_VERSION,
            "measurements": [
                _entry_payload(entry)
                for entry in sorted(
                    entries.values(),
                    key=lambda item: (
                        item.model_id,
                        item.state_precision,
                        item.return_curves,
                        item.mode,
                    ),
                )
            ],
        },
    )


def cached_process_many_speed(
    path: Path,
    *,
    model_id: str,
    return_curves: bool,
    mode: str,
    state_precision: str = "fp32",
) -> CachedProcessManySpeed | None:
    return _matching_cached_speed(
        _read_entries(path),
        model_id=model_id,
        return_curves=return_curves,
        mode=mode,
        state_precision=state_precision,
    )


def checkpoint_build_speed_estimate(
    path: Path,
    *,
    model_id: str,
    return_curves: bool,
    mode: str,
    state_precision: str = "fp32",
) -> CheckpointBuildSpeedEstimate:
    """Choose the best defensible checkpoint-build throughput estimate.

    Prefer an exact selected-mode measurement. If that is absent, a measured
    no-curve result for that same mode is reduced by 25% before considering
    another executor. A requested GPU build can then fall back to measured CPU
    Fast curve throughput, followed by CPU Fast without curves at the same 25%
    discount. With no applicable evidence, use the explicitly unmeasured 150k
    reviews/minute fallback.
    """

    entries = tuple(
        entry for entry in _read_entries(Path(path))
        if entry.state_precision == state_precision
    )
    wanted_model = str(model_id)
    wanted_curves = bool(return_curves)
    wanted_mode = str(mode)

    matching = _matching_cached_speed(
        entries,
        model_id=wanted_model,
        return_curves=wanted_curves,
        mode=wanted_mode,
        state_precision=state_precision,
    )
    if matching is not None:
        return CheckpointBuildSpeedEstimate(
            reviews_per_minute=matching.reviews_per_minute,
            basis=CHECKPOINT_SPEED_MATCHING_MEASUREMENT,
            measurement=matching,
        )

    if wanted_curves:
        selected_without_curves = _matching_cached_speed(
            entries,
            model_id=wanted_model,
            return_curves=False,
            mode=wanted_mode,
            state_precision=state_precision,
        )
        if selected_without_curves is not None:
            return CheckpointBuildSpeedEstimate(
                reviews_per_minute=(
                    selected_without_curves.reviews_per_minute
                    * CHECKPOINT_CURVE_SPEED_FACTOR
                ),
                basis=CHECKPOINT_SPEED_WITHOUT_CURVES,
                measurement=selected_without_curves,
            )

    if wanted_mode != PROCESS_MANY_FAST_MODE:
        cpu_matching = _matching_cached_speed(
            entries,
            model_id=wanted_model,
            return_curves=wanted_curves,
            mode=PROCESS_MANY_FAST_MODE,
            state_precision=state_precision,
        )
        if cpu_matching is not None:
            return CheckpointBuildSpeedEstimate(
                reviews_per_minute=cpu_matching.reviews_per_minute,
                basis=CHECKPOINT_SPEED_CPU_MEASUREMENT,
                measurement=cpu_matching,
            )

        if wanted_curves:
            cpu_without_curves = _matching_cached_speed(
                entries,
                model_id=wanted_model,
                return_curves=False,
                mode=PROCESS_MANY_FAST_MODE,
                state_precision=state_precision,
            )
            if cpu_without_curves is not None:
                return CheckpointBuildSpeedEstimate(
                    reviews_per_minute=(
                        cpu_without_curves.reviews_per_minute
                        * CHECKPOINT_CURVE_SPEED_FACTOR
                    ),
                    basis=CHECKPOINT_SPEED_WITHOUT_CURVES,
                    measurement=cpu_without_curves,
                )

    return CheckpointBuildSpeedEstimate(
        reviews_per_minute=float(UNMEASURED_PROCESS_MANY_REVIEWS_PER_MINUTE),
        basis=CHECKPOINT_SPEED_UNMEASURED,
    )


def _matching_cached_speed(
    entries: tuple[CachedProcessManySpeed, ...],
    *,
    model_id: object,
    return_curves: object,
    mode: object,
    state_precision: str = "fp32",
) -> CachedProcessManySpeed | None:
    wanted = (str(model_id), bool(return_curves), str(mode), state_precision)
    return next((entry for entry in entries if _entry_key(entry) == wanted), None)


def _read_entries(path: Path) -> tuple[CachedProcessManySpeed, ...]:
    try:
        payload = read_json(Path(path))
    except (OSError, ValueError, TypeError):
        return ()
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PROCESS_MANY_SPEED_CACHE_SCHEMA_VERSION
    ):
        return ()
    raw_measurements = payload.get("measurements")
    if not isinstance(raw_measurements, list):
        return ()

    entries: list[CachedProcessManySpeed] = []
    for raw in raw_measurements:
        entry = _parse_entry(raw)
        if entry is not None:
            entries.append(entry)
    return tuple(entries)


def _parse_entry(raw: Any) -> CachedProcessManySpeed | None:
    if not isinstance(raw, dict):
        return None
    try:
        model_id = str(raw["model_id"])
        return_curves = raw["return_curves"]
        mode = str(raw["mode"])
        review_count = int(raw["review_count"])
        reviews_per_minute = float(raw["reviews_per_minute"])
        measured_at = float(raw.get("measured_at", 0.0))
        source = str(raw.get("source", "speed_test"))
        precision = raw.get("state_precision", "fp32")
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (
        not model_id
        or precision not in ("fp16", "fp32")
        or not isinstance(return_curves, bool)
        or mode not in PROCESS_MANY_MODES
        or review_count <= 0
        or not math.isfinite(reviews_per_minute)
        or reviews_per_minute <= 0
        or not math.isfinite(measured_at)
        or measured_at < 0
        or not source
    ):
        return None
    return CachedProcessManySpeed(
        model_id=model_id,
        return_curves=return_curves,
        mode=mode,
        review_count=review_count,
        reviews_per_minute=reviews_per_minute,
        measured_at=measured_at,
        source=source,
        state_precision=precision,
    )


def _largest_measurement_by_mode(
    measurements: tuple[ProcessManySpeedMeasurement, ...],
) -> tuple[ProcessManySpeedMeasurement, ...]:
    by_mode: dict[str, ProcessManySpeedMeasurement] = {}
    for measurement in measurements:
        existing = by_mode.get(measurement.mode)
        if existing is None or measurement.review_count > existing.review_count:
            by_mode[measurement.mode] = measurement
    return tuple(by_mode.values())


def _entry_key(entry: CachedProcessManySpeed) -> tuple[str, bool, str, str]:
    return (entry.model_id, entry.return_curves, entry.mode, entry.state_precision)


def _entry_payload(entry: CachedProcessManySpeed) -> dict[str, Any]:
    return {
        "model_id": entry.model_id,
        "state_precision": entry.state_precision,
        "return_curves": entry.return_curves,
        "mode": entry.mode,
        "review_count": entry.review_count,
        "reviews_per_minute": entry.reviews_per_minute,
        "measured_at": entry.measured_at,
        "source": entry.source,
    }


def _measurement_time(value: float | None) -> float:
    measured_at = time.time() if value is None else float(value)
    if not math.isfinite(measured_at) or measured_at < 0:
        raise ValueError("Process speed measurement time must be finite and non-negative.")
    return measured_at
