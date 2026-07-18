from __future__ import annotations

import math
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Current Rust F32 .bin checkpoints store fixed-shape recurrent state by unique
# processed card, note, deck, and preset IDs. Note type is not a separate state.
RUST_CHECKPOINT_CARD_BYTES = 52_600
RUST_CHECKPOINT_NOTE_BYTES = 35_100
RUST_CHECKPOINT_DECK_BYTES = 70_100
RUST_CHECKPOINT_PRESET_BYTES = 52_600
RUST_CHECKPOINT_GLOBAL_BYTES = 70_000
RUST_CHECKPOINT_SAFETY_MARGIN = 0.30
RUST_PROCESS_MANY_REVIEWS_PER_MINUTE = 33_000
RUST_PROCESS_MANY_TIME_TOLERANCE = 0.15
BENCHMARK_PROCESS_MANY_SPEED_TOLERANCE = 0.10


class InsufficientCheckpointDiskSpaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class RustCheckpointStorageEstimate:
    unique_card_count: int
    unique_note_count: int
    unique_deck_count: int
    unique_preset_count: int
    estimated_checkpoint_bytes: int
    required_free_bytes: int


@dataclass(frozen=True)
class RustCheckpointIdentityCounts:
    card_count: int
    note_count: int
    deck_count: int
    preset_count: int

    def as_kwargs(self) -> dict[str, int]:
        return {
            "card_count": self.card_count,
            "note_count": self.note_count,
            "deck_count": self.deck_count,
            "preset_count": self.preset_count,
        }


@dataclass(frozen=True)
class RustCheckpointProcessingTimeEstimate:
    review_count: int
    estimated_seconds: float
    lower_seconds: float
    upper_seconds: float


def estimate_rust_checkpoint_storage(
    rows: Iterable[dict[str, Any]],
    *,
    safety_margin: float = RUST_CHECKPOINT_SAFETY_MARGIN,
    expected_checkpoint_bytes: int | None = None,
) -> RustCheckpointStorageEstimate:
    """Estimate current F32 Rust .bin checkpoint disk usage from processed IDs."""

    counts = rust_checkpoint_identity_counts(rows)
    return estimate_rust_checkpoint_storage_from_counts(
        counts,
        safety_margin=safety_margin,
        expected_checkpoint_bytes=expected_checkpoint_bytes,
    )


def rust_checkpoint_identity_counts(
    rows: Iterable[dict[str, Any]],
) -> RustCheckpointIdentityCounts:
    """Count the normalized identities retained by a Rust checkpoint."""

    card_ids: set[int] = set()
    note_ids: set[int] = set()
    deck_ids: set[int] = set()
    preset_ids: set[int] = set()
    missing_note_card_ids: set[int] = set()
    has_missing_deck = False
    has_missing_preset = False
    for row in rows:
        card_id = _optional_int(row.get("card_id"))
        if card_id is not None:
            card_ids.add(card_id)
        note_id = _optional_int(row.get("note_id"))
        if note_id is None:
            if card_id is not None:
                missing_note_card_ids.add(card_id)
        else:
            note_ids.add(note_id)
        deck_id = _optional_int(row.get("deck_id"))
        if deck_id is None:
            has_missing_deck = has_missing_deck or card_id is not None
        else:
            deck_ids.add(deck_id)
        preset_id = _optional_int(row.get("preset_id"))
        if preset_id is None:
            has_missing_preset = has_missing_preset or card_id is not None
        else:
            preset_ids.add(preset_id)
    return RustCheckpointIdentityCounts(
        card_count=len(card_ids),
        note_count=len(note_ids) + len(missing_note_card_ids),
        deck_count=len(deck_ids) + int(has_missing_deck),
        preset_count=len(preset_ids) + int(has_missing_preset),
    )


def estimate_rust_checkpoint_storage_from_counts(
    counts: RustCheckpointIdentityCounts,
    *,
    safety_margin: float = RUST_CHECKPOINT_SAFETY_MARGIN,
    expected_checkpoint_bytes: int | None = None,
) -> RustCheckpointStorageEstimate:
    """Build a storage estimate from normalized identity counts.

    ``expected_checkpoint_bytes`` should come from RWKV-SRS's public
    ``expected_checkpoint_size()`` helper. The fixed-size arithmetic remains a
    compatibility fallback for test backends and older unpackaged runtimes.
    """

    unique_card_count = max(0, int(counts.card_count))
    unique_note_count = max(0, int(counts.note_count))
    unique_deck_count = max(0, int(counts.deck_count))
    unique_preset_count = max(0, int(counts.preset_count))
    estimated = (
        int(expected_checkpoint_bytes)
        if expected_checkpoint_bytes is not None
        else (
            unique_card_count * RUST_CHECKPOINT_CARD_BYTES
            + unique_note_count * RUST_CHECKPOINT_NOTE_BYTES
            + unique_deck_count * RUST_CHECKPOINT_DECK_BYTES
            + unique_preset_count * RUST_CHECKPOINT_PRESET_BYTES
            + RUST_CHECKPOINT_GLOBAL_BYTES
        )
    )
    if estimated < 0:
        raise ValueError("Expected checkpoint size must be non-negative.")
    required = math.ceil(estimated * (1.0 + max(0.0, float(safety_margin))))
    return RustCheckpointStorageEstimate(
        unique_card_count=unique_card_count,
        unique_note_count=unique_note_count,
        unique_deck_count=unique_deck_count,
        unique_preset_count=unique_preset_count,
        estimated_checkpoint_bytes=estimated,
        required_free_bytes=required,
    )


def estimate_rust_checkpoint_processing_time(
    review_count: int,
    *,
    reviews_per_minute: int = RUST_PROCESS_MANY_REVIEWS_PER_MINUTE,
    tolerance: float = RUST_PROCESS_MANY_TIME_TOLERANCE,
) -> RustCheckpointProcessingTimeEstimate:
    """Estimate Rust process_many() time for checkpoint initialization.

    This is a planning estimate for the release Rust backend after review rows
    are already available and the model is loaded. It deliberately excludes
    checkpoint save/load time and GUI overhead.
    """

    count = max(0, int(review_count))
    rpm = max(1, int(reviews_per_minute))
    estimated_seconds = count * 60.0 / rpm
    tolerance = max(0.0, float(tolerance))
    return RustCheckpointProcessingTimeEstimate(
        review_count=count,
        estimated_seconds=estimated_seconds,
        lower_seconds=estimated_seconds * (1.0 - tolerance),
        upper_seconds=estimated_seconds * (1.0 + tolerance),
    )


def estimate_checkpoint_processing_time_from_benchmark(
    review_count: int,
    reviews_per_minute: float,
    *,
    speed_tolerance: float = BENCHMARK_PROCESS_MANY_SPEED_TOLERANCE,
) -> RustCheckpointProcessingTimeEstimate:
    """Estimate build time from a measured rate and a throughput tolerance.

    The quickest time uses the cached rate plus the tolerance, while the
    slowest time uses the cached rate minus it. This intentionally differs
    from applying the tolerance directly to elapsed time.
    """

    count = max(0, int(review_count))
    rate = float(reviews_per_minute)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Benchmark reviews per minute must be finite and positive.")
    tolerance = min(0.99, max(0.0, float(speed_tolerance)))
    estimated_seconds = count * 60.0 / rate
    fastest_rate = rate * (1.0 + tolerance)
    slowest_rate = rate * (1.0 - tolerance)
    return RustCheckpointProcessingTimeEstimate(
        review_count=count,
        estimated_seconds=estimated_seconds,
        lower_seconds=count * 60.0 / fastest_rate,
        upper_seconds=count * 60.0 / slowest_rate,
    )


def ensure_rust_checkpoint_disk_space(
    rows: Iterable[dict[str, Any]],
    checkpoint_path: Path,
    *,
    safety_margin: float = RUST_CHECKPOINT_SAFETY_MARGIN,
    expected_checkpoint_bytes: int | None = None,
) -> RustCheckpointStorageEstimate:
    estimate = estimate_rust_checkpoint_storage(
        rows,
        safety_margin=safety_margin,
        expected_checkpoint_bytes=expected_checkpoint_bytes,
    )
    ensure_rust_checkpoint_disk_space_for_estimate(
        estimate,
        checkpoint_path,
        safety_margin=safety_margin,
    )
    return estimate


def ensure_rust_checkpoint_disk_space_for_estimate(
    estimate: RustCheckpointStorageEstimate,
    checkpoint_path: Path,
    *,
    safety_margin: float = RUST_CHECKPOINT_SAFETY_MARGIN,
) -> None:
    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    available = int(shutil.disk_usage(str(checkpoint_dir)).free)
    if available < estimate.required_free_bytes:
        raise InsufficientCheckpointDiskSpaceError(
            _low_disk_space_message(
                checkpoint_path=Path(checkpoint_path),
                estimate=estimate,
                available_bytes=available,
                safety_margin=safety_margin,
            )
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _low_disk_space_message(
    *,
    checkpoint_path: Path,
    estimate: RustCheckpointStorageEstimate,
    available_bytes: int,
    safety_margin: float,
) -> str:
    percent = int(round(max(0.0, float(safety_margin)) * 100))
    return (
        "Not enough free disk space to save the RWKV checkpoint. "
        "The checkpoint write was canceled before creating the temporary "
        "checkpoint file.\n\n"
        f"Checkpoint folder: {checkpoint_path.parent}\n"
        f"Estimated checkpoint size: {format_storage_bytes(estimate.estimated_checkpoint_bytes)}\n"
        f"Required free space with {percent}% safety margin: "
        f"{format_storage_bytes(estimate.required_free_bytes)}\n"
        f"Available free space: {format_storage_bytes(available_bytes)}\n\n"
        "Free disk space, then try updating or rebuilding the checkpoint again."
    )


def format_storage_bytes(value: int) -> str:
    amount = float(max(0, int(value)))
    units = ("bytes", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            if unit == "bytes":
                return f"{int(amount)} bytes"
            return f"{amount:.1f} {unit}"
        amount /= 1024.0


def format_processing_time_range(
    estimate: RustCheckpointProcessingTimeEstimate,
) -> str:
    if estimate.review_count <= 0:
        return "less than 1 second"
    lower = _format_duration(estimate.lower_seconds)
    upper = _format_duration(estimate.upper_seconds)
    if lower == upper:
        return lower
    return f"{lower} to {upper}"


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 1.0:
        return "less than 1 second"
    if seconds < 90.0:
        rounded = max(1, int(round(seconds)))
        return f"{rounded} second" if rounded == 1 else f"{rounded} seconds"
    minutes = seconds / 60.0
    if minutes < 90.0:
        return _format_decimal_unit(minutes, "minute")
    hours = minutes / 60.0
    if hours < 48.0:
        return _format_decimal_unit(hours, "hour")
    days = hours / 24.0
    return _format_decimal_unit(days, "day")


def _format_decimal_unit(value: float, unit: str) -> str:
    rounded = round(float(value), 1)
    text = str(int(rounded)) if rounded.is_integer() else f"{rounded:.1f}"
    if text == "1":
        return f"{text} {unit}"
    return f"{text} {unit}s"
