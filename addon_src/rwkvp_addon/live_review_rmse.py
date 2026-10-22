"""Recover benchmark bin identities for recorded Live forecasts without model replay."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .live_review_stats import LiveRetentionRecord, LiveRmseBinContext
from .progress import ProgressReporter


def live_rmse_bin_context(
    records: Iterable[LiveRetentionRecord],
    rows: Iterable[Mapping[str, Any]],
    progress: ProgressReporter,
) -> dict[tuple[int, int], LiveRmseBinContext]:
    """Match review/card/outcome and saved elapsed days; never invent counters."""
    wanted = {
        (record.review_id, record.card_id): record for record in _checked(records, progress)
    }
    result: dict[tuple[int, int], LiveRmseBinContext] = {}
    for index, row in enumerate(rows):
        if index % 10_000 == 0:
            progress.check_cancelled()
        key = (int(row["review_id"]), int(row["card_id"]))
        record = wanted.get(key)
        if record is None:
            continue
        try:
            elapsed = float(row["elapsed_days"])
            count = int(row["review_count"])
            lapses = int(row["prior_lapses"])
            rating = int(row["rating"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        context = _matching_context(record, elapsed, count, lapses, rating)
        if context is not None:
            result[key] = context
    progress.check_cancelled()
    return result


def _matching_context(
    record: LiveRetentionRecord, elapsed: float, count: int, lapses: int, rating: int,
) -> LiveRmseBinContext | None:
    if (
        not math.isfinite(elapsed) or elapsed <= 0 or count < 1 or lapses < 0
        or rating not in {1, 2, 3, 4} or (rating > 1) != record.remembered
        or (record.rating is not None and record.rating != rating)
        or (record.elapsed_days is not None and record.elapsed_days != elapsed)
    ):
        return None
    return LiveRmseBinContext(elapsed, count, lapses)


def live_rmse_bin_context_from_dataset(
    records: Iterable[LiveRetentionRecord], dataset: Any, progress: ProgressReporter,
) -> dict[tuple[int, int], LiveRmseBinContext]:
    """Recover only the requested bins, not model rows or hierarchy metadata.

    Match review_data_from_dataset's default benchmark path, independently of
    the active model generation. Full earlier history supplies counters; later
    Learning starts can invalidate earlier records. Keep all history for wanted
    cards until the canonical trimming boundaries have been established.
    """
    from .review_rows import (
        _learning_sequence_retention_boundaries,
        day_offset_for_timestamp,
        rwkv_processable_revlog,
    )

    wanted = {
        (record.review_id, record.card_id): record for record in _checked(records, progress)
    }
    if not wanted:
        return {}
    card_ids = {card_id for _, card_id in wanted}
    entries = sorted(
        (entry for entry in _checked(dataset.revlogs, progress) if int(entry.cid) in card_ids),
        key=lambda entry: int(entry.id),
    )
    starts, _ = _learning_sequence_retention_boundaries(_checked(entries, progress))
    cutoff = int(dataset.next_day_at)
    # Per-card (previous raw day, positive-day review count + 1, prior lapses).
    # These are the same counters as _append_review_values_to_context. The
    # collection-wide day origin cancels in elapsed-day differences.
    counters: dict[int, tuple[int, int, int]] = {}
    result: dict[tuple[int, int], LiveRmseBinContext] = {}
    for entry in _checked(entries, progress):
        card_id, review_id = int(entry.cid), int(entry.id)
        start = starts.get(card_id)
        if start is None or review_id < start or not rwkv_processable_revlog(entry):
            continue
        rating = int(entry.button_chosen)
        day = day_offset_for_timestamp(review_id / 1000.0, cutoff)
        previous = counters.get(card_id)
        if previous is None:
            elapsed, count, lapses = -1, 1, 0
        else:
            previous_day, count, lapses = previous
            elapsed = day - previous_day
            if elapsed > 0:
                count += 1
        key = (review_id, card_id)
        record = wanted.get(key)
        if record is not None:
            context = _matching_context(record, float(elapsed), count, lapses, rating)
            if context is not None:
                result[key] = context
        if elapsed > 0 and rating == 1:
            lapses += 1
        counters[card_id] = (day, count, lapses)
    return result


def _checked(items, progress):
    for index, item in enumerate(items):
        if index % 512 == 0:
            progress.check_cancelled()
        yield item
    progress.check_cancelled()


def load_live_rmse_bin_context(col, store, records, progress: ProgressReporter):
    # This is a collection read through Anki's public export API. Exported data
    # is temporary in the add-on cache; no checkpoint or runtime is opened.
    from .review_rows import load_research_dataset

    progress.check_cancelled()
    progress.update(0, 3, "Reading Anki history for RMSE(bins)…")
    store.cache_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="live-rmse-", dir=store.cache_dir) as temporary:
        path = Path(temporary) / "history.pb"
        col.export_dataset_for_research(str(path))
        progress.check_cancelled()
        progress.update(1, 3, "Preparing benchmark review counts and prior lapses…")
        dataset = load_research_dataset(path)
    progress.check_cancelled()
    progress.update(2, 3, "Matching recorded Live reviews to benchmark bins…")
    context = live_rmse_bin_context_from_dataset(records, dataset, progress)
    progress.update(3, 3, "RMSE(bins) review context ready.")
    return context
