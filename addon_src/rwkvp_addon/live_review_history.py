from __future__ import annotations

import math
import sqlite3
import time
import uuid
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .live_review_stats import (
    LiveRetentionCategory,
    LiveRetentionRecord,
    LiveRetentionSummary,
    summarize_live_retention_records,
)
from .progress import CancelledError, ProgressReporter

SCHEMA_VERSION = 2
SECONDS_PER_DAY = 86_400


class LiveReviewPeriod(str, Enum):
    WEEK = "week"
    MONTH = "month"
    THREE_MONTHS = "three_months"
    YEAR = "year"
    ALL = "all"


class LiveReviewScheduling(str, Enum):
    ALL = "all"
    RWKV = "rwkv"
    RANDOM = "random"

    def includes(self, record: LiveRetentionRecord) -> bool:
        if self is LiveReviewScheduling.ALL:
            return True
        return bool(record.randomly_selected) is (self is LiveReviewScheduling.RANDOM)


LIVE_REVIEW_SCHEDULING_LABELS = {
    LiveReviewScheduling.ALL: "All",
    LiveReviewScheduling.RWKV: "RWKV-Scheduled",
    LiveReviewScheduling.RANDOM: "Randomly Scheduled",
}


LIVE_REVIEW_PERIOD_DAYS: dict[LiveReviewPeriod, int | None] = {
    LiveReviewPeriod.WEEK: 7,
    # Match the range boundaries used by Anki's Reviews graph.
    LiveReviewPeriod.MONTH: 31,
    LiveReviewPeriod.THREE_MONTHS: 90,
    LiveReviewPeriod.YEAR: 365,
    LiveReviewPeriod.ALL: None,
}


@dataclass(frozen=True)
class LiveReviewSessionContext:
    started_at_ms: int | None = None
    ended_at_ms: int | None = None
    source_deck_id: int | None = None
    source_deck_name: str | None = None
    same_day_only: bool = False
    allow_same_day_repeats: bool = False
    review_limit: int | None = None
    minimum_review_limit: int = 0
    order_index: int | None = None
    model_id: str | None = None
    fsrs_comparison_enabled: bool = True


@dataclass(frozen=True)
class LiveReviewHistorySession:
    session_id: str
    started_at_ms: int
    ended_at_ms: int
    source_deck_id: int | None
    source_deck_name: str | None
    same_day_only: bool
    allow_same_day_repeats: bool
    review_limit: int | None
    minimum_review_limit: int
    order_index: int | None
    model_id: str | None
    fsrs_comparison_enabled: bool
    skipped_count: int
    summary: LiveRetentionSummary

    @property
    def review_count(self) -> int:
        return self.summary.review_count


@dataclass(frozen=True)
class LiveReviewHistoryOverview:
    summary: LiveRetentionSummary
    sessions: tuple[LiveReviewHistorySession, ...]


def records_in_live_review_period(
    records: tuple[LiveRetentionRecord, ...],
    period: LiveReviewPeriod,
    *,
    next_day_at_seconds: int,
) -> tuple[LiveRetentionRecord, ...]:
    """Return records in an Anki-style calendar-day reporting window."""

    days = LIVE_REVIEW_PERIOD_DAYS[period]
    if days is None:
        return records
    end_ms = int(next_day_at_seconds) * 1000
    start_ms = end_ms - days * SECONDS_PER_DAY * 1000
    return tuple(record for record in records if start_ms < _record_answered_at_ms(record) < end_ms)


def _record_answered_at_ms(record: LiveRetentionRecord) -> int:
    answered_at_ms = record.answered_at_ms
    if answered_at_ms is not None and int(answered_at_ms) >= 0:
        return int(answered_at_ms)
    return int(record.review_id)


def append_live_review_history_session(
    path: Path,
    summary: LiveRetentionSummary,
    context: LiveReviewSessionContext,
    *,
    session_id: str | None = None,
) -> str | None:
    if summary.review_count <= 0:
        return None
    resolved_session_id = session_id or uuid.uuid4().hex
    now_ms = int(time.time() * 1000)
    started_at_ms = _non_negative_int(context.started_at_ms) or now_ms
    ended_at_ms = _non_negative_int(context.ended_at_ms) or now_ms
    if ended_at_ms < started_at_ms:
        ended_at_ms = started_at_ms

    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        _ensure_schema(db)
        db.execute(
            """
            INSERT OR REPLACE INTO live_review_sessions (
                session_id,
                started_at_ms,
                ended_at_ms,
                source_deck_id,
                source_deck_name,
                same_day_only,
                allow_same_day_repeats,
                review_limit,
                minimum_review_limit,
                order_index,
                model_id,
                fsrs_comparison_enabled,
                review_count,
                skipped_count,
                created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_session_id,
                started_at_ms,
                ended_at_ms,
                _optional_int(context.source_deck_id),
                context.source_deck_name,
                _bool_int(context.same_day_only),
                _bool_int(context.allow_same_day_repeats),
                _optional_int(context.review_limit),
                int(context.minimum_review_limit),
                _optional_int(context.order_index),
                context.model_id,
                _bool_int(context.fsrs_comparison_enabled),
                summary.review_count,
                int(summary.skipped_count),
                now_ms,
            ),
        )
        db.execute(
            "DELETE FROM live_review_reviews WHERE session_id = ?",
            (resolved_session_id,),
        )
        db.executemany(
            """
            INSERT INTO live_review_reviews (
                session_id,
                review_id,
                card_id,
                category,
                rating,
                elapsed_days,
                remembered,
                rwkv_predicted_retrievability,
                fsrs_predicted_retrievability,
                source_deck_id,
                desired_retention,
                active_desired_retention,
                rwkv_stability_days,
                fsrs_difficulty,
                answered_at_ms,
                randomly_selected,
                ordinal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _review_values(resolved_session_id, record, ordinal)
                for ordinal, record in enumerate(summary.records)
            ),
        )
    return resolved_session_id


def load_live_review_history(
    path: Path,
    *,
    session_limit: int | None = 200,
    progress: ProgressReporter | None = None,
) -> LiveReviewHistoryOverview:
    progress = progress or ProgressReporter()
    progress.check_cancelled()
    if not path.exists():
        return LiveReviewHistoryOverview(
            summary=summarize_live_retention_records(()),
            sessions=(),
        )
    with closing(sqlite3.connect(path)) as db, db:
        db.row_factory = sqlite3.Row
        cancelled: CancelledError | None = None

        def check_sqlite_cancelled() -> int:
            nonlocal cancelled
            try:
                progress.check_cancelled()
            except CancelledError as exc:
                cancelled = exc
                return 1
            return 0

        # SQLite may spend time sorting before it yields its first row.
        db.set_progress_handler(check_sqlite_cancelled, 10_000)
        try:
            _ensure_schema(db)
            session_rows = _load_session_rows(db, session_limit=session_limit)
            selected_sessions = {str(row["session_id"]) for row in session_rows}
            total = int(db.execute("SELECT count(*) FROM live_review_reviews").fetchone()[0])
            all_records: list[LiveRetentionRecord] = []
            grouped: dict[str, list[tuple[int, int, LiveRetentionRecord]]] = defaultdict(list)
            progress.update(0, total, "Reading saved Live reviews…")
            for index, row in enumerate(db.execute(
                """
                SELECT *
                FROM live_review_reviews
                ORDER BY answered_at_ms, ordinal, review_id
                """
            )):
                if index % 512 == 0:
                    progress.update(index, total, "Reading saved Live reviews…")
                record = _record_from_row(row)
                if record is None:
                    continue
                all_records.append(record)
                session_id = str(row["session_id"])
                if session_id in selected_sessions:
                    grouped[session_id].append((int(row["ordinal"]), record.review_id, record))
            progress.update(total, total, "Read saved Live reviews")
        except sqlite3.OperationalError:
            if cancelled is not None:
                raise cancelled from None
            raise
        finally:
            db.set_progress_handler(None, 0)

    sessions = []
    for index, row in enumerate(session_rows):
        progress.update(index, len(session_rows), "Summarizing saved Live sessions…")
        # Preserve the independent session ordinal order, even when answer
        # timestamps are not monotonic. Hydrate each immutable record only once.
        session_records = grouped.get(str(row["session_id"]), [])
        session_records.sort(key=lambda item: (item[0], item[1]))
        sessions.append(_history_session_from_row(
            row, tuple(item[2] for item in session_records), progress=progress,
        ))
    progress.update(len(sessions), len(sessions), "Summarizing all saved Live reviews…")
    summary = summarize_live_retention_records(all_records, progress=progress)
    progress.check_cancelled()
    return LiveReviewHistoryOverview(
        summary=summary,
        sessions=tuple(sessions),
    )


def _ensure_schema(db: sqlite3.Connection) -> None:
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS live_review_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS live_review_sessions (
            session_id TEXT PRIMARY KEY,
            started_at_ms INTEGER NOT NULL,
            ended_at_ms INTEGER NOT NULL,
            source_deck_id INTEGER,
            source_deck_name TEXT,
            same_day_only INTEGER NOT NULL DEFAULT 0,
            allow_same_day_repeats INTEGER NOT NULL DEFAULT 0,
            review_limit INTEGER,
            minimum_review_limit INTEGER NOT NULL DEFAULT 0,
            order_index INTEGER,
            model_id TEXT,
            fsrs_comparison_enabled INTEGER NOT NULL DEFAULT 1,
            review_count INTEGER NOT NULL,
            skipped_count INTEGER NOT NULL DEFAULT 0,
            created_at_ms INTEGER NOT NULL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS live_review_reviews (
            session_id TEXT NOT NULL,
            review_id INTEGER NOT NULL,
            card_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            rating INTEGER,
            elapsed_days REAL,
            remembered INTEGER NOT NULL,
            rwkv_predicted_retrievability REAL NOT NULL,
            fsrs_predicted_retrievability REAL,
            source_deck_id INTEGER,
            desired_retention REAL,
            active_desired_retention REAL,
            rwkv_stability_days REAL,
            fsrs_difficulty REAL,
            answered_at_ms INTEGER,
            randomly_selected INTEGER NOT NULL DEFAULT 0,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY (session_id, review_id),
            FOREIGN KEY (session_id)
                REFERENCES live_review_sessions(session_id)
                ON DELETE CASCADE
        )
        """
    )
    # Existing users' saved history predates random sampling. Missing flags
    # belong to RWKV-Scheduled; only explicit random draws form the random cohort.
    columns = {row[1] for row in db.execute("PRAGMA table_info(live_review_reviews)")}
    if "randomly_selected" not in columns:
        db.execute(
            "ALTER TABLE live_review_reviews "
            "ADD COLUMN randomly_selected INTEGER NOT NULL DEFAULT 0"
        )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS live_review_reviews_session_idx
        ON live_review_reviews(session_id, ordinal)
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS live_review_sessions_ended_idx
        ON live_review_sessions(ended_at_ms DESC)
        """
    )
    db.execute(
        "INSERT OR REPLACE INTO live_review_metadata(key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )


def _load_session_rows(
    db: sqlite3.Connection,
    *,
    session_limit: int | None,
) -> list[sqlite3.Row]:
    query = """
        SELECT *
        FROM live_review_sessions
        ORDER BY ended_at_ms DESC, started_at_ms DESC
    """
    if session_limit is None:
        return list(db.execute(query).fetchall())
    return list(db.execute(query + " LIMIT ?", (max(0, int(session_limit)),)).fetchall())


def _history_session_from_row(
    row: sqlite3.Row,
    records: tuple[LiveRetentionRecord, ...],
    *,
    progress: ProgressReporter | None = None,
) -> LiveReviewHistorySession:
    return LiveReviewHistorySession(
        session_id=str(row["session_id"]),
        started_at_ms=int(row["started_at_ms"]),
        ended_at_ms=int(row["ended_at_ms"]),
        source_deck_id=_optional_int(row["source_deck_id"]),
        source_deck_name=row["source_deck_name"],
        same_day_only=bool(row["same_day_only"]),
        allow_same_day_repeats=bool(row["allow_same_day_repeats"]),
        review_limit=_optional_int(row["review_limit"]),
        minimum_review_limit=int(row["minimum_review_limit"]),
        order_index=_optional_int(row["order_index"]),
        model_id=row["model_id"],
        fsrs_comparison_enabled=bool(row["fsrs_comparison_enabled"]),
        skipped_count=int(row["skipped_count"]),
        summary=summarize_live_retention_records(
            records,
            skipped_count=int(row["skipped_count"]),
            progress=progress,
        ),
    )


def _review_values(
    session_id: str,
    record: LiveRetentionRecord,
    ordinal: int,
) -> tuple:
    return (
        session_id,
        int(record.review_id),
        int(record.card_id),
        record.category.value,
        _optional_int(record.rating),
        _sqlite_float(record.elapsed_days),
        _bool_int(record.remembered),
        float(record.predicted_retrievability),
        _sqlite_float(record.fsrs_predicted_retrievability),
        _optional_int(record.source_deck_id),
        _sqlite_float(record.desired_retention),
        _sqlite_float(record.active_desired_retention),
        _sqlite_float(record.rwkv_stability_days),
        _sqlite_float(record.fsrs_difficulty),
        _optional_int(record.answered_at_ms) or int(record.review_id),
        _bool_int(record.randomly_selected),
        int(ordinal),
    )


def _record_from_row(row: sqlite3.Row) -> LiveRetentionRecord | None:
    try:
        category = LiveRetentionCategory(str(row["category"]))
        prediction = float(row["rwkv_predicted_retrievability"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(prediction):
        return None
    return LiveRetentionRecord(
        review_id=int(row["review_id"]),
        card_id=int(row["card_id"]),
        category=category,
        predicted_retrievability=prediction,
        remembered=bool(row["remembered"]),
        fsrs_predicted_retrievability=_sqlite_optional_float(
            row["fsrs_predicted_retrievability"]
        ),
        rating=_optional_int(row["rating"]),
        elapsed_days=_sqlite_optional_float(row["elapsed_days"]),
        source_deck_id=_optional_int(row["source_deck_id"]),
        desired_retention=_sqlite_optional_float(row["desired_retention"]),
        active_desired_retention=_sqlite_optional_float(
            row["active_desired_retention"]
        ),
        rwkv_stability_days=_sqlite_optional_float(row["rwkv_stability_days"]),
        fsrs_difficulty=_sqlite_optional_float(row["fsrs_difficulty"]),
        answered_at_ms=_optional_int(row["answered_at_ms"]),
        randomly_selected=row["randomly_selected"] == 1,
    )


def _sqlite_float(value: float | None) -> float | None:
    parsed = _sqlite_optional_float(value)
    return parsed if parsed is not None else None


def _sqlite_optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _non_negative_int(value) -> int | None:
    parsed = _optional_int(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _bool_int(value) -> int:
    return 1 if bool(value) else 0
