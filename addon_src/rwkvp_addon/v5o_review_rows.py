"""V5o retained-history inputs; learned clocks and summaries stay native."""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .compact_v5o_history import PackedV5oReviewRows
from .review_rows import CardInfo, LastReviewInfo, ReviewData
from .timestamp_context import MILLISECONDS_PER_DAY, rollover_phase_ms_from_next_day_at

V5O_PREDICT_FIELDS = ("card_id", "target_timestamp_ms", "rollover_phase_ms")
V5O_OPTIONAL_FIELDS = ("note_id", "deck_id", "preset_id", "card_creation_ms")
V5O_PROCESS_FIELDS = (
    "review_id",
    "card_id",
    *V5O_OPTIONAL_FIELDS,
    "review_start_ms",
    "review_finish_ms",
    "duration_ms",
    "source_ordinal",
    "rating",
    "is_original_first_review",
    "rollover_phase_ms",
)


def exact_i64(value: Any, field: str) -> int:
    if type(value) is int:
        result = value
    else:
        if isinstance(value, bool):
            raise TypeError(f"{field} must be an integer, not bool")
        try:
            result = int(operator.index(value))
        except TypeError:
            raise TypeError(f"{field} must be an exact integer, not a float") from None
    if not -(1 << 63) <= result < (1 << 63):
        raise OverflowError(f"{field} must fit a signed 64-bit integer")
    return result


def _phase(value: Any) -> int:
    result = exact_i64(value, "rollover_phase_ms")
    if not 0 <= result < MILLISECONDS_PER_DAY:
        raise ValueError("rollover_phase_ms must be in [0, 86400000)")
    return result


def _card_id(value: Any) -> int:
    result = exact_i64(value, "card_id")
    if result < 0:
        raise ValueError("card_id must be nonnegative")
    return result


def v5o_answer_input_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a complete answer without accepting legacy float clocks."""

    answer = {
        field: exact_i64(row[field], field)
        for field in (
            "review_id",
            "review_start_ms",
            "review_finish_ms",
            "duration_ms",
            "source_ordinal",
            "rating",
        )
    }
    answer["card_id"] = _card_id(row["card_id"])
    answer["rollover_phase_ms"] = _phase(row["rollover_phase_ms"])
    first = row["is_original_first_review"]
    if type(first) is not bool:
        raise TypeError("is_original_first_review must be a Boolean")
    answer["is_original_first_review"] = first
    if answer["rating"] not in {1, 2, 3, 4}:
        raise ValueError("rating must be an integer from 1 through 4")
    if answer["duration_ms"] < 0 or (
        answer["review_finish_ms"] - answer["review_start_ms"] != answer["duration_ms"]
    ):
        raise ValueError("duration_ms must be nonnegative and equal finish minus start")
    for field in V5O_OPTIONAL_FIELDS:
        value = row.get(field)
        answer[field] = None if value is None else exact_i64(value, field)
    return answer


def v5o_prediction_input_row(row: Mapping[str, Any]) -> dict[str, int]:
    return {
        "card_id": _card_id(row["card_id"]),
        "target_timestamp_ms": exact_i64(row["target_timestamp_ms"], "target_timestamp_ms"),
        "rollover_phase_ms": _phase(row["rollover_phase_ms"]),
    }


def v5o_live_candidate_input_row(row: Mapping[str, Any]) -> dict[str, int]:
    return {"card_id": _card_id(row["card_id"])}


@dataclass(frozen=True)
class V5oInputContext:
    """Immutable replay catalog and phase, with current inventory kept separate.

    Deleted-card reviews can contribute to V5o-a's shared state. Their original
    hierarchy and creation time are unknown, never inferred from opaque IDs.
    They belong to the native replay catalog, not Anki's selectable inventory.
    """

    rollover_phase_ms: int
    cards: Mapping[int, CardInfo]
    deleted_card_ids: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rollover_phase_ms", _phase(self.rollover_phase_ms))
        catalog: dict[int, CardInfo] = {}
        for key, card in self.cards.items():
            card_id = _card_id(key)
            if card_id != _card_id(card.card_id):
                raise ValueError("Card catalog key and card identity disagree")
            values = [
                None if (value := getattr(card, field)) is None else exact_i64(value, field)
                for field in V5O_OPTIONAL_FIELDS
            ]
            catalog[card_id] = CardInfo(card_id, *values)
        object.__setattr__(self, "cards", MappingProxyType(catalog))
        deleted_ids = frozenset(_card_id(value) for value in self.deleted_card_ids)
        if not deleted_ids.issubset(catalog):
            raise ValueError("Deleted-history cards must belong to the replay catalog")
        object.__setattr__(self, "deleted_card_ids", deleted_ids)

    @classmethod
    def from_review_data(cls, review_data: ReviewData) -> V5oInputContext:
        catalog = dict(review_data.cards)
        # Prepared V5 history already indexes every processed identity. Do not
        # rescan all reviews when refreshing a context after a Live transition.
        history_ids = (
            review_data.row_build_context.previous_by_card
            if review_data.row_build_context is not None
            else (row["card_id"] for row in review_data.rows)
        )
        for value in history_ids:
            card_id = _card_id(value)
            if card_id not in catalog:
                catalog[card_id] = CardInfo(card_id, None, None, None, None)
        return cls(
            rollover_phase_ms_from_next_day_at(review_data.next_day_at), catalog,
            frozenset(catalog.keys() - review_data.cards.keys()),
        )

    def without_deleted_history(self) -> V5oInputContext:
        """Keep current unreviewed cards, but no historical-only identities."""
        return V5oInputContext(
            self.rollover_phase_ms,
            {card_id: card for card_id, card in self.cards.items()
             if card_id not in self.deleted_card_ids},
        )

    def catalog_rows(self) -> list[dict[str, Any]]:
        return [
            {"card_id": card_id, **{field: getattr(card, field) for field in V5O_OPTIONAL_FIELDS}}
            for card_id, card in sorted(self.cards.items())
        ]

    def catalog_fingerprint(self) -> str:
        encoded = json.dumps(self.catalog_rows(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def process_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        card_id = _card_id(row["card_id"])
        card = self.cards.get(card_id)
        if card is None:
            raise ValueError(f"V5o answer card {card_id} is outside the replay catalog")
        # Source ordinals and first-learning flags are owned by history preparation,
        # never inferred from an already filtered V1-shaped row here.
        finish = exact_i64(row.get("review_finish_ms", row["review_id"]), "review_finish_ms")
        duration = exact_i64(
            row["duration_ms"] if "duration_ms" in row else row["taken_millis"], "duration_ms"
        )
        return v5o_answer_input_row(
            {
                "review_id": row["review_id"],
                "card_id": card_id,
                **{field: getattr(card, field) for field in V5O_OPTIONAL_FIELDS},
                "review_start_ms": row.get("review_start_ms", finish - duration),
                "review_finish_ms": finish,
                "duration_ms": duration,
                "source_ordinal": row["source_ordinal"],
                "is_original_first_review": row["is_original_first_review"],
                "rating": row["rating"],
                "rollover_phase_ms": self.rollover_phase_ms,
            }
        )

    def prediction_row(self, row: Mapping[str, Any]) -> dict[str, int]:
        target = row["target_timestamp_ms"] if "target_timestamp_ms" in row else row["review_id"]
        return v5o_prediction_input_row(
            {
                "card_id": row["card_id"],
                "target_timestamp_ms": target,
                "rollover_phase_ms": self.rollover_phase_ms,
            }
        )

    def prediction_row_for_card(
        self,
        card: CardInfo,
        last_review: LastReviewInfo | None,
        *,
        target_timestamp_ms: int,
        next_day_at: int,
        day_offset_origin: int = 0,
    ) -> dict[str, int]:
        if last_review is None:
            raise ValueError("V5o cannot query a card before its first retained answer")
        return self.prediction_row(
            {"card_id": card.card_id, "target_timestamp_ms": target_timestamp_ms}
        )

    def live_candidate_row(self, row: Mapping[str, Any]) -> dict[str, int]:
        return v5o_live_candidate_input_row(row)

    def process_columns(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
        return v5o_process_input_columns(rows, context=self)


def v5o_process_input_columns(
    rows: Sequence[Mapping[str, Any]], *, context: V5oInputContext | None = None
) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {field: [] for field in V5O_PROCESS_FIELDS}
    if isinstance(rows, PackedV5oReviewRows) and context is not None:
        # Prepared records already retain exact i64 source facts. Read their
        # scalar buffers once, rather than rebuilding/validating a dictionary
        # per answer. The native V5oReviewBatch still validates every column.
        (
            review_ids, card_ids, note_ids, deck_ids, preset_ids, creation_times,
            starts, finishes, durations, ordinals, ratings, first_flags, phases,
        ) = columns.values()
        for finish, card_id, duration, rating, ordinal, first in rows.iter_answer_values():
            if card_id < 0:
                _card_id(card_id)
            card = context.cards.get(card_id)
            if card is None:
                raise ValueError(f"V5o answer card {card_id} is outside the replay catalog")
            if rating not in {1, 2, 3, 4}:
                raise ValueError("rating must be an integer from 1 through 4")
            if duration < 0:
                raise ValueError("duration_ms must be nonnegative and equal finish minus start")
            start = exact_i64(finish - duration, "review_start_ms")
            review_ids.append(finish)
            card_ids.append(card_id)
            note_ids.append(card.note_id)
            deck_ids.append(card.deck_id)
            preset_ids.append(card.preset_id)
            creation_times.append(card.card_creation_ms)
            starts.append(start)
            finishes.append(finish)
            durations.append(duration)
            ordinals.append(ordinal)
            ratings.append(rating)
            first_flags.append(first)
            phases.append(context.rollover_phase_ms)
        return columns
    for source in rows:
        row = context.process_row(source) if context is not None else v5o_answer_input_row(source)
        for field in columns:
            columns[field].append(row[field])
    return columns


def v5o_prediction_input_columns(
    rows: Sequence[Mapping[str, Any]], *, context: V5oInputContext | None = None
) -> dict[str, list[int]]:
    columns: dict[str, list[int]] = {field: [] for field in V5O_PREDICT_FIELDS}
    for source in rows:
        row = (
            context.prediction_row(source)
            if context is not None
            else v5o_prediction_input_row(source)
        )
        for field in columns:
            columns[field].append(row[field])
    return columns
