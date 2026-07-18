from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from .review_rows import (
    SECONDS_PER_DAY,
    CardInfo,
    LastReviewInfo,
    day_offset_for_timestamp,
    prediction_row_for_card,
    rwkv_processable_revlog_where_sql,
)


@dataclass(frozen=True)
class CardInfoRetrievabilityValues:
    immediate: float | None
    forgetting_curve: float | None
    stability_interval_seconds: float | None = None
    desired_retention_interval_seconds: float | None = None
    desired_retention: float | None = None
    forgetting_curve_graph: CardInfoForgettingCurveGraph | None = None


@dataclass(frozen=True)
class CardInfoCurvePoint:
    elapsed_seconds: float
    retrievability: float


@dataclass(frozen=True)
class CardInfoForgettingCurveGraph:
    points: tuple[CardInfoCurvePoint, ...]
    desired_retention: float
    minimum_retrievability: float
    desired_retention_interval_seconds: float | None
    last_review_timestamp_seconds: float
    now_timestamp_seconds: float


def rwkv_card_info_retrievability(
    col,
    manager,
    card_id: int,
    *,
    runtime=None,
    target_timestamp_seconds: float | None = None,
    curve_predictor=None,
    interval_predictor=None,
    include_immediate: bool = True,
    include_forgetting_curve: bool = True,
    include_intervals: bool = False,
    include_forgetting_curve_graph: bool = False,
    forgetting_curve_graph_lower_bound: float = 0.6,
) -> CardInfoRetrievabilityValues | None:
    """Calculate Card Info values using an already-open runtime.

    This unit-testable core never loads a checkpoint itself. The GUI passes an
    already-loaded runtime owned by an active Live Session or by an explicit
    Browser Card Info load.
    """

    prediction_runtime = manager if runtime is None else runtime
    if runtime is None and not bool(getattr(manager, "runtime_loaded", False)):
        return None
    next_day_at = int(col.sched.day_cutoff)
    day_offset_origin = _day_offset_origin_from_loaded_review_data(
        manager,
        next_day_at=next_day_at,
    )
    if day_offset_origin is None:
        return None

    target_timestamp_seconds = (
        time.time() if target_timestamp_seconds is None else float(target_timestamp_seconds)
    )
    row = prediction_row_for_card_from_collection(
        col,
        int(card_id),
        target_timestamp_seconds=target_timestamp_seconds,
        day_offset_origin=day_offset_origin,
    )
    if row is None:
        return None

    curve = None
    forgetting_curve = None
    if include_forgetting_curve:
        curve = _latest_curve_for_card(prediction_runtime, int(card_id))
        forgetting_curve = _finite_prediction(
            _predict_forgetting_curve(
                curve,
                elapsed_seconds=float(row["elapsed_seconds"]),
                curve_predictor=curve_predictor,
            )
        )
    stability_interval = None
    desired_retention_interval = None
    desired_retention = None
    forgetting_curve_graph = None
    if (
        (include_intervals or include_forgetting_curve_graph)
        and interval_predictor is not None
        and curve is not None
    ):
        desired_retention = _desired_retention_for_card(col, int(card_id))
        desired_retention_interval = _finite_interval(
            _predict_curve_interval(
                curve,
                desired_retention,
                interval_predictor=interval_predictor,
            )
        )
        if include_intervals:
            stability_interval = _finite_interval(
                _predict_curve_interval(curve, 0.9, interval_predictor=interval_predictor)
            )
        if include_forgetting_curve_graph:
            forgetting_curve_graph = _build_forgetting_curve_graph(
                curve,
                desired_retention=desired_retention,
                desired_retention_interval_seconds=desired_retention_interval,
                current_elapsed_seconds=float(row["elapsed_seconds"]),
                current_retrievability=forgetting_curve,
                target_timestamp_seconds=target_timestamp_seconds,
                interval_predictor=interval_predictor,
                minimum_retrievability=forgetting_curve_graph_lower_bound,
            )

    return CardInfoRetrievabilityValues(
        immediate=(
            _finite_prediction(_predict_immediate(prediction_runtime, row))
            if include_immediate
            else None
        ),
        forgetting_curve=forgetting_curve,
        stability_interval_seconds=stability_interval,
        desired_retention_interval_seconds=desired_retention_interval,
        desired_retention=desired_retention,
        forgetting_curve_graph=forgetting_curve_graph,
    )


def prediction_row_for_card_from_collection(
    col,
    card_id: int,
    *,
    target_timestamp_seconds: float,
    day_offset_origin: int | None = None,
) -> dict[str, Any] | None:
    card = _card_info_from_collection(col, int(card_id))
    if card is None:
        return None
    next_day_at = int(col.sched.day_cutoff)
    if day_offset_origin is None:
        day_offset_origin = _day_offset_origin_from_collection(col, next_day_at)
    return prediction_row_for_card(
        card,
        _last_review_info_from_collection(
            col,
            int(card_id),
            next_day_at=next_day_at,
            day_offset_origin=day_offset_origin,
        ),
        target_timestamp_seconds=target_timestamp_seconds,
        next_day_at=next_day_at,
        day_offset_origin=day_offset_origin,
    )


def _day_offset_origin_from_loaded_review_data(
    manager,
    *,
    next_day_at: int,
) -> int | None:
    if not hasattr(manager, "cached_day_offset_origin"):
        return None
    try:
        try:
            value = manager.cached_day_offset_origin(next_day_at=next_day_at)
        except TypeError:
            # Compatibility for manager-like adapters that predate cutoff-aware
            # origin rebasing.
            value = manager.cached_day_offset_origin()
    except Exception:
        return None
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_retrievability_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.0f}%"


def format_rwkv_interval(value: float | None) -> str:
    if value is None:
        return "-"
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        return "-"
    units = (
        ("second", 1.0, 60.0),
        ("minute", 60.0, 60.0),
        ("hour", 3_600.0, 24.0),
        ("day", float(SECONDS_PER_DAY), 30.0),
        ("month", float(SECONDS_PER_DAY * 30), 12.0),
        ("year", float(SECONDS_PER_DAY * 365), math.inf),
    )
    name = "second"
    amount = seconds
    for candidate_name, scale, next_threshold in units:
        candidate_amount = seconds / scale
        name = candidate_name
        amount = candidate_amount
        if candidate_amount < next_threshold:
            break
    if name in {"month", "year"}:
        total_days = max(0, int(round(seconds / SECONDS_PER_DAY)))
        day_label = "day" if total_days == 1 else "days"
        return f"{amount:.2f} {name}s ({total_days} {day_label})"
    rounded = max(0, int(round(amount)))
    if rounded != 1:
        name = f"{name}s"
    return f"{rounded} {name}"


def _card_info_from_collection(col, card_id: int) -> CardInfo | None:
    row = col.db.first(
        """
        SELECT id, nid, CASE WHEN odid = 0 THEN did ELSE odid END AS did
        FROM cards
        WHERE id = ?
        """,
        int(card_id),
    )
    if row is None:
        return None
    deck_id = int(row[2]) if row[2] is not None else None
    return CardInfo(
        card_id=int(row[0]),
        note_id=int(row[1]) if row[1] is not None else None,
        deck_id=deck_id,
        preset_id=_preset_id_for_deck(col, deck_id),
    )


def _preset_id_for_deck(col, deck_id: int | None) -> int | None:
    if deck_id is None:
        return None
    try:
        deck = col.decks.get(int(deck_id))
    except Exception:
        return None
    if not deck:
        return None
    preset_id = deck.get("conf")
    return int(preset_id) if preset_id is not None else None


def _day_offset_origin_from_collection(col, next_day_at: int) -> int:
    review_id = col.db.scalar(
        f"""
        SELECT min(id)
        FROM revlog
        WHERE {rwkv_processable_revlog_where_sql("revlog")}
        """
    )
    if review_id is None:
        return 0
    return day_offset_for_timestamp(int(review_id) / 1000.0, next_day_at)


def _last_review_info_from_collection(
    col,
    card_id: int,
    *,
    next_day_at: int,
    day_offset_origin: int,
) -> LastReviewInfo | None:
    row = col.db.first(
        f"""
        SELECT id, ivl
        FROM revlog
        WHERE cid = ?
          AND {rwkv_processable_revlog_where_sql("revlog")}
        ORDER BY id DESC
        LIMIT 1
        """,
        int(card_id),
    )
    if row is None:
        return None
    review_id = int(row[0])
    raw_day_offset = day_offset_for_timestamp(review_id / 1000.0, next_day_at)
    return LastReviewInfo(
        review_id=review_id,
        day_offset=raw_day_offset - day_offset_origin,
        timestamp_seconds=review_id / 1000.0,
        interval=int(row[1]),
        lapse_count=0,
    )


def _predict_immediate(manager, row: dict[str, Any]) -> float | None:
    try:
        # Card Info is a one-card interactive query. Keep it on the CPU fallback
        # even when the active Live Session retains a GPU cache for bulk work.
        if hasattr(manager, "predict_many_progress_chunk_size"):
            predictions = manager.predict_many([row], allow_gpu=False)
        else:
            predictions = manager.predict_many([row])
    except Exception:
        return None
    if not predictions:
        return None
    return float(predictions[0])


def _predict_forgetting_curve(
    curve,
    *,
    elapsed_seconds: float,
    curve_predictor,
) -> float | None:
    if elapsed_seconds < 0:
        return None
    if curve is None:
        return None
    try:
        return float(curve_predictor(curve, elapsed_seconds))
    except Exception:
        return None


def _latest_curve_for_card(manager, card_id: int):
    if hasattr(manager, "latest_curve_for_card"):
        return manager.latest_curve_for_card(int(card_id))
    return manager.latest_curves_by_card().get(int(card_id))


def _predict_curve_interval(curve, retention: float, *, interval_predictor) -> float | None:
    try:
        return float(interval_predictor(curve, float(retention)))
    except Exception:
        return None


def _build_forgetting_curve_graph(
    curve,
    *,
    desired_retention: float,
    desired_retention_interval_seconds: float | None,
    current_elapsed_seconds: float,
    current_retrievability: float | None,
    target_timestamp_seconds: float,
    interval_predictor,
    minimum_retrievability: float = 0.6,
) -> CardInfoForgettingCurveGraph | None:
    if current_elapsed_seconds < 0:
        return None
    minimum_retrievability = _normalized_graph_lower_bound(minimum_retrievability)
    last_review_timestamp = float(target_timestamp_seconds) - float(current_elapsed_seconds)
    points = [CardInfoCurvePoint(elapsed_seconds=0.0, retrievability=1.0)]
    lower_percent = int(math.ceil(minimum_retrievability * 100.0))
    for percent in range(99, lower_percent - 1, -1):
        retrievability = percent / 100.0
        interval = _finite_interval(
            _predict_curve_interval(
                curve,
                retrievability,
                interval_predictor=interval_predictor,
            )
        )
        if interval is None:
            continue
        points.append(
            CardInfoCurvePoint(
                elapsed_seconds=interval,
                retrievability=retrievability,
            )
        )
    if (
        current_retrievability is not None
        and minimum_retrievability <= current_retrievability <= 1.0
    ):
        points.append(
            CardInfoCurvePoint(
                elapsed_seconds=float(current_elapsed_seconds),
                retrievability=float(current_retrievability),
            )
        )

    deduped = _dedupe_curve_points(points)
    if len(deduped) < 2:
        return None
    return CardInfoForgettingCurveGraph(
        points=tuple(deduped),
        desired_retention=float(desired_retention),
        minimum_retrievability=minimum_retrievability,
        desired_retention_interval_seconds=desired_retention_interval_seconds,
        last_review_timestamp_seconds=last_review_timestamp,
        now_timestamp_seconds=float(target_timestamp_seconds),
    )


def _dedupe_curve_points(points: list[CardInfoCurvePoint]) -> list[CardInfoCurvePoint]:
    deduped: dict[float, CardInfoCurvePoint] = {}
    for point in points:
        elapsed = float(point.elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0:
            continue
        deduped[elapsed] = point
    return [
        deduped[elapsed]
        for elapsed in sorted(deduped)
        if math.isfinite(float(deduped[elapsed].retrievability))
    ]


def _desired_retention_for_card(col, card_id: int) -> float:
    card = _card_info_from_collection(col, int(card_id))
    if card is None or card.deck_id is None:
        return 0.9
    try:
        configs = col.decks.get_deck_configs_for_update(int(card.deck_id))
        desired = None
        limits = configs.current_deck.limits
        try:
            if limits.HasField("desired_retention"):
                desired = float(limits.desired_retention)
        except Exception:
            desired = float(getattr(limits, "desired_retention", 0) or 0) or None
        if desired is not None:
            return float(desired)
        config_id = int(configs.current_deck.config_id)
        for config_with_extra in configs.all_config:
            config = config_with_extra.config
            if int(config.id) == config_id:
                return float(getattr(config.config, "desired_retention", 0.9) or 0.9)
    except Exception:
        return 0.9
    return 0.9


def _finite_prediction(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _finite_interval(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _normalized_graph_lower_bound(value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.6
    if not math.isfinite(parsed):
        return 0.6
    return max(0.0, min(0.99, parsed))
