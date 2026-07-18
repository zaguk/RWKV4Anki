from __future__ import annotations

import math
import random
import re
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

from .review_rows import (
    NEW_CARD_ELAPSED,
    CardInfo,
    LastReviewInfo,
    ReviewData,
    day_offset_for_timestamp,
)

BEHAVIOR_LAB_SCHEMA_VERSION = 1
MAX_SCENARIOS = 32
MAX_EVENTS_PER_SCENARIO = 2_000
MAX_CONTEXT_REVIEWS = 60_000
MAX_TRACKED_CARDS = 100_000
MAX_REVIEW_DURATION_SECONDS = 60.0
SIMULATION_CONTEXT_CHUNK_SIZE = 10_000

EventKind = Literal["review", "wait", "context", "observe"]
ReviewContext = Literal["new", "learning", "review", "relearning", "filtered", "collection"]
ContextScope = Literal[
    "unrelated",
    "siblings",
    "same_deck",
    "same_preset",
    "selection",
    "collection",
]
ContextRatingMode = Literal["collection", "fixed"]

REVIEW_CONTEXT_STATES: dict[str, int] = {
    "new": 0,
    "learning": 1,
    "review": 2,
    "relearning": 3,
    "filtered": 4,
}

DEFAULT_CURVE_SAMPLE_SECONDS = (
    1.0,
    10.0,
    60.0,
    10.0 * 60.0,
    60.0 * 60.0,
    6.0 * 60.0 * 60.0,
    24.0 * 60.0 * 60.0,
    3.0 * 24.0 * 60.0 * 60.0,
    7.0 * 24.0 * 60.0 * 60.0,
    30.0 * 24.0 * 60.0 * 60.0,
    90.0 * 24.0 * 60.0 * 60.0,
    365.0 * 24.0 * 60.0 * 60.0,
)

SCENARIO_COLORS = (
    "#607d8b",
    "#1976d2",
    "#ef6c00",
    "#2e7d32",
    "#c62828",
    "#6a1b9a",
    "#00838f",
    "#ad1457",
)

_SCENARIO_COLOR_PATTERN = re.compile(r"^#[0-9a-f]{6}$")


class BehaviorLabValidationError(ValueError):
    pass


def normalize_behavior_lab_color(value: Any) -> str:
    """Return a safe canonical scenario color used by inline HTML/SVG styles.

    An empty value deliberately means "use the renderer's palette fallback".
    Explicit colors use the one supported interchange format: ``#RRGGBB``.
    """

    if not isinstance(value, str):
        raise BehaviorLabValidationError(
            "Scenario colors must be empty or use the #RRGGBB format."
        )
    normalized = value.strip().lower()
    if normalized and _SCENARIO_COLOR_PATTERN.fullmatch(normalized) is None:
        raise BehaviorLabValidationError(
            "Scenario colors must be empty or use the #RRGGBB format."
        )
    return normalized


class BehaviorLabRuntime(Protocol):
    def predict_many(
        self,
        rows: list[dict[str, Any]],
        *,
        batch_size: int | None = None,
        allow_gpu: bool = True,
    ) -> list[float]: ...

    def process_simulation_one(
        self,
        row: dict[str, Any],
        *,
        return_curves: bool,
    ) -> tuple[float, Any | None]: ...

    def process_simulation_many(self, rows: list[dict[str, Any]]) -> list[float]: ...


@dataclass(frozen=True)
class BehaviorLabEvent:
    kind: EventKind
    label: str = ""
    after_seconds: float = 0.0
    card_id: int | None = None
    rating: int = 3
    review_context: ReviewContext = "review"
    duration_seconds: float = 5.0
    capture_curve: bool = True
    context_scope: ContextScope = "unrelated"
    context_count: int = 100
    context_spacing_seconds: float = 1.0
    context_rating_mode: ContextRatingMode = "collection"
    context_seed: int = 5489
    context_card_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "after_seconds": self.after_seconds,
            "card_id": self.card_id,
            "rating": self.rating,
            "review_context": self.review_context,
            "duration_seconds": self.duration_seconds,
            "capture_curve": self.capture_curve,
            "context_scope": self.context_scope,
            "context_count": self.context_count,
            "context_spacing_seconds": self.context_spacing_seconds,
            "context_rating_mode": self.context_rating_mode,
            "context_seed": self.context_seed,
            "context_card_ids": list(self.context_card_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BehaviorLabEvent:
        if not isinstance(value, Mapping):
            raise BehaviorLabValidationError("Every timeline event must be an object.")
        event = cls(
            kind=str(value.get("kind", "")),  # type: ignore[arg-type]
            label=str(value.get("label", "")),
            after_seconds=_finite_float(value.get("after_seconds", 0.0), "after_seconds"),
            card_id=_optional_int(value.get("card_id"), "card_id"),
            rating=_int_value(value.get("rating", 3), "rating"),
            review_context=str(value.get("review_context", "review")),  # type: ignore[arg-type]
            duration_seconds=_finite_float(
                value.get("duration_seconds", 5.0),
                "duration_seconds",
            ),
            capture_curve=bool(value.get("capture_curve", True)),
            context_scope=str(value.get("context_scope", "unrelated")),  # type: ignore[arg-type]
            context_count=_int_value(value.get("context_count", 100), "context_count"),
            context_spacing_seconds=_finite_float(
                value.get("context_spacing_seconds", 1.0),
                "context_spacing_seconds",
            ),
            context_rating_mode=str(  # type: ignore[arg-type]
                value.get("context_rating_mode", "collection")
            ),
            context_seed=_int_value(value.get("context_seed", 5489), "context_seed"),
            context_card_ids=tuple(
                _int_value(item, "context_card_ids") for item in value.get("context_card_ids", ())
            ),
        )
        validate_event(event)
        return event


@dataclass(frozen=True)
class BehaviorLabScenario:
    name: str
    events: tuple[BehaviorLabEvent, ...]
    color: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "color", normalize_behavior_lab_color(self.color))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "color": self.color,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BehaviorLabScenario:
        if not isinstance(value, Mapping):
            raise BehaviorLabValidationError("Every scenario must be an object.")
        scenario = cls(
            name=str(value.get("name", "Scenario")).strip() or "Scenario",
            color=value.get("color", ""),
            events=tuple(BehaviorLabEvent.from_dict(event) for event in value.get("events", ())),
        )
        validate_scenario(scenario)
        return scenario


@dataclass(frozen=True)
class BehaviorLabExperiment:
    name: str
    focal_card_id: int
    scenarios: tuple[BehaviorLabScenario, ...]
    tracked_card_ids: tuple[int, ...] = ()
    selection_card_ids: tuple[int, ...] = ()
    track_siblings: bool = True
    track_collection: bool = False
    tracked_search: str = ""
    baseline_timestamp_seconds: float | None = None
    schema_version: int = BEHAVIOR_LAB_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "focal_card_id": self.focal_card_id,
            "tracked_card_ids": list(self.tracked_card_ids),
            "selection_card_ids": list(self.selection_card_ids),
            "track_siblings": self.track_siblings,
            "track_collection": self.track_collection,
            "tracked_search": self.tracked_search,
            "baseline_timestamp_seconds": self.baseline_timestamp_seconds,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BehaviorLabExperiment:
        if not isinstance(value, Mapping):
            raise BehaviorLabValidationError("The experiment must be an object.")
        schema_version = _int_value(value.get("schema_version", 1), "schema_version")
        if schema_version != BEHAVIOR_LAB_SCHEMA_VERSION:
            raise BehaviorLabValidationError(
                f"Unsupported Behavior Lab schema version {schema_version}."
            )
        timestamp_value = value.get("baseline_timestamp_seconds")
        experiment = cls(
            schema_version=schema_version,
            name=str(value.get("name", "Untitled experiment")).strip() or "Untitled experiment",
            focal_card_id=_int_value(value.get("focal_card_id"), "focal_card_id"),
            tracked_card_ids=tuple(
                _int_value(item, "tracked_card_ids") for item in value.get("tracked_card_ids", ())
            ),
            selection_card_ids=tuple(
                _int_value(item, "selection_card_ids")
                for item in value.get("selection_card_ids", ())
            ),
            track_siblings=bool(value.get("track_siblings", True)),
            track_collection=bool(value.get("track_collection", False)),
            tracked_search=str(value.get("tracked_search", "")),
            baseline_timestamp_seconds=(
                None
                if timestamp_value is None
                else _finite_float(timestamp_value, "baseline_timestamp_seconds")
            ),
            scenarios=tuple(
                BehaviorLabScenario.from_dict(scenario) for scenario in value.get("scenarios", ())
            ),
        )
        validate_experiment(experiment)
        return experiment


@dataclass(frozen=True)
class ResolvedBehaviorLabEvent:
    event: BehaviorLabEvent
    context_card_ids: tuple[int, ...] = ()
    context_review_values: tuple[tuple[int, float, str], ...] = ()


@dataclass(frozen=True)
class ResolvedBehaviorLabScenario:
    scenario: BehaviorLabScenario
    events: tuple[ResolvedBehaviorLabEvent, ...]


@dataclass(frozen=True)
class BehaviorLabCard:
    card_id: int
    note_id: int | None
    deck_id: int | None
    preset_id: int | None
    relation: str


@dataclass(frozen=True)
class ResolvedBehaviorLabExperiment:
    experiment: BehaviorLabExperiment
    baseline_timestamp_seconds: float
    tracked_card_ids: tuple[int, ...]
    runtime_card_ids: tuple[int, ...]
    cards: tuple[BehaviorLabCard, ...]
    scenarios: tuple[ResolvedBehaviorLabScenario, ...]

    @property
    def uses_complete_collection_scope(self) -> bool:
        return self.experiment.track_collection


@dataclass(frozen=True)
class BehaviorLabCurvePoint:
    elapsed_seconds: float
    probability: float


@dataclass(frozen=True)
class BehaviorLabReviewResult:
    event_index: int
    label: str
    card_id: int
    timestamp_seconds: float
    prediction_before_answer: float
    rating: int
    review_context: str
    duration_seconds: float
    elapsed_days: float
    elapsed_seconds: float
    curve_points: tuple[BehaviorLabCurvePoint, ...]


@dataclass(frozen=True)
class BehaviorLabObservationResult:
    event_index: int
    ordinal: int
    label: str
    timestamp_seconds: float
    predictions: tuple[tuple[int, float], ...]

    def prediction_for(self, card_id: int) -> float | None:
        normalized = int(card_id)
        for candidate_id, prediction in self.predictions:
            if candidate_id == normalized:
                return prediction
        return None


@dataclass(frozen=True)
class BehaviorLabScenarioResult:
    name: str
    color: str
    reviews: tuple[BehaviorLabReviewResult, ...]
    observations: tuple[BehaviorLabObservationResult, ...]
    processed_context_reviews: int
    ending_timestamp_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "color", normalize_behavior_lab_color(self.color))


@dataclass(frozen=True)
class BehaviorLabResult:
    experiment_name: str
    experiment: BehaviorLabExperiment
    model_id: str
    checkpoint_fingerprint: str
    generated_timestamp_seconds: float
    baseline_timestamp_seconds: float
    focal_card_id: int
    cards: tuple[BehaviorLabCard, ...]
    scenarios: tuple[BehaviorLabScenarioResult, ...]

    @property
    def control(self) -> BehaviorLabScenarioResult:
        return self.scenarios[0]


def validate_event(event: BehaviorLabEvent) -> None:
    if event.kind not in {"review", "wait", "context", "observe"}:
        raise BehaviorLabValidationError(f"Unsupported timeline event: {event.kind!r}.")
    _require_nonnegative(event.after_seconds, "after_seconds")
    if event.rating not in {1, 2, 3, 4}:
        raise BehaviorLabValidationError("Ratings must be between 1 and 4.")
    if event.review_context not in {*REVIEW_CONTEXT_STATES, "collection"}:
        raise BehaviorLabValidationError(f"Unsupported review context: {event.review_context!r}.")
    if not 0.0 <= event.duration_seconds <= MAX_REVIEW_DURATION_SECONDS:
        raise BehaviorLabValidationError(
            f"Review duration must be between 0 and {MAX_REVIEW_DURATION_SECONDS:g} seconds."
        )
    if event.context_scope not in {
        "unrelated",
        "siblings",
        "same_deck",
        "same_preset",
        "selection",
        "collection",
    }:
        raise BehaviorLabValidationError(f"Unsupported context scope: {event.context_scope!r}.")
    if event.context_rating_mode not in {"collection", "fixed"}:
        raise BehaviorLabValidationError(
            f"Unsupported context rating mode: {event.context_rating_mode!r}."
        )
    if not 0 <= event.context_count <= MAX_CONTEXT_REVIEWS:
        raise BehaviorLabValidationError(
            f"Context count must be between 0 and {MAX_CONTEXT_REVIEWS:,}."
        )
    _require_nonnegative(event.context_spacing_seconds, "context_spacing_seconds")


def validate_scenario(scenario: BehaviorLabScenario) -> None:
    if not scenario.name.strip():
        raise BehaviorLabValidationError("Scenario names cannot be blank.")
    if len(scenario.events) > MAX_EVENTS_PER_SCENARIO:
        raise BehaviorLabValidationError(
            f"A scenario may contain at most {MAX_EVENTS_PER_SCENARIO:,} events."
        )
    normalize_behavior_lab_color(scenario.color)
    for event in scenario.events:
        validate_event(event)


def validate_experiment(experiment: BehaviorLabExperiment) -> None:
    if experiment.focal_card_id <= 0:
        raise BehaviorLabValidationError("Choose a focal card before running the experiment.")
    if not experiment.scenarios:
        raise BehaviorLabValidationError("An experiment needs at least one scenario.")
    if len(experiment.scenarios) > MAX_SCENARIOS:
        raise BehaviorLabValidationError(
            f"An experiment may contain at most {MAX_SCENARIOS} scenarios."
        )
    if len(experiment.tracked_card_ids) > MAX_TRACKED_CARDS:
        raise BehaviorLabValidationError(
            f"An experiment may explicitly track at most {MAX_TRACKED_CARDS:,} cards."
        )
    if experiment.baseline_timestamp_seconds is not None:
        _finite_float(experiment.baseline_timestamp_seconds, "baseline_timestamp_seconds")
    names: set[str] = set()
    for scenario in experiment.scenarios:
        validate_scenario(scenario)
        normalized = scenario.name.casefold()
        if normalized in names:
            raise BehaviorLabValidationError(f"Duplicate scenario name: {scenario.name!r}.")
        names.add(normalized)


def resolve_experiment(
    experiment: BehaviorLabExperiment,
    review_data: ReviewData,
    *,
    searched_card_ids: Iterable[int] = (),
    now_timestamp_seconds: float | None = None,
) -> ResolvedBehaviorLabExperiment:
    validate_experiment(experiment)
    focal_id = int(experiment.focal_card_id)
    focal = review_data.cards.get(focal_id)
    if focal is None:
        raise BehaviorLabValidationError(
            f"Focal card {focal_id} is not present in the current collection snapshot."
        )

    latest_review_timestamp = max(
        (info.timestamp_seconds for info in review_data.last_by_card.values()),
        default=0.0,
    )
    baseline = (
        max(time.time(), latest_review_timestamp + 0.001)
        if experiment.baseline_timestamp_seconds is None
        else float(experiment.baseline_timestamp_seconds)
    )
    if baseline + 1e-9 < latest_review_timestamp:
        raise BehaviorLabValidationError(
            "The Behavior Lab currently starts from the present checkpoint. "
            "Choose a baseline at or after the latest collection review."
        )
    if now_timestamp_seconds is not None and experiment.baseline_timestamp_seconds is None:
        baseline = max(float(now_timestamp_seconds), latest_review_timestamp + 0.001)

    sibling_ids = _sibling_card_ids(focal, review_data.cards)
    tracked: set[int] = {focal_id, *experiment.tracked_card_ids}
    tracked.update(experiment.selection_card_ids)
    tracked.update(int(card_id) for card_id in searched_card_ids)
    if experiment.track_siblings:
        tracked.update(sibling_ids)
    if experiment.track_collection:
        tracked.update(review_data.cards)
    tracked.intersection_update(review_data.cards)
    if len(tracked) > MAX_TRACKED_CARDS:
        raise BehaviorLabValidationError(
            f"The resolved tracked set has {len(tracked):,} cards; the limit is "
            f"{MAX_TRACKED_CARDS:,}."
        )

    resolved_scenarios: list[ResolvedBehaviorLabScenario] = []
    runtime_ids = set(tracked)
    latest_rows_by_card: dict[int, Mapping[str, Any]] | None = None
    for scenario in experiment.scenarios:
        resolved_events: list[ResolvedBehaviorLabEvent] = []
        for event in scenario.events:
            context_ids: tuple[int, ...] = ()
            context_values: tuple[tuple[int, float, str], ...] = ()
            if event.kind == "context" and event.context_count:
                context_ids = resolve_context_card_ids(
                    event,
                    experiment=experiment,
                    review_data=review_data,
                    focal=focal,
                    sibling_ids=sibling_ids,
                )
                if event.context_rating_mode == "collection" and latest_rows_by_card is None:
                    latest_rows_by_card = _latest_rows_by_card(review_data.rows)
                context_values = tuple(
                    _context_review_values(
                        event,
                        card_id,
                        latest_rows_by_card or {},
                    )
                    for card_id in context_ids
                )
                runtime_ids.update(context_ids)
            if event.kind == "review":
                review_card_id = focal_id if event.card_id is None else int(event.card_id)
                if review_card_id not in review_data.cards:
                    raise BehaviorLabValidationError(
                        f"Review event card {review_card_id} is not present in the "
                        "current collection snapshot."
                    )
                runtime_ids.add(review_card_id)
            resolved_events.append(
                ResolvedBehaviorLabEvent(
                    event=event,
                    context_card_ids=context_ids,
                    context_review_values=context_values,
                )
            )
        resolved_scenarios.append(
            ResolvedBehaviorLabScenario(
                scenario=scenario,
                events=tuple(resolved_events),
            )
        )

    runtime_ids.intersection_update(review_data.cards)
    cards = tuple(
        _card_descriptor(card_id, focal=focal, cards=review_data.cards)
        for card_id in sorted(tracked)
    )
    return ResolvedBehaviorLabExperiment(
        experiment=experiment,
        baseline_timestamp_seconds=baseline,
        tracked_card_ids=tuple(sorted(tracked)),
        runtime_card_ids=tuple(sorted(runtime_ids)),
        cards=cards,
        scenarios=tuple(resolved_scenarios),
    )


def resolve_context_card_ids(
    event: BehaviorLabEvent,
    *,
    experiment: BehaviorLabExperiment,
    review_data: ReviewData,
    focal: CardInfo,
    sibling_ids: set[int],
) -> tuple[int, ...]:
    explicit = [card_id for card_id in event.context_card_ids if card_id in review_data.cards]
    if event.context_card_ids:
        if not explicit:
            raise BehaviorLabValidationError(
                "None of the explicit context card IDs are present in the current "
                "collection snapshot."
            )
        candidates = sorted(set(explicit))
    else:
        candidates = _context_candidates(
            event.context_scope,
            experiment=experiment,
            review_data=review_data,
            focal=focal,
            sibling_ids=sibling_ids,
        )
    if not candidates:
        raise BehaviorLabValidationError(
            f"No cards are available for the {event.context_scope.replace('_', ' ')} context block."
        )

    rng = random.Random(int(event.context_seed))
    resolved: list[int] = []
    remaining = int(event.context_count)
    while remaining > 0:
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        take = min(remaining, len(shuffled))
        resolved.extend(shuffled[:take])
        remaining -= take
    return tuple(resolved)


def experiment_work_units(resolved: ResolvedBehaviorLabExperiment) -> int:
    total = 0
    tracked = len(resolved.tracked_card_ids)
    for scenario in resolved.scenarios:
        for resolved_event in scenario.events:
            event = resolved_event.event
            if event.kind == "observe":
                total += max(1, tracked)
            elif event.kind == "context":
                total += len(resolved_event.context_card_ids)
            elif event.kind == "review":
                total += 1
    return max(1, total)


def runtime_scope_card_ids(resolved: ResolvedBehaviorLabExperiment) -> tuple[int, ...] | None:
    if resolved.uses_complete_collection_scope:
        return None
    return resolved.runtime_card_ids


class BehaviorLabScenarioRunner:
    def __init__(
        self,
        *,
        runtime: BehaviorLabRuntime,
        review_data: ReviewData,
        resolved_experiment: ResolvedBehaviorLabExperiment,
        curve_predictor: Callable[[Any, float], float],
        progress_step: Callable[[int, str], None] | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.review_data = review_data
        self.resolved_experiment = resolved_experiment
        self.curve_predictor = curve_predictor
        self.progress_step = progress_step or (lambda _amount, _label: None)
        self.check_cancelled = check_cancelled or (lambda: None)

    def run(self, scenario: ResolvedBehaviorLabScenario) -> BehaviorLabScenarioResult:
        history = _SimulationHistory(
            self.review_data,
            baseline_timestamp_seconds=self.resolved_experiment.baseline_timestamp_seconds,
        )
        current_timestamp = self.resolved_experiment.baseline_timestamp_seconds
        reviews: list[BehaviorLabReviewResult] = []
        observations: list[BehaviorLabObservationResult] = []
        context_review_count = 0
        observation_ordinal = 0

        for event_index, resolved_event in enumerate(scenario.events):
            self.check_cancelled()
            event = resolved_event.event
            if event.kind == "wait":
                current_timestamp += event.after_seconds
                continue
            current_timestamp += event.after_seconds

            if event.kind == "observe":
                rows = [
                    history.prediction_row(card_id, current_timestamp)
                    for card_id in self.resolved_experiment.tracked_card_ids
                ]
                predictions = self.runtime.predict_many(
                    rows,
                    allow_gpu=len(rows) >= 256,
                )
                if len(predictions) != len(rows):
                    raise RuntimeError(
                        "RWKV-SRS returned a different number of predictions than "
                        "the Behavior Lab requested."
                    )
                observations.append(
                    BehaviorLabObservationResult(
                        event_index=event_index,
                        ordinal=observation_ordinal,
                        label=event.label or f"Observation {observation_ordinal + 1}",
                        timestamp_seconds=current_timestamp,
                        predictions=tuple(
                            (card_id, float(prediction))
                            for card_id, prediction in zip(
                                self.resolved_experiment.tracked_card_ids,
                                predictions,
                                strict=True,
                            )
                        ),
                    )
                )
                observation_ordinal += 1
                self.progress_step(
                    max(1, len(rows)),
                    f"{scenario.scenario.name}: {event.label or 'observing cards'}",
                )
                continue

            if event.kind == "review":
                card_id = (
                    self.resolved_experiment.experiment.focal_card_id
                    if event.card_id is None
                    else int(event.card_id)
                )
                row = history.processing_row(
                    card_id,
                    current_timestamp,
                    rating=event.rating,
                    review_context=event.review_context,
                    duration_seconds=event.duration_seconds,
                )
                prediction, curve = self.runtime.process_simulation_one(
                    row,
                    return_curves=event.capture_curve,
                )
                curve_points = (
                    sample_curve(curve, self.curve_predictor)
                    if event.capture_curve and curve is not None
                    else ()
                )
                history.record_review(row, current_timestamp)
                reviews.append(
                    BehaviorLabReviewResult(
                        event_index=event_index,
                        label=event.label or f"Review card {card_id}",
                        card_id=card_id,
                        timestamp_seconds=current_timestamp,
                        prediction_before_answer=float(prediction),
                        rating=int(event.rating),
                        review_context=str(event.review_context),
                        duration_seconds=float(event.duration_seconds),
                        elapsed_days=float(row["elapsed_days"]),
                        elapsed_seconds=float(row["elapsed_seconds"]),
                        curve_points=curve_points,
                    )
                )
                self.progress_step(1, f"{scenario.scenario.name}: processing review")
                continue

            context_rows: list[dict[str, Any]] = []
            for context_index, (card_id, review_values) in enumerate(
                zip(
                    resolved_event.context_card_ids,
                    resolved_event.context_review_values,
                    strict=True,
                )
            ):
                if context_index:
                    current_timestamp += event.context_spacing_seconds
                rating, duration_seconds, review_context = review_values
                row = history.processing_row(
                    int(card_id),
                    current_timestamp,
                    rating=rating,
                    review_context=review_context,
                    duration_seconds=duration_seconds,
                )
                history.record_review(row, current_timestamp)
                context_rows.append(row)
                if len(context_rows) >= SIMULATION_CONTEXT_CHUNK_SIZE:
                    self._process_context_chunk(scenario, context_rows)
                    context_review_count += len(context_rows)
                    context_rows = []
            if context_rows:
                self._process_context_chunk(scenario, context_rows)
                context_review_count += len(context_rows)

        return BehaviorLabScenarioResult(
            name=scenario.scenario.name,
            color=scenario.scenario.color,
            reviews=tuple(reviews),
            observations=tuple(observations),
            processed_context_reviews=context_review_count,
            ending_timestamp_seconds=current_timestamp,
        )

    def _process_context_chunk(
        self,
        scenario: ResolvedBehaviorLabScenario,
        rows: list[dict[str, Any]],
    ) -> None:
        self.check_cancelled()
        predictions = self.runtime.process_simulation_many(rows)
        if len(predictions) != len(rows):
            raise RuntimeError(
                "RWKV-SRS returned a different number of processed context "
                "reviews than the Behavior Lab requested."
            )
        self.progress_step(
            len(rows),
            f"{scenario.scenario.name}: processed {len(rows):,} context reviews",
        )


def build_behavior_lab_result(
    resolved: ResolvedBehaviorLabExperiment,
    scenario_results: Sequence[BehaviorLabScenarioResult],
    *,
    model_id: str,
    checkpoint_fingerprint: str = "",
    generated_timestamp_seconds: float | None = None,
) -> BehaviorLabResult:
    if len(scenario_results) != len(resolved.scenarios):
        raise ValueError("Every resolved scenario must have one result.")
    return BehaviorLabResult(
        experiment_name=resolved.experiment.name,
        experiment=resolved.experiment,
        model_id=str(model_id),
        checkpoint_fingerprint=str(checkpoint_fingerprint),
        generated_timestamp_seconds=(
            time.time()
            if generated_timestamp_seconds is None
            else float(generated_timestamp_seconds)
        ),
        baseline_timestamp_seconds=resolved.baseline_timestamp_seconds,
        focal_card_id=resolved.experiment.focal_card_id,
        cards=resolved.cards,
        scenarios=tuple(scenario_results),
    )


def sample_curve(
    curve: Any,
    curve_predictor: Callable[[Any, float], float],
    *,
    elapsed_seconds: Iterable[float] = DEFAULT_CURVE_SAMPLE_SECONDS,
) -> tuple[BehaviorLabCurvePoint, ...]:
    points: list[BehaviorLabCurvePoint] = []
    for elapsed in elapsed_seconds:
        try:
            probability = float(curve_predictor(curve, float(elapsed)))
        except (RuntimeError, ValueError):
            break
        if math.isfinite(probability):
            points.append(
                BehaviorLabCurvePoint(
                    elapsed_seconds=float(elapsed),
                    probability=probability,
                )
            )
    return tuple(points)


def observation_deltas(
    result: BehaviorLabResult,
    scenario: BehaviorLabScenarioResult,
    observation: BehaviorLabObservationResult,
) -> tuple[tuple[int, float], ...]:
    control_observation = _matching_control_observation(result.control, observation)
    if control_observation is None:
        return ()
    control = dict(control_observation.predictions)
    return tuple(
        (card_id, prediction - control[card_id])
        for card_id, prediction in observation.predictions
        if card_id in control
    )


def cohort_delta_summary(
    result: BehaviorLabResult,
    scenario: BehaviorLabScenarioResult,
    observation: BehaviorLabObservationResult,
) -> dict[str, float | int] | None:
    values = [delta for _card_id, delta in observation_deltas(result, scenario, observation)]
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def behavior_lab_template(
    template: str,
    *,
    focal_card_id: int,
    selection_card_ids: Iterable[int] = (),
    delay_seconds: float = 7.0 * 24.0 * 60.0 * 60.0,
    duration_seconds: float = 5.0,
    context_count: int = 100,
) -> BehaviorLabExperiment:
    focal = int(focal_card_id)
    selection = tuple(dict.fromkeys(int(card_id) for card_id in selection_card_ids))
    before = BehaviorLabEvent(kind="observe", label="Baseline")
    after = BehaviorLabEvent(kind="observe", label="Immediately after")
    minute = BehaviorLabEvent(kind="observe", label="After 1 minute", after_seconds=60)

    if template == "rating_comparison":
        scenarios = [_control_scenario(delay_seconds, minute_after=True)]
        for rating, name in ((1, "Again"), (2, "Hard"), (3, "Good"), (4, "Easy")):
            scenarios.append(
                BehaviorLabScenario(
                    name=name,
                    events=(
                        before,
                        BehaviorLabEvent(
                            kind="review",
                            label=f"{name} review",
                            after_seconds=delay_seconds,
                            rating=rating,
                            duration_seconds=duration_seconds,
                        ),
                        after,
                        minute,
                    ),
                )
            )
        name = "Rating comparison"
    elif template == "review_context":
        scenarios = [_control_scenario(delay_seconds, minute_after=True)]
        for context, name in (("review", "Review"), ("filtered", "Filtered")):
            scenarios.append(
                BehaviorLabScenario(
                    name=name,
                    events=(
                        before,
                        BehaviorLabEvent(
                            kind="review",
                            label=f"Good · {name}",
                            after_seconds=delay_seconds,
                            rating=3,
                            review_context=context,  # type: ignore[arg-type]
                            duration_seconds=duration_seconds,
                        ),
                        after,
                        minute,
                    ),
                )
            )
        name = "Review versus filtered"
    elif template == "sibling_spillover":
        scenarios = (
            _control_scenario(delay_seconds, minute_after=True),
            BehaviorLabScenario(
                name="Good review",
                events=(
                    before,
                    BehaviorLabEvent(
                        kind="review",
                        label="Review focal card",
                        after_seconds=delay_seconds,
                        rating=3,
                        duration_seconds=duration_seconds,
                    ),
                    after,
                    minute,
                ),
            ),
        )
        name = "Sibling spillover"
    elif template == "intervening_reviews":
        context = BehaviorLabEvent(
            kind="context",
            label=f"{context_count:,} unrelated reviews",
            context_count=context_count,
            context_scope="unrelated",
            context_rating_mode="collection",
        )
        scenarios = (
            BehaviorLabScenario(
                name="Control",
                events=(
                    before,
                    BehaviorLabEvent(kind="wait", after_seconds=delay_seconds),
                    context,
                    BehaviorLabEvent(kind="observe", label="After context"),
                ),
            ),
            BehaviorLabScenario(
                name="Focal review",
                events=(
                    before,
                    BehaviorLabEvent(
                        kind="review",
                        label="Review focal card",
                        after_seconds=delay_seconds,
                        rating=3,
                        duration_seconds=duration_seconds,
                    ),
                    context,
                    BehaviorLabEvent(kind="observe", label="After context"),
                ),
            ),
        )
        name = "Intervening reviews"
    elif template == "good_streak":

        def streak(name: str, mode: ContextRatingMode, rating: int) -> BehaviorLabScenario:
            return BehaviorLabScenario(
                name=name,
                events=(
                    before,
                    BehaviorLabEvent(kind="wait", after_seconds=delay_seconds),
                    BehaviorLabEvent(
                        kind="context",
                        label=f"{context_count:,} collection reviews",
                        context_count=context_count,
                        context_scope="unrelated",
                        context_rating_mode=mode,
                        rating=rating,
                        duration_seconds=duration_seconds,
                    ),
                    BehaviorLabEvent(kind="observe", label="After streak"),
                ),
            )

        scenarios = (
            streak("Collection ratings", "collection", 3),
            streak("All Good", "fixed", 3),
            streak("All Easy", "fixed", 4),
        )
        name = "Good-review streak"
    elif template == "custom":
        scenarios = (
            BehaviorLabScenario(name="Control", events=(before,)),
            BehaviorLabScenario(
                name="Scenario",
                events=(
                    before,
                    BehaviorLabEvent(
                        kind="review",
                        after_seconds=delay_seconds,
                        duration_seconds=duration_seconds,
                    ),
                    after,
                ),
            ),
        )
        name = "Custom experiment"
    else:
        raise BehaviorLabValidationError(f"Unknown Behavior Lab template: {template!r}.")

    normalized_scenarios = tuple(
        replace(scenario, color=SCENARIO_COLORS[index % len(SCENARIO_COLORS)])
        for index, scenario in enumerate(scenarios)
    )
    return BehaviorLabExperiment(
        name=name,
        focal_card_id=focal,
        selection_card_ids=selection,
        tracked_card_ids=selection,
        scenarios=normalized_scenarios,
    )


def apply_sweep(
    experiment: BehaviorLabExperiment,
    *,
    scenario_index: int,
    field: str,
    values: Sequence[str | int | float],
) -> BehaviorLabExperiment:
    try:
        source = experiment.scenarios[int(scenario_index)]
    except (IndexError, ValueError) as exc:
        raise BehaviorLabValidationError("Choose a valid scenario to sweep.") from exc
    if not values:
        raise BehaviorLabValidationError("Enter at least one sweep value.")

    scenarios: list[BehaviorLabScenario] = []
    for value in values:
        events = list(source.events)
        event_index = _sweep_event_index(events, field)
        event = events[event_index]
        rendered_value: str
        if field == "rating":
            parsed = _int_value(value, "rating")
            if parsed not in {1, 2, 3, 4}:
                raise BehaviorLabValidationError("Rating sweep values must be 1, 2, 3, or 4.")
            events[event_index] = replace(event, rating=parsed)
            rendered_value = {1: "Again", 2: "Hard", 3: "Good", 4: "Easy"}[parsed]
        elif field == "review_context":
            parsed_context = str(value).strip().lower()
            if parsed_context not in REVIEW_CONTEXT_STATES:
                raise BehaviorLabValidationError(
                    f"Unsupported review-context sweep value: {value!r}."
                )
            events[event_index] = replace(
                event,
                review_context=parsed_context,  # type: ignore[arg-type]
            )
            rendered_value = parsed_context.title()
        elif field == "delay":
            parsed_float = _finite_float(value, "delay")
            _require_nonnegative(parsed_float, "delay")
            events[event_index] = replace(event, after_seconds=parsed_float)
            rendered_value = _format_seconds_short(parsed_float)
        elif field == "duration":
            parsed_float = _finite_float(value, "duration")
            if not 0 <= parsed_float <= MAX_REVIEW_DURATION_SECONDS:
                raise BehaviorLabValidationError("Duration sweep values must be 0-60 seconds.")
            events[event_index] = replace(event, duration_seconds=parsed_float)
            rendered_value = f"{parsed_float:g}s"
        elif field == "context_count":
            parsed = _int_value(value, "context_count")
            if not 0 <= parsed <= MAX_CONTEXT_REVIEWS:
                raise BehaviorLabValidationError(
                    f"Context counts must be 0-{MAX_CONTEXT_REVIEWS:,}."
                )
            events[event_index] = replace(event, context_count=parsed)
            rendered_value = f"{parsed:,} reviews"
        else:
            raise BehaviorLabValidationError(f"Unsupported sweep field: {field!r}.")
        scenarios.append(
            BehaviorLabScenario(
                name=f"{source.name} · {rendered_value}",
                color=SCENARIO_COLORS[len(scenarios) % len(SCENARIO_COLORS)],
                events=tuple(events),
            )
        )
    swept = replace(experiment, scenarios=tuple(scenarios))
    validate_experiment(swept)
    return swept


class _SimulationHistory:
    def __init__(self, review_data: ReviewData, *, baseline_timestamp_seconds: float) -> None:
        self.review_data = review_data
        self.last_by_card = dict(review_data.last_by_card)
        latest_review_id = max(
            (int(row["review_id"]) for row in review_data.rows),
            default=int(baseline_timestamp_seconds * 1000) - 1,
        )
        self.next_review_id = max(latest_review_id + 1, int(baseline_timestamp_seconds * 1000))

    def prediction_row(self, card_id: int, timestamp_seconds: float) -> dict[str, Any]:
        card = self.review_data.cards.get(int(card_id))
        if card is None:
            raise BehaviorLabValidationError(f"Card {card_id} no longer exists.")
        return self._base_row(card, timestamp_seconds, consume_review_id=False)

    def processing_row(
        self,
        card_id: int,
        timestamp_seconds: float,
        *,
        rating: int,
        review_context: str,
        duration_seconds: float,
    ) -> dict[str, Any]:
        card = self.review_data.cards.get(int(card_id))
        if card is None:
            raise BehaviorLabValidationError(f"Card {card_id} no longer exists.")
        row = self._base_row(card, timestamp_seconds, consume_review_id=True)
        row.update(
            {
                "rating": int(rating),
                "duration": float(duration_seconds) * 1000.0,
                "state": _review_state(review_context, card.card_id, self.review_data),
            }
        )
        return row

    def record_review(self, row: Mapping[str, Any], timestamp_seconds: float) -> None:
        card_id = int(row["card_id"])
        previous = self.last_by_card.get(card_id)
        lapse_count = 0 if previous is None else previous.lapse_count
        if int(row["rating"]) == 1 and float(row["elapsed_days"]) > 0:
            lapse_count += 1
        self.last_by_card[card_id] = LastReviewInfo(
            review_id=int(row["review_id"]),
            day_offset=int(row["day_offset"]),
            timestamp_seconds=float(timestamp_seconds),
            interval=0,
            lapse_count=lapse_count,
        )

    def _base_row(
        self,
        card: CardInfo,
        timestamp_seconds: float,
        *,
        consume_review_id: bool,
    ) -> dict[str, Any]:
        raw_day = day_offset_for_timestamp(timestamp_seconds, self.review_data.next_day_at)
        day_offset = raw_day - self.review_data.day_offset_origin
        previous = self.last_by_card.get(card.card_id)
        if previous is None:
            elapsed_days = NEW_CARD_ELAPSED
            elapsed_seconds = NEW_CARD_ELAPSED
        else:
            elapsed_days = day_offset - previous.day_offset
            elapsed_seconds = timestamp_seconds - previous.timestamp_seconds
        review_id = max(self.next_review_id, int(timestamp_seconds * 1000))
        if consume_review_id:
            self.next_review_id = review_id + 1
        return {
            "review_id": review_id,
            "card_id": card.card_id,
            "note_id": card.note_id,
            "deck_id": card.deck_id,
            "preset_id": card.preset_id,
            "day_offset": day_offset,
            "elapsed_days": elapsed_days,
            "elapsed_seconds": elapsed_seconds,
        }


def _context_candidates(
    scope: str,
    *,
    experiment: BehaviorLabExperiment,
    review_data: ReviewData,
    focal: CardInfo,
    sibling_ids: set[int],
) -> list[int]:
    cards = review_data.cards
    focal_id = focal.card_id
    if scope == "siblings":
        return sorted(sibling_ids)
    if scope == "selection":
        return sorted(
            card_id
            for card_id in set(experiment.selection_card_ids)
            if card_id in cards and card_id != focal_id
        )
    if scope == "same_deck":
        return sorted(
            card_id
            for card_id, card in cards.items()
            if card_id != focal_id and card.deck_id == focal.deck_id
        )
    if scope == "same_preset":
        return sorted(
            card_id
            for card_id, card in cards.items()
            if card_id != focal_id and card.preset_id == focal.preset_id
        )
    if scope == "unrelated":
        return sorted(
            card_id
            for card_id, card in cards.items()
            if card_id != focal_id and card.note_id != focal.note_id
        )
    return sorted(card_id for card_id in cards if card_id != focal_id)


def _context_review_values(
    event: BehaviorLabEvent,
    card_id: int,
    latest_rows_by_card: Mapping[int, Mapping[str, Any]],
) -> tuple[int, float, str]:
    if event.context_rating_mode == "fixed":
        context = event.review_context
        if context == "collection":
            context = "review"
        return event.rating, event.duration_seconds, context
    latest = latest_rows_by_card.get(int(card_id))
    if latest is None:
        return event.rating, event.duration_seconds, "new"
    duration_ms = min(
        MAX_REVIEW_DURATION_SECONDS * 1000.0,
        max(0.0, float(latest.get("duration", event.duration_seconds * 1000.0))),
    )
    state = int(latest.get("state", REVIEW_CONTEXT_STATES["review"]))
    context = next(
        (name for name, value in REVIEW_CONTEXT_STATES.items() if value == state),
        "review",
    )
    return int(latest.get("rating", event.rating)), duration_ms / 1000.0, context


def _latest_rows_by_card(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    latest: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        latest[int(row["card_id"])] = row
    return latest


def _latest_row_for_card(
    card_id: int,
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for row in reversed(rows):
        if int(row["card_id"]) == int(card_id):
            return row
    return None


def _review_state(context: str, card_id: int, review_data: ReviewData) -> int:
    if context != "collection":
        return REVIEW_CONTEXT_STATES[context]
    latest = _latest_row_for_card(card_id, review_data.rows)
    return (
        REVIEW_CONTEXT_STATES["new"]
        if latest is None
        else int(latest.get("state", REVIEW_CONTEXT_STATES["review"]))
    )


def _sibling_card_ids(focal: CardInfo, cards: Mapping[int, CardInfo]) -> set[int]:
    if focal.note_id is None:
        return set()
    return {
        card_id
        for card_id, card in cards.items()
        if card_id != focal.card_id and card.note_id == focal.note_id
    }


def _card_descriptor(
    card_id: int,
    *,
    focal: CardInfo,
    cards: Mapping[int, CardInfo],
) -> BehaviorLabCard:
    card = cards[card_id]
    if card_id == focal.card_id:
        relation = "focal"
    elif focal.note_id is not None and card.note_id == focal.note_id:
        relation = "sibling"
    elif focal.deck_id is not None and card.deck_id == focal.deck_id:
        relation = "same deck"
    elif focal.preset_id is not None and card.preset_id == focal.preset_id:
        relation = "same preset"
    else:
        relation = "global only"
    return BehaviorLabCard(
        card_id=card.card_id,
        note_id=card.note_id,
        deck_id=card.deck_id,
        preset_id=card.preset_id,
        relation=relation,
    )


def _matching_control_observation(
    control: BehaviorLabScenarioResult,
    observation: BehaviorLabObservationResult,
) -> BehaviorLabObservationResult | None:
    for candidate in control.observations:
        if candidate.ordinal == observation.ordinal:
            return candidate
    for candidate in control.observations:
        if candidate.label == observation.label:
            return candidate
    return None


def _control_scenario(delay_seconds: float, *, minute_after: bool) -> BehaviorLabScenario:
    events: list[BehaviorLabEvent] = [BehaviorLabEvent(kind="observe", label="Baseline")]
    events.append(BehaviorLabEvent(kind="wait", after_seconds=delay_seconds))
    events.append(BehaviorLabEvent(kind="observe", label="Immediately after"))
    if minute_after:
        events.append(BehaviorLabEvent(kind="observe", label="After 1 minute", after_seconds=60))
    return BehaviorLabScenario(name="Control", events=tuple(events))


def _sweep_event_index(events: Sequence[BehaviorLabEvent], field: str) -> int:
    required_kind = "context" if field == "context_count" else "review"
    for index, event in enumerate(events):
        if event.kind == required_kind:
            return index
    raise BehaviorLabValidationError(
        f"The selected scenario has no {required_kind} event to sweep."
    )


def _format_seconds_short(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:g}s"
    if seconds < 3600:
        return f"{seconds / 60:g}m"
    if seconds < 86_400:
        return f"{seconds / 3600:g}h"
    return f"{seconds / 86_400:g}d"


def _finite_float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BehaviorLabValidationError(f"{field} must be a number.") from exc
    if not math.isfinite(number):
        raise BehaviorLabValidationError(f"{field} must be finite.")
    return number


def _int_value(value: Any, field: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise BehaviorLabValidationError(f"{field} must be an integer.") from exc
    return number


def _optional_int(value: Any, field: str) -> int | None:
    return None if value in {None, ""} else _int_value(value, field)


def _require_nonnegative(value: float, field: str) -> None:
    if value < 0:
        raise BehaviorLabValidationError(f"{field} cannot be negative.")
