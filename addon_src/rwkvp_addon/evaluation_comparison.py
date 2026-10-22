"""Generation comparisons using normal checkpoint validation and cache ownership."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .anki_api import FsrsEvaluationMode, fsrs_evaluation_review_ids_for_plans
from .checkpoint_manager import (
    InconsistentCheckpointError,
    LegacyCheckpointError,
    MissingCheckpointError,
    RWKVCheckpointManager,
)
from .dataset_export import (
    ReviewDataLoad,
    ensure_checkpoint_ready_from_load,
    initialize_or_update_checkpoint_from_load,
    load_review_data_for_checkpoint,
)
from .evaluation_predictions import (
    EvaluationPredictionSnapshot,
    load_current_evaluation_prediction_rows,
    load_current_evaluation_prediction_snapshot,
)
from .metrics import (
    EvaluationScope,
    RWKVPredictionMode,
    ScopedMetricResults,
    calculate_aligned_scoped_metrics,
    row_matches_scope,
)
from .model_generation import (
    compatible_evaluation_generations,
    model_generation_label,
    model_generation_supports_curves,
)
from .model_storage import available_model_ids, default_model_id_for_generation
from .prediction_cache import (
    PER_REVIEW_CACHE_SPEC,
    StalePredictionCacheError,
    evaluation_cache_has_specs,
    prediction_cache_specs,
)
from .profile_store import ProfileStore
from .progress import CancelledError, ProgressReporter
from .review_rows import ReviewData


class ComparisonCheckpointRequired(RuntimeError):
    """The alternate checkpoint needs an explicitly confirmed rebuild."""


class ComparisonEvaluationError(RuntimeError):
    """An alternate-state failure must never invoke active-checkpoint recovery."""


@dataclass(frozen=True)
class GenerationEvaluationResults:
    comparison: ScopedMetricResults
    active: ScopedMetricResults


class GenerationEvaluation:
    """One alternate generation, without changing the profile's selected model.

    Collection loading is a separate stage. Native work reserves the active
    manager's slot, uses a private manager, and releases it before returning.
    Existing valid checkpoints are updated even when retention is disabled.
    Rebuilds are staged and only published after successful scoring if opted in.
    """

    def __init__(
        self,
        active_manager: RWKVCheckpointManager,
        generation: str,
        *,
        prediction_mode: RWKVPredictionMode,
        configure_manager: Callable[[RWKVCheckpointManager], None] | None = None,
    ) -> None:
        if generation not in compatible_evaluation_generations(
            active_manager.model_generation,
            prediction_mode=prediction_mode.value,
            include_deleted_reviews=not active_manager.exclude_deleted_card_revlogs,
        ):
            raise ValueError("This generation cannot use the active evaluation settings.")
        self.active_manager = active_manager
        self.generation = generation
        self.prediction_mode = prediction_mode
        self.configure_manager = configure_manager
        self.store = active_manager.store.for_generation(generation)
        self.store.ensure()
        saved_model = self.store.manifest().get("model_id")
        self.model_id = (
            str(saved_model)
            if saved_model in available_model_ids(generation)
            else default_model_id_for_generation(generation)
        )
        self.manager = self._new_manager(self.store, preserve_curves=True)

    def _new_manager(self, store: ProfileStore, *, preserve_curves: bool = False):
        specs = (PER_REVIEW_CACHE_SPEC,)
        # Keep a saved V1 curve stream current when updating its Immediate cache.
        # Fresh Immediate comparisons need only the Immediate stream.
        if (
            preserve_curves
            and model_generation_supports_curves(self.generation)
            and evaluation_cache_has_specs(store.evaluation_cache_path, prediction_cache_specs())
        ):
            specs = prediction_cache_specs()
        manager = RWKVCheckpointManager(
            store,
            model_id=self.model_id,
            model_generation=self.generation,
            state_precision=getattr(self.active_manager, "state_precision", "fp32"),
            async_saves=False,
            prediction_cache_specs=specs,
            process_many_mode=self.active_manager.process_many_mode,
            exclude_deleted_card_revlogs=self.active_manager.exclude_deleted_card_revlogs,
            filtered_review_normalization_policy=(
                self.active_manager.filtered_review_normalization_policy
            ),
        )
        if self.configure_manager is not None:
            self.configure_manager(manager)
        return manager

    @property
    def needs_build(self) -> bool:
        return self.manager.status() != "valid"

    def validate_active_checkpoint(self) -> None:
        status = self.active_manager.status()
        if status == "missing":
            raise MissingCheckpointError("Initialize the active RWKV checkpoint first.")
        if status not in {"valid", "partial"}:
            raise RuntimeError(
                "The active RWKV checkpoint does not match its model, history or cache settings. "
                "Rebuild it before comparing generations."
            )

    def build_warning(self, *, retain: bool) -> str:
        label = model_generation_label(self.generation)
        persistence = (
            f"Retain Multiple Model Generations is on. The completed {label} checkpoint "
            "and evaluation cache will replace its saved generation slot."
            if retain
            else "Retain Multiple Model Generations is off. The new model state will be "
            "discarded after this comparison."
        )
        return (
            f"No usable cached {label} model matches this evaluation. Building it from "
            f"your review history may take some time. {persistence}"
        )

    def load_review_data(self, col, progress: ProgressReporter) -> ReviewDataLoad:
        return load_review_data_for_checkpoint(
            col,
            self.store,
            self.manager,
            progress,
            allow_incremental=False,
        )

    def compare(
        self,
        review_load: ReviewDataLoad,
        active_data: ReviewData,
        active_snapshot: EvaluationPredictionSnapshot,
        plans,
        active_targets: dict[str, set[int]],
        progress: ProgressReporter,
        *,
        allow_build: bool,
        retain: bool,
    ) -> GenerationEvaluationResults:
        with self._comparison_errors():
            return self._compare(
                review_load,
                active_data,
                active_snapshot,
                plans,
                active_targets,
                progress,
                allow_build=allow_build,
                retain=retain,
            )

    @contextmanager
    def _comparison_errors(self):
        try:
            yield
        except (ComparisonCheckpointRequired, CancelledError):
            raise
        except Exception as exc:
            raise ComparisonEvaluationError(
                f"RWKV {model_generation_label(self.generation)} comparison failed: {exc}"
            ) from exc

    def _compare(
        self,
        review_load: ReviewDataLoad,
        active_data: ReviewData,
        active_snapshot: EvaluationPredictionSnapshot,
        plans,
        active_targets: dict[str, set[int]],
        progress: ProgressReporter,
        *,
        allow_build: bool,
        retain: bool,
    ) -> GenerationEvaluationResults:
        with self._prepared_checkpoint(
            review_load,
            progress,
            allow_build=allow_build,
            retain=retain,
        ) as (readiness, manager):
            snapshot = load_current_evaluation_prediction_snapshot(
                readiness,
                manager.store,
                manager,
                progress,
                prediction_mode=self.prediction_mode,
            )
            other_targets = fsrs_evaluation_review_ids_for_plans(
                readiness.review_data,
                plans,
                mode=FsrsEvaluationMode.TIME_SERIES,
            )
            return compare_generation_predictions(
                active_data,
                active_snapshot,
                readiness.review_data,
                snapshot,
                [plan.scope for plan in plans],
                active_targets,
                other_targets,
                progress,
            )

    def calibration_rows(
        self,
        review_load: ReviewDataLoad,
        active_rows: Sequence[dict],
        progress: ProgressReporter,
        *,
        allow_build: bool,
        retain: bool,
    ) -> list[dict]:
        """Pair graph predictions without imposing Evaluate's time-series split.

        Search limits the cache read, while the checkpoint processes full history.
        Same-day/interval filters remain presentation choices in both graphs.
        """
        with (
            self._comparison_errors(),
            self._prepared_checkpoint(
                review_load,
                progress,
                allow_build=allow_build,
                retain=retain,
            ) as (readiness, manager),
        ):
            rows = load_current_evaluation_prediction_rows(
                readiness,
                manager.store,
                manager,
                progress,
                prediction_mode=self.prediction_mode,
                card_ids={int(row["card_id"]) for row in active_rows},
            )
            return pair_generation_calibration_rows(active_rows, rows, progress)

    @contextmanager
    def _prepared_checkpoint(self, review_load, progress, *, allow_build, retain):
        """Publish a staged state only after the caller successfully consumes it."""
        with ExitStack() as stack:
            stack.enter_context(self.active_manager.reserve_runtime_slot(progress))
            stack.callback(self.manager.unload)
            progress.check_cancelled()
            staged = None
            if allow_build:
                temporary = stack.enter_context(
                    TemporaryDirectory(
                        prefix=".evaluation-",
                        dir=self.store.checkpoints_dir.parent,
                    )
                )
                staged = self.store.with_checkpoint_directory(Path(temporary) / "checkpoint")
                manager = self._new_manager(staged)
                stack.callback(manager.unload)
                initialize_or_update_checkpoint_from_load(
                    manager,
                    review_load,
                    progress,
                    rebuild=True,
                )
            else:
                manager = self.manager
                if self.needs_build:
                    raise ComparisonCheckpointRequired(
                        "The saved comparison checkpoint is unavailable."
                    )
            # Re-prime and verify full history, including old manifests without an
            # explicit deleted-history flag. Never acknowledge an inconsistent state.
            try:
                readiness = ensure_checkpoint_ready_from_load(
                    manager,
                    review_load,
                    progress,
                    force_save=True,
                    wait_for_save=True,
                    capture_prediction_tail=True,
                )
                progress.check_cancelled()
                yield readiness, manager
            except (
                InconsistentCheckpointError,
                MissingCheckpointError,
                LegacyCheckpointError,
                StalePredictionCacheError,
            ) as exc:
                if allow_build:
                    raise
                raise ComparisonCheckpointRequired(str(exc)) from exc
            progress.check_cancelled()
            if staged is not None and retain:
                # Close all handles before renaming the slot, including on Windows.
                manager.unload()
                self.store.publish_checkpoint(staged)


def pair_generation_calibration_rows(
    active_rows: Sequence[dict],
    comparison_rows: Sequence[dict],
    progress: ProgressReporter,
) -> list[dict]:
    """Join exact answers; missing/anchor predictions remain explicit for graph counts.

    The graph's existing ``fsrs_prediction`` field holds the selected comparator.
    Neither source sequence nor a durable prediction cache is modified.
    """

    def answer_key(row):
        return int(row["review_id"]), int(row["card_id"]), int(row["rating"])

    predictions = {}
    total = len(active_rows) + len(comparison_rows)
    for index, row in enumerate(comparison_rows):
        if index % 10_000 == 0:
            progress.check_cancelled()
            progress.update(index, total, "Matching generation predictions")
        key = answer_key(row)
        if key in predictions:
            raise ValueError("Comparison history contains duplicate answers.")
        predictions[key] = row.get("prediction")
    result = []
    for index, row in enumerate(active_rows):
        if index % 10_000 == 0:
            progress.check_cancelled()
            progress.update(len(comparison_rows) + index, total, "Matching generation predictions")
        result.append(dict(row, fsrs_prediction=predictions.get(answer_key(row))))
    progress.check_cancelled()
    progress.update(total, total, "Matched generation predictions")
    return result


def compare_generation_predictions(
    active_data: ReviewData,
    active_snapshot: EvaluationPredictionSnapshot,
    comparison_data: ReviewData,
    comparison_snapshot: EvaluationPredictionSnapshot,
    scopes: Sequence[EvaluationScope],
    active_targets: dict[str, set[int]],
    comparison_targets: dict[str, set[int]],
    progress: ProgressReporter,
) -> GenerationEvaluationResults:
    """Use the same eligible review IDs, scopes and outcomes for both models.

    V5 anchors can have no prediction and each generation has its own history
    preprocessing. Intersect finite predictions as well as time-series targets;
    otherwise the two columns can silently score different cohorts.
    """

    def eligible(data, snapshot):
        if len(data.rows) != len(snapshot.predictions):
            raise ValueError("Comparison predictions do not cover the full review history.")
        indices = {}
        pairs = zip(data.rows, snapshot.predictions, strict=True)
        for index, (row, probability) in enumerate(pairs):
            if index % 10_000 == 0:
                progress.check_cancelled()
            if (
                probability is not None
                and math.isfinite(probability)
                and int(row["rating"]) in {1, 2, 3, 4}
                and float(row["elapsed_days"]) > 0
            ):
                indices[int(row["review_id"])] = index
        return indices

    # Packed histories decode dictionaries on access. Retain only integer indices,
    # never one decoded dictionary per review for both full histories.
    active_indices = eligible(active_data, active_snapshot)
    comparison_indices = eligible(comparison_data, comparison_snapshot)
    eligible_ids = active_indices.keys() & comparison_indices.keys()

    def matches_shared_answer(active_row, comparison_row, scope):
        # The two histories may have been loaded on different collection stages.
        # A retained review ID alone cannot prove that both forecasts refer to
        # the same card and outcome after a history edit between those stages.
        return (
            int(active_row["card_id"]) == int(comparison_row["card_id"])
            and int(active_row["rating"]) == int(comparison_row["rating"])
            and row_matches_scope(active_row, scope)
            and row_matches_scope(comparison_row, scope)
        )

    shared = {}
    for index, scope in enumerate(scopes):
        progress.check_cancelled()
        progress.update(index, max(1, len(scopes)), "Selecting shared evaluation reviews")
        targets = (
            active_targets.get(scope.key, set())
            & comparison_targets.get(scope.key, set())
            & eligible_ids
        )
        shared[scope.key] = {
            review_id
            for review_id in targets
            if matches_shared_answer(
                active_data.rows[active_indices[review_id]],
                comparison_data.rows[comparison_indices[review_id]],
                scope,
            )
        }
    del active_indices, comparison_indices, eligible_ids
    active = calculate_aligned_scoped_metrics(
        active_data.rows,
        active_snapshot.predictions,
        scopes,
        target_review_ids_by_scope=shared,
        progress=progress,
    )
    comparison = calculate_aligned_scoped_metrics(
        comparison_data.rows,
        comparison_snapshot.predictions,
        scopes,
        target_review_ids_by_scope=shared,
        progress=progress,
    )
    progress.update(len(scopes), max(1, len(scopes)), "Compared RWKV generations")
    return GenerationEvaluationResults(comparison=comparison, active=active)
