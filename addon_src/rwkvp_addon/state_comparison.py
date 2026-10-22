from __future__ import annotations

import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

from .checkpoint_storage import rust_checkpoint_identity_counts
from .compact_review_data import PackedProcessReviewRows
from .compact_v5o_history import PackedV5oReviewRows
from .constants import USER_FILES_ROOT
from .metrics import (
    EvaluationScope,
    MetricResult,
    ScopedMetricResults,
    calculate_aligned_scoped_metrics,
)
from .model_generation import (
    DEFAULT_MODEL_GENERATION,
    MODEL_GENERATIONS,
    model_generation_label,
    model_generation_uses_v5o,
    normalize_model_generation,
)
from .model_storage import default_model_id_for_generation
from .prediction_cache import PredictionRecordSet
from .profile_store import ProfileStore
from .progress import ProgressReporter
from .review_rows import ReviewData, day_offset_origin_from_rows
from .review_type_normalization import filtered_review_normalization_policy_for_store
from .runtime_inputs import RuntimeInputContext
from .rwkv_performance_modes import normalize_process_many_mode
from .rwkv_processing import new_rwkvp_runtime, process_review_rows_with_predictions
from .rwkv_runtime_resources import release_runtime_resources
from .v5o_review_rows import V5oInputContext

DELETED_REVIEWS_COMPARISON = "deleted-reviews"
GENERATIONS_COMPARISON = "generations"
MODELS_COMPARISON = "models"
MODELS_AND_DELETED_REVIEWS_COMPARISON = "models-and-deleted-reviews"

_COLLECTION_SCOPE_KEY = "current-collection"

ComparisonReviewRows = list[dict[str, Any]] | PackedProcessReviewRows


class RuntimeFactory(Protocol):
    def __call__(
        self,
        *,
        model_id: str,
        model_generation: str,
        process_many_mode: str,
        state_precision: str,
    ) -> object: ...


@dataclass(frozen=True)
class ImmediateStateComparisonVariant:
    key: str
    label: str
    model_id: str
    include_deleted_reviews: bool
    rows: ComparisonReviewRows
    day_offset_adjustment: int = 0
    input_context: RuntimeInputContext | None = None
    model_generation: str | None = None


@dataclass(frozen=True)
class ImmediateStateComparisonPlan:
    comparison: str
    baseline_key: str
    process_many_mode: str
    current_card_ids: frozenset[int]
    current_review_count: int
    variants: tuple[ImmediateStateComparisonVariant, ...]
    model_generation: str = DEFAULT_MODEL_GENERATION
    target_review_ids: frozenset[int] | None = None
    state_precision: str = "fp32"


@dataclass(frozen=True)
class ImmediateStateComparisonMeasurement:
    key: str
    label: str
    model_id: str
    include_deleted_reviews: bool
    processed_review_count: int
    evaluated_review_count: int
    metrics: MetricResult
    expected_checkpoint_bytes: int | None = None
    model_generation: str = DEFAULT_MODEL_GENERATION


@dataclass(frozen=True)
class ImmediateStateVariantEvaluation:
    """Scalar result from one disposable, curve-free RWKV state build."""

    processed_review_count: int
    evaluated_review_count: int
    metrics: MetricResult
    expected_checkpoint_bytes: int | None = None


@dataclass(frozen=True)
class ImmediateStateComparisonResult:
    comparison: str
    baseline_key: str
    process_many_mode: str
    current_card_count: int
    current_review_count: int
    measurements: tuple[ImmediateStateComparisonMeasurement, ...]
    retained_variant_keys: tuple[str, ...] = ()

    @property
    def baseline(self) -> ImmediateStateComparisonMeasurement:
        return self.measurement(self.baseline_key)

    def measurement(self, key: str) -> ImmediateStateComparisonMeasurement:
        for measurement in self.measurements:
            if measurement.key == str(key):
                return measurement
        raise KeyError(key)


def deleted_review_comparison_plan(
    *,
    current_rows: list[dict[str, Any]],
    all_rows: ComparisonReviewRows,
    current_card_ids: Iterable[int],
    model_id: str,
    model_generation: str = DEFAULT_MODEL_GENERATION,
    current_includes_deleted_reviews: bool,
    process_many_mode: str,
    without_deleted_day_offset_adjustment: int = 0,
    input_context: RuntimeInputContext | None = None,
    state_precision: str = "fp32",
) -> ImmediateStateComparisonPlan:
    without_deleted = ImmediateStateComparisonVariant(
        key="without-deleted-reviews",
        label="Without deleted-card history",
        model_id=str(model_id),
        include_deleted_reviews=False,
        rows=current_rows,
        day_offset_adjustment=int(without_deleted_day_offset_adjustment),
        input_context=_history_policy_context(input_context, include_deleted=False),
    )
    with_deleted = ImmediateStateComparisonVariant(
        key="with-deleted-reviews",
        label="With deleted-card history",
        model_id=str(model_id),
        include_deleted_reviews=True,
        rows=all_rows,
        input_context=input_context,
    )
    baseline_key = with_deleted.key if current_includes_deleted_reviews else without_deleted.key
    return ImmediateStateComparisonPlan(
        comparison=DELETED_REVIEWS_COMPARISON,
        baseline_key=baseline_key,
        process_many_mode=normalize_process_many_mode(process_many_mode),
        current_card_ids=frozenset(int(card_id) for card_id in current_card_ids),
        current_review_count=len(current_rows),
        variants=_baseline_first((without_deleted, with_deleted), baseline_key),
        model_generation=normalize_model_generation(model_generation),
        state_precision=state_precision,
    )


def model_deleted_review_matrix_plan(
    *,
    current_rows: list[dict[str, Any]],
    all_rows: ComparisonReviewRows,
    current_card_ids: Iterable[int],
    model_ids: Sequence[str],
    current_model_id: str,
    model_generation: str = DEFAULT_MODEL_GENERATION,
    current_includes_deleted_reviews: bool,
    process_many_mode: str,
    without_deleted_day_offset_adjustment: int = 0,
    input_context: RuntimeInputContext | None = None,
    state_precision: str = "fp32",
) -> ImmediateStateComparisonPlan:
    """Compare every model with both deleted-history policies.

    All without-deleted variants deliberately share ``current_rows`` and all
    with-deleted variants share ``all_rows``. Setup therefore retains only two
    potentially large histories while still constructing and releasing one
    disposable runtime per matrix cell.
    """

    normalized_models = tuple(dict.fromkeys(str(model_id) for model_id in model_ids))
    if not normalized_models:
        raise ValueError("No RWKV models are available for comparison.")
    selected_model = str(current_model_id)
    if selected_model not in normalized_models:
        raise ValueError("The selected RWKV model is not available for comparison.")

    variants: list[ImmediateStateComparisonVariant] = []
    baseline_key = ""
    for index, model_id in enumerate(normalized_models):
        for include_deleted, history_key, history_label in (
            (False, "without-deleted-reviews", "Without deleted-card history"),
            (True, "with-deleted-reviews", "With deleted-card history"),
        ):
            key = f"model-{index}-{history_key}"
            variants.append(
                ImmediateStateComparisonVariant(
                    key=key,
                    label=f"{model_id} · {history_label}",
                    model_id=model_id,
                    include_deleted_reviews=include_deleted,
                    rows=all_rows if include_deleted else current_rows,
                    day_offset_adjustment=(
                        0 if include_deleted else int(without_deleted_day_offset_adjustment)
                    ),
                    input_context=_history_policy_context(input_context, include_deleted),
                )
            )
            if model_id == selected_model and include_deleted == bool(
                current_includes_deleted_reviews
            ):
                baseline_key = key

    if not baseline_key:  # Defensive: the selected model was validated above.
        raise ValueError("The selected matrix baseline is unavailable.")
    return ImmediateStateComparisonPlan(
        comparison=MODELS_AND_DELETED_REVIEWS_COMPARISON,
        baseline_key=baseline_key,
        process_many_mode=normalize_process_many_mode(process_many_mode),
        current_card_ids=frozenset(int(card_id) for card_id in current_card_ids),
        current_review_count=len(current_rows),
        # Keep the two history policies adjacent for each model. Unlike the
        # standalone comparisons, Setup's matrix result does not use its
        # baseline for relative coloring, so moving that row first would only
        # make the table and progress sequence harder to scan.
        variants=tuple(variants),
        model_generation=normalize_model_generation(model_generation),
        state_precision=state_precision,
    )


def generation_comparison_plan(
    *,
    rows: ComparisonReviewRows,
    current_card_ids: Iterable[int],
    model_ids_by_generation: Mapping[str, Sequence[str]],
    current_model_generation: str,
    include_deleted_reviews: bool,
    current_review_count: int,
    process_many_mode: str,
    input_context: RuntimeInputContext | None = None,
    v5o_review_data: ReviewData | None = None,
    state_precision: str = "fp32",
) -> ImmediateStateComparisonPlan:
    """Compare one representative default model from every available generation."""

    generation_models = _normalized_generation_models(model_ids_by_generation)
    variants = tuple(
        ImmediateStateComparisonVariant(
            key=f"generation-{generation}",
            label=f"{_generation_label(generation)} · {model_id}",
            model_id=model_id,
            include_deleted_reviews=bool(include_deleted_reviews),
            rows=(
                _v5_history_rows(_v5_data(v5o_review_data), include_deleted_reviews)
                if generation == "v5o" else rows
            ),
            input_context=(
                _history_policy_context(
                    V5oInputContext.from_review_data(_v5_data(v5o_review_data)),
                    include_deleted_reviews,
                )
                if generation == "v5o"
                else input_context
            ),
            model_generation=generation,
        )
        for generation, model_ids in generation_models
        for model_id in (_default_available_model(generation, model_ids),)
    )
    selected_generation = normalize_model_generation(current_model_generation)
    baseline_key = f"generation-{selected_generation}"
    if baseline_key not in {variant.key for variant in variants}:
        baseline_key = variants[0].key
    return ImmediateStateComparisonPlan(
        comparison=GENERATIONS_COMPARISON,
        baseline_key=baseline_key,
        process_many_mode=normalize_process_many_mode(process_many_mode),
        current_card_ids=frozenset(int(card_id) for card_id in current_card_ids),
        current_review_count=max(0, int(current_review_count)),
        variants=_baseline_first(variants, baseline_key),
        # Cross-generation plans carry the authoritative generation on each
        # variant. Keep this fallback only for older single-generation callers.
        model_generation=selected_generation,
        target_review_ids=_common_review_ids(variants),
        state_precision=state_precision,
    )


def generation_model_deleted_review_matrix_plan(
    *,
    current_rows: list[dict[str, Any]],
    all_rows: ComparisonReviewRows,
    current_card_ids: Iterable[int],
    model_ids_by_generation: Mapping[str, Sequence[str]],
    current_model_generation: str,
    current_model_id: str,
    current_includes_deleted_reviews: bool,
    process_many_mode: str,
    without_deleted_day_offset_adjustment: int = 0,
    input_context: RuntimeInputContext | None = None,
    v5o_review_data: ReviewData | None = None,
    state_precision: str = "fp32",
) -> ImmediateStateComparisonPlan:
    """Compare every valid generation/model/history combination.

    Model identifiers remain paired with the generation that owns their bundled
    artifact. The matrix therefore never tries to load a V1 model through V5,
    or vice versa.
    """

    generation_models = _normalized_generation_models(model_ids_by_generation)
    selected_generation = normalize_model_generation(current_model_generation)
    generation_model_map = dict(generation_models)
    if selected_generation not in generation_model_map:
        selected_generation = generation_models[0][0]
    models_for_selected_generation = generation_model_map[selected_generation]
    selected_model = str(current_model_id)
    if selected_model not in models_for_selected_generation:
        selected_model = _default_available_model(
            selected_generation,
            models_for_selected_generation,
        )

    variants: list[ImmediateStateComparisonVariant] = []
    baseline_key = ""
    for generation, model_ids in generation_models:
        for model_index, model_id in enumerate(model_ids):
            for include_deleted, history_key, history_label in (
                (False, "without-deleted-reviews", "Without deleted-card history"),
                (True, "with-deleted-reviews", "With deleted-card history"),
            ):
                key = f"{generation}-model-{model_index}-{history_key}"
                variants.append(
                    ImmediateStateComparisonVariant(
                        key=key,
                        label=(f"{_generation_label(generation)} · {model_id} · {history_label}"),
                        model_id=model_id,
                        include_deleted_reviews=include_deleted,
                        rows=(
                            _v5_history_rows(_v5_data(v5o_review_data), include_deleted)
                            if generation == "v5o"
                            else all_rows
                            if include_deleted
                            else current_rows
                        ),
                        day_offset_adjustment=(
                            0
                            if include_deleted or generation == "v5o"
                            else int(without_deleted_day_offset_adjustment)
                        ),
                        input_context=(
                            _history_policy_context(
                                V5oInputContext.from_review_data(_v5_data(v5o_review_data)),
                                include_deleted,
                            )
                            if generation == "v5o"
                            else input_context
                        ),
                        model_generation=generation,
                    )
                )
                if (
                    generation == selected_generation
                    and model_id == selected_model
                    and include_deleted == bool(current_includes_deleted_reviews)
                ):
                    baseline_key = key

    if not baseline_key:
        raise ValueError("The selected generation/model matrix baseline is unavailable.")
    return ImmediateStateComparisonPlan(
        comparison=MODELS_AND_DELETED_REVIEWS_COMPARISON,
        baseline_key=baseline_key,
        process_many_mode=normalize_process_many_mode(process_many_mode),
        current_card_ids=frozenset(int(card_id) for card_id in current_card_ids),
        current_review_count=len(current_rows),
        variants=tuple(variants),
        model_generation=selected_generation,
        target_review_ids=_common_review_ids(variants),
        state_precision=state_precision,
    )


def current_review_rows_from_included_history(
    rows: Sequence[Mapping[str, Any]],
    current_card_ids: Iterable[int],
    *,
    source_day_offset_origin: int,
) -> tuple[list[dict[str, Any]], int]:
    """Select current-card rows and return the normal filtered-history day shift."""

    current_ids = frozenset(int(card_id) for card_id in current_card_ids)
    # Comparison variants temporarily adjust normalized day offsets.  Keep that
    # mutation isolated from the canonical compact history.
    current_rows = [dict(row) for row in rows if int(row["card_id"]) in current_ids]
    adjustment = int(source_day_offset_origin) - day_offset_origin_from_rows(current_rows)
    return current_rows, adjustment


def model_comparison_plan(
    *,
    rows: ComparisonReviewRows,
    current_card_ids: Iterable[int],
    model_ids: Sequence[str],
    current_model_id: str,
    model_generation: str = DEFAULT_MODEL_GENERATION,
    include_deleted_reviews: bool,
    current_review_count: int,
    process_many_mode: str,
    input_context: RuntimeInputContext | None = None,
    state_precision: str = "fp32",
) -> ImmediateStateComparisonPlan:
    normalized_models = tuple(dict.fromkeys(str(model_id) for model_id in model_ids))
    if not normalized_models:
        raise ValueError("No RWKV models are available for comparison.")
    baseline_key = str(current_model_id)
    if baseline_key not in normalized_models:
        raise ValueError("The selected RWKV model is not available for comparison.")
    variants = tuple(
        ImmediateStateComparisonVariant(
            key=model_id,
            label=model_id,
            model_id=model_id,
            include_deleted_reviews=bool(include_deleted_reviews),
            rows=rows,
            input_context=input_context,
        )
        for model_id in normalized_models
    )
    return ImmediateStateComparisonPlan(
        comparison=MODELS_COMPARISON,
        baseline_key=baseline_key,
        process_many_mode=normalize_process_many_mode(process_many_mode),
        current_card_ids=frozenset(int(card_id) for card_id in current_card_ids),
        current_review_count=max(0, int(current_review_count)),
        variants=_baseline_first(variants, baseline_key),
        model_generation=normalize_model_generation(model_generation),
        state_precision=state_precision,
    )


def run_immediate_state_comparison(
    plan: ImmediateStateComparisonPlan,
    progress: ProgressReporter,
    *,
    runtime_factory: RuntimeFactory | None = None,
    process_rows: Callable[..., PredictionRecordSet] | None = None,
    calculate_metrics: Callable[..., ScopedMetricResults] | None = None,
    release_runtime: Callable[[object], None] | None = None,
    retention_store: ProfileStore | None = None,
) -> ImmediateStateComparisonResult:
    """Compare serially, optionally publishing one complete state per generation.

    Every cross-generation entry point uses this retention boundary. Failed or
    cancelled comparisons discard all staged candidates before publication.
    """

    generations = {
        normalize_model_generation(v.model_generation or plan.model_generation)
        for v in plan.variants
    }
    if retention_store is not None and len(generations) > 1:
        directory = retention_store.root / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        staging = TemporaryDirectory(prefix=".comparison-", dir=directory)
    else:
        retention_store = None
        staging = nullcontext(None)
    with staging as staging_root:
        retention = (
            _ComparisonRetention(retention_store, Path(staging_root), plan)
            if retention_store is not None else None
        )
        return _run_comparison_variants(
            plan, progress, runtime_factory=runtime_factory, process_rows=process_rows,
            calculate_metrics=calculate_metrics, release_runtime=release_runtime,
            retention=retention,
        )


def _run_comparison_variants(
    plan, progress, *, runtime_factory, process_rows, calculate_metrics, release_runtime, retention
) -> ImmediateStateComparisonResult:

    variants = tuple(plan.variants)
    _validate_plan(plan, variants)
    total_work = sum(len(variant.rows) + 1 for variant in variants)
    completed_work = 0
    measurements: list[ImmediateStateComparisonMeasurement] = []

    for index, variant in enumerate(variants, start=1):
        prefix = f"{variant.label} ({index}/{len(variants)})"
        variant_progress = _VariantProgress(
            progress,
            offset=completed_work,
            span=len(variant.rows),
            total_work=total_work,
            prefix=prefix,
        )
        model_generation = normalize_model_generation(
            variant.model_generation or plan.model_generation
        )
        evaluation = evaluate_immediate_state_variant(
            variant.rows,
            plan.current_card_ids,
            model_id=variant.model_id,
            model_generation=model_generation,
            process_many_mode=plan.process_many_mode,
            state_precision=plan.state_precision,
            progress=variant_progress,
            day_offset_adjustment=variant.day_offset_adjustment,
            input_context=variant.input_context,
            target_review_ids=plan.target_review_ids,
            runtime_factory=runtime_factory,
            process_rows=process_rows,
            calculate_metrics=calculate_metrics,
            release_runtime=release_runtime,
            on_complete=(
                lambda runtime, rows, records, evaluation,
                variant=variant, variant_progress=variant_progress: retention.consider(
                    variant, runtime, rows, records, evaluation, variant_progress
                )
            ) if retention is not None else None,
        )
        measurements.append(
            ImmediateStateComparisonMeasurement(
                key=variant.key,
                label=variant.label,
                model_id=variant.model_id,
                include_deleted_reviews=variant.include_deleted_reviews,
                processed_review_count=evaluation.processed_review_count,
                evaluated_review_count=evaluation.evaluated_review_count,
                metrics=evaluation.metrics,
                model_generation=model_generation,
                expected_checkpoint_bytes=evaluation.expected_checkpoint_bytes,
            )
        )
        completed_work += len(variant.rows) + 1
        progress.update(completed_work, total_work, f"{prefix}: complete")

    progress.check_cancelled()
    retained = retention.publish(progress) if retention is not None else ()
    return ImmediateStateComparisonResult(
        comparison=plan.comparison,
        baseline_key=plan.baseline_key,
        process_many_mode=plan.process_many_mode,
        current_card_count=len(plan.current_card_ids),
        current_review_count=plan.current_review_count,
        measurements=tuple(measurements),
        retained_variant_keys=retained,
    )


class _ComparisonRetention:
    def __init__(self, store: ProfileStore, directory: Path, plan: ImmediateStateComparisonPlan):
        self.store = store
        self.directory = directory
        self.plan = plan
        self.policy = filtered_review_normalization_policy_for_store(store)
        self.best: dict[str, tuple[ImmediateStateComparisonVariant, ImmediateStateVariantEvaluation,
                                   ProfileStore]] = {}
        self.sequence = 0

    def consider(self, variant, runtime, rows, records, evaluation, progress):
        from .checkpoint_manager import write_comparison_checkpoint
        from .setup_wizard import select_metric_winner

        generation = normalize_model_generation(
            variant.model_generation or self.plan.model_generation
        )
        previous = self.best.get(generation)
        if previous is not None:
            old_variant, old_evaluation, _old_store = previous
            candidates = (old_variant, variant)
            default = next(
                (v.key for v in candidates if v.key == self.plan.baseline_key),
                next((v.key for v in candidates
                      if v.model_id == default_model_id_for_generation(generation)
                      and not v.include_deleted_reviews), old_variant.key),
            )
            winner = select_metric_winner(
                {old_variant.key: old_evaluation.metrics, variant.key: evaluation.metrics},
                default_key=default,
                checkpoint_sizes_by_key={
                    old_variant.key: old_evaluation.expected_checkpoint_bytes,
                    variant.key: evaluation.expected_checkpoint_bytes,
                },
            ).winner_key
            if winner != variant.key:
                return
        progress.check_cancelled()
        progress.update(len(rows), max(1, len(rows)), "Saving comparison checkpoint and cache")
        self.sequence += 1
        staged = self.store.for_generation(generation).with_checkpoint_directory(
            self.directory / str(self.sequence)
        )
        write_comparison_checkpoint(
            staged, runtime, rows, records, model_id=variant.model_id,
            model_generation=generation, context=variant.input_context,
            normalization_policy=self.policy,
            exclude_deleted_card_revlogs=not variant.include_deleted_reviews,
        )
        self.best[generation] = (variant, evaluation, staged)
        if previous is not None:
            shutil.rmtree(previous[2].checkpoints_dir)

    def publish(self, progress) -> tuple[str, ...]:
        progress.check_cancelled()
        # Publication is a short non-cancellable critical section. Each slot is
        # a complete pair; never interrupt between a checkpoint and its cache.
        for generation, (_variant, _evaluation, staged) in self.best.items():
            self.store.for_generation(generation).publish_checkpoint(staged)
        return tuple(variant.key for variant, _evaluation, _staged in self.best.values())


def evaluate_immediate_state_variant(
    rows: ComparisonReviewRows,
    current_card_ids: Iterable[int],
    *,
    model_id: str,
    model_generation: str = DEFAULT_MODEL_GENERATION,
    process_many_mode: str,
    progress: ProgressReporter,
    day_offset_adjustment: int = 0,
    input_context: RuntimeInputContext | None = None,
    target_review_ids: frozenset[int] | None = None,
    runtime_factory: RuntimeFactory | None = None,
    process_rows: Callable[..., PredictionRecordSet] | None = None,
    calculate_metrics: Callable[..., ScopedMetricResults] | None = None,
    release_runtime: Callable[[object], None] | None = None,
    on_complete: Callable | None = None,
    state_precision: str = "fp32",
) -> ImmediateStateVariantEvaluation:
    """Build, evaluate, and release one disposable Immediate-only state.

    Both the comparison dialog and Setup Wizard use this shared in-process path,
    so resource cleanup and Evaluate-compatible metrics cannot drift apart.
    """

    make_runtime = runtime_factory or new_rwkvp_runtime
    process = process_rows or process_review_rows_with_predictions
    score = calculate_metrics or calculate_aligned_scoped_metrics
    release = release_runtime or release_runtime_resources
    scope = EvaluationScope(
        key=_COLLECTION_SCOPE_KEY,
        label="Current collection",
        kind="collection",
        search="",
        card_ids=frozenset(int(card_id) for card_id in current_card_ids),
    )
    runtime = None
    records = PredictionRecordSet.empty()
    effective_rows = rows
    rows_adjusted = False
    total = max(1, len(rows))
    try:
        progress.check_cancelled()
        progress.update(0, total, f"Loading {model_id}")
        if day_offset_adjustment:
            if isinstance(rows, (PackedProcessReviewRows, PackedV5oReviewRows)):
                effective_rows = [dict(row) for row in rows]
            _adjust_day_offsets(effective_rows, day_offset_adjustment)
            rows_adjusted = True
        runtime = make_runtime(
            model_id=str(model_id),
            model_generation=normalize_model_generation(model_generation),
            process_many_mode=str(process_many_mode),
            state_precision=state_precision,
        )
        progress.check_cancelled()
        if isinstance(input_context, V5oInputContext):
            runtime.rebuild_retained_history(
                [], cards=input_context.catalog_rows(),
                rollover_phase_ms=input_context.rollover_phase_ms,
                mode=str(process_many_mode),
            )
        records = process(
            runtime,
            effective_rows,
            progress,
            label="Building disposable RWKV state",
            record_set=records,
            latest_curves_by_card=None,
            process_many_mode=str(process_many_mode),
            calculate_curves=False,
            input_context=input_context,
        )
        progress.check_cancelled()
        progress.update(len(rows), total, "Calculating RMSE(bins), LogLoss and AUC")
        if target_review_ids is None and model_generation_uses_v5o(model_generation):
            target_review_ids = frozenset(
                int(row["review_id"]) for row in effective_rows
                if float(row["elapsed_days"]) > 0
            )
        scoring_kwargs = (
            {}
            if target_review_ids is None
            else {"target_review_ids_by_scope": {scope.key: set(target_review_ids)}}
        )
        if calculate_metrics is None:
            scoring_kwargs["progress"] = progress
        scoped = score(effective_rows, records.immediate_predictions, (scope,), **scoring_kwargs)
        expected_checkpoint_bytes = _expected_checkpoint_bytes(runtime, effective_rows)
        progress.check_cancelled()
        evaluation = ImmediateStateVariantEvaluation(
            processed_review_count=len(rows),
            # This count comes from the same aligned metric calculation as
            # Evaluate; a separate preprocessing count can disagree.
            evaluated_review_count=scoped.counts[scope.key],
            metrics=scoped.metrics[scope.key],
            expected_checkpoint_bytes=expected_checkpoint_bytes,
        )
        if on_complete is not None:
            on_complete(runtime, effective_rows, records, evaluation)
        return evaluation
    finally:
        # Prediction arrays can be large. Clear them before loading another
        # model/state, then explicitly close the native runtime on its worker.
        records.immediate_predictions.clear()
        records.predict_ahead_predictions.clear()
        try:
            if runtime is not None:
                release(runtime)
        finally:
            if rows_adjusted:
                _adjust_day_offsets(effective_rows, -day_offset_adjustment)


class _VariantProgress(ProgressReporter):
    def __init__(
        self,
        parent: ProgressReporter,
        *,
        offset: int,
        span: int,
        total_work: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self._parent = parent
        self._offset = max(0, int(offset))
        self._span = max(0, int(span))
        self._total_work = max(1, int(total_work))
        self._prefix = str(prefix)

    def check_cancelled(self) -> None:
        self._parent.check_cancelled()

    def update(self, current: int, total: int, label: str = "") -> None:
        source_total = max(1, int(total))
        source_current = min(source_total, max(0, int(current)))
        mapped = self._offset + round(self._span * source_current / source_total)
        text = f"{self._prefix}: {label} ({source_current:,}/{max(0, int(total)):,} reviews)"
        self._parent.update(mapped, self._total_work, text)


def _validate_plan(
    plan: ImmediateStateComparisonPlan,
    variants: tuple[ImmediateStateComparisonVariant, ...],
) -> None:
    if plan.comparison not in {
        DELETED_REVIEWS_COMPARISON,
        GENERATIONS_COMPARISON,
        MODELS_COMPARISON,
        MODELS_AND_DELETED_REVIEWS_COMPARISON,
    }:
        raise ValueError(f"Unsupported state comparison: {plan.comparison!r}.")
    if not variants:
        raise ValueError("The state comparison has no variants.")
    keys = tuple(variant.key for variant in variants)
    if len(set(keys)) != len(keys):
        raise ValueError("State comparison variant keys must be unique.")
    if plan.baseline_key not in keys:
        raise ValueError("The state comparison baseline is missing.")
    if not plan.current_card_ids:
        raise ValueError("The collection has no current cards to evaluate.")
    if plan.current_review_count <= 0:
        raise ValueError("The collection has no current-card reviews to evaluate.")
    if any(not variant.rows for variant in variants):
        raise ValueError("The collection has no processable reviews to compare.")


def _normalized_generation_models(
    model_ids_by_generation: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    supplied: dict[str, list[str]] = {generation: [] for generation in MODEL_GENERATIONS}
    for raw_generation, raw_model_ids in model_ids_by_generation.items():
        if raw_generation not in MODEL_GENERATIONS:
            raise ValueError(f"Unsupported RWKV generation: {raw_generation}")
        generation = normalize_model_generation(raw_generation)
        supplied.setdefault(generation, []).extend(str(model_id) for model_id in raw_model_ids)
    normalized = tuple(
        (generation, tuple(dict.fromkeys(supplied.get(generation, ()))))
        for generation in MODEL_GENERATIONS
        if supplied.get(generation)
    )
    if not normalized:
        raise ValueError("No RWKV generations are available for comparison.")
    return normalized


def _default_available_model(generation: str, model_ids: Sequence[str]) -> str:
    if not model_ids:
        raise ValueError(f"No models are available for {_generation_label(generation)}.")
    preferred = default_model_id_for_generation(generation)
    return preferred if preferred in model_ids else str(model_ids[0])


def _generation_label(generation: str) -> str:
    return model_generation_label(generation)


def _v5_data(data: ReviewData | None) -> ReviewData:
    if data is None or data.history_generation != "v5o":
        raise ValueError("V5 comparison requires separately prepared V5o-a history")
    return data


def _history_policy_context(
    context: RuntimeInputContext | None, include_deleted: bool,
) -> RuntimeInputContext | None:
    if isinstance(context, V5oInputContext) and not include_deleted:
        return context.without_deleted_history()
    return context


def _v5_history_rows(data: ReviewData, include_deleted: bool) -> ComparisonReviewRows:
    if include_deleted or all(int(row["card_id"]) in data.cards for row in data.rows):
        return data.rows
    return [row for row in data.rows if int(row["card_id"]) in data.cards]


def _common_review_ids(
    variants: Sequence[ImmediateStateComparisonVariant],
) -> frozenset[int] | None:
    """Score a common cohort while replaying each generation's full history."""
    if not any(model_generation_uses_v5o(v.model_generation) for v in variants):
        return None
    common: set[int] | None = None
    for variant in variants:
        ids: set[int] = set()
        seen: set[int] = set()
        for row in variant.rows:
            card_id = int(row["card_id"])
            if float(row["elapsed_days"]) > 0 and (
                variant.model_generation != "v5o" or card_id in seen
            ):
                ids.add(int(row["review_id"]))
            seen.add(card_id)
        common = ids if common is None else common.intersection(ids)
    return frozenset(common or ())


def _baseline_first(
    variants: Sequence[ImmediateStateComparisonVariant],
    baseline_key: str,
) -> tuple[ImmediateStateComparisonVariant, ...]:
    return tuple(
        sorted(
            variants,
            key=lambda variant: (variant.key != baseline_key,),
        )
    )


def _adjust_day_offsets(rows: Sequence[dict[str, Any]], adjustment: int) -> None:
    changed = 0
    try:
        for row in rows:
            row["day_offset"] = int(row["day_offset"]) + int(adjustment)
            changed += 1
    except BaseException:
        for row in rows[:changed]:
            row["day_offset"] = int(row["day_offset"]) - int(adjustment)
        raise


def _expected_checkpoint_bytes(
    runtime: object,
    rows: Sequence[Mapping[str, Any]],
) -> int | None:
    if getattr(runtime, "model_version", None) == "v5o":
        # API 76 exposes the facade method for every model, but its native
        # implementation still supports V1 only. Measure the disposable V5
        # checkpoint rather than treating method presence as a capability.
        USER_FILES_ROOT.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="v5-state-size-", dir=USER_FILES_ROOT) as root:
            path = Path(root) / "state.bin"
            runtime.save_checkpoint(path)
            return path.stat().st_size
    expected_checkpoint_size = getattr(runtime, "expected_checkpoint_size", None)
    if not callable(expected_checkpoint_size):
        return None
    counts = rust_checkpoint_identity_counts(rows)
    return int(expected_checkpoint_size(**counts.as_kwargs()))
