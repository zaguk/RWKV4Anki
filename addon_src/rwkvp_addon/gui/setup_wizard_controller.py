from __future__ import annotations

# Inline wizard HTML is intentionally kept readable as browser source.
# ruff: noqa: E501
# This controller intentionally keeps Qt wiring thin.  The wizard's durable
# choices are ordinary config values, benchmark work runs through background
# stages, and the webview receives only escaped presentation fragments.
import html
import json
import math
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aqt import mw

from ..addon_config import (
    EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY,
    LIVE_REVIEW_PREDICTION_REFRESH_LIMIT_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    PREDICT_MANY_MODE_CONFIG_KEY,
    PROCESS_MANY_MODE_CONFIG_KEY,
    default_addon_config,
    predict_many_batch_size,
)
from ..anki_api import find_cards
from ..checkpoint_storage import format_storage_bytes
from ..constants import DEFAULT_MODEL_ID
from ..dataset_export import (
    latest_collection_review_timestamp_seconds,
    load_review_data_for_checkpoint,
    load_review_data_from_collection,
    open_checkpoint_runtime_from_load,
)
from ..glossary import GlossaryRenderer
from ..live_prediction_benchmark import open_live_prediction_benchmark_session
from ..metrics import MetricResult
from ..process_many_benchmark import benchmark_process_many_rows
from ..process_speed_cache import (
    cache_setup_process_mode_benchmark,
    process_many_speed_cache_path,
)
from ..progress import TimedProgressReporter, format_eta
from ..review_rows import (
    checkpoint_scope_cards_for_card_ids,
    prediction_rows_for_card_ids,
)
from ..rwkv_performance_modes import (
    PREDICT_MANY_FAST_MODE,
    PROCESS_MANY_FAST_MODE,
)
from ..rwkv_processing import (
    _begin_bulk_gpu_process,
    _finish_bulk_gpu_process,
    _runtime_process_many,
    new_rwkvp_runtime,
)
from ..rwkv_runtime_resources import release_runtime_resources
from ..setup_faq_html import render_setup_faq_link
from ..setup_wizard import (
    METRIC_DISPLAY_DECIMAL_PLACES,
    apply_setup_feature_choices,
    apply_setup_immediate_choice,
    estimate_optimization_durations,
    lower_is_better_improvement_percent,
    metric_values_tie_at_display_precision,
    select_metric_winner,
    summarize_setup_config,
    valid_metric_candidate_keys,
)
from ..setup_wizard_benchmarks import (
    SETUP_PROCESS_REVIEW_LIMIT,
    NoProcessableReviewsError,
    PredictionRefreshSearchResult,
    SetupPredictionModeBenchmarkResult,
    SetupProcessModeBenchmarkResult,
    process_history_state_only,
    run_prediction_mode_benchmark,
    run_process_mode_benchmark,
    search_prediction_refresh_count,
)
from ..speed_test import (
    ProcessManyCurveSpeedTestResult,
    capped_curve_speed_test_rows,
    run_process_many_curve_speed_test,
    speed_test_checkpoint_is_usable,
    state_backed_prediction_card_ids,
)
from ..state_comparison import (
    ImmediateStateComparisonPlan,
    ImmediateStateComparisonResult,
    current_review_rows_from_included_history,
    deleted_review_comparison_plan,
    model_comparison_plan,
    model_deleted_review_matrix_plan,
    run_immediate_state_comparison,
)
from .background_stages import ProgressStage, run_background_stages

if TYPE_CHECKING:
    from .config_dialog import RWKVConfigDialog


_MAX_TUNED_PREDICTION_CARDS = 99_999
_SETUP_SPEED_TEST_EVENTS = frozenset(
    {"run-performance", "run-process-performance", "run-curve-test"}
)
_NO_PROCESSABLE_SPEED_TESTS_REASON = (
    "Unavailable for the rest of this setup session because this collection has no "
    "processable reviews to benchmark. Setup will use the recommended defaults."
)
_SETUP_HISTORY_EXCLUDED_EVENTS = frozenset(
    {
        "back",
        "cancel-run",
        "exit",
        "exit-discard",
        "exit-keep",
        "exit-resume",
        "finish",
        "initial-build",
        "initial-done",
        "retry",
    }
)


@dataclass(frozen=True)
class _PerformanceInput:
    review_load: object
    process_rows: Sequence[Mapping[str, Any]]
    prediction_rows: tuple[dict[str, Any], ...]
    eligible_card_ids: tuple[int, ...]
    eligible_card_count: int
    collection_card_count: int
    target_timestamp_seconds: float


@dataclass(frozen=True)
class _PerformanceResult:
    process: SetupProcessModeBenchmarkResult
    prediction: SetupPredictionModeBenchmarkResult | None
    refresh_search: PredictionRefreshSearchResult | None
    available_review_count: int = 0


@dataclass(frozen=True)
class _OptimizationCounts:
    current_review_count: int
    all_review_count: int
    selected_history_review_count: int
    approximate: bool = False


@dataclass
class _ComparisonInput:
    plan: ImmediateStateComparisonPlan

    def release_rows(self) -> None:
        seen: set[int] = set()
        for variant in self.plan.variants:
            identity = id(variant.rows)
            if identity in seen:
                continue
            seen.add(identity)
            variant.rows.clear()


@dataclass(frozen=True)
class _WizardHistoryEntry:
    """Restorable state from immediately before one forward wizard action."""

    step: str
    working_config: dict[str, Any]
    immediate_enabled: bool | None
    curves_enabled: bool | None
    process_mode_decided: bool
    process_result: SetupProcessModeBenchmarkResult | None
    prediction_result: SetupPredictionModeBenchmarkResult | None
    refresh_result: PredictionRefreshSearchResult | None
    curve_result: ProcessManyCurveSpeedTestResult | None
    selected_history_review_count: int | None
    optimization_counts: _OptimizationCounts | None
    model_result: ImmediateStateComparisonResult | None
    deleted_result: ImmediateStateComparisonResult | None
    matrix_result: ImmediateStateComparisonResult | None
    interruption_message: str | None
    interruption_retryable: bool


@dataclass
class _WizardSession:
    entry_config: dict[str, Any]
    working_config: dict[str, Any]
    initial_launch: bool = False
    immediate_enabled: bool | None = None
    curves_enabled: bool | None = None
    step: str = "immediate"
    return_step: str | None = None
    process_mode_decided: bool = False
    process_result: SetupProcessModeBenchmarkResult | None = None
    prediction_result: SetupPredictionModeBenchmarkResult | None = None
    refresh_result: PredictionRefreshSearchResult | None = None
    curve_result: ProcessManyCurveSpeedTestResult | None = None
    selected_history_review_count: int | None = None
    optimization_counts: _OptimizationCounts | None = None
    model_result: ImmediateStateComparisonResult | None = None
    deleted_result: ImmediateStateComparisonResult | None = None
    matrix_result: ImmediateStateComparisonResult | None = None
    interruption_message: str | None = None
    interruption_retryable: bool = True
    speed_tests_unavailable_reason: str | None = None
    history: list[_WizardHistoryEntry] = field(default_factory=list)


class SetupWizardController:
    def __init__(
        self,
        dialog: RWKVConfigDialog,
        *,
        predict_gpu_available: bool,
        process_gpu_available: bool,
        model_ids: Sequence[str],
    ) -> None:
        self.dialog = dialog
        self._mw = mw
        self.predict_gpu_available = bool(predict_gpu_available)
        self.process_gpu_available = bool(process_gpu_available)
        self.model_ids = tuple(dict.fromkeys(str(model_id) for model_id in model_ids))
        self.session: _WizardSession | None = None
        self._reporter: TimedProgressReporter | None = None
        self._action_name: str | None = None
        self._action_generation = 0
        self._progress_sequence = 0
        self._last_progress_sequence = 0
        self._exit_after_cancel = False
        self._shutting_down = False

    @property
    def active(self) -> bool:
        return self.session is not None

    @property
    def running(self) -> bool:
        return self._reporter is not None

    def handle(self, event: str, value: object | None = None) -> None:
        event = str(event).strip().lower()
        if event == "open":
            self.open(initial_launch=str(value or "").strip().lower() == "initial")
            return
        if self.session is None:
            raise ValueError("The RWKV Setup Wizard is not open.")
        if self.running and event not in {"cancel-run", "exit"}:
            return
        if event == "back":
            self._go_back()
            return

        handlers = {
            "initial-start": self._show_immediate_question,
            "initial-build": lambda: self._finish_initial_window(build_state=True),
            "initial-done": lambda: self._finish_initial_window(build_state=False),
            "immediate-yes": lambda: self._choose_immediate(True),
            "immediate-no": lambda: self._choose_immediate(False),
            "run-performance": lambda: self._run_performance(include_predictions=True),
            "continue-performance": self._show_curve_question,
            "skip-performance": self._skip_performance,
            "curves-yes": lambda: self._choose_curves(True),
            "curves-no": lambda: self._choose_curves(False),
            "run-process-performance": lambda: self._run_performance(include_predictions=False),
            "continue-process-performance": self._after_process_performance,
            "skip-process-performance": self._skip_process_performance,
            "run-curve-test": self._run_curve_test,
            "disable-curves-after-test": lambda: self._finish_curve_test(False),
            "keep-curves-after-test": lambda: self._finish_curve_test(True),
            "skip-curve-test": self._skip_curve_test,
            "optimize-deleted-only": self._show_deleted_test_intro,
            "optimize-models-only": self._show_model_test_intro,
            "optimize-all": self._show_matrix_test_intro,
            "optimize-no": self._show_summary,
            "run-model-test": self._run_model_test,
            "continue-model-test": self._show_summary,
            "skip-model-test": self._skip_model_test,
            "run-deleted-test": self._run_deleted_test,
            "continue-deleted-test": self._show_summary,
            "show-summary": self._show_summary,
            "skip-deleted-test": self._skip_deleted_test,
            "run-matrix-test": self._run_matrix_test,
            "continue-matrix-test": self._show_summary,
            "skip-matrix-test": self._skip_matrix_test,
            "retry": self._retry_current_action,
            "skip-current": self._skip_current_action,
            "cancel-run": self.cancel_running,
            "exit": self.request_exit,
            "exit-keep": self._keep_and_exit,
            "exit-discard": self._discard_and_exit,
            "exit-resume": self._resume_from_exit,
            "finish": self._finish,
        }
        handler = handlers.get(event)
        if handler is None:
            raise ValueError(f"Unknown RWKV Setup Wizard action: {event!r}.")
        history_entry = (
            self._remember_current_step()
            if event not in _SETUP_HISTORY_EXCLUDED_EVENTS
            else None
        )
        starting_step = self.session.step
        try:
            handler()
        except Exception:
            self._discard_history_entry(history_entry)
            raise
        if (
            history_entry is not None
            and self.session is not None
            and not self.running
            and self.session.step == starting_step
        ):
            # A disabled/no-op action may simply rerender its current page. It
            # must not manufacture a duplicate Back destination.
            self._discard_history_entry(history_entry)

    def open(self, *, initial_launch: bool = False) -> None:
        if self.session is not None:
            if initial_launch:
                self.session.initial_launch = True
            self._render_current_step()
            return
        entry = deepcopy(self.dialog._draft)
        self.session = _WizardSession(
            entry_config=entry,
            working_config=deepcopy(entry),
            initial_launch=bool(initial_launch),
            step="welcome" if initial_launch else "immediate",
        )
        self.dialog._set_setup_active(True)
        if initial_launch:
            self._show_initial_welcome()
        else:
            self._show_immediate_question()

    def request_exit(self) -> None:
        if self.session is None:
            return
        if self.running:
            self._exit_after_cancel = True
            self.cancel_running(label="Cancelling before exiting setup…")
            return
        if self.session.step.startswith("initial-complete"):
            self._finish_initial_window(build_state=False)
            return
        if self.session.step == "exit":
            self._render_exit_review()
            return
        if self.session.history and self.session.history[-1].step == self.session.step:
            # Cancelling an in-flight action in order to exit leaves the
            # action's intro as both the current page and the latest history
            # entry. Remove that duplicate so Continue Setup followed by Back
            # advances to the genuinely preceding page in one click.
            self.session.history.pop()
        self.session.return_step = self.session.return_step or self.session.step
        self.session.step = "exit"
        self._render_exit_review()

    def cancel_running(self, *, label: str = "Cancelling after the current operation…") -> None:
        reporter = self._reporter
        if reporter is None:
            return
        reporter.cancel()
        self._eval_progress(0, 0, label, None)

    def shutdown(self) -> None:
        self._shutting_down = True
        if self._reporter is not None:
            self._reporter.cancel()
        self.session = None

    def _remember_current_step(self) -> _WizardHistoryEntry:
        session = self._require_session()
        entry = _WizardHistoryEntry(
            step=session.step,
            working_config=deepcopy(session.working_config),
            immediate_enabled=session.immediate_enabled,
            curves_enabled=session.curves_enabled,
            process_mode_decided=session.process_mode_decided,
            process_result=session.process_result,
            prediction_result=session.prediction_result,
            refresh_result=session.refresh_result,
            curve_result=session.curve_result,
            selected_history_review_count=session.selected_history_review_count,
            optimization_counts=session.optimization_counts,
            model_result=session.model_result,
            deleted_result=session.deleted_result,
            matrix_result=session.matrix_result,
            interruption_message=session.interruption_message,
            interruption_retryable=session.interruption_retryable,
        )
        session.history.append(entry)
        return entry

    def _discard_history_entry(self, entry: _WizardHistoryEntry | None) -> None:
        if entry is None or self.session is None:
            return
        if self.session.history and self.session.history[-1] is entry:
            self.session.history.pop()

    def _go_back(self) -> None:
        session = self._require_session()
        if (
            self.running
            or not session.history
            or session.step == "exit"
            or session.step.startswith("initial-complete")
        ):
            return
        previous = session.history.pop()
        session.step = previous.step
        session.working_config = deepcopy(previous.working_config)
        session.immediate_enabled = previous.immediate_enabled
        session.curves_enabled = previous.curves_enabled
        session.process_mode_decided = previous.process_mode_decided
        session.process_result = previous.process_result
        session.prediction_result = previous.prediction_result
        session.refresh_result = previous.refresh_result
        session.curve_result = previous.curve_result
        session.selected_history_review_count = previous.selected_history_review_count
        session.optimization_counts = previous.optimization_counts
        session.model_result = previous.model_result
        session.deleted_result = previous.deleted_result
        session.matrix_result = previous.matrix_result
        session.interruption_message = previous.interruption_message
        session.interruption_retryable = previous.interruption_retryable
        session.return_step = None
        self._render_current_step()

    def _back_is_available(self) -> bool:
        session = self.session
        return bool(
            session is not None
            and session.history
            and not self.running
            and session.step != "exit"
            and not session.step.startswith("initial-complete")
        )

    # ------------------------------------------------------------------
    # Questions and step transitions

    def _show_initial_welcome(self) -> None:
        session = self._require_session()
        session.step = "welcome"
        glossary = GlossaryRenderer("setup-initial-welcome")
        body = (
            '<div class="setup-callout"><strong>Why this opened</strong><br>'
            "This is the first time RWKV4Anki has opened with this Anki profile, "
            "so Guided Setup is being offered once after installation.</div>"
            f'<p class="setup-copy">{glossary.render("The wizard explains RWKV Immediate and RWKV Forgetting Curve, runs short speed tests with your collection, and recommends settings for this computer.")}</p>'
            f'<p class="setup-copy">{glossary.render("The tests use temporary RWKV state built from your review history. They do not change cards or save a checkpoint. You will review the proposed settings before anything is saved.")}</p>'
            '<p class="setup-help-hint">Hover or focus a dotted term for a quick explanation.</p>'
            '<p class="setup-copy">If you close setup, it will not open automatically '
            "again for this profile. You can always rerun it from "
            "<strong>RWKV → Settings → General → Guided Setup</strong>.</p>"
        )
        self._show(
            "Welcome",
            "Set up RWKV4Anki",
            body,
            self._buttons(
                ("Not Now", "exit", "secondary"),
                ("Start Guided Setup", "initial-start", "primary"),
            ),
        )

    def _show_immediate_question(self) -> None:
        session = self._require_session()
        session.step = "immediate"
        glossary = GlossaryRenderer("setup-immediate")
        gpu_note = (
            "Setup can compare CPU Fast with the available GPU modes, then choose how "
            "many cards RWKV can check in about 100 ms between reviews."
            if self.predict_gpu_available or self.process_gpu_available
            else "Setup will measure CPU Fast and choose how many cards RWKV can check "
            "in about 100 ms between reviews."
        )
        body = (
            f'<p class="setup-copy"><strong>{glossary.render("RWKV Immediate")}</strong>'
            f"{glossary.render(' estimates the chance that you would remember a card if Anki showed it now. In benchmark tests, it has been this add-on’s most accurate way to choose what you study next.')}</p>"
            f'<p class="setup-copy">{glossary.render("Live Session updates RWKV state after every answer, estimates recall for a group of available cards just before the next review, and uses each card’s retrievability to choose what to show. Card Info can display the same estimate.")}</p>'
            '<p class="setup-help-hint">Hover or focus a dotted term for a quick explanation.</p>'
            f'<div class="setup-callout">{glossary.render(gpu_note)}</div>'
            + self._choice_grid(
                (
                    (
                        "No, not right now",
                        "Hide Immediate evaluation, retrievability reports, filtered decks, Live Session, and Card Info fields, then continue to the Forgetting Curve choice.",
                        "immediate-no",
                    ),
                    (
                        "Yes, enable RWKV Immediate",
                        "Show all Immediate tools, turn on Live Session and Immediate Card Info, then tune performance.",
                        "immediate-yes",
                    ),
                ),
                initial_focus_event="immediate-yes",
            )
        )
        self._show("Step 1 of 4", "RWKV Immediate", body)

    def _choose_immediate(self, enabled: bool) -> None:
        session = self._require_session()
        session.immediate_enabled = bool(enabled)
        session.working_config = apply_setup_immediate_choice(
            session.working_config,
            immediate_enabled=bool(enabled),
        )
        if enabled:
            self._show_performance_intro(include_predictions=True)
        else:
            self._show_curve_question()

    def _show_performance_intro(self, *, include_predictions: bool) -> None:
        session = self._require_session()
        session.step = "performance" if include_predictions else "process-performance"
        glossary = GlossaryRenderer(
            "setup-immediate-performance" if include_predictions else "setup-curve-performance"
        )
        if include_predictions:
            title = "Tune Immediate performance"
            if self.predict_gpu_available and self.process_gpu_available:
                mode_purpose = (
                    "GPU acceleration is available for both jobs, so Setup will compare "
                    "CPU Fast and GPU for each one and use the fastest successful mode. "
                )
            elif self.process_gpu_available:
                mode_purpose = (
                    "For State Building, Setup will compare CPU Fast and GPU and use the "
                    "fastest successful mode. Live Session predictions will use CPU Fast. "
                )
            elif self.predict_gpu_available:
                mode_purpose = (
                    "For Live Session predictions, Setup will compare CPU Fast and GPU and "
                    "use the fastest successful mode. State Building will use CPU Fast. "
                )
            else:
                mode_purpose = (
                    "CPU Fast is the only available mode for both jobs, so this test "
                    "calibrates performance rather than choosing between modes. "
                )
            text = glossary.render(
                "Setup will measure CPU Fast for two jobs: State Building with a sample "
                "of up to 10,000 reviews, and a between-review Live Session update "
                "using up to 1,000 reviewed cards. "
                + mode_purpose
                + "Successful State Building results are saved to improve future "
                "build-time estimates."
            )
            detail = glossary.render(
                "Setup will also try several card counts and choose one that takes roughly "
                "100 ms between reviews. This determines how many cards Live Session "
                "checks after each answer. Each measurement includes a representative "
                "answer and prediction refresh; Live Session setup and its initial full "
                "prediction are not included."
            )
            action = "run-performance"
        else:
            title = "Choose a State Building mode"
            text = glossary.render(
                "Your computer can use its GPU for State Building. Setup will process "
                "up to 10,000 reviews once with CPU Fast and once with GPU, then select "
                "the faster successful mode."
            )
            detail = glossary.render(
                "Because you enabled RWKV Forgetting Curve, both runs calculate "
                "forgetting curves so this comparison matches your state builds."
            )
            action = "run-process-performance"
        body = (
            f'<p class="setup-copy">{text}</p>'
            f'<p class="setup-copy">{detail}</p>'
            f'<div class="setup-callout warning">{glossary.render("The test uses temporary RWKV state. It does not change your cards or save a checkpoint. If no usable checkpoint exists yet, preparing the test may take about as long as processing the collection once.")}</div>'
        )
        self._show(
            "Performance check",
            title,
            body,
            (
                self._buttons(
                    ("Exit Setup", "exit", "secondary"),
                    ("Start Speed Test", action, "primary"),
                    (
                        "Use Recommended Defaults",
                        ("skip-performance" if include_predictions else "skip-process-performance"),
                        "primary",
                    ),
                )
                if session.speed_tests_unavailable_reason
                else self._buttons(
                    ("Exit Setup", "exit", "secondary"),
                    ("Start Speed Test", action, "primary"),
                )
            ),
        )

    def _show_curve_question(self) -> None:
        session = self._require_session()
        session.step = "curves"
        glossary = GlossaryRenderer("setup-forgetting-curve")
        selected_process = str(
            session.working_config.get(PROCESS_MANY_MODE_CONFIG_KEY, PROCESS_MANY_FAST_MODE)
        )
        cpu_note = (
            '<div class="setup-callout">'
            + glossary.render(
                "Because CPU Fast was selected, the next step can measure how much "
                "forgetting-curve calculation affects State Building speed."
            )
            + "</div>"
            if selected_process != "gpu" and session.process_result is not None
            else ""
        )
        body = (
            f'<p class="setup-copy"><strong>{glossary.render("RWKV Forgetting Curve")}</strong>'
            f"{glossary.render(' estimates how a card’s retrievability is expected to change as time passes after an answer. It adds stability, suggested interval, and graph details to Card Info.')}</p>"
            f'<p class="setup-copy">{glossary.render("On benchmark data, it has outperformed known FSRS versions at both short and long intervals.")}</p>'
            '<div class="setup-callout warning">This feature has received less add-on '
            "testing than RWKV Immediate and can sometimes produce surprising results.</div>"
            + cpu_note
            + self._choice_grid(
                (
                    (
                        "Yes, calculate forgetting curves",
                        "Turn on curve calculation and all curve-related Card Info fields.",
                        "curves-yes",
                    ),
                    (
                        "No, turn forgetting curves off",
                        "Skip curve calculation and hide curve-dependent features.",
                        "curves-no",
                    ),
                ),
                initial_focus_event="curves-no",
            )
        )
        self._show("Step 2 of 4", "RWKV Forgetting Curve", body)

    def _choose_curves(self, enabled: bool) -> None:
        session = self._require_session()
        session.curves_enabled = bool(enabled)
        session.working_config = apply_setup_feature_choices(
            session.working_config,
            immediate_enabled=bool(session.immediate_enabled),
            curves_enabled=bool(session.curves_enabled),
        )
        if not session.immediate_enabled and not session.curves_enabled:
            self._show_neither_selected()
            return
        if session.curves_enabled and not session.process_mode_decided:
            if self.process_gpu_available:
                self._show_performance_intro(include_predictions=False)
            else:
                # There is no hardware choice to benchmark. CPU Fast is the safe
                # setup default; continue to the curve-cost explanation directly.
                self._skip_process_performance()
            return
        self._after_process_performance()

    def _after_process_performance(self) -> None:
        session = self._require_session()
        if session.curves_enabled and self._selected_process_mode() != "gpu":
            self._show_curve_test_intro()
            return
        self._prepare_optimization_estimate()

    def _show_neither_selected(self) -> None:
        session = self._require_session()
        session.step = "neither"
        glossary = GlossaryRenderer("setup-no-features")
        body = (
            f'<div class="setup-callout">{glossary.render("RWKV Immediate, Live Session, RWKV Card Info, RWKV Forgetting Curve, and curve-dependent experimental features will all be turned off.")}</div>'
            '<p class="setup-copy">Nothing else needs to be measured. You can review '
            "these proposed changes before applying them, and run Guided Setup again "
            "whenever you are ready to use RWKV.</p>"
        )
        self._show(
            "Setup paused",
            "RWKV features are disabled",
            body,
            self._buttons(
                ("Exit Setup", "exit", "secondary"),
                ("Review Settings", "show-summary", "primary"),
            ),
        )

    def _show_curve_test_intro(self) -> None:
        session = self._require_session()
        session.step = "curve-test"
        glossary = GlossaryRenderer("setup-curve-speed")
        body = (
            f'<p class="setup-copy">{glossary.render("On a CPU, calculating forgetting curves can slow State Building. This quick test processes up to 3,000 reviews once with curves and once without them in CPU Fast mode.")}</p>'
            '<p class="setup-copy">The test will not change any cards. After seeing the '
            "result, you can keep forgetting curves enabled or turn them off.</p>"
        )
        self._show(
            "Curve performance",
            "Measure curve calculation cost",
            body,
            (
                self._buttons(
                    ("Run Quick Speed Test", "run-curve-test", "primary"),
                    ("Skip Test", "skip-curve-test", "primary"),
                )
                if session.speed_tests_unavailable_reason
                else self._buttons(
                    ("Skip Test", "skip-curve-test", "secondary"),
                    ("Run Quick Speed Test", "run-curve-test", "primary"),
                )
            ),
        )

    # ------------------------------------------------------------------
    # Hardware and prediction performance

    def _run_performance(self, *, include_predictions: bool) -> None:
        session = self._require_session()
        if session.speed_tests_unavailable_reason:
            self._show_performance_intro(include_predictions=include_predictions)
            return
        if self._rwkv_busy():
            self._render_inline_error(
                "Stop the active RWKV operation, Browser load, Live Session, or checkpoint write before running setup tests.",
                retry_action=("performance" if include_predictions else "process-performance"),
            )
            return

        manager, store = self._manager_and_store()
        model_id = str(session.working_config[MODEL_CONFIG_KEY])
        return_curves = bool(not include_predictions and session.curves_enabled)

        def collect(col, progress, _previous):
            progress.update(0, 1, "Reading review history and current cards")
            review_load = load_review_data_for_checkpoint(
                col,
                store,
                manager,
                progress,
                allow_incremental=True,
            )
            all_card_ids = sorted(int(card_id) for card_id in find_cards(col, ""))
            eligible_ids = state_backed_prediction_card_ids(
                all_card_ids,
                review_load.review_data.last_by_card,
            )
            selected_ids = eligible_ids[:_MAX_TUNED_PREDICTION_CARDS]
            target_timestamp = max(
                time.time(),
                float(latest_collection_review_timestamp_seconds(col) or 0.0),
            )
            prediction_rows = (
                prediction_rows_for_card_ids(
                    selected_ids,
                    review_load.review_data,
                    target_timestamp_seconds=target_timestamp,
                )
                if include_predictions
                else []
            )
            progress.update(1, 1, "Review history ready")
            return _PerformanceInput(
                review_load=review_load,
                process_rows=review_load.review_data.rows,
                prediction_rows=tuple(prediction_rows),
                eligible_card_ids=tuple(selected_ids),
                eligible_card_count=len(eligible_ids),
                collection_card_count=len(all_card_ids),
                target_timestamp_seconds=target_timestamp,
            )

        def benchmark(_col, progress, previous):
            input_data: _PerformanceInput = previous
            process_rows = input_data.process_rows[:SETUP_PROCESS_REVIEW_LIMIT]
            slot = manager.reserve_runtime_slot(progress)
            try:
                process_result = run_process_mode_benchmark(
                    process_rows,
                    gpu_available=self.process_gpu_available,
                    run_once=lambda mode, rows, *, return_curves: (
                        benchmark_process_many_rows(
                            rows,
                            model_id=model_id,
                            mode=mode,
                            return_curves=return_curves,
                            warm_gpu_before_timing=True,
                        )
                    ),
                    progress=progress,
                    return_curves=return_curves,
                )
                # This disposable estimate cache must never turn a successful
                # setup benchmark into a user-visible failure. Store every
                # successful mode under the exact workload that was measured so
                # "Build State Now" can immediately use the selected rate.
                with suppress(OSError, TypeError, ValueError):
                    cache_setup_process_mode_benchmark(
                        process_many_speed_cache_path(store.cache_dir),
                        process_result,
                        model_id=model_id,
                        return_curves=return_curves,
                    )
            finally:
                slot.close()
            if not include_predictions or not input_data.prediction_rows:
                return _PerformanceResult(
                    process_result,
                    None,
                    None,
                    available_review_count=len(input_data.process_rows),
                )
            prediction_result, refresh_result = self._benchmark_predictions(
                input_data,
                progress=progress,
                manager=manager,
                model_id=model_id,
                process_many_mode=process_result.selected_mode,
            )
            return _PerformanceResult(
                process=process_result,
                prediction=prediction_result,
                refresh_search=refresh_result,
                available_review_count=len(input_data.process_rows),
            )

        self._start_action(
            "performance" if include_predictions else "process-performance",
            title="Testing RWKV performance",
            stages=[
                ProgressStage(collect, uses_collection=True),
                ProgressStage(benchmark, uses_collection=False),
            ],
            on_success=lambda result: self._performance_succeeded(
                result,
                include_predictions=include_predictions,
            ),
        )

    def _benchmark_predictions(
        self,
        input_data: _PerformanceInput,
        *,
        progress,
        manager,
        model_id: str,
        process_many_mode: str,
    ) -> tuple[SetupPredictionModeBenchmarkResult, PredictionRefreshSearchResult]:
        lease = None
        runtime = None
        slot = None
        try:
            if speed_test_checkpoint_is_usable(manager) and str(manager.model_id) == str(model_id):
                _readiness, lease = open_checkpoint_runtime_from_load(
                    manager,
                    input_data.review_load,
                    progress,
                    scope_cards=checkpoint_scope_cards_for_card_ids(
                        input_data.eligible_card_ids,
                        input_data.review_load.review_data,
                    ),
                )

                def open_session(rows, mode, _batch_size, refresh_limit):
                    return open_live_prediction_benchmark_session(
                        lease,
                        rows,
                        mode=mode,
                        batch_size=predict_many_batch_size(
                            self._require_session().working_config,
                            str(mode),
                        ),
                        refresh_limit=refresh_limit,
                        target_timestamp_seconds=input_data.target_timestamp_seconds,
                    )

            else:
                slot = manager.reserve_runtime_slot(progress)
                progress.update(0, 1, "Preparing temporary data for prediction tuning")
                runtime = new_rwkvp_runtime(
                    model_id=str(model_id),
                    process_many_mode=process_many_mode,
                )
                use_gpu = _begin_bulk_gpu_process(
                    runtime,
                    process_many_mode=process_many_mode,
                    row_count=len(input_data.process_rows),
                )
                try:
                    process_history_state_only(
                        input_data.process_rows,
                        process=lambda chunk: _runtime_process_many(
                            runtime,
                            list(chunk),
                            return_curves=False,
                            use_gpu=use_gpu,
                        ),
                        progress=progress,
                        label="Preparing temporary prediction data",
                    )
                finally:
                    if use_gpu:
                        _finish_bulk_gpu_process(runtime)

                def open_session(rows, mode, _batch_size, refresh_limit):
                    return open_live_prediction_benchmark_session(
                        runtime,
                        rows,
                        mode=mode,
                        batch_size=predict_many_batch_size(
                            self._require_session().working_config,
                            str(mode),
                        ),
                        refresh_limit=refresh_limit,
                        target_timestamp_seconds=input_data.target_timestamp_seconds,
                    )

            prediction_result = run_prediction_mode_benchmark(
                input_data.prediction_rows,
                eligible_card_count=min(
                    input_data.eligible_card_count,
                    len(input_data.prediction_rows),
                ),
                gpu_available=self.predict_gpu_available,
                open_session=open_session,
                progress=progress,
            )
            selected_measurement = prediction_result.measurement(prediction_result.selected_mode)
            seeded = (
                {prediction_result.card_count: float(selected_measurement.average_seconds)}
                if selected_measurement.average_seconds is not None
                else None
            )
            refresh_result = search_prediction_refresh_count(
                input_data.prediction_rows,
                eligible_card_count=min(
                    input_data.eligible_card_count,
                    len(input_data.prediction_rows),
                ),
                mode=prediction_result.selected_mode,
                open_session=open_session,
                progress=progress,
                seeded_durations=seeded,
            )
            return prediction_result, refresh_result
        finally:
            if lease is not None:
                lease.close()
            if runtime is not None:
                release_runtime_resources(runtime)
            if slot is not None:
                slot.close()

    def _performance_succeeded(
        self,
        result: _PerformanceResult,
        *,
        include_predictions: bool,
    ) -> None:
        session = self._require_session()
        session.process_result = result.process
        session.process_mode_decided = True
        available_review_count = max(0, int(result.available_review_count))
        session.selected_history_review_count = available_review_count or None
        session.working_config[PROCESS_MANY_MODE_CONFIG_KEY] = result.process.selected_mode
        if include_predictions:
            # A rerun replaces the complete performance decision. Do not let a
            # cancelled/partial rerun retain measurements from an older run.
            session.prediction_result = result.prediction
            session.refresh_result = result.refresh_search
        if session.prediction_result is not None:
            session.working_config[PREDICT_MANY_MODE_CONFIG_KEY] = (
                session.prediction_result.selected_mode
            )
        if session.refresh_result is not None:
            session.working_config[LIVE_REVIEW_PREDICTION_REFRESH_LIMIT_CONFIG_KEY] = (
                session.refresh_result.selected_card_count
            )
        elif include_predictions:
            defaults = default_addon_config()
            session.working_config[PREDICT_MANY_MODE_CONFIG_KEY] = PREDICT_MANY_FAST_MODE
            session.working_config[LIVE_REVIEW_PREDICTION_REFRESH_LIMIT_CONFIG_KEY] = int(
                defaults[LIVE_REVIEW_PREDICTION_REFRESH_LIMIT_CONFIG_KEY]
            )
        session.step = "performance-result" if include_predictions else "process-result"
        body = self._render_performance_result(result, include_predictions=include_predictions)
        next_event = (
            "continue-performance" if include_predictions else "continue-process-performance"
        )
        self._show(
            "Performance result",
            "Performance settings selected",
            body,
            self._buttons(
                (
                    "Run Again",
                    "run-performance" if include_predictions else "run-process-performance",
                    "secondary",
                ),
                ("Continue", next_event, "primary"),
            ),
        )

    def _skip_performance(self) -> None:
        session = self._require_session()
        defaults = default_addon_config()
        session.process_mode_decided = True
        session.process_result = None
        session.prediction_result = None
        session.refresh_result = None
        session.selected_history_review_count = None
        session.working_config[PROCESS_MANY_MODE_CONFIG_KEY] = PROCESS_MANY_FAST_MODE
        session.working_config[PREDICT_MANY_MODE_CONFIG_KEY] = PREDICT_MANY_FAST_MODE
        session.working_config[LIVE_REVIEW_PREDICTION_REFRESH_LIMIT_CONFIG_KEY] = int(
            defaults[LIVE_REVIEW_PREDICTION_REFRESH_LIMIT_CONFIG_KEY]
        )
        self._show_curve_question()

    def _skip_process_performance(self) -> None:
        session = self._require_session()
        session.process_mode_decided = True
        session.process_result = None
        session.selected_history_review_count = None
        session.working_config[PROCESS_MANY_MODE_CONFIG_KEY] = PROCESS_MANY_FAST_MODE
        self._after_process_performance()

    # ------------------------------------------------------------------
    # Curves and optional accuracy comparisons

    def _run_curve_test(self) -> None:
        session = self._require_session()
        if session.speed_tests_unavailable_reason:
            self._show_curve_test_intro()
            return
        if self._rwkv_busy():
            self._render_inline_error(
                "Stop the active RWKV operation before testing curve calculation.",
                retry_action="curve-test",
            )
            return
        manager, store = self._manager_and_store()
        model_id = str(session.working_config[MODEL_CONFIG_KEY])

        def collect(col, progress, _previous):
            progress.update(0, 1, "Preparing 3,000 reviews")
            review_load = load_review_data_for_checkpoint(
                col,
                store,
                manager,
                progress,
                allow_incremental=True,
            )
            rows = tuple(capped_curve_speed_test_rows(review_load.review_data.rows)[:3000])
            progress.update(1, 1, "Curve speed-test rows ready")
            return rows, len(review_load.review_data.rows)

        def benchmark(_col, progress, previous):
            rows, available_count = previous
            slot = manager.reserve_runtime_slot(progress)
            try:
                return run_process_many_curve_speed_test(
                    review_count=len(rows),
                    available_review_count=available_count,
                    model_id=model_id,
                    mode=PROCESS_MANY_FAST_MODE,
                    run_once=lambda curves: benchmark_process_many_rows(
                        rows,
                        model_id=model_id,
                        mode=PROCESS_MANY_FAST_MODE,
                        return_curves=curves,
                    ),
                    progress=progress,
                    repetitions=1,
                )
            finally:
                slot.close()

        self._start_action(
            "curve-test",
            title="Testing curve calculation",
            stages=[
                ProgressStage(collect, uses_collection=True),
                ProgressStage(benchmark, uses_collection=False),
            ],
            on_success=self._curve_test_succeeded,
        )

    def _curve_test_succeeded(self, result: ProcessManyCurveSpeedTestResult) -> None:
        session = self._require_session()
        session.curve_result = result
        available_review_count = max(0, int(result.available_review_count))
        session.selected_history_review_count = available_review_count or None
        session.step = "curve-result"
        glossary = GlossaryRenderer("setup-curve-speed-result")
        with_curves = result.measurement(True)
        without_curves = result.measurement(False)
        gain = lower_is_better_improvement_percent(
            with_curves.average_seconds,
            without_curves.average_seconds,
        )
        direction = (
            f"{abs(gain):.1f}% faster"
            if gain is not None and gain >= 0
            else f"{abs(gain or 0):.1f}% slower"
        )
        body = (
            '<div class="setup-results">'
            + self._result_row(
                "Forgetting curves on",
                self._format_duration(with_curves.average_seconds),
                f"{with_curves.reviews_per_minute:,.0f} reviews/minute",
            )
            + self._result_row(
                "Forgetting curves off",
                self._format_duration(without_curves.average_seconds),
                f"{without_curves.reviews_per_minute:,.0f} reviews/minute",
            )
            + "</div>"
            f'<div class="setup-callout">{glossary.render("On this sample, turning forgetting curves off was")} <strong>{html.escape(direction)}</strong>. '
            "Your feature choice has not changed.</div>"
            '<p class="setup-copy">Choose whether to keep forgetting curves enabled '
            "before continuing.</p>"
        )
        self._show(
            "Curve performance",
            "Curve calculation measured",
            body,
            self._buttons(
                ("Run Again", "run-curve-test", "secondary"),
                ("Disable Forgetting Curves", "disable-curves-after-test", "secondary"),
                ("Keep Forgetting Curves Enabled", "keep-curves-after-test", "primary"),
            ),
        )

    def _finish_curve_test(self, curves_enabled: bool) -> None:
        session = self._require_session()
        session.curves_enabled = bool(curves_enabled)
        session.working_config = apply_setup_feature_choices(
            session.working_config,
            immediate_enabled=bool(session.immediate_enabled),
            curves_enabled=bool(session.curves_enabled),
        )
        if not session.immediate_enabled and not session.curves_enabled:
            self._show_neither_selected()
            return
        self._prepare_optimization_estimate()

    def _skip_curve_test(self) -> None:
        self._prepare_optimization_estimate()

    def _prepare_optimization_estimate(self) -> None:
        session = self._require_session()
        if not session.immediate_enabled and not session.curves_enabled:
            self._show_summary()
            return
        measured_count = session.selected_history_review_count
        counts = (
            _OptimizationCounts(
                current_review_count=measured_count,
                all_review_count=measured_count,
                selected_history_review_count=measured_count,
                approximate=True,
            )
            if measured_count is not None
            else None
        )
        self._show_optimization_choices(counts)

    def _show_optimization_choices(self, counts: _OptimizationCounts | None) -> None:
        session = self._require_session()
        session.optimization_counts = counts
        rate = self._selected_process_reviews_per_minute()
        estimate = (
            estimate_optimization_durations(
                reviews_per_minute=rate,
                model_count=len(self.model_ids),
                model_review_count=counts.selected_history_review_count,
                without_deleted_review_count=counts.current_review_count,
                with_deleted_review_count=counts.all_review_count,
            )
            if counts is not None
            else None
        )
        session.step = "optimization"
        glossary = GlossaryRenderer("setup-accuracy-choice")
        if estimate is None or estimate.matrix_comparison_seconds is None:
            estimate_copy = (
                "The setup speed test was skipped or did not produce a usable rate, so "
                "a time estimate is unavailable. Comparisons remain cancellable."
            )
        elif counts is not None and counts.approximate:
            estimate_copy = (
                "Based on the review history already measured: "
                f"Models only, about {self._format_duration(estimate.model_comparison_seconds)}. "
                f"Deleted history only, roughly {self._format_duration(estimate.deleted_reviews_comparison_seconds)}. "
                f"Test All, roughly {self._format_duration(estimate.matrix_comparison_seconds)}. "
                "The latter two may change because including deleted-card history changes "
                "how many reviews are processed."
            )
        else:
            estimate_copy = (
                f"Models only: about {self._format_duration(estimate.model_comparison_seconds)}. "
                f"Deleted history only: about {self._format_duration(estimate.deleted_reviews_comparison_seconds)}. "
                f"Test All: about {self._format_duration(estimate.matrix_comparison_seconds)}."
            )
        body = (
            '<p class="setup-copy">RWKV is already configured. These optional tests '
            "process your full review history several times to look for small, "
            "collection-specific accuracy gains. <strong>Skipping them is recommended "
            "for most users.</strong></p>"
            f'<div class="setup-callout warning">{html.escape(estimate_copy)}</div>'
            f'<p class="setup-copy">{glossary.render("Each RWKV model was trained on a different subset of benchmark data, and deleted-card history can affect the result. Test All checks every combination. Setup compares LogLoss and then RMSE(bins), both as shown to four decimal places; Test All uses the smaller checkpoint for any remaining tie. Lower metric scores are better.")}</p>'
            + self._choice_grid(
                (
                    (
                        "Compare Deleted-Card History",
                        "Test with and without deleted-card history using the current model.",
                        "optimize-deleted-only",
                    ),
                    (
                        "Test Models Only",
                        "Compare every bundled model using the current setting for deleted-card history.",
                        "optimize-models-only",
                    ),
                    (
                        "Test All",
                        "Test every model with and without deleted-card history.",
                        "optimize-all",
                    ),
                    (
                        "Skip accuracy tuning (recommended)",
                        "Keep the current model and deleted-card history setting.",
                        "optimize-no",
                    ),
                ),
                initial_focus_event="optimize-no",
            )
        )
        self._show("Step 3 of 4", "Optional accuracy tuning", body)

    def _show_model_test_intro(self) -> None:
        session = self._require_session()
        session.step = "model-test"
        glossary = GlossaryRenderer("setup-model-comparison")
        history_policy = (
            "with deleted-card history"
            if not bool(
                session.working_config[EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY]
            )
            else "without deleted-card history"
        )
        body = (
            f'<p class="setup-copy">Setup will test all {len(self.model_ids):,} bundled '
            f"{glossary.render(f'RWKV models one at a time, {history_policy}. Only reviews belonging to cards still in your collection are scored. Each temporary state is discarded before the next model is tested.')}</p>"
            f'<p class="setup-copy">{glossary.render("The model with the lowest LogLoss shown to four decimal places wins. RMSE(bins) is used only when LogLoss is tied; if both scores tie, Setup prefers the default model. Lower scores are better.")}</p>'
        )
        self._show(
            "Accuracy tuning",
            "Compare RWKV models",
            body,
            self._buttons(
                ("Skip and Use Default Model", "skip-model-test", "secondary"),
                ("Compare Models", "run-model-test", "primary"),
            ),
        )

    def _run_model_test(self) -> None:
        if len(self.model_ids) < 2:
            self._skip_model_test()
            return
        manager, store = self._manager_and_store()
        session = self._require_session()
        include_deleted = not bool(session.working_config[EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY])

        def collect(col, progress, _previous):
            progress.update(0, 2, "Finding current cards")
            current_ids = frozenset(int(card_id) for card_id in find_cards(col, ""))
            progress.update(1, 2, "Reading review history for model comparison")
            data = load_review_data_from_collection(
                col,
                store,
                exclude_deleted_card_revlogs=not include_deleted,
            )
            reviewed_ids: set[int] = set()
            current_review_count = 0
            for row in data.rows:
                card_id = int(row["card_id"])
                if card_id in current_ids:
                    reviewed_ids.add(card_id)
                    current_review_count += 1
            plan = model_comparison_plan(
                rows=data.rows,
                current_card_ids=reviewed_ids,
                model_ids=self.model_ids,
                current_model_id=(
                    str(session.working_config.get(MODEL_CONFIG_KEY))
                    if str(session.working_config.get(MODEL_CONFIG_KEY)) in self.model_ids
                    else self.model_ids[0]
                ),
                include_deleted_reviews=include_deleted,
                current_review_count=current_review_count,
                process_many_mode=self._selected_process_mode(),
            )
            progress.update(2, 2, "Model comparison ready")
            return _ComparisonInput(plan)

        self._start_comparison_action(
            "model-test",
            title="Comparing RWKV models",
            collect=collect,
            on_success=self._model_test_succeeded,
            manager=manager,
        )

    def _model_test_succeeded(self, result: ImmediateStateComparisonResult) -> None:
        session = self._require_session()
        session.model_result = result
        metrics = {measurement.key: measurement.metrics for measurement in result.measurements}
        default_key = DEFAULT_MODEL_ID if DEFAULT_MODEL_ID in metrics else self.model_ids[0]
        selection = select_metric_winner(metrics, default_key=default_key)
        winner = selection.winner_key
        valid_keys = valid_metric_candidate_keys(metrics)
        session.working_config[MODEL_CONFIG_KEY] = winner
        session.step = "model-result"
        glossary = GlossaryRenderer("setup-model-result")
        rows = "".join(
            self._result_row(
                measurement.label,
                self._metric_pair(measurement.metrics),
                "Selected" if measurement.key == winner else "",
                selected=measurement.key == winner,
            )
            for measurement in result.measurements
        )
        if not valid_keys:
            selection_copy = (
                "No model produced usable metrics. Setup retained the default model "
                f"<strong>{html.escape(winner)}</strong>."
            )
            callout_class = "warning"
        elif len(valid_keys) == 1:
            selection_copy = (
                f"Selected <strong>{html.escape(winner)}</strong>. It was the only model "
                "that produced usable metrics; failed results were not treated as ties."
            )
            callout_class = "success"
        else:
            alternative = result.measurement(str(selection.comparison_key))
            selected = result.measurement(winner)
            metric_changes = self._metric_change_summary(
                selected.metrics,
                alternative.metrics,
            )
            selection_copy = (
                f"Selected <strong>{html.escape(winner)}</strong>. "
                f"{self._metric_selection_reason(selection.reason)}<br>"
                f"Compared with <strong>{html.escape(alternative.label)}</strong>, "
                f"{html.escape(metric_changes)}."
            )
            callout_class = "success"
        body = (
            f'<div class="setup-callout {callout_class}">{selection_copy}</div>'
            f'<div class="setup-results">{rows}</div>'
            f'<p class="setup-copy">{glossary.render("Scores are compared as shown, to four decimal places. LogLoss is considered first; RMSE(bins) is a tiebreaker only when LogLoss is tied. Lower scores are better.")}</p>'
        )
        buttons = []
        if len(valid_keys) < len(metrics):
            buttons.append(("Retry Comparison", "run-model-test", "secondary"))
        buttons.append(("Continue", "continue-model-test", "primary"))
        self._show(
            "Accuracy tuning",
            "Model selected",
            body,
            self._buttons(*buttons),
        )

    def _skip_model_test(self) -> None:
        session = self._require_session()
        session.working_config[MODEL_CONFIG_KEY] = (
            DEFAULT_MODEL_ID if DEFAULT_MODEL_ID in self.model_ids else self.model_ids[0]
        )
        self._show_summary()

    def _show_deleted_test_intro(self) -> None:
        session = self._require_session()
        session.step = "deleted-test"
        glossary = GlossaryRenderer("setup-deleted-history")
        model_id = html.escape(str(session.working_config[MODEL_CONFIG_KEY]))
        body = (
            f'<p class="setup-copy">{glossary.render("Deleted-card history can give RWKV more context and may improve predictions for cards you still have. Including it also means more State Building work and may make the checkpoint larger.")}</p>'
            f'<p class="setup-copy">Using <strong>{model_id}</strong>, {glossary.render("Setup will score the same current-card reviews once without deleted history and once with it, then estimate each checkpoint’s disk space.")}</p>'
            f'<p class="setup-copy">{glossary.render("The lower LogLoss wins. RMSE(bins) breaks a LogLoss tie. If both scores tie at four decimal places, Setup excludes deleted-card history to save space.")}</p>'
        )
        self._show(
            "Accuracy tuning",
            "Compare deleted-card history",
            body,
            self._buttons(
                ("Skip and Exclude Deleted History", "skip-deleted-test", "secondary"),
                ("Run Comparison", "run-deleted-test", "primary"),
            ),
        )

    def _run_deleted_test(self) -> None:
        manager, store = self._manager_and_store()
        session = self._require_session()

        def collect(col, progress, _previous):
            current_rows, all_rows, reviewed_ids, adjustment = (
                self._collect_deleted_history_comparison_rows(
                    col,
                    progress,
                    store,
                )
            )
            plan = deleted_review_comparison_plan(
                current_rows=current_rows,
                all_rows=all_rows,
                current_card_ids=reviewed_ids,
                model_id=str(session.working_config[MODEL_CONFIG_KEY]),
                current_includes_deleted_reviews=not bool(
                    session.working_config[EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY]
                ),
                process_many_mode=self._selected_process_mode(),
                without_deleted_day_offset_adjustment=adjustment,
            )
            progress.update(2, 2, "Deleted-history comparison ready")
            return _ComparisonInput(plan)

        self._start_comparison_action(
            "deleted-test",
            title="Comparing deleted-card history",
            collect=collect,
            on_success=self._deleted_test_succeeded,
            manager=manager,
        )

    @staticmethod
    def _collect_deleted_history_comparison_rows(col, progress, store):
        progress.update(0, 2, "Finding current cards")
        current_ids = frozenset(int(card_id) for card_id in find_cards(col, ""))
        progress.update(1, 2, "Reading history including deleted cards")
        data = load_review_data_from_collection(
            col,
            store,
            exclude_deleted_card_revlogs=False,
        )
        current_rows, adjustment = current_review_rows_from_included_history(
            data.rows,
            current_ids,
            source_day_offset_origin=data.day_offset_origin,
        )
        reviewed_ids = frozenset(int(row["card_id"]) for row in current_rows)
        return current_rows, data.rows, reviewed_ids, adjustment

    def _deleted_test_succeeded(self, result: ImmediateStateComparisonResult) -> None:
        session = self._require_session()
        session.deleted_result = result
        session.step = "deleted-result"
        glossary = GlossaryRenderer("setup-deleted-history-result")
        metrics = {measurement.key: measurement.metrics for measurement in result.measurements}
        default_key = "without-deleted-reviews"
        selection = select_metric_winner(metrics, default_key=default_key)
        winner = selection.winner_key
        valid_keys = valid_metric_candidate_keys(metrics)
        session.working_config[EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY] = (
            winner != "with-deleted-reviews"
        )
        selected = result.measurement(winner)
        alternative = next(
            measurement for measurement in result.measurements if measurement.key != winner
        )
        size_without = result.measurement("without-deleted-reviews").expected_checkpoint_bytes
        size_with = result.measurement("with-deleted-reviews").expected_checkpoint_bytes
        size_text = self._checkpoint_size_comparison(size_without, size_with)
        rows = "".join(
            self._result_row(
                measurement.label,
                self._metric_pair(measurement.metrics),
                self._format_storage(measurement.expected_checkpoint_bytes),
                selected=measurement.key == winner,
            )
            for measurement in result.measurements
        )
        if not valid_keys:
            improvement_text = (
                "Neither policy produced usable metrics. Setup retained the default "
                "policy of excluding deleted-card history."
            )
            callout_class = "warning"
        elif len(valid_keys) == 1:
            improvement_text = (
                f"Only {selected.label} produced usable metrics; the unavailable result "
                "was not treated as a tie."
            )
            callout_class = "success"
        else:
            metric_changes = self._metric_change_summary(
                selected.metrics,
                alternative.metrics,
            )
            selection_reason = (
                "Both metrics tied at four decimal places, so Setup excluded "
                "deleted-card history to save checkpoint space."
                if selection.reason == "default-tie"
                else self._metric_selection_reason(selection.reason)
            )
            improvement_text = (
                f"{selection_reason} Compared with {alternative.label}, {metric_changes}."
            )
            callout_class = "success"
        body = (
            f'<div class="setup-callout {callout_class}">Selected <strong>{html.escape(selected.label)}</strong>. '
            f"{html.escape(improvement_text)}</div>"
            f'<div class="setup-results">{rows}</div>'
            f'<div class="setup-callout">{html.escape(size_text)}</div>'
            f'<p class="setup-copy">{glossary.render("Scores are compared as shown, to four decimal places. LogLoss is considered first; RMSE(bins) is a tiebreaker only when LogLoss is tied. Lower scores are better.")}</p>'
        )
        buttons = []
        if len(valid_keys) < len(metrics):
            buttons.append(("Retry Comparison", "run-deleted-test", "secondary"))
        buttons.append(("Review Setup", "continue-deleted-test", "primary"))
        self._show(
            "Accuracy tuning",
            "Deleted-card history selected",
            body,
            self._buttons(*buttons),
        )

    def _skip_deleted_test(self) -> None:
        session = self._require_session()
        session.working_config[EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY] = True
        self._show_summary()

    def _show_matrix_test_intro(self) -> None:
        session = self._require_session()
        session.step = "matrix-test"
        glossary = GlossaryRenderer("setup-model-history-matrix")
        combination_count = len(self.model_ids) * 2
        body = (
            f'<p class="setup-copy">Setup will build and score all {combination_count:,} '
            f"{glossary.render('combinations of bundled RWKV model and setting for deleted-card history. Each temporary state is discarded before the next combination is tested.')}</p>"
            f'<p class="setup-copy">{glossary.render("The lowest LogLoss shown to four decimal places wins. RMSE(bins) breaks a LogLoss tie; if both metrics tie, the smaller expected checkpoint wins. If every value still ties, Setup prefers the default model without deleted-card history.")}</p>'
        )
        self._show(
            "Accuracy tuning",
            "Test all model and history combinations",
            body,
            self._buttons(
                ("Skip and Use Defaults", "skip-matrix-test", "secondary"),
                ("Test All Combinations", "run-matrix-test", "primary"),
            ),
        )

    def _run_matrix_test(self) -> None:
        manager, store = self._manager_and_store()
        session = self._require_session()

        def collect(col, progress, _previous):
            current_rows, all_rows, reviewed_ids, adjustment = (
                self._collect_deleted_history_comparison_rows(
                    col,
                    progress,
                    store,
                )
            )
            selected_model = str(session.working_config.get(MODEL_CONFIG_KEY))
            if selected_model not in self.model_ids:
                selected_model = (
                    DEFAULT_MODEL_ID
                    if DEFAULT_MODEL_ID in self.model_ids
                    else self.model_ids[0]
                )
            plan = model_deleted_review_matrix_plan(
                current_rows=current_rows,
                all_rows=all_rows,
                current_card_ids=reviewed_ids,
                model_ids=self.model_ids,
                current_model_id=selected_model,
                current_includes_deleted_reviews=not bool(
                    session.working_config[EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY]
                ),
                process_many_mode=self._selected_process_mode(),
                without_deleted_day_offset_adjustment=adjustment,
            )
            progress.update(2, 2, "Full accuracy comparison ready")
            return _ComparisonInput(plan)

        self._start_comparison_action(
            "matrix-test",
            title="Testing every model and history combination",
            collect=collect,
            on_success=self._matrix_test_succeeded,
            manager=manager,
        )

    def _matrix_test_succeeded(self, result: ImmediateStateComparisonResult) -> None:
        session = self._require_session()
        session.matrix_result = result
        session.step = "matrix-result"
        metrics = {
            measurement.key: measurement.metrics
            for measurement in result.measurements
        }
        checkpoint_sizes = {
            measurement.key: measurement.expected_checkpoint_bytes
            for measurement in result.measurements
        }
        default_model = (
            DEFAULT_MODEL_ID
            if any(
                measurement.model_id == DEFAULT_MODEL_ID
                for measurement in result.measurements
            )
            else self.model_ids[0]
        )
        default_measurement = next(
            measurement
            for measurement in result.measurements
            if measurement.model_id == default_model
            and not measurement.include_deleted_reviews
        )
        selection = select_metric_winner(
            metrics,
            default_key=default_measurement.key,
            checkpoint_sizes_by_key=checkpoint_sizes,
        )
        winner = result.measurement(selection.winner_key)
        valid_keys = valid_metric_candidate_keys(metrics)
        session.working_config[MODEL_CONFIG_KEY] = winner.model_id
        session.working_config[EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY] = (
            not winner.include_deleted_reviews
        )
        rows = "".join(
            self._result_row(
                measurement.label,
                self._metric_pair(measurement.metrics),
                self._format_storage(measurement.expected_checkpoint_bytes),
                selected=measurement.key == winner.key,
            )
            for measurement in result.measurements
        )
        if not valid_keys:
            selection_copy = (
                "No combination produced usable metrics. Setup retained the default "
                "model without deleted-card history."
            )
            callout_class = "warning"
        elif len(valid_keys) == 1:
            selection_copy = (
                f"Selected <strong>{html.escape(winner.label)}</strong>. It was the "
                "only combination that produced usable metrics."
            )
            callout_class = "success"
        else:
            alternative = result.measurement(str(selection.comparison_key))
            metric_changes = self._metric_change_summary(
                winner.metrics,
                alternative.metrics,
            )
            selection_copy = (
                f"Selected <strong>{html.escape(winner.label)}</strong>. "
                f"{self._metric_selection_reason(selection.reason)}<br>"
                f"Compared with <strong>{html.escape(alternative.label)}</strong>, "
                f"{html.escape(metric_changes)}."
            )
            callout_class = "success"
        body = (
            f'<div class="setup-callout {callout_class}">{selection_copy}</div>'
            f'<div class="setup-results">{rows}</div>'
            '<p class="setup-copy">Scores are compared as shown, to four decimal '
            "places. Lower scores are better. Checkpoint size is considered only "
            "after both displayed metrics tie.</p>"
        )
        buttons = []
        if len(valid_keys) < len(metrics):
            buttons.append(("Retry Comparison", "run-matrix-test", "secondary"))
        buttons.append(("Review Setup", "continue-matrix-test", "primary"))
        self._show(
            "Accuracy tuning",
            "Model and history selected",
            body,
            self._buttons(*buttons),
        )

    def _skip_matrix_test(self) -> None:
        session = self._require_session()
        session.working_config[MODEL_CONFIG_KEY] = (
            DEFAULT_MODEL_ID if DEFAULT_MODEL_ID in self.model_ids else self.model_ids[0]
        )
        session.working_config[EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY] = True
        self._show_summary()

    def _start_comparison_action(
        self,
        action_name: str,
        *,
        title: str,
        collect,
        on_success,
        manager,
    ) -> None:
        if self._rwkv_busy():
            self._render_inline_error(
                "Stop the active RWKV operation before starting this comparison.",
                retry_action=action_name,
            )
            return

        def compare(_col, progress, previous):
            input_data: _ComparisonInput = previous
            slot = None
            try:
                slot = manager.reserve_runtime_slot(progress)
                return run_immediate_state_comparison(input_data.plan, progress)
            finally:
                input_data.release_rows()
                if slot is not None:
                    slot.close()

        self._start_action(
            action_name,
            title=title,
            stages=[
                ProgressStage(collect, uses_collection=True),
                ProgressStage(compare, uses_collection=False),
            ],
            on_success=on_success,
        )

    # ------------------------------------------------------------------
    # Transaction summary and exit

    def _show_summary(self) -> None:
        session = self._require_session()
        session.step = "summary"
        summary = summarize_setup_config(
            session.entry_config,
            session.working_config,
            restart_context=self.dialog._restart_requirement_context(),
        )
        changes = self._change_list(summary.changes)
        notices = self._summary_notices(
            summary,
            checkpoint_exists=self.dialog._checkpoint_exists(),
        )
        if session.initial_launch:
            introduction = (
                '<p class="setup-copy">Review the suggested settings below. Choosing '
                "<strong>Save Setup</strong> saves them for this profile. You will then "
                "see whether Anki needs to restart or whether RWKV can build its state "
                "immediately.</p>"
            )
            finish_label = "Save Setup"
        else:
            introduction = (
                '<p class="setup-copy">Review the suggested settings below. Choosing '
                "<strong>Use These Settings</strong> copies them to the main Settings page; "
                "nothing is saved until you choose <strong>Apply</strong> or <strong>OK</strong>.</p>"
            )
            finish_label = "Use These Settings"
        body = introduction + changes + notices + render_setup_faq_link()
        self._show(
            "Step 4 of 4",
            "Review setup",
            body,
            self._buttons(
                ("Discard Setup", "exit-discard", "danger"),
                (finish_label, "finish", "primary"),
            ),
        )

    def _finish(self) -> None:
        session = self._require_session()
        if session.initial_launch:
            summary = summarize_setup_config(
                session.entry_config,
                session.working_config,
                restart_context=self.dialog._restart_requirement_context(),
            )
            if not self.dialog._save_initial_setup_config(session.working_config):
                return
            if not self.dialog._record_initial_setup_seen():
                return
            self._render_initial_completion(summary)
            return
        self.dialog._accept_setup_config(session.working_config)
        self._close()

    def _render_initial_completion(self, summary) -> None:
        session = self._require_session()
        features_enabled = bool(session.immediate_enabled or session.curves_enabled)
        restart_required = bool(summary.restart_required_labels)
        if restart_required:
            session.step = "initial-complete-restart"
            restart_items = "".join(
                f"<li>{html.escape(label)}</li>" for label in summary.restart_required_labels
            )
            rebuild_guidance = (
                '<p class="setup-copy">After restarting, open '
                "<strong>RWKV → Manage Checkpoint</strong>. Choose "
                "<strong>Initialize Checkpoint</strong> if this profile has no RWKV "
                "state yet, or <strong>Rebuild Checkpoint</strong> when replacing an "
                "existing state.</p>"
                if features_enabled
                else (
                    '<p class="setup-copy">No RWKV state needs to be built while both '
                    "Immediate and Forgetting Curve features remain off.</p>"
                )
            )
            body = (
                '<div class="setup-callout success"><strong>Your setup choices are saved for this profile.</strong></div>'
                '<div class="setup-callout warning"><strong>Restart Anki before using these changes:</strong>'
                f'<ul class="setup-change-list">{restart_items}</ul></div>'
                '<p class="setup-copy">Building now could still use the old State '
                "Building Mode or other settings held by the current Anki process, so "
                "setup will wait until after the restart.</p>" + rebuild_guidance
            )
            title = "Restart Anki to finish"
        elif features_enabled:
            session.step = "initial-complete-build"
            body = (
                '<div class="setup-callout success"><strong>Your setup choices are saved for this profile.</strong></div>'
                '<p class="setup-copy">RWKV now needs a local state built from your '
                "review history. <strong>Build State Now</strong> closes Guided Setup, "
                "shows the estimated checkpoint size and build time, and asks for "
                "confirmation before processing.</p>"
                '<p class="setup-copy">You can also do this later from '
                "<strong>RWKV → Manage Checkpoint → Initialize Checkpoint</strong>. "
                "If a prior state needs replacement, use <strong>Rebuild Checkpoint</strong>.</p>"
            )
            title = "Build the RWKV state?"
        else:
            session.step = "initial-complete-disabled"
            body = (
                '<div class="setup-callout success"><strong>Your setup choices are saved for this profile.</strong></div>'
                '<p class="setup-copy">RWKV Immediate and RWKV Forgetting Curve are '
                "both off, so there is no state to build. Run Guided Setup again from "
                "RWKV Settings whenever you want to enable them.</p>"
            )
            title = "Setup complete"

        buttons = (
            self._buttons(
                ("Do It Later", "initial-done", "secondary"),
                ("Build State Now", "initial-build", "primary"),
            )
            if not restart_required and features_enabled
            else self._buttons(("Close Setup", "initial-done", "primary"))
        )
        self._show("Setup complete", title, body, buttons)

    def _render_exit_review(self) -> None:
        session = self._require_session()
        summary = summarize_setup_config(
            session.entry_config,
            session.working_config,
            restart_context=self.dialog._restart_requirement_context(),
        )
        if not summary.changes:
            self._discard_and_exit()
            return
        if session.initial_launch:
            introduction = (
                '<p class="setup-copy">You can save the choices made so far, discard '
                "every Guided Setup change, or return to the wizard. Closing either way "
                "records that setup was shown, so it will not open automatically again "
                "for this profile.</p>"
            )
            keep_label = "Save Changes and Exit"
        else:
            introduction = (
                '<p class="setup-copy">You can copy the choices made so far to the main '
                "Settings page, discard every Guided Setup change, or return to the wizard. "
                "Nothing is saved until you later choose <strong>Apply</strong> or "
                "<strong>OK</strong>.</p>"
            )
            keep_label = "Keep Changes and Exit"
        body = (
            introduction
            + self._change_list(summary.changes)
            + self._summary_notices(
                summary,
                checkpoint_exists=self.dialog._checkpoint_exists(),
            )
            + render_setup_faq_link()
        )
        self._show(
            "Exit setup",
            "Keep your current setup changes?",
            body,
            self._buttons(
                ("Abandon All Setup Changes", "exit-discard", "danger"),
                ("Continue Setup", "exit-resume", "secondary"),
                (keep_label, "exit-keep", "primary"),
            ),
            show_close_button=False,
        )

    def _keep_and_exit(self) -> None:
        session = self._require_session()
        if session.initial_launch:
            self.dialog._accept_setup_config(session.working_config, render=False)
            if not self.dialog.apply():
                return
            if not self.dialog._record_initial_setup_seen():
                return
            self._finish_initial_window(build_state=False)
            return
        self.dialog._accept_setup_config(session.working_config)
        self._close()

    def _discard_and_exit(self) -> None:
        session = self._require_session()
        if session.initial_launch:
            if not self.dialog._record_initial_setup_seen():
                return
            self._finish_initial_window(build_state=False)
            return
        self._close()

    def _resume_from_exit(self) -> None:
        session = self._require_session()
        if session.return_step is None:
            self._render_current_step()
            return
        session.step = session.return_step or "immediate"
        session.return_step = None
        self._render_current_step()

    def _close(self) -> None:
        self._eval("window.rwkvSetupClose && window.rwkvSetupClose();")
        self.session = None
        self._exit_after_cancel = False
        self.dialog._set_setup_active(False)

    def _finish_initial_window(self, *, build_state: bool) -> None:
        session = self._require_session()
        if not session.initial_launch:
            self._close()
            return
        self._close()
        self.dialog._finish_initial_setup_window(build_state=build_state)

    # ------------------------------------------------------------------
    # Background lifecycle

    def _start_action(
        self,
        action_name: str,
        *,
        title: str,
        stages: list[ProgressStage],
        on_success,
    ) -> None:
        if self.running:
            return
        session = self._require_session()
        session.step = action_name
        self._action_name = action_name
        self._action_generation += 1
        generation = self._action_generation
        self._progress_sequence = 0
        self._last_progress_sequence = 0

        def progress_callback(current: int, total: int, label: str, eta: float | None):
            self._progress_sequence += 1
            sequence = self._progress_sequence

            def apply() -> None:
                if (
                    self._shutting_down
                    or generation != self._action_generation
                    or sequence <= self._last_progress_sequence
                ):
                    return
                self._last_progress_sequence = sequence
                self._eval_progress(current, total, label, eta)

            self._mw.taskman.run_on_main(apply)

        reporter = TimedProgressReporter(progress_callback)
        self._reporter = reporter
        self._render_progress(title)

        def success(result) -> None:
            if not self._finish_action(generation):
                return
            if self._exit_after_cancel:
                self._exit_after_cancel = False
                self.request_exit()
                return
            try:
                on_success(result)
            except Exception as exc:
                self._render_interrupted(
                    action_name,
                    str(exc) or exc.__class__.__name__,
                )

        def failure(exc: Exception) -> None:
            if not self._finish_action(generation):
                return
            if self._exit_after_cancel:
                self._exit_after_cancel = False
                self.request_exit()
                return
            no_processable_reviews = isinstance(exc, NoProcessableReviewsError)
            if no_processable_reviews:
                session.speed_tests_unavailable_reason = _NO_PROCESSABLE_SPEED_TESTS_REASON
            self._render_interrupted(
                action_name,
                str(exc) or exc.__class__.__name__,
                retryable=not no_processable_reviews,
            )

        def cancelled() -> None:
            if not self._finish_action(generation):
                return
            if self._exit_after_cancel:
                self._exit_after_cancel = False
                self.request_exit()
                return
            self._render_interrupted(action_name, "The test was cancelled.")

        run_background_stages(
            stages=stages,
            reporter=reporter,
            on_success=success,
            on_failure=failure,
            on_cancel=cancelled,
        )

    def _finish_action(self, generation: int) -> bool:
        if generation != self._action_generation or self._reporter is None:
            return False
        self._reporter = None
        self._action_name = None
        return not self._shutting_down and self.session is not None

    def _render_progress(self, title: str) -> None:
        body = (
            '<div class="setup-progress-wrap">'
            '<div class="setup-progress-track indeterminate" id="rwkv-setup-progress-track" role="progressbar" aria-label="Setup progress" aria-valuemin="0" aria-valuemax="1" aria-valuenow="0">'
            '<div class="setup-progress-bar" id="rwkv-setup-progress-bar"></div></div>'
            '<div class="setup-progress-label" id="rwkv-setup-progress-label" aria-live="polite">Preparing…</div>'
            '<div class="setup-progress-meta"><span id="rwkv-setup-progress-count">Working…</span><span id="rwkv-setup-progress-eta">ETA unknown</span></div>'
            "</div>"
            '<p class="setup-copy">Cancellation is checked between processing chunks and '
            "short speed-test runs. Reading Anki's exported history may need to finish "
            "its current operation before cancellation completes.</p>"
        )
        self._show(
            "Working",
            title,
            body,
            self._buttons(("Cancel", "cancel-run", "danger")),
            busy=True,
        )

    def _eval_progress(
        self,
        current: int,
        total: int,
        label: str,
        eta: float | None,
    ) -> None:
        payload = {
            "current": max(0, int(current)),
            "total": max(0, int(total)),
            "label": str(label or "Working…"),
            "eta": format_eta(eta),
        }
        self._eval(
            "window.rwkvSetupProgress && window.rwkvSetupProgress("
            + self._script_json(payload)
            + ");"
        )

    def _render_interrupted(
        self,
        action_name: str,
        message: str,
        *,
        retryable: bool = True,
    ) -> None:
        session = self._require_session()
        session.step = f"interrupted:{action_name}"
        session.interruption_message = str(message)
        session.interruption_retryable = bool(retryable)
        guidance = (
            '<p class="setup-copy">You can rerun this step, or use the recommended '
            "defaults for it and continue. No partial measurement has been applied.</p>"
            if retryable
            else '<p class="setup-copy">Use the recommended defaults to continue, or '
            "exit setup. No partial measurement has been applied.</p>"
        )
        body = f'<div class="setup-callout warning">{html.escape(message)}</div>' + guidance
        buttons = [("Exit Setup", "exit", "secondary")]
        if retryable:
            buttons.append(("Run Again", "retry", "secondary"))
        buttons.append(("Use Defaults and Continue", "skip-current", "primary"))
        self._show(
            "Test interrupted",
            "Rerun or continue with defaults" if retryable else "Continue with defaults",
            body,
            self._buttons(*buttons),
        )

    def _retry_current_action(self) -> None:
        session = self._require_session()
        action = session.step.removeprefix("interrupted:")
        {
            "performance": lambda: self._run_performance(include_predictions=True),
            "process-performance": lambda: self._run_performance(include_predictions=False),
            "curve-test": self._run_curve_test,
            "model-test": self._run_model_test,
            "deleted-test": self._run_deleted_test,
            "matrix-test": self._run_matrix_test,
        }.get(action, self._show_summary)()

    def _skip_current_action(self) -> None:
        session = self._require_session()
        action = session.step.removeprefix("interrupted:")
        {
            "performance": self._skip_performance,
            "process-performance": self._skip_process_performance,
            "curve-test": self._skip_curve_test,
            "model-test": self._skip_model_test,
            "deleted-test": self._skip_deleted_test,
            "matrix-test": self._skip_matrix_test,
        }.get(action, self._show_summary)()

    # ------------------------------------------------------------------
    # Rendering helpers

    def _render_current_step(self) -> None:
        session = self._require_session()
        step = session.step
        if step == "welcome":
            self._show_initial_welcome()
        elif step == "immediate":
            self._show_immediate_question()
        elif step in {"performance", "performance-result"}:
            if step == "performance-result" and session.process_result is not None:
                self._performance_succeeded(
                    _PerformanceResult(
                        session.process_result,
                        session.prediction_result,
                        session.refresh_result,
                        available_review_count=(
                            session.selected_history_review_count or 0
                        ),
                    ),
                    include_predictions=True,
                )
            else:
                self._show_performance_intro(include_predictions=True)
        elif step == "curves":
            self._show_curve_question()
        elif step in {"process-performance", "process-result"}:
            if step == "process-result" and session.process_result is not None:
                self._performance_succeeded(
                    _PerformanceResult(
                        session.process_result,
                        None,
                        None,
                        available_review_count=(
                            session.selected_history_review_count or 0
                        ),
                    ),
                    include_predictions=False,
                )
            else:
                self._show_performance_intro(include_predictions=False)
        elif step in {"curve-test", "curve-result"}:
            if step == "curve-result" and session.curve_result is not None:
                self._curve_test_succeeded(session.curve_result)
            else:
                self._show_curve_test_intro()
        elif step == "optimization":
            if session.optimization_counts is not None:
                self._show_optimization_choices(session.optimization_counts)
            else:
                self._prepare_optimization_estimate()
        elif step in {"model-test", "model-result"}:
            if step == "model-result" and session.model_result is not None:
                self._model_test_succeeded(session.model_result)
            else:
                self._show_model_test_intro()
        elif step in {"deleted-test", "deleted-result"}:
            if step == "deleted-result" and session.deleted_result is not None:
                self._deleted_test_succeeded(session.deleted_result)
            else:
                self._show_deleted_test_intro()
        elif step in {"matrix-test", "matrix-result"}:
            if step == "matrix-result" and session.matrix_result is not None:
                self._matrix_test_succeeded(session.matrix_result)
            else:
                self._show_matrix_test_intro()
        elif step == "summary":
            self._show_summary()
        elif step == "neither":
            self._show_neither_selected()
        elif step == "exit":
            self._render_exit_review()
        elif step.startswith("interrupted:"):
            self._render_interrupted(
                step.removeprefix("interrupted:"),
                session.interruption_message or "The test was interrupted.",
                retryable=session.interruption_retryable,
            )
        else:
            self._show_immediate_question()

    def _show(
        self,
        step: str,
        title: str,
        body: str,
        footer: str = "",
        *,
        busy: bool = False,
        show_close_button: bool = True,
    ) -> None:
        if not busy and self._back_is_available():
            footer = self._buttons(("Back", "back", "quiet")) + footer
        payload = {
            "step": str(step),
            "title": str(title),
            "body": str(body),
            "footer": str(footer),
            "busy": bool(busy),
            "showCloseButton": bool(show_close_button and not busy),
        }
        self._eval(
            "window.rwkvSetupRender && window.rwkvSetupRender(" + self._script_json(payload) + ");"
        )

    def _render_performance_result(
        self,
        result: _PerformanceResult,
        *,
        include_predictions: bool,
    ) -> str:
        glossary = GlossaryRenderer("setup-performance-result")
        process_rows = "".join(
            self._result_row(
                f"{self._mode_label(item.mode)} state building",
                "Failed" if not item.succeeded else self._format_duration(item.average_seconds),
                item.error
                or (
                    f"{item.items_per_second * 60:,.0f} reviews/minute"
                    if item.items_per_second is not None
                    else ""
                ),
                selected=item.mode == result.process.selected_mode,
            )
            for item in result.process.measurements
        )
        selected_process_mode = glossary.render(self._mode_label(result.process.selected_mode))
        sample_context = [
            f"{glossary.render('State Building')} used <strong>{result.process.review_count:,} reviews</strong>."
        ]
        if include_predictions and result.prediction is not None:
            sample_context.append(
                "Prediction Mode used "
                f"<strong>{result.prediction.card_count:,} reviewed cards</strong>."
            )
        if include_predictions and result.refresh_search is not None:
            probed_counts = sorted(
                {int(measurement.card_count) for measurement in result.refresh_search.measurements}
            )
            if len(probed_counts) == 1:
                sample_context.append(
                    f"Between-review tuning tested <strong>{probed_counts[0]:,} cards</strong>."
                )
            elif probed_counts:
                sample_context.append(
                    "Between-review tuning tried "
                    f"<strong>{len(probed_counts)} card counts</strong> from "
                    f"<strong>{probed_counts[0]:,}</strong> to "
                    f"<strong>{probed_counts[-1]:,} cards</strong>."
                )
        sections = [
            '<div class="setup-callout"><strong>Test size:</strong> '
            + " ".join(sample_context)
            + "</div>",
            f'<div class="setup-callout success">Selected <strong>{selected_process_mode}</strong> for {glossary.render("State Building")}.</div>',
            f'<div class="setup-results">{process_rows}</div>',
        ]
        if include_predictions and result.prediction is not None:
            prediction_rows = "".join(
                self._result_row(
                    f"{self._mode_label(item.mode)} Live Session cycle",
                    "Failed" if not item.succeeded else self._format_duration(item.average_seconds),
                    item.error
                    or (
                        f"{item.items_per_second:,.0f} cards/second"
                        if item.items_per_second is not None
                        else ""
                    ),
                    selected=item.mode == result.prediction.selected_mode,
                )
                for item in result.prediction.measurements
            )
            sections.extend(
                (
                    f'<div class="setup-callout success">Selected <strong>{glossary.render(self._mode_label(result.prediction.selected_mode))}</strong> for Live Session predictions.</div>',
                    f'<div class="setup-results">{prediction_rows}</div>',
                )
            )
        if include_predictions and result.refresh_search is not None:
            selected = result.refresh_search.measurement(result.refresh_search.selected_card_count)
            sections.append(
                '<div class="setup-callout"><strong>'
                f"{result.refresh_search.selected_card_count:,} Cards Checked Between Reviews</strong> "
                f"was closest to the 100 ms target ({self._format_duration(selected.duration_seconds)} measured). "
                "At 60 Hz, 100 ms is about six frames; delays below this are usually subtle."
                "</div>"
            )
        elif include_predictions:
            sections.append(
                '<div class="setup-callout warning">No reviewed cards were available for '
                "prediction tuning, so the recommended default prediction settings remain in use.</div>"
            )
        return "".join(sections)

    def _render_inline_error(
        self,
        message: str,
        *,
        retry_action: str | None = None,
    ) -> None:
        if retry_action:
            self._require_session().step = f"interrupted:{retry_action}"
        self._show(
            "Setup unavailable",
            "RWKV is currently busy",
            f'<div class="setup-callout warning">{html.escape(message)}</div>',
            self._buttons(("Exit Setup", "exit", "secondary"), ("Try Again", "retry", "primary")),
        )

    def _buttons(self, *buttons: tuple[str, str, str]) -> str:
        speed_tests_unavailable_reason = (
            self.session.speed_tests_unavailable_reason if self.session else None
        )
        normalized_buttons = [
            (
                label,
                event,
                "secondary"
                if speed_tests_unavailable_reason and event in _SETUP_SPEED_TEST_EVENTS
                else kind,
                (speed_tests_unavailable_reason if event in _SETUP_SPEED_TEST_EVENTS else None),
            )
            for label, event, kind in buttons
        ]
        primary_indices = [
            index
            for index, (_label, _event, kind, _disabled_reason) in enumerate(normalized_buttons)
            if kind == "primary"
        ]
        if len(primary_indices) > 1:
            raise ValueError("Setup Wizard footers may only have one primary action")
        if primary_indices and primary_indices[0] != len(normalized_buttons) - 1:
            raise ValueError("The Setup Wizard primary action must be the rightmost footer button")
        rendered: list[str] = []
        for label, event, kind, disabled_reason in normalized_buttons:
            variant = {
                "primary": "primary",
                "danger": "destructive",
                "quiet": "quiet",
            }.get(kind, "secondary")
            classes = f"rwkv-button rwkv-button--{variant}"
            if event == "back":
                classes += " setup-back-button"
            initial_focus = (
                " data-setup-initial-focus" if kind == "primary" and disabled_reason is None else ""
            )
            escaped_event = html.escape(event, quote=True)
            disabled = ' disabled aria-disabled="true"' if disabled_reason is not None else ""
            button = (
                f'<button type="button" class="{classes}" '
                f'data-setup-event="{escaped_event}"{initial_focus}{disabled}>'
                f"{html.escape(label)}</button>"
            )
            if disabled_reason is not None:
                escaped_label = html.escape(label, quote=True)
                escaped_reason = html.escape(disabled_reason, quote=True)
                button = (
                    '<span class="rwkv-disabled-control-help rwkv-help-surface" '
                    f'tabindex="0" aria-label="{escaped_label}. {escaped_reason}" '
                    f'data-rwkv-tooltip="{escaped_reason}">{button}</span>'
                )
            rendered.append(button)
        return "".join(rendered)

    @staticmethod
    def _choice_grid(
        choices: Sequence[tuple[str, str, str]],
        *,
        initial_focus_event: str | None = None,
    ) -> str:
        rendered: list[str] = []
        for title, description, event in choices:
            initial_focus = " data-setup-initial-focus" if event == initial_focus_event else ""
            rendered.append(
                '<button type="button" class="setup-choice" '
                f'data-setup-event="{html.escape(event, quote=True)}"'
                f"{initial_focus}>"
                f"<strong>{html.escape(title)}</strong>"
                f"<span>{html.escape(description)}</span></button>"
            )
        return '<div class="setup-choice-grid">' + "".join(rendered) + "</div>"

    @staticmethod
    def _result_row(
        label: str,
        value: str,
        detail: str = "",
        *,
        selected: bool = False,
    ) -> str:
        selected_class = " setup-selected" if selected else ""
        selected_text = " · selected" if selected else ""
        return (
            '<div class="setup-result-row">'
            f'<div class="{selected_class.strip()}"><strong>{html.escape(label)}</strong>'
            f"<small>{html.escape(detail)}{html.escape(selected_text)}</small></div>"
            f'<div class="setup-result-value{selected_class}">{html.escape(value)}</div>'
            "</div>"
        )

    @staticmethod
    def _change_list(changes) -> str:
        if not changes:
            return '<div class="setup-callout">No settings differ from when setup started.</div>'
        glossary = GlossaryRenderer("setup-change-list")
        return (
            '<ul class="setup-change-list">'
            + "".join(f"<li>{glossary.render(change.description)}</li>" for change in changes)
            + "</ul>"
        )

    @staticmethod
    def _summary_notices(summary, *, checkpoint_exists: bool) -> str:
        glossary = GlossaryRenderer("setup-summary-notices")
        restart = (
            '<div class="setup-callout warning"><strong>Restart Anki after applying:</strong><ul class="setup-change-list">'
            + "".join(f"<li>{html.escape(label)}</li>" for label in summary.restart_required_labels)
            + "</ul></div>"
            if summary.restart_required_labels
            else '<div class="setup-callout success">No Anki restart is required by these changes.</div>'
        )
        rebuild = (
            f'<div class="setup-callout warning"><strong>{glossary.render("Rebuild the checkpoint after applying:")}</strong><ul class="setup-change-list">'
            + "".join(
                f"<li>{glossary.render(reason.explanation)}</li>"
                for reason in summary.checkpoint_rebuild_reasons
            )
            + "</ul></div>"
            if checkpoint_exists and summary.checkpoint_rebuild_reasons
            else ""
        )
        return restart + rebuild

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None or not math.isfinite(float(seconds)):
            return "Unavailable"
        value = max(0.0, float(seconds))
        if value < 1.0:
            return f"{value * 1000:.1f} ms"
        if value < 60.0:
            return f"{value:.2f} s"
        minutes, remaining = divmod(round(value), 60)
        if minutes < 60:
            return f"{minutes}m {remaining}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m"

    @staticmethod
    def _metric_pair(metrics: MetricResult) -> str:
        if metrics.error or metrics.log_loss is None or metrics.rmse_bins is None:
            return metrics.error or "Unavailable"
        return (
            f"LogLoss {metrics.log_loss:.{METRIC_DISPLAY_DECIMAL_PLACES}f} · "
            f"RMSE(bins) {metrics.rmse_bins:.{METRIC_DISPLAY_DECIMAL_PLACES}f}"
        )

    @staticmethod
    def _metric_selection_reason(reason: str) -> str:
        return {
            "log-loss": "It had the lowest LogLoss.",
            "rmse-bins": (
                "LogLoss was tied at four decimal places, so the lower "
                "RMSE(bins) decided the selection."
            ),
            "checkpoint-size": (
                "LogLoss and RMSE(bins) tied at four decimal places, so the "
                "smaller checkpoint decided the selection."
            ),
            "default-tie": ("Both metrics tied at four decimal places, so Setup kept the default."),
            "stable-tie": (
                "The leading choices tied on both metrics at four decimal places; "
                "Setup selected the first available one."
            ),
        }.get(reason, "Setup selected the best available result.")

    @staticmethod
    def _metric_change_summary(
        selected: MetricResult,
        alternative: MetricResult,
    ) -> str:
        return "; ".join(
            (
                SetupWizardController._metric_change_phrase(
                    "LogLoss",
                    alternative.log_loss,
                    selected.log_loss,
                ),
                SetupWizardController._metric_change_phrase(
                    "RMSE(bins)",
                    alternative.rmse_bins,
                    selected.rmse_bins,
                ),
            )
        )

    @staticmethod
    def _metric_change_phrase(
        label: str,
        baseline_value: float | None,
        selected_value: float | None,
    ) -> str:
        if baseline_value is None or selected_value is None:
            return f"{label} comparison was unavailable"
        baseline = float(baseline_value)
        selected = float(selected_value)
        if metric_values_tie_at_display_precision(baseline, selected):
            return f"{label} tied at four decimal places"
        change = lower_is_better_improvement_percent(baseline, selected)
        if change is None:
            return f"{label} comparison was unavailable"
        direction = "improved" if change > 0 else "worsened"
        return f"{label} {direction} by {abs(change):.2f}%"

    @staticmethod
    def _mode_label(mode: str) -> str:
        return "GPU" if str(mode).lower() == "gpu" else str(mode).title()

    @staticmethod
    def _format_storage(value: int | None) -> str:
        return "Size unavailable" if value is None else format_storage_bytes(value)

    @staticmethod
    def _checkpoint_size_comparison(
        without_deleted: int | None,
        with_deleted: int | None,
    ) -> str:
        if without_deleted is None or with_deleted is None:
            return "RWKV-SRS could not report both expected checkpoint sizes."
        difference = int(with_deleted) - int(without_deleted)
        percent = difference / int(without_deleted) * 100.0 if int(without_deleted) else 0.0
        return (
            f"Expected checkpoint without deleted history: {format_storage_bytes(without_deleted)}. "
            f"With deleted history: {format_storage_bytes(with_deleted)} "
            f"({format_storage_bytes(abs(difference))} and {abs(percent):.1f}% "
            f"{'larger' if difference >= 0 else 'smaller'})."
        )

    @staticmethod
    def _script_json(value: object) -> str:
        return (
            json.dumps(value, ensure_ascii=False)
            .replace("&", "\\u0026")
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
        )

    def _eval(self, script: str) -> None:
        if self._shutting_down or self.dialog._cleaned_up:
            return
        try:
            self.dialog.web.eval(script)
        except RuntimeError:
            return

    # ------------------------------------------------------------------
    # Small state helpers

    def _require_session(self) -> _WizardSession:
        if self.session is None:
            raise RuntimeError("The RWKV Setup Wizard is not open.")
        return self.session

    def _selected_process_mode(self) -> str:
        return str(
            self._require_session().working_config.get(
                PROCESS_MANY_MODE_CONFIG_KEY,
                PROCESS_MANY_FAST_MODE,
            )
        )

    def _selected_process_reviews_per_minute(self) -> float | None:
        session = self._require_session()
        result = session.process_result
        selected_mode = self._selected_process_mode()
        if result is not None:
            try:
                measurement = result.measurement(selected_mode)
            except KeyError:
                pass
            else:
                rate = measurement.items_per_second
                if rate is not None and math.isfinite(rate) and rate > 0:
                    return rate * 60.0

        # A curve-only setup without processing-GPU support deliberately skips
        # the meaningless hardware-choice screen. Its curve-cost test is still
        # a measured CPU Fast state build, so use that rate for the optional
        # accuracy-test estimate instead of claiming no estimate is available.
        curve_result = session.curve_result
        if curve_result is None or selected_mode != str(curve_result.mode):
            return None
        try:
            curve_rate = float(curve_result.measurement(True).reviews_per_minute)
        except (KeyError, TypeError, ValueError):
            return None
        return curve_rate if math.isfinite(curve_rate) and curve_rate > 0 else None

    def _manager_and_store(self):
        from ..runtime import manager_for_mw, store_for_mw

        return manager_for_mw(self._mw), store_for_mw(self._mw)

    def _rwkv_busy(self) -> bool:
        manager, _store = self._manager_and_store()
        return bool(
            getattr(manager, "runtime_scope_active", False)
            or getattr(manager, "runtime_loaded", False)
            or getattr(manager, "save_in_progress", False)
        )


__all__ = ["SetupWizardController"]
