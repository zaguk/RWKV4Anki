from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .adaptive_retention import AdaptiveRetentionSettings
from .anki_api import DeckRetention
from .filtered_deck import retentions_with_desired_values
from .study_session_html import render_study_session_dialog_html
from .web_dialog_bridge import BridgePayloadError, WebDialogCommand
from .web_dialog_controller import BaseWebDialogController, CloseReason

UPDATE_FORM_ACTION = "update-form"
CHECK_SEARCH_ACTION = "check-search"
RESTORE_DEFAULTS_ACTION = "restore-defaults"
TOGGLE_RETENTION_EDITOR_ACTION = "toggle-retention-editor"

ADAPTIVE_RETENTION_SETTINGS_KEY = "adaptive_retention"


@dataclass(frozen=True)
class StudySessionFormSpec:
    """Workflow labels, ranges, actions, and storage keys for the shared form."""

    title: str
    intro: str
    size_section_title: str
    size_section_intro: str
    minimum_label: str
    minimum_description: str
    maximum_label: str
    maximum_description: str
    minimum_default: int
    maximum_default: int
    maximum_value: int
    default_order_index: int
    minimum_storage_key: str
    maximum_storage_key: str
    primary_action: str
    primary_label: str
    same_day_action: str
    same_day_label: str
    same_day_description: str
    split_same_day_action: bool = False

    def __post_init__(self) -> None:
        if self.maximum_value < 1:
            raise ValueError("maximum_value must be positive")
        if not 0 <= self.minimum_default <= self.maximum_value:
            raise ValueError("minimum_default is outside the form range")
        if not 1 <= self.maximum_default <= self.maximum_value:
            raise ValueError("maximum_default is outside the form range")
        if self.primary_action == self.same_day_action:
            raise ValueError("study-session actions must be distinct")


@dataclass(frozen=True)
class StudySessionDraft:
    search_filter: str
    minimum: str
    maximum: str
    order_index: str
    desired_retentions: tuple[str, ...]
    same_day_desired_retentions: tuple[str, ...]
    override_enabled: bool
    override_value: str
    same_day_override_enabled: bool
    same_day_override_value: str
    adaptive_enabled: bool = False
    adaptive_flat: str = "0"
    adaptive_s_multi: str = "0"
    adaptive_d_multi: str = "0"


@dataclass(frozen=True)
class StudySessionRequest:
    retentions: tuple[DeckRetention, ...]
    minimum: int
    maximum: int
    order_index: int
    same_day_only: bool
    extra_search: str
    adaptive_retention_settings: AdaptiveRetentionSettings | None

    def saved_settings(self, spec: StudySessionFormSpec) -> dict[str, Any]:
        saved: dict[str, Any] = {
            spec.maximum_storage_key: self.maximum,
            spec.minimum_storage_key: self.minimum,
            "order_index": self.order_index,
        }
        if self.extra_search:
            saved["search_filter"] = self.extra_search
        if self.adaptive_retention_settings is not None:
            adaptive = self.adaptive_retention_settings
            saved[ADAPTIVE_RETENTION_SETTINGS_KEY] = {
                "enabled": adaptive.enabled,
                "flat": adaptive.flat,
                "s_multi": adaptive.s_multi,
                "d_multi": adaptive.d_multi,
            }
        return saved


SubmitCallback = Callable[[StudySessionRequest], bool]
CheckCallback = Callable[[str], int]
VoidCallback = Callable[[], None]
WarningCallback = Callable[[str], None]


class StudySessionFormController(BaseWebDialogController):
    """Shared Qt-independent editor for Filtered Deck and Live Session setup."""

    def __init__(
        self,
        *,
        spec: StudySessionFormSpec,
        source_name: str,
        retentions: Sequence[DeckRetention],
        order_options: Sequence[tuple[int, str]],
        saved_settings: Mapping[str, Any] | None,
        adaptive_available: bool,
        on_submit_requested: SubmitCallback,
        on_check_requested: CheckCallback,
        on_restore_defaults: VoidCallback,
        on_warning: WarningCallback,
        background_submission: bool,
        is_dark: bool = False,
    ) -> None:
        if not callable(on_submit_requested):
            raise TypeError("on_submit_requested must be callable")
        if not callable(on_check_requested):
            raise TypeError("on_check_requested must be callable")
        if not callable(on_restore_defaults):
            raise TypeError("on_restore_defaults must be callable")
        if not callable(on_warning):
            raise TypeError("on_warning must be callable")

        self.spec = spec
        self.source_name = str(source_name)
        self.retentions = tuple(retentions)
        self.order_options = _normalize_order_options(order_options)
        self.adaptive_available = bool(adaptive_available)
        self.background_submission = bool(background_submission)
        self.is_dark = bool(is_dark)
        self.running = False
        self.filter_count: int | None = None
        self.focus_target: str | None = None
        self.retention_editor_expanded = False
        self.draft = _draft_from_saved(
            spec,
            self.retentions,
            self.order_options,
            saved_settings,
            adaptive_available=self.adaptive_available,
        )

        self.actions = frozenset(
            {
                UPDATE_FORM_ACTION,
                CHECK_SEARCH_ACTION,
                RESTORE_DEFAULTS_ACTION,
                TOGGLE_RETENTION_EDITOR_ACTION,
                spec.primary_action,
                spec.same_day_action,
            }
        )
        self._on_submit_requested = on_submit_requested
        self._on_check_requested = on_check_requested
        self._on_restore_defaults = on_restore_defaults
        self._on_warning = on_warning
        self._rerender: Callable[[], Any] | None = None

    def attach_rerender(self, rerender: Callable[[], Any]) -> None:
        if not callable(rerender):
            raise TypeError("study-session rerender callback must be callable")
        self._rerender = rerender

    def render_html(self, generation: int) -> str:
        return render_study_session_dialog_html(
            spec=self.spec,
            source_name=self.source_name,
            retentions=self.retentions,
            order_options=self.order_options,
            draft=self.draft,
            adaptive_available=self.adaptive_available,
            filter_count=self.filter_count,
            running=self.running,
            focus_target=self.focus_target,
            retention_editor_expanded=self.retention_editor_expanded,
            is_dark=self.is_dark,
            generation=generation,
        )

    def handle_command(self, command: WebDialogCommand) -> dict[str, bool]:
        if command.action == UPDATE_FORM_ACTION:
            self._apply_form_payload(command.payload)
            self.focus_target = None
            return {"updated": True}

        if command.action == CHECK_SEARCH_ACTION:
            self._apply_form_payload(command.payload)
            self._check_search()
            return {"checked": self.filter_count is not None}

        if command.action == RESTORE_DEFAULTS_ACTION:
            self.restore_defaults()
            return {"restored": True}

        if command.action == TOGGLE_RETENTION_EDITOR_ACTION:
            self._set_retention_editor_expanded(command.payload)
            return {"expanded": self.retention_editor_expanded}

        if command.action in {self.spec.primary_action, self.spec.same_day_action}:
            self._apply_form_payload(command.payload)
            return {
                "started": self.request_submission(
                    same_day_only=command.action == self.spec.same_day_action
                )
            }

        raise BridgePayloadError(f"Unhandled study-session action: {command.action}")

    def is_action_enabled(self, action: str) -> bool:
        return action in self.actions and not self.running

    def can_close(self, reason: CloseReason) -> bool:
        del reason
        return not self.running

    def request_submission(self, *, same_day_only: bool) -> bool:
        if self.running:
            return False
        try:
            request = self._request_from_draft(same_day_only=same_day_only)
        except ValueError as exc:
            self._on_warning(str(exc))
            self.focus_target = "retention_table"
            self._request_rerender()
            return False

        self.draft = replace(
            self.draft,
            minimum=str(request.minimum),
            maximum=str(request.maximum),
            order_index=str(request.order_index),
        )
        return self._start_request(request)

    def retry_submission(self, request: StudySessionRequest) -> bool:
        """Restart a previously validated request after checkpoint recovery."""

        if self.running:
            return False
        return self._start_request(request)

    def restore_defaults(self) -> None:
        if self.running:
            return
        self._on_restore_defaults()
        self.draft = _default_draft(
            self.spec,
            self.retentions,
            self.order_options,
        )
        self.filter_count = None
        self.focus_target = "search_filter"
        self.retention_editor_expanded = False
        self._request_rerender()

    def finish_submission(self, *, rerender: bool = True) -> None:
        if not self.running:
            return
        self.running = False
        if rerender:
            self._request_rerender()

    def clear_filter_count(self, *, rerender: bool = True) -> None:
        self.filter_count = None
        if rerender:
            self._request_rerender()

    def current_request(self, *, same_day_only: bool = False) -> StudySessionRequest:
        return self._request_from_draft(same_day_only=same_day_only)

    def _start_request(self, request: StudySessionRequest) -> bool:
        if self.background_submission:
            self.running = True
            self.focus_target = None
            self._request_rerender()
        started = bool(self._on_submit_requested(request))
        if not started:
            if self.background_submission:
                self.running = False
            self._request_rerender()
        return started

    def _check_search(self) -> None:
        try:
            self.filter_count = max(
                0,
                int(self._on_check_requested(self.draft.search_filter.strip())),
            )
        except Exception as exc:
            self.filter_count = None
            self._on_warning(str(exc))
        self.focus_target = "search_filter"
        self._request_rerender()

    def _request_from_draft(self, *, same_day_only: bool) -> StudySessionRequest:
        minimum = _form_int(
            self.draft.minimum,
            label=self.spec.minimum_label,
            minimum=0,
            maximum=self.spec.maximum_value,
        )
        maximum = _form_int(
            self.draft.maximum,
            label=self.spec.maximum_label,
            minimum=1,
            maximum=self.spec.maximum_value,
        )
        maximum = max(1, minimum, maximum)
        order_index = _valid_order_index(
            self.draft.order_index,
            self.order_options,
            default=self.spec.default_order_index,
        )
        desired_values = (
            tuple(self.draft.override_value for _retention in self.retentions)
            if self.draft.override_enabled
            else self.draft.desired_retentions
        )
        same_day_values = (
            tuple(self.draft.same_day_override_value for _retention in self.retentions)
            if self.draft.same_day_override_enabled
            else self.draft.same_day_desired_retentions
        )
        desired_fractions = tuple(_retention_fraction(value) for value in desired_values)
        same_day_fractions = tuple(_retention_fraction(value) for value in same_day_values)
        retentions = tuple(
            retentions_with_desired_values(
                self.retentions,
                desired_fractions,
                same_day_fractions,
            )
        )
        adaptive = None
        if self.adaptive_available:
            adaptive = AdaptiveRetentionSettings(
                enabled=self.draft.adaptive_enabled,
                flat=_bounded_float(self.draft.adaptive_flat, minimum=-10.0, maximum=10.0),
                s_multi=_bounded_float(
                    self.draft.adaptive_s_multi,
                    minimum=-10.0,
                    maximum=10.0,
                ),
                d_multi=_bounded_float(
                    self.draft.adaptive_d_multi,
                    minimum=-10.0,
                    maximum=10.0,
                ),
            )
        return StudySessionRequest(
            retentions=retentions,
            minimum=minimum,
            maximum=maximum,
            order_index=order_index,
            same_day_only=bool(same_day_only),
            extra_search=self.draft.search_filter.strip(),
            adaptive_retention_settings=adaptive,
        )

    def _apply_form_payload(self, payload: Mapping[str, Any]) -> None:
        expected = _form_keys(len(self.retentions), self.adaptive_available)
        actual = set(payload)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            details = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if extra:
                details.append(f"unsupported {', '.join(extra)}")
            raise BridgePayloadError(
                "Study-session form fields are invalid"
                + (f": {'; '.join(details)}" if details else ".")
            )

        string_keys = expected - _checkbox_keys(self.adaptive_available)
        for key in string_keys:
            if not isinstance(payload[key], str):
                raise BridgePayloadError(f"{key} must be text.")
        for key in _checkbox_keys(self.adaptive_available):
            if not isinstance(payload[key], bool):
                raise BridgePayloadError(f"{key} must be true or false.")

        self.draft = StudySessionDraft(
            search_filter=payload["search_filter"],
            minimum=payload["minimum"],
            maximum=payload["maximum"],
            order_index=payload["order_index"],
            desired_retentions=tuple(
                payload[f"desired_retention[{index}]"] for index in range(len(self.retentions))
            ),
            same_day_desired_retentions=tuple(
                payload[f"same_day_desired_retention[{index}]"]
                for index in range(len(self.retentions))
            ),
            override_enabled=payload["override_enabled"],
            override_value=payload["override_value"],
            same_day_override_enabled=payload["same_day_override_enabled"],
            same_day_override_value=payload["same_day_override_value"],
            adaptive_enabled=(payload["adaptive_enabled"] if self.adaptive_available else False),
            adaptive_flat=(payload["adaptive_flat"] if self.adaptive_available else "0"),
            adaptive_s_multi=(payload["adaptive_s_multi"] if self.adaptive_available else "0"),
            adaptive_d_multi=(payload["adaptive_d_multi"] if self.adaptive_available else "0"),
        )

    def _set_retention_editor_expanded(self, payload: Mapping[str, Any]) -> None:
        if set(payload) != {"expanded"} or not isinstance(payload["expanded"], bool):
            raise BridgePayloadError("Retention-editor expanded state must be true or false.")
        self.retention_editor_expanded = payload["expanded"]

    def _request_rerender(self) -> None:
        if self._rerender is not None:
            self._rerender()


def _draft_from_saved(
    spec: StudySessionFormSpec,
    retentions: tuple[DeckRetention, ...],
    order_options: tuple[tuple[int, str], ...],
    saved_settings: Mapping[str, Any] | None,
    *,
    adaptive_available: bool,
) -> StudySessionDraft:
    draft = _default_draft(spec, retentions, order_options)
    saved = saved_settings if isinstance(saved_settings, Mapping) else {}
    adaptive = (
        _adaptive_from_saved(saved.get(ADAPTIVE_RETENTION_SETTINGS_KEY))
        if adaptive_available
        else AdaptiveRetentionSettings()
    )
    return replace(
        draft,
        search_filter=(str(saved["search_filter"]) if "search_filter" in saved else ""),
        minimum=str(
            _bounded_int(
                saved.get(spec.minimum_storage_key),
                default=spec.minimum_default,
                minimum=0,
                maximum=spec.maximum_value,
            )
        ),
        maximum=str(
            _bounded_int(
                saved.get(spec.maximum_storage_key),
                default=spec.maximum_default,
                minimum=1,
                maximum=spec.maximum_value,
            )
        ),
        order_index=str(
            _valid_order_index(
                saved.get("order_index"),
                order_options,
                default=spec.default_order_index,
            )
        ),
        adaptive_enabled=adaptive.enabled,
        adaptive_flat=_number_text(adaptive.flat),
        adaptive_s_multi=_number_text(adaptive.s_multi),
        adaptive_d_multi=_number_text(adaptive.d_multi),
    )


def _default_draft(
    spec: StudySessionFormSpec,
    retentions: tuple[DeckRetention, ...],
    order_options: tuple[tuple[int, str], ...],
) -> StudySessionDraft:
    desired = tuple(_retention_percent_text(item.desired_retention) for item in retentions)
    same_day = tuple(_retention_percent_text(_same_day_retention(item)) for item in retentions)
    default_desired = desired[0] if desired else "90"
    default_same_day = same_day[0] if same_day else "90"
    return StudySessionDraft(
        search_filter="",
        minimum=str(spec.minimum_default),
        maximum=str(spec.maximum_default),
        order_index=str(
            _valid_order_index(
                spec.default_order_index,
                order_options,
                default=spec.default_order_index,
            )
        ),
        desired_retentions=desired,
        same_day_desired_retentions=same_day,
        override_enabled=False,
        override_value=default_desired,
        same_day_override_enabled=False,
        same_day_override_value=default_same_day,
    )


def _adaptive_from_saved(value: Any) -> AdaptiveRetentionSettings:
    saved = value if isinstance(value, Mapping) else {}
    return AdaptiveRetentionSettings(
        enabled=bool(saved.get("enabled", False)),
        flat=_saved_float(saved.get("flat")),
        s_multi=_saved_float(saved.get("s_multi")),
        d_multi=_saved_float(saved.get("d_multi")),
    )


def _same_day_retention(retention: DeckRetention) -> float:
    value = retention.same_day_desired_retention
    return float(retention.desired_retention if value is None else value)


def _retention_percent_text(value: float) -> str:
    percentage = min(99.0, max(1.0, float(value) * 100.0))
    return _number_text(round(percentage, 6))


def _retention_fraction(value: str) -> str:
    try:
        percentage = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Desired retention must be a percentage between 1 and 99.") from exc
    if not math.isfinite(percentage):
        raise ValueError("Desired retention must be a finite percentage.")
    clamped = min(99.0, max(1.0, percentage))
    return _number_text(clamped / 100.0)


def _saved_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return min(10.0, max(-10.0, parsed))


def _bounded_float(value: str, *, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Adaptive DR parameters must be numbers.") from exc
    if not math.isfinite(parsed):
        raise ValueError("Adaptive DR parameters must be finite numbers.")
    return min(maximum, max(minimum, parsed))


def _bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return min(int(maximum), max(int(minimum), parsed))


def _form_int(
    value: str,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a whole number.") from exc
    return min(int(maximum), max(int(minimum), parsed))


def _valid_order_index(
    value: Any,
    order_options: Sequence[tuple[int, str]],
    *,
    default: int,
) -> int:
    allowed = {int(option_value) for option_value, _label in order_options}
    if not allowed:
        raise ValueError("At least one sort order must be available.")
    resolved_default = int(default)
    if resolved_default not in allowed:
        resolved_default = int(order_options[0][0])
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return resolved_default
    return parsed if parsed in allowed else resolved_default


def _normalize_order_options(
    order_options: Sequence[tuple[int, str]],
) -> tuple[tuple[int, str], ...]:
    normalized = tuple((int(value), str(label)) for value, label in order_options)
    if not normalized:
        raise ValueError("At least one sort order must be available.")
    values = [value for value, _label in normalized]
    if len(values) != len(set(values)):
        raise ValueError("Sort-order values must be unique.")
    if any(not label.strip() for _value, label in normalized):
        raise ValueError("Sort-order labels must not be empty.")
    return normalized


def _form_keys(retention_count: int, adaptive_available: bool) -> set[str]:
    keys = {
        "search_filter",
        "minimum",
        "maximum",
        "order_index",
        "override_enabled",
        "override_value",
        "same_day_override_enabled",
        "same_day_override_value",
    }
    for index in range(retention_count):
        keys.add(f"desired_retention[{index}]")
        keys.add(f"same_day_desired_retention[{index}]")
    if adaptive_available:
        keys.update(
            {
                "adaptive_enabled",
                "adaptive_flat",
                "adaptive_s_multi",
                "adaptive_d_multi",
            }
        )
    return keys


def _checkbox_keys(adaptive_available: bool) -> set[str]:
    keys = {"override_enabled", "same_day_override_enabled"}
    if adaptive_available:
        keys.add("adaptive_enabled")
    return keys


def _number_text(value: float) -> str:
    return f"{float(value):g}"
