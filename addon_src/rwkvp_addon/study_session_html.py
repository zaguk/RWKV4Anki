"""Serializable study-form presentation; Svelte owns the visible controls."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING

from .anki_api import DeckRetention
from .search_filters import candidate_filter_count_label
from .svelte_ui import render_svelte_document

if TYPE_CHECKING:
    from .study_session_controller import StudySessionDraft, StudySessionFormSpec


def render_study_session_dialog_html(
    *,
    spec: StudySessionFormSpec,
    source_name: str,
    retentions: Sequence[DeckRetention],
    order_options: Sequence[tuple[int, str]],
    draft: StudySessionDraft,
    adaptive_available: bool,
    filter_count: int | None,
    running: bool = False,
    focus_target: str | None = None,
    retention_editor_expanded: bool = False,
    is_dark: bool = False,
    generation: int = 1,
) -> str:
    root_classes = "rwkv-study-session-dialog"
    if retention_editor_expanded:
        root_classes += " rwkv-retention-editor-expanded"
    return render_svelte_document(
        "study-session",
        {
            "spec": asdict(spec),
            "source_name": source_name,
            "retentions": [{"name": retention.name} for retention in retentions],
            "order_options": list(order_options),
            "draft": asdict(draft),
            "adaptive_available": adaptive_available,
            "count_label": (
                candidate_filter_count_label(filter_count) if filter_count is not None else None
            ),
            "running": running,
            "focus_target": focus_target,
            "retention_editor_expanded": retention_editor_expanded,
        },
        title=spec.title,
        intro=spec.intro,
        generation=generation,
        is_dark=is_dark,
        root_extra_classes=root_classes,
    )
