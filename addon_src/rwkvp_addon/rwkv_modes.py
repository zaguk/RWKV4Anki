from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .metrics import RWKVPredictionMode


class RetrievabilityMode(Enum):
    IMMEDIATE = "immediate"
    FORGETTING_CURVE = "forgetting_curve"

    @property
    def window_title(self) -> str:
        return mode_spec(self).retrievability_window_title

    @property
    def graph_title(self) -> str:
        return mode_spec(self).retrievability_graph_title

    @property
    def warning_title(self) -> str:
        return mode_spec(self).warning_title


@dataclass(frozen=True)
class RWKVModeSpec:
    mode: RetrievabilityMode
    deck_menu_title: str
    evaluate_title: str
    evaluate_label: str
    retrievability_window_title: str
    retrievability_graph_title: str
    warning_title: str
    filtered_deck_title: str
    generated_deck_prefix_label: str
    legacy_generated_deck_prefix_labels: tuple[str, ...] = ()

    def generated_deck_name_prefix(self, source_deck_name: str) -> str:
        return f"{self.generated_deck_prefix_label} - {top_level_deck_label(source_deck_name)} - "

    def generated_deck_name_prefixes_for_collision(
        self,
        source_deck_name: str,
    ) -> tuple[str, ...]:
        deck_label = top_level_deck_label(source_deck_name)
        labels = (
            self.generated_deck_prefix_label,
            *self.legacy_generated_deck_prefix_labels,
        )
        return tuple(f"{label} - {deck_label} - " for label in labels)


RETRIEVABILITY_MODES = (
    RetrievabilityMode.IMMEDIATE,
    RetrievabilityMode.FORGETTING_CURVE,
)

_MODE_SPECS = {
    RetrievabilityMode.IMMEDIATE: RWKVModeSpec(
        mode=RetrievabilityMode.IMMEDIATE,
        deck_menu_title="RWKV (Immediate)",
        evaluate_title="RWKV Immediate Evaluation",
        evaluate_label="RWKV Immediate",
        retrievability_window_title="RWKV Immediate Average Retrievability",
        retrievability_graph_title="RWKV Immediate Retrievability",
        warning_title="RWKV Immediate",
        filtered_deck_title="Generate RWKV Immediate Filtered Deck",
        generated_deck_prefix_label="- RWKV Immediate",
        legacy_generated_deck_prefix_labels=("- RWKV-P",),
    ),
    RetrievabilityMode.FORGETTING_CURVE: RWKVModeSpec(
        mode=RetrievabilityMode.FORGETTING_CURVE,
        deck_menu_title="RWKV (Forgetting Curve)",
        evaluate_title="RWKV Forgetting Curve Evaluation",
        evaluate_label="RWKV Forgetting Curve",
        retrievability_window_title="RWKV Forgetting Curve Average Retrievability",
        retrievability_graph_title="RWKV Forgetting Curve Retrievability",
        warning_title="RWKV Forgetting Curve",
        filtered_deck_title="Generate RWKV Forgetting Curve Filtered Deck",
        generated_deck_prefix_label="- RWKV Forgetting Curve",
    ),
}


def mode_spec(mode: RetrievabilityMode) -> RWKVModeSpec:
    return _MODE_SPECS[mode]


def top_level_deck_label(source_deck_name: str) -> str:
    return " - ".join(part.strip() for part in source_deck_name.split("::") if part.strip())


def generated_deck_name_prefix_for_mode(
    mode: RetrievabilityMode,
    source_deck_name: str,
) -> str:
    return mode_spec(mode).generated_deck_name_prefix(source_deck_name)


def generated_deck_name_prefixes_for_mode_collision(
    mode: RetrievabilityMode,
    source_deck_name: str,
) -> tuple[str, ...]:
    return mode_spec(mode).generated_deck_name_prefixes_for_collision(source_deck_name)


def filtered_deck_settings_key(mode: RetrievabilityMode, deck_id: int) -> str:
    return f"{mode.value}:{int(deck_id)}"


def enabled_prediction_modes_for_retrievability_mode(
    mode: RetrievabilityMode,
    config: Mapping[str, Any] | None,
) -> tuple[RWKVPredictionMode, ...]:
    from .addon_config import (
        enabled_forgetting_curve_rwkv_prediction_modes,
        enabled_immediate_rwkv_prediction_modes,
    )

    if mode == RetrievabilityMode.FORGETTING_CURVE:
        return enabled_forgetting_curve_rwkv_prediction_modes(config)
    return enabled_immediate_rwkv_prediction_modes(config)
