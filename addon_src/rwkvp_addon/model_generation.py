from __future__ import annotations

from dataclasses import dataclass
from typing import Any

V1_MODEL_GENERATION = "v1"
V5O_MODEL_GENERATION = "v5o"
DEFAULT_MODEL_GENERATION = V1_MODEL_GENERATION
MODEL_GENERATIONS = (V1_MODEL_GENERATION, V5O_MODEL_GENERATION)


@dataclass(frozen=True)
class ModelGenerationCapabilities:
    label: str
    curves: bool = False
    deleted_card_history: bool = True


# One compatibility policy for settings and cross-generation evaluation. Native
# row formats remain generation-specific even when the evaluation settings match.
MODEL_GENERATION_CAPABILITIES = {
    V1_MODEL_GENERATION: ModelGenerationCapabilities("V1", curves=True),
    V5O_MODEL_GENERATION: ModelGenerationCapabilities("V5"),
}


def model_generation_capabilities(value: Any) -> ModelGenerationCapabilities:
    return MODEL_GENERATION_CAPABILITIES[normalize_model_generation(value)]


def model_generation_evaluation_incompatibility(
    value: str,
    *,
    prediction_mode: str,
    include_deleted_reviews: bool,
) -> str | None:
    """Explain why a generation cannot use the active training/evaluation policy."""
    if value not in MODEL_GENERATIONS:
        return "Unsupported RWKV generation."
    capabilities = model_generation_capabilities(value)
    if prediction_mode not in {"per_review", "predict_ahead"}:
        return "Unsupported RWKV evaluation mode."
    if prediction_mode == "predict_ahead" and not capabilities.curves:
        return f"{capabilities.label} does not support Curve evaluation."
    if include_deleted_reviews and not capabilities.deleted_card_history:
        return f"{capabilities.label} cannot include deleted-card history."
    return None


def compatible_evaluation_generations(
    active_generation: str,
    *,
    prediction_mode: str,
    include_deleted_reviews: bool,
) -> tuple[str, ...]:
    settings = dict(
        prediction_mode=prediction_mode,
        include_deleted_reviews=include_deleted_reviews,
    )
    if model_generation_evaluation_incompatibility(active_generation, **settings):
        return ()
    return tuple(
        generation
        for generation in MODEL_GENERATIONS
        if generation != active_generation
        and model_generation_evaluation_incompatibility(generation, **settings) is None
    )


def normalize_model_generation(value: Any) -> str:
    """Return the canonical internal RWKV model generation.

    ``v5`` is accepted as hand-edited-config aliases; persisted
    configuration uses RWKV-SRS's public ``v5o`` name.
    Unknown values retain the safe, release-compatible V1 default.
    """

    normalized = str(value or "").strip().casefold()
    if normalized in {"v5", V5O_MODEL_GENERATION}:
        return V5O_MODEL_GENERATION
    return V1_MODEL_GENERATION


def model_generation_uses_v5o(value: Any) -> bool:
    return normalize_model_generation(value) == V5O_MODEL_GENERATION


def model_generation_uses_exact_timestamps(value: Any) -> bool:
    return normalize_model_generation(value) == V5O_MODEL_GENERATION


def model_generation_supports_curves(value: Any) -> bool:
    return model_generation_capabilities(value).curves


def model_generation_label(value: Any) -> str:
    return model_generation_capabilities(value).label


__all__ = [
    "DEFAULT_MODEL_GENERATION",
    "MODEL_GENERATIONS",
    "MODEL_GENERATION_CAPABILITIES",
    "ModelGenerationCapabilities",
    "V1_MODEL_GENERATION",
    "V5O_MODEL_GENERATION",
    "model_generation_label",
    "model_generation_capabilities",
    "model_generation_evaluation_incompatibility",
    "compatible_evaluation_generations",
    "model_generation_supports_curves",
    "model_generation_uses_exact_timestamps",
    "model_generation_uses_v5o",
    "normalize_model_generation",
]
