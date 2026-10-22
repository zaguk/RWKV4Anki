from __future__ import annotations

from pathlib import Path, PurePath

from .constants import DEFAULT_MODEL_ID, V5O_MODEL_ID, VENDOR_ROOT
from .model_generation import (
    MODEL_GENERATIONS,
    V1_MODEL_GENERATION,
    V5O_MODEL_GENERATION,
    normalize_model_generation,
)

DEFAULT_BUNDLED_MODEL_GENERATION = V1_MODEL_GENERATION


def default_model_id_for_generation(model_generation: object) -> str:
    return {
        V1_MODEL_GENERATION: DEFAULT_MODEL_ID,
        V5O_MODEL_GENERATION: V5O_MODEL_ID,
    }[normalize_model_generation(model_generation)]


def model_storage_subdirectory(model_generation: object) -> str:
    return {
        V1_MODEL_GENERATION: "v1",
        V5O_MODEL_GENERATION: "v5",
    }[normalize_model_generation(model_generation)]


def legacy_model_generation(model_id: object) -> str:
    """Classify only known non-V1 names in old flat model directories."""

    return {
        V5O_MODEL_ID: V5O_MODEL_GENERATION,
        "rwkv-p-v5o": V5O_MODEL_GENERATION,
    }.get(str(model_id), V1_MODEL_GENERATION)


def available_model_ids(
    model_generation: object = DEFAULT_BUNDLED_MODEL_GENERATION,
) -> tuple[str, ...]:
    generation = normalize_model_generation(model_generation)
    pretrained_dir = VENDOR_ROOT / "rwkv_srs" / "pretrained"
    try:
        model_ids = tuple(
            sorted(
                path.stem
                for path in (
                    pretrained_dir / model_storage_subdirectory(generation)
                ).glob("*.safetensors")
                if path.is_file()
                and path.stem not in {"rwkv-p-v2o", "rwkv-p-v5o"}
            )
        )
        # Read old development bundles without perpetuating their flat layout.
        # New bundles use one directory for each generation.
        legacy_ids = tuple(
            sorted(
                path.stem
                for path in pretrained_dir.glob("*.safetensors")
                if path.is_file()
                and path.stem not in {"rwkv-p-v2o", "rwkv-p-v5o"}
                and legacy_model_generation(path.stem) == generation
            )
        )
        if not model_ids:
            model_ids = legacy_ids
    except OSError:
        model_ids = ()
    if model_ids:
        return model_ids
    return (default_model_id_for_generation(generation),)


def available_model_ids_by_generation() -> dict[str, tuple[str, ...]]:
    return {
        generation: available_model_ids(generation)
        for generation in MODEL_GENERATIONS
    }


def bundled_model_path(
    model_id: object,
    *,
    model_generation: object | None = None,
) -> Path | None:
    """Return a bundled safetensors path, including legacy flat bundles.

    ``None`` generation is useful for old call sites that only know a logical
    model id: a path is returned only when that id resolves unambiguously.
    """

    value = str(model_id or "").strip()
    if value in {"rwkv-p-v2o", "rwkv-p-v5o"}:
        return None
    if not value or value != PurePath(value).name:
        return None
    pretrained_dir = VENDOR_ROOT / "rwkv_srs" / "pretrained"
    generations = (
        tuple(MODEL_GENERATIONS)
        if model_generation is None
        else (normalize_model_generation(model_generation),)
    )
    candidates = [
        pretrained_dir
        / model_storage_subdirectory(generation)
        / f"{value}.safetensors"
        for generation in generations
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        return None
    legacy = pretrained_dir / f"{value}.safetensors"
    if legacy.is_file():
        if model_generation is None:
            return legacy
        inferred = legacy_model_generation(value)
        if inferred == normalize_model_generation(model_generation):
            return legacy
    return None


__all__ = [
    "DEFAULT_BUNDLED_MODEL_GENERATION",
    "available_model_ids",
    "available_model_ids_by_generation",
    "bundled_model_path",
    "default_model_id_for_generation",
    "legacy_model_generation",
    "model_storage_subdirectory",
]
