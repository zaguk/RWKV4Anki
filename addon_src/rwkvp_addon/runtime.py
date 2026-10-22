from __future__ import annotations

from .addon_config import (
    addon_config_for_mw,
    checkpoint_save_interval,
    configured_model_generation,
    configured_model_id,
    enabled_prediction_cache_specs,
    exclude_deleted_card_revlogs,
    gpu_prediction_math,
    predict_many_batch_size,
    predict_many_mode,
    process_many_mode,
    state_precision,
)
from .anki_api import profile_name
from .checkpoint_manager import CheckpointBusyError, RWKVCheckpointManager
from .profile_store import ProfileStore
from .review_type_normalization import filtered_review_normalization_policy_for_store
from .rwkv_backend import configured_rwkv_backend
from .rwkv_performance_modes import (
    PREDICT_MANY_ORACLE_MODE,
)

_manager: RWKVCheckpointManager | None = None
_manager_profile: str | None = None
_manager_model_id: str | None = None
_manager_model_generation: str | None = None


def manager_for_mw(mw) -> RWKVCheckpointManager:
    global _manager, _manager_model_generation, _manager_model_id, _manager_profile
    name = profile_name(mw)
    config = addon_config_for_mw(mw)
    model_generation = configured_model_generation(config)
    model_id = configured_model_id(config)
    prediction_cache_specs = enabled_prediction_cache_specs(config)
    configured_predict_many_mode = predict_many_mode(config)
    configured_batch_size = predict_many_batch_size(
        config,
        (
            configured_predict_many_mode
            if configured_rwkv_backend() == "rust"
            else PREDICT_MANY_ORACLE_MODE
        ),
    )
    configured_process_many_mode = process_many_mode(config)
    configured_precision = state_precision(config)
    configured_math = gpu_prediction_math(config)
    math_changed = bool(
        _manager is not None
        and getattr(_manager, "gpu_prediction_math", "fp32") != configured_math
    )
    precision_changed = bool(
        _manager is not None
        and getattr(_manager, "state_precision", "fp32") != configured_precision
    )
    configured_save_interval = checkpoint_save_interval(config)
    configured_exclude_deleted = exclude_deleted_card_revlogs(config)
    store = (
        _manager.store
        if _manager is not None and _manager_profile == name
        else ProfileStore.for_profile(name)
    )
    normalization_policy = filtered_review_normalization_policy_for_store(store)
    policy_changed = bool(
        _manager is not None
        and _manager_profile == name
        and getattr(_manager, "filtered_review_normalization_policy", None) != normalization_policy
    )
    if (
        _manager is None
        or _manager_profile != name
        or _manager_model_id != model_id
        or _manager_model_generation != model_generation
        or policy_changed
        or precision_changed
        or math_changed
    ):
        if _manager is not None:
            if (
                _manager_profile == name
                and (
                    _manager_model_id != model_id
                    or _manager_model_generation != model_generation
                    or policy_changed
                    or precision_changed
                    or math_changed
                )
                and bool(getattr(_manager, "runtime_scope_active", False))
            ):
                setting = (
                    "the RWKV model"
                    if (
                        _manager_model_id != model_id
                        or _manager_model_generation != model_generation
                        or precision_changed
                        or math_changed
                    )
                    else "Filtered-review interpretation"
                )
                raise CheckpointBusyError(
                    f"Stop the active RWKV operation or Live Session before changing {setting}."
                )
            _manager.unload()
        _manager = RWKVCheckpointManager(
            store,
            model_id=model_id,
            model_generation=model_generation,
            state_precision=configured_precision,
            gpu_prediction_math=configured_math,
            prediction_cache_specs=prediction_cache_specs,
            predict_many_mode=configured_predict_many_mode,
            predict_many_batch_size=configured_batch_size,
            process_many_mode=configured_process_many_mode,
            checkpoint_save_interval=configured_save_interval,
            exclude_deleted_card_revlogs=configured_exclude_deleted,
            filtered_review_normalization_policy=normalization_policy,
        )
        if not hasattr(_manager, "filtered_review_normalization_policy"):
            _manager.filtered_review_normalization_policy = normalization_policy
        _manager_profile = name
        _manager_model_id = model_id
        _manager_model_generation = model_generation
    else:
        # Prediction mode and review-history GPU selection are per-call. Every
        # add-on-owned processing runtime uses the Fast CPU profile, so Fast and
        # GPU can switch without replacing the manager or restarting Anki.
        _manager.set_prediction_cache_specs(prediction_cache_specs)
        _manager.set_predict_many_mode(configured_predict_many_mode)
        _manager.set_predict_many_batch_size(configured_batch_size)
        setter = getattr(_manager, "set_process_many_mode", None)
        if setter is not None:
            setter(configured_process_many_mode)
        else:
            _manager.process_many_mode = configured_process_many_mode
        deleted_setter = getattr(_manager, "set_exclude_deleted_card_revlogs", None)
        if deleted_setter is not None:
            deleted_setter(configured_exclude_deleted)
        else:
            _manager.exclude_deleted_card_revlogs = configured_exclude_deleted
        _manager.set_checkpoint_save_interval(configured_save_interval)
    return _manager


def store_for_mw(mw) -> ProfileStore:
    return manager_for_mw(mw).store


def reset_runtime() -> None:
    global _manager, _manager_model_generation, _manager_model_id, _manager_profile
    if _manager is not None:
        _manager.unload()
    _manager = None
    _manager_profile = None
    _manager_model_id = None
    _manager_model_generation = None
