from __future__ import annotations

from .addon_config import (
    addon_config_for_mw,
    checkpoint_save_interval,
    configured_model_id,
    enabled_prediction_cache_specs,
    exclude_deleted_card_revlogs,
    predict_many_batch_size,
    predict_many_mode,
    process_many_mode,
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


def manager_for_mw(mw) -> RWKVCheckpointManager:
    global _manager, _manager_model_id, _manager_profile
    name = profile_name(mw)
    config = addon_config_for_mw(mw)
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
        or policy_changed
    ):
        if _manager is not None:
            if (
                _manager_profile == name
                and (_manager_model_id != model_id or policy_changed)
                and bool(getattr(_manager, "runtime_scope_active", False))
            ):
                setting = (
                    "the RWKV model"
                    if _manager_model_id != model_id
                    else "Filtered-review interpretation"
                )
                raise CheckpointBusyError(
                    f"Stop the active RWKV operation or Live Session before changing {setting}."
                )
            _manager.unload()
        _manager = RWKVCheckpointManager(
            store,
            model_id=model_id,
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
    global _manager, _manager_model_id, _manager_profile
    if _manager is not None:
        _manager.unload()
    _manager = None
    _manager_profile = None
    _manager_model_id = None
