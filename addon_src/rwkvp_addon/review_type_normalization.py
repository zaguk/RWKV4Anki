from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .profile_store import ProfileStore

FILTERED_REVIEW_NORMALIZATION_SETTINGS_KEY = "filtered_review_type_normalization"
FILTERED_REVIEW_NORMALIZATION_MANIFEST_KEY = "filtered_review_type_normalization"
FILTERED_REVIEW_NORMALIZATION_ENABLED_CONFIG_KEY = "normalize_filtered_review_types"
FILTERED_REVIEW_NORMALIZATION_CUTOFF_CONFIG_KEY = "normalize_filtered_reviews_from"

FILTERED_REVIEW_NORMALIZATION_VERSION = 1

RWKV_STATE_NEW = 0
RWKV_STATE_LEARNING = 1
RWKV_STATE_REVIEW = 2
RWKV_STATE_RELEARNING = 3
RWKV_STATE_FILTERED = 4

_PROFILE_SETTINGS_LOCK = threading.RLock()


@dataclass(frozen=True)
class FilteredReviewNormalizationPolicy:
    """Profile-specific rules for interpreting Filtered rows sent to RWKV.

    ``cutoff_review_id`` is an Anki revlog timestamp in milliseconds. Anki's
    stored row is never changed; the policy controls only the derived ``state``
    value supplied to RWKV.
    """

    enabled: bool
    cutoff_review_id: int

    @classmethod
    def disabled(cls) -> FilteredReviewNormalizationPolicy:
        return cls(enabled=False, cutoff_review_id=0)

    def applies_to(self, review_id: int) -> bool:
        return self.enabled and int(review_id) >= int(self.cutoff_review_id)

    def semantic_signature(self) -> dict[str, Any]:
        signature: dict[str, Any] = {
            "version": FILTERED_REVIEW_NORMALIZATION_VERSION,
            "enabled": bool(self.enabled),
        }
        if self.enabled:
            signature["cutoff_review_id"] = int(self.cutoff_review_id)
        return signature


def filtered_review_normalization_policy_for_store(
    store: ProfileStore,
    *,
    now_millis: int | None = None,
) -> FilteredReviewNormalizationPolicy:
    """Read or initialize the policy for one Anki profile.

    Missing records deliberately start at the current instant. This lets an
    existing checkpoint retain its historical Filtered rows while new reviews
    can use semantic interpretation immediately.
    """

    with _PROFILE_SETTINGS_LOCK:
        read_settings = getattr(store, "settings", None)
        write_settings = getattr(store, "write_settings", None)
        if not callable(read_settings) or not callable(write_settings):
            return FilteredReviewNormalizationPolicy.disabled()
        settings = read_settings()
        record = settings.get(FILTERED_REVIEW_NORMALIZATION_SETTINGS_KEY)
        policy = _policy_from_record(record)
        if policy is not None:
            return policy

        cutoff = int(now_millis) if now_millis is not None else int(time.time_ns() // 1_000_000)
        policy = FilteredReviewNormalizationPolicy(
            enabled=True,
            cutoff_review_id=max(0, cutoff),
        )
        settings[FILTERED_REVIEW_NORMALIZATION_SETTINGS_KEY] = _policy_record(policy)
        write_settings(settings)
        return policy


def write_filtered_review_normalization_policy(
    store: ProfileStore,
    policy: FilteredReviewNormalizationPolicy,
) -> None:
    with _PROFILE_SETTINGS_LOCK:
        settings = store.settings()
        settings[FILTERED_REVIEW_NORMALIZATION_SETTINGS_KEY] = _policy_record(policy)
        store.write_settings(settings)


def profile_config_values_for_filtered_review_policy(
    policy: FilteredReviewNormalizationPolicy,
) -> dict[str, Any]:
    return {
        FILTERED_REVIEW_NORMALIZATION_ENABLED_CONFIG_KEY: bool(policy.enabled),
        FILTERED_REVIEW_NORMALIZATION_CUTOFF_CONFIG_KEY: format_cutoff_datetime(
            policy.cutoff_review_id
        ),
    }


def filtered_review_policy_from_config_values(
    values: Mapping[str, Any],
    *,
    fallback: FilteredReviewNormalizationPolicy,
) -> FilteredReviewNormalizationPolicy:
    enabled = _bool_value(
        values.get(
            FILTERED_REVIEW_NORMALIZATION_ENABLED_CONFIG_KEY,
            fallback.enabled,
        )
    )
    cutoff_value = values.get(FILTERED_REVIEW_NORMALIZATION_CUTOFF_CONFIG_KEY)
    displayed_fallback = format_cutoff_datetime(fallback.cutoff_review_id)
    if cutoff_value is None or str(cutoff_value).strip() == displayed_fallback:
        # The custom local date/time editor is intentionally displayed at
        # second precision.
        # Preserve the exact first-open millisecond unless the user actually
        # edits the field, or an unrelated Settings save would appear to move
        # the cutoff and needlessly invalidate the checkpoint.
        cutoff = fallback.cutoff_review_id
    else:
        cutoff = parse_cutoff_datetime(cutoff_value)
    return FilteredReviewNormalizationPolicy(
        enabled=enabled,
        cutoff_review_id=cutoff,
    )


def strip_filtered_review_profile_config_values(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    stripped = dict(config)
    stripped.pop(FILTERED_REVIEW_NORMALIZATION_ENABLED_CONFIG_KEY, None)
    stripped.pop(FILTERED_REVIEW_NORMALIZATION_CUTOFF_CONFIG_KEY, None)
    return stripped


def format_cutoff_datetime(cutoff_review_id: int) -> str:
    local = datetime.fromtimestamp(max(0, int(cutoff_review_id)) / 1000).astimezone()
    return local.strftime("%Y-%m-%dT%H:%M:%S")


def parse_cutoff_datetime(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Filtered-review start must be a date and time.")
    if isinstance(value, (int, float)):
        parsed = int(value)
        if parsed < 0:
            raise ValueError("Filtered-review start cannot be before 1970.")
        return parsed

    text = str(value).strip()
    if not text:
        raise ValueError("Choose when RWKV should begin interpreting Filtered reviews.")
    try:
        parsed_datetime = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("Filtered-review start must be a valid local date and time.") from exc
    if parsed_datetime.tzinfo is None:
        # ``astimezone()`` resolves the operating system's offset for this
        # particular local date, including daylight-saving transitions. A
        # tzinfo copied from ``now`` can be an hour wrong in the other season.
        parsed_datetime = parsed_datetime.astimezone()
    milliseconds = int(parsed_datetime.timestamp() * 1000)
    if milliseconds < 0:
        raise ValueError("Filtered-review start cannot be before 1970.")
    return milliseconds


def resolve_rwkv_review_state(
    *,
    benchmark_state: int,
    is_filtered: bool,
    elapsed_days: float,
    rating: int,
    previous_phase: int | None,
    normalize_filtered: bool,
) -> tuple[int, int]:
    """Return the emitted RWKV state and the next same-day semantic phase.

    The phase is derived for every Filtered row, including rows before the
    configured cutoff. That provides look-behind context without rewriting the
    older row's emitted state. A Filtered chain therefore needs no recursive
    database reads: its already-resolved phase is carried forward per card.
    """

    raw_state = int(benchmark_state)
    if not is_filtered:
        semantic_state = raw_state
    elif float(elapsed_days) > 0:
        semantic_state = RWKV_STATE_REVIEW
    elif float(elapsed_days) == 0 and previous_phase in {
        RWKV_STATE_LEARNING,
        RWKV_STATE_RELEARNING,
        RWKV_STATE_FILTERED,
    }:
        semantic_state = int(previous_phase)
    else:
        semantic_state = RWKV_STATE_FILTERED

    if semantic_state in {RWKV_STATE_NEW, RWKV_STATE_LEARNING}:
        next_phase = RWKV_STATE_LEARNING
    elif semantic_state == RWKV_STATE_REVIEW:
        next_phase = RWKV_STATE_RELEARNING if int(rating) == 1 else RWKV_STATE_FILTERED
    elif semantic_state == RWKV_STATE_RELEARNING:
        next_phase = RWKV_STATE_RELEARNING
    else:
        next_phase = RWKV_STATE_FILTERED

    emitted_state = semantic_state if is_filtered and normalize_filtered else raw_state
    return emitted_state, next_phase


def checkpoint_policy_matches(
    manifest: Mapping[str, Any],
    policy: FilteredReviewNormalizationPolicy,
) -> bool:
    """Return whether a durable checkpoint can use the current policy.

    Legacy manifests predate normalization metadata. They remain compatible
    when normalization is disabled or when every processed review predates a
    newly initialized cutoff.
    """

    stored = manifest.get(FILTERED_REVIEW_NORMALIZATION_MANIFEST_KEY)
    if isinstance(stored, Mapping):
        return dict(stored) == policy.semantic_signature()
    if not policy.enabled:
        return True
    try:
        last_review_id = int(manifest.get("last_review_id"))
    except (TypeError, ValueError):
        return manifest.get("processed_review_count") in {None, 0, "0"}
    return last_review_id < int(policy.cutoff_review_id)


def _policy_from_record(value: Any) -> FilteredReviewNormalizationPolicy | None:
    if not isinstance(value, Mapping):
        return None
    try:
        cutoff = int(value["cutoff_review_id"])
    except (KeyError, TypeError, ValueError):
        return None
    if cutoff < 0:
        return None
    return FilteredReviewNormalizationPolicy(
        enabled=_bool_value(value.get("enabled", True)),
        cutoff_review_id=cutoff,
    )


def _policy_record(policy: FilteredReviewNormalizationPolicy) -> dict[str, Any]:
    return {
        "record_version": FILTERED_REVIEW_NORMALIZATION_VERSION,
        "enabled": bool(policy.enabled),
        "cutoff_review_id": int(policy.cutoff_review_id),
    }


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
