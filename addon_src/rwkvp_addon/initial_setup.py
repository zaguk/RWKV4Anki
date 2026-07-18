from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .anki_api import profile_name
from .profile_store import ProfileStore

INITIAL_SETUP_SETTINGS_KEY = "guided_initial_setup"
INITIAL_SETUP_RECORD_VERSION = 1


def initial_setup_seen(store: ProfileStore) -> bool:
    """Return whether this profile has dismissed or completed first-run setup."""

    record = store.settings().get(INITIAL_SETUP_SETTINGS_KEY)
    return bool(isinstance(record, Mapping) and record.get("seen"))


def mark_initial_setup_seen(store: ProfileStore) -> None:
    """Durably suppress the automatic wizard without disturbing other profile data."""

    settings: dict[str, Any] = store.settings()
    settings[INITIAL_SETUP_SETTINGS_KEY] = {
        "seen": True,
        "record_version": INITIAL_SETUP_RECORD_VERSION,
    }
    store.write_settings(settings)


def initial_setup_seen_for_mw(mw) -> bool:
    return initial_setup_seen(ProfileStore.for_profile(profile_name(mw)))


def mark_initial_setup_seen_for_mw(mw) -> None:
    mark_initial_setup_seen(ProfileStore.for_profile(profile_name(mw)))
