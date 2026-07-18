from __future__ import annotations

from pathlib import Path
from typing import Any

from .behavior_lab import (
    BEHAVIOR_LAB_SCHEMA_VERSION,
    BehaviorLabExperiment,
    BehaviorLabValidationError,
)
from .profile_store import atomic_write_json, read_json

_STORE_SCHEMA_VERSION = 1


class BehaviorLabExperimentStore:
    """Profile-local, JSON-only storage for reproducible experiment definitions."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._load_payload(), key=str.casefold))

    def experiments(self) -> dict[str, BehaviorLabExperiment]:
        return {
            name: BehaviorLabExperiment.from_dict(value)
            for name, value in self._load_payload().items()
        }

    def get(self, name: str) -> BehaviorLabExperiment | None:
        normalized = str(name).strip()
        if not normalized:
            return None
        value = self._load_payload().get(normalized)
        return None if value is None else BehaviorLabExperiment.from_dict(value)

    def save(self, name: str, experiment: BehaviorLabExperiment) -> None:
        normalized = str(name).strip()
        if not normalized:
            raise BehaviorLabValidationError("Enter a name before saving the experiment.")
        if len(normalized) > 160:
            raise BehaviorLabValidationError("Experiment names may contain at most 160 characters.")
        # Round-trip through the public schema before touching durable storage.
        validated = BehaviorLabExperiment.from_dict(experiment.to_dict())
        payload = self._load_payload()
        payload[normalized] = validated.to_dict()
        self._write_payload(payload)

    def delete(self, name: str) -> bool:
        normalized = str(name).strip()
        payload = self._load_payload()
        if normalized not in payload:
            return False
        del payload[normalized]
        self._write_payload(payload)
        return True

    def _load_payload(self) -> dict[str, dict[str, Any]]:
        raw = read_json(self.path)
        if not raw:
            return {}
        if int(raw.get("schema_version", 0)) != _STORE_SCHEMA_VERSION:
            raise BehaviorLabValidationError("Unsupported Behavior Lab experiment-store version.")
        experiments = raw.get("experiments", {})
        if not isinstance(experiments, dict):
            raise BehaviorLabValidationError("Behavior Lab experiment storage is malformed.")
        result: dict[str, dict[str, Any]] = {}
        for name, value in experiments.items():
            if not isinstance(name, str) or not isinstance(value, dict):
                raise BehaviorLabValidationError("Behavior Lab experiment storage is malformed.")
            if int(value.get("schema_version", 0)) != BEHAVIOR_LAB_SCHEMA_VERSION:
                raise BehaviorLabValidationError(
                    f"Saved experiment {name!r} uses an unsupported schema version."
                )
            result[name] = value
        return result

    def _write_payload(self, experiments: dict[str, dict[str, Any]]) -> None:
        atomic_write_json(
            self.path,
            {
                "schema_version": _STORE_SCHEMA_VERSION,
                "experiments": experiments,
            },
        )
