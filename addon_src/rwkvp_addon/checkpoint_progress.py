from __future__ import annotations

from .progress import ProgressReporter

CHECKPOINT_PREPARATION_TOTAL = 4
CHECKPOINT_COLLECTION_DATA_STEP = 1
CHECKPOINT_LOAD_STEP = 2
CHECKPOINT_CONSISTENCY_STEP = 3
CHECKPOINT_CURVE_DATA_STEP = 4


def update_checkpoint_collection_data(progress: ProgressReporter, label: str) -> None:
    progress.update(CHECKPOINT_COLLECTION_DATA_STEP, CHECKPOINT_PREPARATION_TOTAL, label)


def update_checkpoint_review_history(progress: ProgressReporter, label: str) -> None:
    update_checkpoint_collection_data(progress, label)


def update_checkpoint_load(progress: ProgressReporter, label: str) -> None:
    progress.update(CHECKPOINT_LOAD_STEP, CHECKPOINT_PREPARATION_TOTAL, label)


def update_checkpoint_consistency(progress: ProgressReporter, label: str) -> None:
    progress.update(CHECKPOINT_CONSISTENCY_STEP, CHECKPOINT_PREPARATION_TOTAL, label)


def update_checkpoint_curve_data(progress: ProgressReporter, label: str) -> None:
    progress.update(CHECKPOINT_CURVE_DATA_STEP, CHECKPOINT_PREPARATION_TOTAL, label)
