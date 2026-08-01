"""Shared tabular-schema definitions."""

from typing import Final

PARTICIPANT_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "n_trials",
    "subject_id",
)
