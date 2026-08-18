"""Canonical column names and participant keys for project result tables.

The schema constants identify model, study, convergence, likelihood, trial-count,
and subject columns and define the compound participant and participant-model keys
used throughout fitting, selection, correction, and analysis.
"""

from typing import Final

MODEL_COLUMN: Final[str] = "model"
SOURCE_STUDY_COLUMN: Final[str] = "source_study"
CONVERGED_COLUMN: Final[str] = "converged"
NLL_COLUMN: Final[str] = "negative_log_likelihood"

N_TRIALS_COLUMN: Final[str] = "n_trials"
SUBJECT_ID_COLUMN: Final[str] = "subject_id"
PARTICIPANT_KEY_COLUMNS: Final[tuple[str, ...]] = (
    N_TRIALS_COLUMN,
    SUBJECT_ID_COLUMN,
)

PARTICIPANT_MODEL_KEY_COLUMNS: Final[tuple[str, ...]] = (
    *PARTICIPANT_KEY_COLUMNS,
    MODEL_COLUMN,
)
