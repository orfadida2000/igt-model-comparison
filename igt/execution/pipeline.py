"""Shared orchestration for loading, selecting, and fitting IGT subjects."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from igt.comparison import fit_results_to_dataframe
from igt.constants.fitting import DEFAULT_MAX_ITERATIONS
from igt.models.base import ComputationalModel
from igt.rdata_preprocessing import load_igt_long_table
from igt.subject_selection import filter_subjects_by_keys

from .manager import fit_all_subjects
from .typing import SubjectModelWarmStartsProvider


@dataclass(frozen=True, slots=True)
class FittingPipelineConfig:
    """Resolved inputs for one fitting pipeline execution."""

    rdata_path: Path
    models: Sequence[ComputationalModel] = field(repr=False)
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    n_workers: int | None = None
    show_progress: bool = True
    n_subjects: int | None = None
    subject_keys: pd.DataFrame | None = field(default=None, repr=False)
    subject_model_warm_starts_provider: SubjectModelWarmStartsProvider | None = field(
        default=None,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        """Validate and freeze configuration values."""

        object.__setattr__(self, "rdata_path", Path(self.rdata_path))
        object.__setattr__(self, "models", tuple(self.models))

        if not self.models:
            raise ValueError("At least one model must be supplied.")

        if not isinstance(self.max_iterations, int) or isinstance(self.max_iterations, bool):
            raise TypeError("max_iterations must be an integer.")

        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be greater than zero.")

        if self.n_workers is not None:
            if not isinstance(self.n_workers, int) or isinstance(self.n_workers, bool):
                raise TypeError("n_workers must be an integer or None.")

            if self.n_workers <= 0:
                raise ValueError("n_workers must be greater than zero.")

        if self.n_subjects is not None:
            if not isinstance(self.n_subjects, int) or isinstance(self.n_subjects, bool):
                raise TypeError("n_subjects must be an integer or None.")

            if self.n_subjects < 0:
                raise ValueError("n_subjects must be greater than or equal to zero.")

        if not isinstance(self.show_progress, bool):
            raise TypeError("show_progress must be a boolean.")

        if self.subject_keys is not None:
            if not isinstance(self.subject_keys, pd.DataFrame):
                raise TypeError("subject_keys must be a pandas DataFrame or None.")

            if self.subject_keys.empty:
                raise ValueError("subject_keys was provided but contains no subjects.")

            if self.n_subjects is not None:
                raise ValueError("subject_keys and n_subjects cannot both be provided.")

            object.__setattr__(self, "subject_keys", self.subject_keys.copy())

        if self.subject_model_warm_starts_provider is not None:
            if not callable(self.subject_model_warm_starts_provider):
                raise TypeError("subject_model_warm_starts_provider must be a callable or None.")


def run_fitting_pipeline(config: FittingPipelineConfig) -> pd.DataFrame:
    """Load data, optionally filter participants, fit models, and return a table."""

    if config.subject_keys is not None and config.subject_keys.empty:
        raise ValueError("subject_keys was provided but contains no subjects.")

    logger = logging.getLogger(__name__)

    logger.info("Loading IGT dataset from: %s", config.rdata_path)
    data = load_igt_long_table(config.rdata_path)

    if config.subject_keys is not None:
        logger.info(
            "Filtering the dataset to %d explicitly selected subjects.",
            len(config.subject_keys),
        )
        data = filter_subjects_by_keys(data, config.subject_keys)

    logger.info(
        "Fitting the models: %r",
        [model.name for model in config.models],
    )
    fit_results = fit_all_subjects(
        data,
        config.models,
        optimizer_options={"maxiter": config.max_iterations},
        show_progress=config.show_progress,
        n_workers=config.n_workers,
        n_subjects=config.n_subjects,
    )
    logger.info("Model fitting completed.")

    return fit_results_to_dataframe(fit_results)
