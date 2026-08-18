"""High-level orchestration for loading, selecting, and fitting IGT participants.

`FittingPipelineConfig` captures one run's data source, models, optimizer settings,
participant subset, and optional warm-start provider. `run_fitting_pipeline` then
loads the data, applies selection, executes fitting, and returns the flat result table.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from igt.comparison import fit_results_to_dataframe
from igt.constants.fitting import DEFAULT_MAX_ITERATIONS
from igt.models.base import ComputationalModel
from igt.rdata_preprocessing import load_igt_long_table
from igt.subject_selection import filter_subjects_by_keys
from igt.typing import StrPathLike
from igt.utils.io import normalize_path

from .manager import fit_all_subjects
from .typing import SubjectModelWarmStartsProvider


@dataclass(frozen=True, slots=True)
class FittingPipelineConfig:
    """Configuration for one complete subject-level fitting pipeline run.

    The configuration specifies the source data, models, optimizer execution
    settings, optional participant subset, and optional warm-start provider used by
    [`run_fitting_pipeline`][igt.execution.pipeline.run_fitting_pipeline].

    Attributes:
        rdata_path: Source RData file containing the IGT dataset.
        models: Computational models fitted to each selected participant.
        max_iterations: Maximum optimizer iterations per starting point.
        n_workers: Number of worker processes used for fitting.
        show_progress: Whether fitting progress is displayed.
        n_subjects: Optional cap on the number of participants processed.
        subject_keys: Optional explicit participant-key subset.
        subject_model_warm_starts_provider: Optional callback supplying additional
            model-specific starting points for individual participants.
    """

    rdata_path: StrPathLike
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
        """Validate and normalize one fitting-pipeline configuration.

        The RData path is normalized, the model collection is frozen as a tuple, numeric
        execution limits are validated, and an explicit participant-key table is copied.
        `subject_keys` and `n_subjects` are mutually exclusive selection mechanisms.

        Raises:
            TypeError: If a configuration field has an unsupported type or the warm-start
                provider is not callable.
            ValueError: If no model is supplied, an integer limit is outside its allowed
                range, `subject_keys` is empty, or both participant-selection mechanisms
                are configured simultaneously.
        """

        object.__setattr__(
            self, "rdata_path", normalize_path(self.rdata_path, parameter_name="rdata_path")
        )
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
    """Load IGT data, apply participant selection, fit models, and return result rows.

    The dataset is loaded from `config.rdata_path`; an explicit participant-key table
    is applied when configured. Model fitting is delegated to
    [`fit_all_subjects`][igt.execution.manager.fit_all_subjects] with the configured
    iteration limit, worker count, and optional warm-start provider. Completed records
    are converted to the project's sorted fit-result DataFrame.

    Args:
        config: Validated configuration for the complete fitting run.

    Returns:
        Sorted per-participant, per-model fit-result table.

    Raises:
        ValueError: If an explicitly supplied participant-key table is empty or
            downstream data, participant, or model validation fails.
    """

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
        subject_model_warm_starts_provider=config.subject_model_warm_starts_provider,
    )
    logger.info("Model fitting completed.")

    return fit_results_to_dataframe(fit_results)
