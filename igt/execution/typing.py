"""Typed data structures and callback protocols used by the fitting execution layer.

The module defines logging state for spawned workers, complete per-model fit records,
subject tasks, optional model warm starts, and the callable contract used to provide
subject-specific warm-start matrices.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from igt.constants.schema import (
    CONVERGED_COLUMN,
    MODEL_COLUMN,
    N_TRIALS_COLUMN,
    NLL_COLUMN,
    SOURCE_STUDY_COLUMN,
    SUBJECT_ID_COLUMN,
)
from igt.models.typing import SubjectData
from igt.typing import Float2DArray


@dataclass(frozen=True, slots=True)
class LoggingContext:
    """Snapshot of process-wide logging state needed by worker processes.

    A context captures the global logging disable threshold, exception behavior,
    and root logger level in the parent process. Call [`apply`][igt.execution.typing.LoggingContext.apply]
    inside a spawned worker to restore those settings before fitting begins.

    Attributes:
        disable_level: Global logging disable threshold captured from the parent.
        raise_exceptions: Whether logging handler exceptions should propagate.
        root_logger_level: Effective root logger level to apply in the worker.
    """
    disable_level: int = field(init=False)
    raise_exceptions: bool = field(init=False)
    root_logger_level: int = field(init=False)

    def __post_init__(self) -> None:
        # Captures current global state from the logging module
        """Capture the current process-wide logging state.

        The captured values are stored on the frozen instance so they can later be
        reapplied inside spawned worker processes.
        """
        object.__setattr__(self, "disable_level", logging.root.manager.disable)
        object.__setattr__(self, "raise_exceptions", logging.raiseExceptions)
        object.__setattr__(self, "root_logger_level", logging.getLogger().level)

    def apply(self) -> None:
        """Apply the captured logging state to the current process.

        This restores the global disable threshold, logging exception behavior, and
        root logger level recorded when the context was created.
        """
        logging.disable(self.disable_level)
        logging.raiseExceptions = self.raise_exceptions
        logging.getLogger().setLevel(self.root_logger_level)


@dataclass(frozen=True, slots=True)
class ModelFitResult:
    """Complete fitted-model result for one participant.

    The record contains identifying metadata, fitted parameters, likelihood and
    information-criterion values, optimizer diagnostics, uniform-choice diagnostics,
    and parameter-boundary counts. Use [`to_record`][igt.execution.typing.ModelFitResult.to_record]
    to obtain the flat representation written to result tables.

    Attributes:
        model_name: Canonical fitted model name.
        n_trials: Number of observed trials for the participant.
        subject_id: Participant identifier within the trial-count group.
        source_study: Study from which the participant data originated.
        parameter_names: Ordered model parameter names.
        parameter_values: Fitted parameter values aligned with `parameter_names`.
        negative_log_likelihood: Optimized negative log-likelihood.
        log_likelihood: Optimized log-likelihood.
        aic: Akaike information criterion.
        bic: Bayesian information criterion.
        uniform_choice_nll: Negative log-likelihood under uniform four-deck choice.
        nll_improvement_over_uniform: Improvement over uniform-choice NLL.
        uniform_choice_fit: Whether the optimized NLL is numerically indistinguishable
            from the uniform-choice baseline.
        n_parameters_at_lower_bound: Number of parameters at lower bounds.
        n_parameters_at_upper_bound: Number of parameters at upper bounds.
        n_parameters_at_any_bound: Number of parameters at either bound.
        converged: Whether the selected optimizer result reported success.
        optimizer_message: Optimizer termination message.
        n_function_evaluations: Number of objective evaluations for the selected run.
        n_iterations: Number of optimizer iterations when reported.
        n_starts: Total number of optimized starting points.
    """

    model_name: str
    n_trials: int
    subject_id: int
    source_study: str
    parameter_names: tuple[str, ...]
    parameter_values: tuple[float, ...]
    negative_log_likelihood: float
    log_likelihood: float
    aic: float
    bic: float
    uniform_choice_nll: float
    nll_improvement_over_uniform: float
    uniform_choice_fit: bool
    n_parameters_at_lower_bound: int
    n_parameters_at_upper_bound: int
    n_parameters_at_any_bound: int
    converged: bool
    optimizer_message: str
    n_function_evaluations: int
    n_iterations: int | None
    n_starts: int

    @staticmethod
    def get_result_columns() -> list[str]:
        """Return the column names for a flat table of model-fit results.

        Returns:
            Column names used by the flat model-fit table representation.
        """

        return [
            MODEL_COLUMN,
            N_TRIALS_COLUMN,
            SUBJECT_ID_COLUMN,
            SOURCE_STUDY_COLUMN,
            NLL_COLUMN,
            "log_likelihood",
            "aic",
            "bic",
            "uniform_choice_nll",
            "nll_improvement_over_uniform",
            "uniform_choice_fit",
            "n_parameters_at_lower_bound",
            "n_parameters_at_upper_bound",
            "n_parameters_at_any_bound",
            CONVERGED_COLUMN,
            "optimizer_message",
            "n_function_evaluations",
            "n_iterations",
            "n_starts",
        ]

    def to_record(self) -> dict[str, object]:
        """Return a flat dictionary suitable for a pandas DataFrame row.

        Returns:
            Flat mapping suitable for constructing one DataFrame row.
        """

        record: dict[str, object] = {
            MODEL_COLUMN: self.model_name,
            N_TRIALS_COLUMN: self.n_trials,
            SUBJECT_ID_COLUMN: self.subject_id,
            SOURCE_STUDY_COLUMN: self.source_study,
            NLL_COLUMN: self.negative_log_likelihood,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "uniform_choice_nll": self.uniform_choice_nll,
            "nll_improvement_over_uniform": self.nll_improvement_over_uniform,
            "uniform_choice_fit": self.uniform_choice_fit,
            "n_parameters_at_lower_bound": self.n_parameters_at_lower_bound,
            "n_parameters_at_upper_bound": self.n_parameters_at_upper_bound,
            "n_parameters_at_any_bound": self.n_parameters_at_any_bound,
            CONVERGED_COLUMN: self.converged,
            "optimizer_message": self.optimizer_message,
            "n_function_evaluations": self.n_function_evaluations,
            "n_iterations": self.n_iterations,
            "n_starts": self.n_starts,
        }

        record.update(
            zip(
                self.parameter_names,
                self.parameter_values,
                strict=True,
            )
        )

        return record


type SubjectModelWarmStartsProvider = Callable[
    [str, int, int],
    Float2DArray | None,
]


@dataclass(frozen=True, slots=True)
class ModelWarmStarts:
    """Additional optimizer starting points supplied for one model.

    Attributes:
        model_name: Stable programmatic model name.
        starting_points: Additional parameter vectors, one per row.
    """

    model_name: str
    starting_points: Float2DArray = field(
        repr=False,
        compare=False,
        hash=False,
    )


@dataclass(frozen=True, slots=True)
class SubjectFitTask:
    """Serializable task containing all data required to fit one subject.

    Attributes:
        n_trials: Number of trials for the subject.
        subject_id: Subject identifier within the trial-count dataset.
        source_study: Source study label.
        data: Validated choices, wins, and losses.
        model_warm_starts: Optional model-specific additional starting points.
    """

    n_trials: int
    subject_id: int
    source_study: str
    data: SubjectData = field(
        repr=False,
        compare=False,
        hash=False,
    )
    model_warm_starts: tuple[ModelWarmStarts, ...] = field(
        default=(),
        repr=False,
        compare=False,
        hash=False,
    )
