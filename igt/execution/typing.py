import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from igt.models.typing import SubjectData
from igt.typing import Float2DArray


@dataclass(frozen=True, slots=True)
class LoggingContext:
    disable_level: int = field(init=False)
    raise_exceptions: bool = field(init=False)
    root_logger_level: int = field(init=False)

    def __post_init__(self) -> None:
        # Captures current global state from the logging module
        object.__setattr__(self, "disable_level", logging.root.manager.disable)
        object.__setattr__(self, "raise_exceptions", logging.raiseExceptions)
        object.__setattr__(self, "root_logger_level", logging.getLogger().level)

    def apply(self) -> None:
        """Applies captured global state to the current process."""
        logging.disable(self.disable_level)
        logging.raiseExceptions = self.raise_exceptions
        logging.getLogger().setLevel(self.root_logger_level)


@dataclass(frozen=True, slots=True)
class ModelFitResult:
    """Best optimization result for one model and one subject."""

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
        """Return the column names for a flat table of model-fit results."""

        return [
            "model",
            "n_trials",
            "subject_id",
            "source_study",
            "negative_log_likelihood",
            "log_likelihood",
            "aic",
            "bic",
            "uniform_choice_nll",
            "nll_improvement_over_uniform",
            "uniform_choice_fit",
            "n_parameters_at_lower_bound",
            "n_parameters_at_upper_bound",
            "n_parameters_at_any_bound",
            "converged",
            "optimizer_message",
            "n_function_evaluations",
            "n_iterations",
            "n_starts",
        ]

    def to_record(self) -> dict[str, object]:
        """Return a flat dictionary suitable for a pandas DataFrame row."""

        record: dict[str, object] = {
            "model": self.model_name,
            "n_trials": self.n_trials,
            "subject_id": self.subject_id,
            "source_study": self.source_study,
            "negative_log_likelihood": self.negative_log_likelihood,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "uniform_choice_nll": self.uniform_choice_nll,
            "nll_improvement_over_uniform": self.nll_improvement_over_uniform,
            "uniform_choice_fit": self.uniform_choice_fit,
            "n_parameters_at_lower_bound": self.n_parameters_at_lower_bound,
            "n_parameters_at_upper_bound": self.n_parameters_at_upper_bound,
            "n_parameters_at_any_bound": self.n_parameters_at_any_bound,
            "converged": self.converged,
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
    """Additional optimizer starting points for one model."""

    model_name: str
    starting_points: Float2DArray = field(
        repr=False,
        compare=False,
        hash=False,
    )


@dataclass(frozen=True, slots=True)
class SubjectFitTask:
    """Serializable input required to fit every model to one subject."""

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
