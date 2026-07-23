"""Generic maximum-likelihood fitting utilities for computational models."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from igt.models.base import ComputationalModel, SubjectData
from igt.typing import FloatArray


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
    converged: bool
    optimizer_message: str
    n_function_evaluations: int
    n_iterations: int | None
    n_starts: int

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


def _validate_starting_points(
    model: ComputationalModel,
    starts: FloatArray,
) -> FloatArray:
    """Validate optimizer starting points returned by a model."""

    starts_array = np.asarray(starts, dtype=np.float64)
    expected_columns = model.n_parameters

    if starts_array.ndim != 2:
        raise ValueError("Model starting points must be a two-dimensional array.")

    if starts_array.shape[0] == 0:
        raise ValueError("A model must provide at least one starting point.")

    if starts_array.shape[1] != expected_columns:
        raise ValueError(
            "Starting-point column count must match the number of model "
            f"parameters: got {starts_array.shape[1]} and expected {expected_columns}."
        )

    if not np.isfinite(starts_array).all():
        raise ValueError("Model starting points must contain only finite values.")

    for start_index, start in enumerate(starts_array):
        if not model.parameters_within_bounds(start):
            raise ValueError(f"Starting point at index {start_index} is outside the model bounds.")

    return starts_array


def _result_nll(result: OptimizeResult) -> float:
    """Return a finite comparison value for an optimizer result."""

    value = float(result.fun)
    return value if np.isfinite(value) else float("inf")


def fit_model(
    model: ComputationalModel,
    data: SubjectData,
    *,
    n_trials: int,
    subject_id: int,
    source_study: str,
    optimizer_options: Mapping[str, object] | None = None,
) -> ModelFitResult:
    """Fit one model to one subject by bounded multistart optimization.

    Every starting point supplied by ``model.starting_points`` is optimized
    independently with L-BFGS-B. The result with the lowest finite negative
    log-likelihood is retained, regardless of the optimizer success flag.
    """

    if n_trials != data.n_trials:
        raise ValueError(
            f"n_trials metadata ({n_trials}) does not match SubjectData ({data.n_trials})."
        )

    if not source_study.strip():
        raise ValueError("source_study must not be empty.")

    starts = _validate_starting_points(
        model,
        model.starting_points(data),
    )

    options = dict(optimizer_options) if optimizer_options is not None else None
    optimization_results: list[OptimizeResult] = []

    for start in starts:
        result = minimize(
            fun=model.negative_log_likelihood,
            x0=start,
            args=(data,),
            method="L-BFGS-B",
            bounds=model.parameter_bounds,
            options=options,
        )
        optimization_results.append(result)

    best_result = min(
        optimization_results,
        key=_result_nll,
    )

    best_parameters = model.validate_parameters(np.asarray(best_result.x, dtype=np.float64))
    negative_log_likelihood = _result_nll(best_result)

    if not np.isfinite(negative_log_likelihood):
        raise RuntimeError(
            f"All optimization runs produced non-finite objective values for {model.name}."
        )

    log_likelihood = -negative_log_likelihood
    n_parameters = model.n_parameters

    aic = (2.0 * n_parameters) + (2.0 * negative_log_likelihood)
    bic = (n_parameters * np.log(data.n_trials)) + (2.0 * negative_log_likelihood)

    raw_nit = getattr(best_result, "nit", None)
    n_iterations = int(raw_nit) if raw_nit is not None else None

    return ModelFitResult(
        model_name=model.name,
        n_trials=n_trials,
        subject_id=subject_id,
        source_study=source_study,
        parameter_names=model.parameter_names,
        parameter_values=tuple(float(value) for value in best_parameters),
        negative_log_likelihood=negative_log_likelihood,
        log_likelihood=log_likelihood,
        aic=float(aic),
        bic=float(bic),
        converged=bool(best_result.success),
        optimizer_message=str(best_result.message),
        n_function_evaluations=int(best_result.nfev),
        n_iterations=n_iterations,
        n_starts=int(starts.shape[0]),
    )
