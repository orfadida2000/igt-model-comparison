"""Generic maximum-likelihood fitting utilities for computational models."""

import logging
from collections.abc import Mapping

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from igt.constants.config import DEFAULT_FIT_METHOD
from igt.models.base import ComputationalModel
from igt.models.typing import SubjectData
from igt.typing import FloatArray

from .typing import ModelFitResult


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
    logger: logging.Logger | None = None,
    fit_method: str = DEFAULT_FIT_METHOD,
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

    if logger is not None:
        logger.debug(
            "Fitting model %r for %r with %d starting points: %r",
            model.name,
            {
                "subject_id": subject_id,
                "source_study": source_study,
                "n_trials": n_trials,
                "fit_method": fit_method,
            },
            starts.shape[0],
            [tuple(float(value) for value in start) for start in starts],
        )

    options = dict(optimizer_options) if optimizer_options is not None else None
    optimization_results: list[OptimizeResult] = []

    for start in starts:
        result = minimize(
            fun=model.negative_log_likelihood,
            x0=start,
            args=(data,),
            method=fit_method,
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

    if logger is not None:
        logger.debug(
            "Fitting model %r for %r results for %d starting points: %r",
            model.name,
            {
                "subject_id": subject_id,
                "source_study": source_study,
                "n_trials": n_trials,
                "fit_method": fit_method,
            },
            starts.shape[0],
            {
                tuple(float(value) for value in result.x): _result_nll(result)
                for result in optimization_results
            },
        )

        logger.debug(
            "Fitting model %r for %r with %d starting points completed with best result: %r",
            model.name,
            {
                "subject_id": subject_id,
                "source_study": source_study,
                "n_trials": n_trials,
                "fit_method": fit_method,
            },
            starts.shape[0],
            {
                "best_parameters": tuple(float(value) for value in best_parameters),
                "negative_log_likelihood": negative_log_likelihood,
            },
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
