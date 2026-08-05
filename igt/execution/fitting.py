"""Generic maximum-likelihood fitting utilities for computational models."""

import logging
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from igt.constants.fitting import (
    DEFAULT_FIT_METHOD,
    PARAMETER_BOUNDARY_ABSOLUTE_TOLERANCE_FACTOR,
    UNIFORM_CHOICE_NLL_ABSOLUTE_TOLERANCE,
)
from igt.constants.models import N_IGT_DECKS
from igt.models.base import ComputationalModel
from igt.models.typing import SubjectData
from igt.typing import Float1DArray, Float2DArray, ParameterBounds

from .typing import ModelFitResult


def _validate_starting_points(
    model: ComputationalModel,
    starts: Float2DArray,
) -> Float2DArray:
    """Validate optimizer starting points for a model."""

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


def _select_best_optimization_result(
    optimization_results: Sequence[OptimizeResult],
    *,
    model_name: str,
) -> OptimizeResult:
    """Return the best finite successful result, or the best finite failure.

    Successful finite results are always preferred. A finite unsuccessful
    result is retained only when every optimizer run failed, so the caller can
    save diagnostics while marking the overall fit as nonconverged.
    """

    finite_results = [
        result
        for result in optimization_results
        if np.isfinite(_result_nll(result))
        and np.isfinite(np.asarray(result.x, dtype=np.float64)).all()
    ]

    if not finite_results:
        raise RuntimeError(
            f"All optimization runs produced non-finite objective values for {model_name}."
        )

    successful_results = [result for result in finite_results if bool(result.success)]
    candidate_results = successful_results or finite_results

    return min(candidate_results, key=_result_nll)


def _parameter_boundary_counts(
    parameters: Float1DArray,
    bounds: ParameterBounds,
) -> tuple[int, int, int]:
    """Count fitted parameters at lower, upper, and either bound."""

    lower_count = 0
    upper_count = 0
    any_count = 0

    for value, (lower_bound, upper_bound) in zip(parameters, bounds, strict=True):
        scale = max(1.0, abs(lower_bound), abs(upper_bound), abs(upper_bound - lower_bound))
        absolute_tolerance = PARAMETER_BOUNDARY_ABSOLUTE_TOLERANCE_FACTOR * scale

        at_lower = bool(
            np.isclose(
                value,
                lower_bound,
                rtol=0.0,
                atol=absolute_tolerance,
            )
        )
        at_upper = bool(
            np.isclose(
                value,
                upper_bound,
                rtol=0.0,
                atol=absolute_tolerance,
            )
        )

        lower_count += int(at_lower)
        upper_count += int(at_upper)
        any_count += int(at_lower or at_upper)

    return lower_count, upper_count, any_count


def fit_model(
    model: ComputationalModel,
    data: SubjectData,
    *,
    n_trials: int,
    subject_id: int,
    source_study: str,
    optimizer_options: Mapping[str, object] | None = None,
    warm_starting_points: Float2DArray | None = None,
    logger: logging.Logger | None = None,
    fit_method: str = DEFAULT_FIT_METHOD,
) -> ModelFitResult:
    """Fit one model to one subject by bounded multistart optimization.

    Every starting point supplied by ``model.starting_points`` is optimized
    independently. The finite successful result with the lowest negative
    log-likelihood is retained. A finite unsuccessful result is retained only
    when no run converged, and the overall fit is then marked nonconverged.
    """

    if n_trials != data.n_trials:
        raise ValueError(
            f"n_trials metadata ({n_trials}) does not match SubjectData ({data.n_trials})."
        )

    if not source_study.strip():
        raise ValueError("source_study must not be empty.")

    model_starts = _validate_starting_points(
        model,
        model.starting_points(data),
    )

    if warm_starting_points is None:
        starts = model_starts
    else:
        warm_starts = _validate_starting_points(
            model,
            warm_starting_points,
        )
        starts = np.vstack(
            (
                warm_starts,
                model_starts,
            )
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

    best_result = _select_best_optimization_result(
        optimization_results,
        model_name=model.name,
    )

    best_parameters = model.validate_parameters(np.asarray(best_result.x, dtype=np.float64))
    negative_log_likelihood = _result_nll(best_result)

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
            [
                {
                    "start": tuple(float(value) for value in start),
                    "parameters": tuple(float(value) for value in result.x),
                    "negative_log_likelihood": _result_nll(result),
                    "success": bool(result.success),
                    "message": str(result.message),
                }
                for start, result in zip(starts, optimization_results, strict=True)
            ],
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
                "success": bool(best_result.success),
                "message": str(best_result.message),
            },
        )

    log_likelihood = -negative_log_likelihood
    n_parameters = model.n_parameters

    aic = (2.0 * n_parameters) + (2.0 * negative_log_likelihood)
    bic = (n_parameters * np.log(data.n_trials)) + (2.0 * negative_log_likelihood)

    uniform_choice_nll = float(data.n_trials * np.log(N_IGT_DECKS))
    nll_improvement_over_uniform = uniform_choice_nll - negative_log_likelihood
    uniform_choice_fit = bool(
        np.isclose(
            nll_improvement_over_uniform,
            0.0,
            rtol=0.0,
            atol=UNIFORM_CHOICE_NLL_ABSOLUTE_TOLERANCE,
        )
    )

    (
        n_parameters_at_lower_bound,
        n_parameters_at_upper_bound,
        n_parameters_at_any_bound,
    ) = _parameter_boundary_counts(best_parameters, model.parameter_bounds)

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
        uniform_choice_nll=uniform_choice_nll,
        nll_improvement_over_uniform=float(nll_improvement_over_uniform),
        uniform_choice_fit=uniform_choice_fit,
        n_parameters_at_lower_bound=n_parameters_at_lower_bound,
        n_parameters_at_upper_bound=n_parameters_at_upper_bound,
        n_parameters_at_any_bound=n_parameters_at_any_bound,
        converged=bool(best_result.success),
        optimizer_message=str(best_result.message),
        n_function_evaluations=int(best_result.nfev),
        n_iterations=n_iterations,
        n_starts=int(starts.shape[0]),
    )
