"""Maximum-likelihood fitting utilities shared by computational models.

Each candidate start is optimized with bounded SciPy minimization, validated against
the model parameterization, and compared across starts. The selected fit is converted
into a `ModelFitResult` with likelihood, information-criterion, convergence, and
parameter-boundary diagnostics.
"""

import logging
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from igt.constants.fitting import (
    DEFAULT_FIT_METHOD,
    PARAMETER_BOUNDARY_ABSOLUTE_TOLERANCE_FACTOR,
    UNIFORM_CHOICE_NLL_ABSOLUTE_TOLERANCE,
)
from igt.constants.models import (
    N_IGT_DECKS,
)
from igt.constants.schema import (
    N_TRIALS_COLUMN,
    NLL_COLUMN,
    SOURCE_STUDY_COLUMN,
    SUBJECT_ID_COLUMN,
)
from igt.models.base import ComputationalModel
from igt.models.typing import SubjectData
from igt.typing import Float1DArray, Float2DArray, ParameterBounds

from .typing import ModelFitResult


def _validate_starting_points(
    model: ComputationalModel,
    starts: Float2DArray,
) -> Float2DArray:
    """Validate and normalize optimizer starting points for a model.

    The candidate array must be two-dimensional, contain at least one row, match
    the model's parameter count, contain only finite values, and place every row
    within the model's configured parameter bounds.

    Args:
        model: Model whose parameter count and bounds constrain the starting points.
        starts: Candidate starting-point matrix, with one parameter vector per row.

    Returns:
        A finite starting-point matrix normalized to floating-point values.

    Raises:
        ValueError: If the starting points have an invalid shape, are empty,
            contain non-finite values, have the wrong number of parameters, or
            include a row outside the model bounds.
    """

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
    """Return an optimizer objective value suitable for result comparison.

    Args:
        result: SciPy optimizer result whose objective value is inspected.

    Returns:
        The finite negative log-likelihood, or positive infinity when the stored
        objective value is non-finite.
    """

    value = float(result.fun)
    return value if np.isfinite(value) else float("inf")


def _select_best_optimization_result(
    optimization_results: Sequence[OptimizeResult],
    *,
    model_name: str,
) -> OptimizeResult:
    """Select the preferred result from a multistart optimization.

    Finite successful results are always preferred over unsuccessful results. The
    successful result with the smallest negative log-likelihood is returned. If no
    run converged, the best finite unsuccessful result is retained so its
    diagnostics can still be reported as a nonconverged fit.

    Args:
        optimization_results: Results produced by all optimized starting points.
        model_name: Model name included in failure diagnostics.

    Returns:
        The preferred finite optimizer result.

    Raises:
        RuntimeError: If every optimizer run has a non-finite objective value or
            non-finite parameter vector.
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
    """Count fitted parameters that lie on configured bounds.

    A scale-aware absolute tolerance derived from
    `PARAMETER_BOUNDARY_ABSOLUTE_TOLERANCE_FACTOR` is used independently for each
    parameter.

    Args:
        parameters: Fitted parameter vector.
        bounds: Lower and upper bounds aligned with `parameters`.

    Returns:
        The numbers of parameters at a lower bound, at an upper bound, and at
        either bound, respectively.
    """

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
    """Fit one computational model to one participant by bounded multistart optimization.

    Default model starting points are generated by the model itself. Optional warm
    starts are validated and prepended before every start is optimized independently.
    Among finite results, successful optimizer runs are preferred and the lowest-NLL
    result is retained. The returned record also includes AIC, BIC, uniform-choice
    diagnostics, parameter-boundary counts, and optimizer diagnostics.

    Args:
        model: Computational model to fit.
        data: Validated choices, wins, and losses for one participant.
        n_trials: Trial-count metadata expected to match `data`.
        subject_id: Participant identifier within the trial-count group.
        source_study: Nonempty source-study label for the participant.
        optimizer_options: Optional options forwarded to `scipy.optimize.minimize`.
        warm_starting_points: Optional additional starting points to optimize before
            the model-generated starts.
        logger: Optional logger used for detailed fitting diagnostics.
        fit_method: Optimization method passed to `scipy.optimize.minimize`.

    Returns:
        The complete fitted-model result for the participant.

    Raises:
        ValueError: If the trial-count metadata disagrees with `data`, the study
            label is empty, or any generated or supplied starting point is invalid.
        RuntimeError: If every optimization run produces a non-finite result.
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
                N_TRIALS_COLUMN: n_trials,
                SUBJECT_ID_COLUMN: subject_id,
                SOURCE_STUDY_COLUMN: source_study,
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
                N_TRIALS_COLUMN: n_trials,
                SUBJECT_ID_COLUMN: subject_id,
                SOURCE_STUDY_COLUMN: source_study,
                "fit_method": fit_method,
            },
            starts.shape[0],
            [
                {
                    "start": tuple(float(value) for value in start),
                    "parameters": tuple(float(value) for value in result.x),
                    NLL_COLUMN: _result_nll(result),
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
                N_TRIALS_COLUMN: n_trials,
                SUBJECT_ID_COLUMN: subject_id,
                SOURCE_STUDY_COLUMN: source_study,
                "fit_method": fit_method,
            },
            starts.shape[0],
            {
                "best_parameters": tuple(float(value) for value in best_parameters),
                NLL_COLUMN: negative_log_likelihood,
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
