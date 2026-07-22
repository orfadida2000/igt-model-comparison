"""Utilities for generating optimizer starting points.

This module contains generic starting-point generation methods. It does not
decide which method a model should use.

Current usage:

- Q-learning uses a Cartesian grid of candidate starting points.
- PVL-Delta uses Sobol starting points distributed across its parameter bounds.
"""

from collections.abc import Sequence
from itertools import product

import numpy as np
from scipy.stats import qmc

from igt.typing import FloatArray, ParameterBounds


def bounds_to_arrays(
    bounds: ParameterBounds,
) -> tuple[FloatArray, FloatArray]:
    """Convert parameter bounds into lower- and upper-bound arrays.

    Args:
        bounds: Sequence of ``(lower_bound, upper_bound)`` pairs.

    Returns:
        A pair containing:

        - an array of lower bounds;
        - an array of upper bounds.

    Raises:
        ValueError: If no bounds are provided, a bound is not finite, or a
            lower bound is not smaller than its corresponding upper bound.
    """

    bounds_array = np.asarray(bounds, dtype=np.float64)

    if bounds_array.ndim != 2 or bounds_array.shape[1] != 2:
        raise ValueError("Bounds must be a sequence of (lower_bound, upper_bound) pairs.")

    if bounds_array.shape[0] == 0:
        raise ValueError("At least one parameter bound is required.")

    if not np.isfinite(bounds_array).all():
        raise ValueError("All parameter bounds must be finite.")

    lower_bounds = bounds_array[:, 0]
    upper_bounds = bounds_array[:, 1]

    invalid_bounds = lower_bounds >= upper_bounds

    if invalid_bounds.any():
        invalid_indices = np.flatnonzero(invalid_bounds).tolist()

        raise ValueError(
            "Every lower bound must be smaller than its upper bound. "
            f"Invalid parameter indices: {invalid_indices}"
        )

    return lower_bounds, upper_bounds


def generate_grid_starts(
    parameter_values: Sequence[Sequence[float] | FloatArray],
) -> FloatArray:
    """Generate every combination of supplied parameter values.

    This function constructs a Cartesian product. Each row of the returned
    array is one starting point, and each column corresponds to one model
    parameter.

    Args:
        parameter_values: One sequence of candidate values per parameter.

    Returns:
        Two-dimensional array with shape:

        ``(number_of_combinations, number_of_parameters)``

    Raises:
        ValueError: If no parameters are supplied, a parameter has no candidate
            values, or a candidate value is not finite.

    Example:
        Given:

        ``alpha = [0.1, 0.5]``

        ``beta = [1.0, 2.0, 3.0]``

        the returned starting points are:

        ``[0.1, 1.0]``

        ``[0.1, 2.0]``

        ``[0.1, 3.0]``

        ``[0.5, 1.0]``

        ``[0.5, 2.0]``

        ``[0.5, 3.0]``
    """

    if len(parameter_values) == 0:
        raise ValueError("Candidate values for at least one parameter are required.")

    value_arrays: list[FloatArray] = []

    for parameter_index, values in enumerate(parameter_values):
        values_array = np.asarray(values, dtype=np.float64)

        if values_array.ndim != 1:
            raise ValueError(
                "Candidate values for each parameter must be "
                f"one-dimensional. Parameter index: {parameter_index}"
            )

        if values_array.size == 0:
            raise ValueError(
                "Each parameter must have at least one candidate value. "
                f"Parameter index: {parameter_index}"
            )

        if not np.isfinite(values_array).all():
            raise ValueError(
                f"All candidate parameter values must be finite. Parameter index: {parameter_index}"
            )

        value_arrays.append(values_array)

    combinations = product(*value_arrays)

    return np.asarray(
        list(combinations),
        dtype=np.float64,
    )


def scale_unit_points_to_bounds(
    unit_points: FloatArray,
    bounds: ParameterBounds,
) -> FloatArray:
    """Scale points from the unit interval to parameter bounds.

    Each input coordinate must be in ``[0, 1]``.

    For example, a unit coordinate of ``0.5`` is mapped to the midpoint of the
    corresponding parameter interval.

    Args:
        unit_points: Two-dimensional array with values in ``[0, 1]``.
        bounds: One ``(lower_bound, upper_bound)`` pair per parameter.

    Returns:
        Scaled two-dimensional array with the same shape as ``unit_points``.

    Raises:
        ValueError: If the points are malformed, non-finite, outside the unit
            interval, or incompatible with the number of parameter bounds.
    """

    points = np.asarray(unit_points, dtype=np.float64)

    if points.ndim != 2:
        raise ValueError("Unit points must be a two-dimensional array.")

    if not np.isfinite(points).all():
        raise ValueError("All unit-point values must be finite.")

    if ((points < 0.0) | (points > 1.0)).any():
        raise ValueError("All unit-point values must be within [0, 1].")

    lower_bounds, upper_bounds = bounds_to_arrays(bounds)

    if points.shape[1] != lower_bounds.size:
        raise ValueError(
            "The number of point dimensions must match the number of "
            f"parameter bounds: got {points.shape[1]} dimensions and "
            f"{lower_bounds.size} bounds."
        )

    scaled_points = qmc.scale(
        sample=points,
        l_bounds=lower_bounds,
        u_bounds=upper_bounds,
    )

    return np.asarray(
        scaled_points,
        dtype=np.float64,
    )


def generate_sobol_starts(
    bounds: ParameterBounds,
    *,
    n_starts: int,
    rng: np.random.Generator | int | None = None,
    scramble: bool = True,
) -> FloatArray:
    """Generate Sobol starting points within parameter bounds.

    The number of starts must be a power of two because Sobol sequences have
    their strongest balance properties when generated with
    :meth:`scipy.stats.qmc.Sobol.random_base2`.

    Valid examples include:

    ``8, 16, 32, 64``

    Args:
        bounds: One ``(lower_bound, upper_bound)`` pair per parameter.
        n_starts: Number of starting points. Must be a positive power of two.
        rng: Optional random number generator or seed for the random number generator.
        scramble: Whether to scramble the Sobol sequence.

    Returns:
        Two-dimensional array with shape:

        ``(n_starts, number_of_parameters)``

    Raises:
        TypeError: If ``n_starts`` is not an integer.
        ValueError: If ``n_starts`` is not a positive power of two or if the
            supplied parameter bounds are invalid.
    """

    if not isinstance(n_starts, int):
        raise TypeError("n_starts must be an integer.")

    if n_starts <= 0:
        raise ValueError("n_starts must be greater than zero.")

    if n_starts & (n_starts - 1):
        raise ValueError("n_starts must be a power of two, such as 8, 16, 32, or 64.")

    lower_bounds, upper_bounds = bounds_to_arrays(bounds)
    n_parameters = lower_bounds.size

    exponent = n_starts.bit_length() - 1

    sampler = qmc.Sobol(
        d=n_parameters,
        scramble=scramble,
        rng=rng,
    )

    unit_points = sampler.random_base2(m=exponent)

    scaled_points = qmc.scale(
        sample=unit_points,
        l_bounds=lower_bounds,
        u_bounds=upper_bounds,
    )

    return np.asarray(
        scaled_points,
        dtype=np.float64,
    )
