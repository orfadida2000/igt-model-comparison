"""Two-parameter Q-learning model for the Iowa Gambling Task.

The model learns chosen-deck action values with a delta rule and maps them to choice
probabilities through a softmax inverse temperature. Its multistart initialization
uses a regular parameter grid and selects distinct grid-local likelihood minima.
"""

from collections.abc import Iterable
from math import ceil
from typing import cast

import numpy as np
from scipy.special import logsumexp

from igt.constants.config import DEFAULT_N_Q_STARTS
from igt.constants.models import (
    DEFAULT_MAX_INVERSE_TEMPERATURE,
    DEFAULT_Q_INVERSE_TEMPERATURE_GRID_STEP,
    DEFAULT_Q_LEARNING_PARAMETER_BOUNDS,
    DEFAULT_Q_LEARNING_RATE_GRID_SIZE,
    INVERSE_TEMPERATURE_PARAMETER_NAME,
    LEARNING_RATE_PARAMETER_NAME,
    MIN_INVERSE_TEMPERATURE,
    N_IGT_DECKS,
    PAYOFF_SCALE,
    Q_LEARNING_MODEL_NAME,
    Q_LEARNING_PARAMETER_NAMES,
)
from igt.initialization import (
    generate_grid_starts,
    select_grid_local_minimum_indices,
)

from .base import (
    ComputationalModel,
    Float1DArray,
    Float2DArray,
    ParameterBounds,
)
from .typing import SubjectData


class QLearningModel(ComputationalModel):
    """Delta-rule Q-learning model with a softmax choice rule for the IGT.

    The model maintains one learned value for each of the four decks. After a
    choice, only the selected deck is updated using the scaled objective outcome
    and the learning-rate prediction error. Choice probabilities are produced
    by a softmax parameterized by inverse temperature.

    Parameters are ordered as `learning_rate` and `inverse_temperature`.
    Starting points are selected from distinct local minima on a model-specific
    objective grid before L-BFGS-B optimization.
    """

    def __init__(
        self,
        *,
        n_starts: int = DEFAULT_N_Q_STARTS,
        learning_rate_grid_size: int = DEFAULT_Q_LEARNING_RATE_GRID_SIZE,
        inverse_temperature_grid_size: int | None = None,
        max_inverse_temperature: float = DEFAULT_MAX_INVERSE_TEMPERATURE,
        payoff_scale: float = PAYOFF_SCALE,
    ) -> None:
        """Initialize Q-learning bounds and the objective-screening grid.

        Args:
            n_starts: Maximum number of distinct grid-local minima retained as
                optimizer starting points.
            learning_rate_grid_size: Number of points in the quadratic base
                learning-rate grid before low-value refinements are appended.
            inverse_temperature_grid_size: Optional number of points in the
                linear base inverse-temperature grid. When omitted, the size is
                chosen to keep approximately unit spacing.
            max_inverse_temperature: Upper bound and grid maximum for inverse
                temperature.
            payoff_scale: Divisor applied to monetary outcomes before the
                Q-value update.

        Raises:
            TypeError: If a grid-size or start-count argument has an invalid
                type.
            ValueError: If a count, bound, or payoff scale is invalid, or if
                `n_starts` exceeds the number of grid points.
        """
        if not isinstance(n_starts, int) or isinstance(n_starts, bool):
            raise TypeError("n_starts must be an integer.")

        if n_starts <= 0:
            raise ValueError("n_starts must be greater than zero.")

        if not isinstance(learning_rate_grid_size, int) or isinstance(
            learning_rate_grid_size, bool
        ):
            raise TypeError("learning_rate_grid_size must be an integer.")

        if learning_rate_grid_size < 2:
            raise ValueError("learning_rate_grid_size must be at least 2.")

        if inverse_temperature_grid_size is not None and (
            not isinstance(inverse_temperature_grid_size, int)
            or isinstance(inverse_temperature_grid_size, bool)
        ):
            raise TypeError("inverse_temperature_grid_size must be an integer or None.")

        if inverse_temperature_grid_size is not None and inverse_temperature_grid_size < 2:
            raise ValueError("inverse_temperature_grid_size must be at least 2.")

        if not np.isfinite(max_inverse_temperature):
            raise ValueError("max_inverse_temperature must be finite.")

        _parameter_name_to_bound_map = dict(
            zip(self.parameter_names, self.parameter_bounds, strict=True)
        )

        if (
            max_inverse_temperature
            <= _parameter_name_to_bound_map[INVERSE_TEMPERATURE_PARAMETER_NAME][0]
        ):
            raise ValueError(
                f"max_inverse_temperature must be greater than {_parameter_name_to_bound_map[INVERSE_TEMPERATURE_PARAMETER_NAME][0]}."
            )

        if not np.isfinite(payoff_scale):
            raise ValueError("payoff_scale must be finite.")

        if payoff_scale <= 0.0:
            raise ValueError("payoff_scale must be greater than zero.")

        self._max_inverse_temperature = float(max_inverse_temperature)
        self._payoff_scale = float(payoff_scale)

        unit_learning_rates = np.linspace(
            0.0,
            1.0,
            num=learning_rate_grid_size,
            dtype=np.float64,
        )
        computed_learning_rates = _parameter_name_to_bound_map[LEARNING_RATE_PARAMETER_NAME][0] + (
            (
                _parameter_name_to_bound_map[LEARNING_RATE_PARAMETER_NAME][1]
                - _parameter_name_to_bound_map[LEARNING_RATE_PARAMETER_NAME][0]
            )
            * np.square(unit_learning_rates)
        )

        additional_low_learning_rates = np.array(
            [0.0001, 0.0003, 0.001, 0.003],
            dtype=np.float64,
        )
        additional_low_learning_rates = additional_low_learning_rates[
            (
                additional_low_learning_rates
                >= _parameter_name_to_bound_map[LEARNING_RATE_PARAMETER_NAME][0]
            )
            & (
                additional_low_learning_rates
                <= _parameter_name_to_bound_map[LEARNING_RATE_PARAMETER_NAME][1]
            )
        ]
        learning_rates = np.concatenate(
            [
                computed_learning_rates,
                additional_low_learning_rates,
            ]
        )

        if inverse_temperature_grid_size is None:
            inverse_temperature_grid_size = (
                ceil(
                    (
                        self._max_inverse_temperature
                        - _parameter_name_to_bound_map[INVERSE_TEMPERATURE_PARAMETER_NAME][0]
                    )
                    / DEFAULT_Q_INVERSE_TEMPERATURE_GRID_STEP
                )
                + 1
            )

        computed_inverse_temperatures = np.linspace(
            _parameter_name_to_bound_map[INVERSE_TEMPERATURE_PARAMETER_NAME][0],
            self._max_inverse_temperature,
            num=inverse_temperature_grid_size,
            dtype=np.float64,
        )

        additional_low_inverse_temperatures = np.array(
            [0.1, 0.25, 0.5, 0.75],
            dtype=np.float64,
        )
        additional_low_inverse_temperatures = additional_low_inverse_temperatures[
            (
                additional_low_inverse_temperatures
                >= _parameter_name_to_bound_map[INVERSE_TEMPERATURE_PARAMETER_NAME][0]
            )
            & (additional_low_inverse_temperatures <= self._max_inverse_temperature)
        ]

        inverse_temperatures = np.concatenate(
            [
                computed_inverse_temperatures,
                additional_low_inverse_temperatures,
            ]
        )

        self._grid_shape = (
            learning_rates.size,
            inverse_temperatures.size,
        )
        self._grid = generate_grid_starts([learning_rates, inverse_temperatures])

        if n_starts > self._grid.shape[0]:
            raise ValueError(
                "n_starts must not exceed the number of Q-learning grid points: "
                f"got {n_starts} and maximum {self._grid.shape[0]}."
            )

        self._n_starts = n_starts

    @classmethod
    def get_name(cls) -> str:
        """Return the canonical Q-learning model name.

        Returns:
            Stable Q-learning model identifier used throughout the project.
        """

        return Q_LEARNING_MODEL_NAME

    @classmethod
    def get_parameter_names(cls) -> tuple[str, ...]:
        """Return Q-learning parameter names in optimizer-array order.

        Returns:
            `learning_rate` followed by `inverse_temperature`.
        """

        return Q_LEARNING_PARAMETER_NAMES

    @property
    def parameter_bounds(self) -> ParameterBounds:
        """Return the Q-learning parameter bounds.

        The learning-rate interval is fixed by the project constants, while the upper
        inverse-temperature bound is the value configured for this model instance.

        Returns:
            Bounds aligned with `learning_rate` and `inverse_temperature`.
        """

        return (
            DEFAULT_Q_LEARNING_PARAMETER_BOUNDS[0],
            (
                MIN_INVERSE_TEMPERATURE,
                self._max_inverse_temperature,
            ),
        )

    def negative_log_likelihood(
        self,
        parameters: Float1DArray,
        data: SubjectData,
    ) -> float:
        """Compute the Q-learning negative log-likelihood for one subject.

        On each trial, the current deck values define softmax choice
        probabilities; the observed choice contributes its negative
        log-probability; and only the chosen deck is updated by the scaled
        outcome prediction error.

        Args:
            parameters: Learning rate and inverse temperature in optimizer
                order.
            data: Validated subject choices and outcomes.

        Returns:
            Negative log-likelihood of the observed choice sequence, or
            positive infinity when the parameter vector is outside bounds.
        """

        parameter_array = self.validate_parameters(parameters)

        if not self.parameters_within_bounds(parameter_array):
            return float("inf")

        learning_rate = float(parameter_array[0])
        inverse_temperature = float(parameter_array[1])

        deck_values = np.zeros(N_IGT_DECKS, dtype=np.float64)
        scaled_outcomes = data.outcomes / self._payoff_scale

        negative_log_likelihood = 0.0

        for choice, outcome in zip(
            cast(Iterable[np.int64], data.choices),
            scaled_outcomes,
            strict=True,
        ):
            chosen_deck = int(choice) - 1

            logits = inverse_temperature * deck_values
            chosen_log_probability = logits[chosen_deck] - logsumexp(logits)

            negative_log_likelihood -= float(chosen_log_probability)

            prediction_error = float(outcome) - deck_values[chosen_deck]

            deck_values[chosen_deck] += learning_rate * prediction_error

        return negative_log_likelihood

    def starting_points(self, data: SubjectData) -> Float2DArray:
        """Select distinct grid-local NLL minima as Q-learning optimizer starts.

        Connected tied minima are represented by one point so a flat plateau
        does not contribute redundant starts. Fewer than the configured maximum
        may be returned when fewer distinct local-minimum regions exist.

        Args:
            data: Subject data used to evaluate every grid point.

        Returns:
            Selected parameter vectors ordered from lower to higher grid NLL.
        """

        nll_values = np.fromiter(
            (self.negative_log_likelihood(start, data) for start in self._grid),
            dtype=np.float64,
            count=self._grid.shape[0],
        )

        best_indices = select_grid_local_minimum_indices(
            nll_values,
            grid_shape=self._grid_shape,
            max_starts=self._n_starts,
        )

        return self._grid[best_indices]
