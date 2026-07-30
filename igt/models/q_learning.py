"""Two-parameter Q-learning model for the Iowa Gambling Task."""

import numpy as np
from scipy.special import logsumexp

from igt.constants.config import (
    DEFAULT_N_Q_STARTS,
    GENERAL_MAX_LEARNING_RATE,
    GENERAL_MIN_LEARNING_RATE,
    MAX_INVERSE_TEMPERATURE,
    MIN_INVERSE_TEMPERATURE,
    PAYOFF_SCALE,
)
from igt.initialization import (
    generate_grid_starts,
    select_grid_local_minimum_indices,
)

from .base import (
    ComputationalModel,
    FloatArray,
    ParameterBounds,
)
from .typing import SubjectData


class QLearningModel(ComputationalModel):
    """Four-deck delta-rule Q-learning model with a softmax choice rule.

    Parameters, in optimizer-array order:

    1. ``learning_rate``
    2. ``inverse_temperature``

    The model evaluates a coarse parameter grid and selects up to ``n_starts``
    distinct grid-local minima as local-optimizer starting points.
    """

    def __init__(
        self,
        *,
        n_starts: int = DEFAULT_N_Q_STARTS,
        learning_rate_grid_size: int = 21,
        inverse_temperature_grid_size: int = 21,
        max_inverse_temperature: float = MAX_INVERSE_TEMPERATURE,
        payoff_scale: float = PAYOFF_SCALE,
    ) -> None:
        if not isinstance(n_starts, int) or isinstance(n_starts, bool):
            raise TypeError("n_starts must be an integer.")

        if n_starts <= 0:
            raise ValueError("n_starts must be greater than zero.")

        if learning_rate_grid_size < 2:
            raise ValueError("learning_rate_grid_size must be at least 2.")

        if inverse_temperature_grid_size < 2:
            raise ValueError("inverse_temperature_grid_size must be at least 2.")

        if not np.isfinite(max_inverse_temperature):
            raise ValueError("max_inverse_temperature must be finite.")

        if max_inverse_temperature <= MIN_INVERSE_TEMPERATURE:
            raise ValueError(
                f"max_inverse_temperature must be greater than {MIN_INVERSE_TEMPERATURE}."
            )

        if not np.isfinite(payoff_scale):
            raise ValueError("payoff_scale must be finite.")

        if payoff_scale <= 0.0:
            raise ValueError("payoff_scale must be greater than zero.")

        self._max_inverse_temperature = float(max_inverse_temperature)
        self._payoff_scale = float(payoff_scale)

        learning_rates = np.linspace(
            GENERAL_MIN_LEARNING_RATE,
            GENERAL_MAX_LEARNING_RATE,
            num=learning_rate_grid_size,
            dtype=np.float64,
        )

        inverse_temperatures = np.linspace(
            MIN_INVERSE_TEMPERATURE,
            self._max_inverse_temperature,
            num=inverse_temperature_grid_size,
            dtype=np.float64,
        )

        self._grid_shape = (
            learning_rate_grid_size,
            inverse_temperature_grid_size,
        )
        self._grid = generate_grid_starts([learning_rates, inverse_temperatures])

        if n_starts > self._grid.shape[0]:
            raise ValueError(
                "n_starts must not exceed the number of Q-learning grid points: "
                f"got {n_starts} and maximum {self._grid.shape[0]}."
            )

        self._n_starts = n_starts

    @property
    def name(self) -> str:
        """Return the model name."""

        return "q_learning"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return parameter names in optimizer-array order."""

        return (
            "learning_rate",
            "inverse_temperature",
        )

    @property
    def parameter_bounds(self) -> ParameterBounds:
        """Return the parameter bounds."""

        return (
            (GENERAL_MIN_LEARNING_RATE, GENERAL_MAX_LEARNING_RATE),
            (MIN_INVERSE_TEMPERATURE, self._max_inverse_temperature),
        )

    def negative_log_likelihood(
        self,
        parameters: FloatArray,
        data: SubjectData,
    ) -> float:
        """Calculate the subject's negative log-likelihood.

        On each trial:

        1. Compute softmax choice probabilities from the current deck values.
        2. Add the observed choice's negative log-probability.
        3. Update only the chosen deck using its prediction error.
        """

        parameter_array = self.validate_parameters(parameters)

        if not self.parameters_within_bounds(parameter_array):
            return float("inf")

        learning_rate = float(parameter_array[0])
        inverse_temperature = float(parameter_array[1])

        deck_values = np.zeros(4, dtype=np.float64)
        scaled_outcomes = data.outcomes / self._payoff_scale

        negative_log_likelihood = 0.0

        for choice, outcome in zip(
            data.choices,
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

    def starting_points(self, data: SubjectData) -> FloatArray:
        """Return up to ``n_starts`` distinct grid-local NLL minima.

        Every selected point is no worse than its immediate horizontal,
        vertical, and diagonal grid neighbors. Connected tied minima are
        collapsed to one representative so a flat plateau does not supply
        redundant starts.

        Fewer than ``n_starts`` points are returned when the grid contains
        fewer distinct local-minimum regions.
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
