"""Two-parameter Q-learning model for the Iowa Gambling Task."""

import numpy as np
from scipy.special import logsumexp

from igt.initialization import generate_grid_starts

from .base import (
    ComputationalModel,
    FloatArray,
    ParameterBounds,
    SubjectData,
)


class QLearningModel(ComputationalModel):
    """Four-deck delta-rule Q-learning model with a softmax choice rule.

    Parameters, in optimizer-array order:

    1. ``learning_rate``
    2. ``inverse_temperature``

    The model evaluates a coarse parameter grid and returns the best grid
    point as its single local-optimizer starting point.
    """

    def __init__(
        self,
        *,
        n_starts: int = 1,
        learning_rate_grid_size: int = 21,
        inverse_temperature_grid_size: int = 21,
        max_inverse_temperature: float = 20.0,
        payoff_scale: float = 100.0,
    ) -> None:
        if learning_rate_grid_size < 2:
            raise ValueError("learning_rate_grid_size must be at least 2.")

        if inverse_temperature_grid_size < 2:
            raise ValueError("inverse_temperature_grid_size must be at least 2.")

        if not np.isfinite(max_inverse_temperature):
            raise ValueError("max_inverse_temperature must be finite.")

        if max_inverse_temperature <= 0.0:
            raise ValueError("max_inverse_temperature must be greater than zero.")

        if not np.isfinite(payoff_scale):
            raise ValueError("payoff_scale must be finite.")

        if payoff_scale <= 0.0:
            raise ValueError("payoff_scale must be greater than zero.")

        self._max_inverse_temperature = float(max_inverse_temperature)
        self._payoff_scale = float(payoff_scale)

        learning_rates = np.linspace(
            0.0,
            1.0,
            num=learning_rate_grid_size,
            dtype=np.float64,
        )

        inverse_temperatures = np.linspace(
            0.0,
            self._max_inverse_temperature,
            num=inverse_temperature_grid_size,
            dtype=np.float64,
        )

        self._grid = generate_grid_starts([learning_rates, inverse_temperatures])

        if n_starts <= 0 or n_starts > self._grid.shape[0]:
            n_starts = 1

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
            (0.0, 1.0),
            (0.0, self._max_inverse_temperature),
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
        """Return the grid point with the lowest unoptimized NLL.

        A two-dimensional array with shape ``(n_starts, 2)`` is returned so the
        generic fitter can handle this model and multistart models through
        the same interface even when ``n_starts`` is 1.
        """

        nll_values = np.fromiter(
            (self.negative_log_likelihood(start, data) for start in self._grid),
            dtype=np.float64,
            count=self._grid.shape[0],
        )

        best_indices = np.argpartition(nll_values, self._n_starts - 1)[: self._n_starts]

        return self._grid[best_indices]
