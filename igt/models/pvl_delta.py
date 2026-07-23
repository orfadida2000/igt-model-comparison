"""Prospect Valence Learning model with the delta update rule."""

import numpy as np
from scipy.special import logsumexp

from igt.initialization import generate_sobol_starts

from .base import (
    ComputationalModel,
    FloatArray,
    ParameterBounds,
    SubjectData,
)


class PVLDeltaModel(ComputationalModel):
    """Four-parameter PVL-Delta model for the Iowa Gambling Task.

    Parameters, in optimizer-array order:

    1. ``learning_rate`` (A)
    2. ``outcome_sensitivity`` (alpha)
    3. ``loss_aversion`` (lambda)
    4. ``response_consistency`` (c)

    Subjective utility is calculated from the net outcome. Only the chosen
    deck expectancy is updated. Choice probabilities use a softmax rule with:

        theta = 3**c - 1

    Sobol points are generated once when the model object is created and are
    reused as the local-optimizer starting points for every subject.
    """

    _OPEN_BOUND_EPSILON = 1e-6

    def __init__(
        self,
        *,
        n_starts: int = 32,
        rng: np.random.Generator | int | None = 42,
        scramble: bool = True,
        payoff_scale: float = 100.0,
    ) -> None:
        if not np.isfinite(payoff_scale):
            raise ValueError("payoff_scale must be finite.")

        if payoff_scale <= 0.0:
            raise ValueError("payoff_scale must be greater than zero.")

        self._payoff_scale = float(payoff_scale)

        self._starts = generate_sobol_starts(
            bounds=self.parameter_bounds,
            n_starts=n_starts,
            rng=rng,
            scramble=scramble,
        )

    @property
    def name(self) -> str:
        """Return the model name."""

        return "pvl_delta"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return parameter names in optimizer-array order."""

        return (
            "learning_rate",
            "outcome_sensitivity",
            "loss_aversion",
            "response_consistency",
        )

    @property
    def parameter_bounds(self) -> ParameterBounds:
        """Return numerically closed approximations of the model bounds."""

        epsilon = self._OPEN_BOUND_EPSILON

        return (
            (epsilon, 1.0 - epsilon),
            (epsilon, 2.0),
            (epsilon, 10.0),
            (0.0, 5.0),
        )

    def negative_log_likelihood(
        self,
        parameters: FloatArray,
        data: SubjectData,
    ) -> float:
        """Calculate the subject's negative log-likelihood.

        On each trial:

        1. Compute choice probabilities from the current deck expectancies.
        2. Add the observed choice's negative log-probability.
        3. Transform the trial's net payoff into subjective utility.
        4. Update only the chosen deck with the delta rule.
        """

        parameter_array = self.validate_parameters(parameters)

        if not self.parameters_within_bounds(parameter_array):
            return float("inf")

        learning_rate = float(parameter_array[0])
        outcome_sensitivity = float(parameter_array[1])
        loss_aversion = float(parameter_array[2])
        response_consistency = float(parameter_array[3])

        theta = (3.0**response_consistency) - 1.0

        expectancies = np.zeros(4, dtype=np.float64)
        scaled_outcomes = data.outcomes / self._payoff_scale

        negative_log_likelihood = 0.0

        for choice, outcome in zip(
            data.choices,
            scaled_outcomes,
            strict=True,
        ):
            chosen_deck = int(choice) - 1

            logits = theta * expectancies
            chosen_log_probability = logits[chosen_deck] - logsumexp(logits)

            negative_log_likelihood -= float(chosen_log_probability)

            numeric_outcome = float(outcome)

            if numeric_outcome >= 0.0:
                utility = numeric_outcome**outcome_sensitivity
            else:
                utility = -loss_aversion * ((-numeric_outcome) ** outcome_sensitivity)

            prediction_error = utility - expectancies[chosen_deck]

            expectancies[chosen_deck] += learning_rate * prediction_error

        return negative_log_likelihood

    def starting_points(
        self,
        data: SubjectData,
    ) -> FloatArray:
        """Return all Sobol starting points.

        ``data`` is accepted to satisfy the common model interface. PVL-Delta
        starting points depend only on the parameter bounds, not on a
        particular subject.
        """

        _ = data
        return self._starts.copy()
