"""Prospect Valence Learning Delta model for the Iowa Gambling Task.

The model transforms monetary outcomes with outcome sensitivity and loss aversion,
updates the chosen deck with a delta rule, and converts learned expectancies to choice
probabilities using response consistency. Sobol points provide multistart initialization.
"""

from collections.abc import Iterable
from typing import cast

import numpy as np
from scipy.special import logsumexp

from igt.constants.config import DEFAULT_N_PVL_STARTS, FIXED_SEED
from igt.constants.models import (
    N_IGT_DECKS,
    PAYOFF_SCALE,
    PVL_DELTA_MODEL_NAME,
    PVL_DELTA_PARAMETER_BOUNDS,
    PVL_DELTA_PARAMETER_NAMES,
)
from igt.initialization import generate_sobol_starts

from .base import (
    ComputationalModel,
    Float1DArray,
    Float2DArray,
    ParameterBounds,
)
from .typing import SubjectData


class PVLDeltaModel(ComputationalModel):
    """Prospect Valence Learning Delta model for the Iowa Gambling Task.

    Monetary outcomes are transformed by subjective outcome sensitivity and
    loss aversion before a delta-rule update of the chosen deck. Choice
    consistency is mapped to the softmax sensitivity used for deck selection.

    Parameters are ordered as `learning_rate`, `outcome_sensitivity`,
    `loss_aversion`, and `response_consistency`. Sobol points provide the
    multi-start initialization set.
    """

    def __init__(
        self,
        *,
        n_starts: int = DEFAULT_N_PVL_STARTS,
        rng: np.random.Generator | int | None = FIXED_SEED,
        scramble: bool = True,
        payoff_scale: float = PAYOFF_SCALE,
    ) -> None:
        """Initialize PVL-Delta fitting configuration and Sobol starting points.

        Args:
            n_starts: Number of Sobol starting points. The initializer requires
                a positive power of two.
            rng: Random generator or seed used by the scrambled Sobol sequence.
            scramble: Whether to scramble the Sobol sequence.
            payoff_scale: Divisor applied to monetary outcomes before utility
                transformation.

        Raises:
            TypeError: If the Sobol-start configuration has an invalid type.
            ValueError: If the payoff scale or Sobol-start configuration is
                invalid.
        """
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

    @classmethod
    def get_name(cls) -> str:
        """Return the canonical PVL-Delta model name.

        Returns:
            Stable PVL-Delta model identifier used throughout the project.
        """

        return PVL_DELTA_MODEL_NAME

    @classmethod
    def get_parameter_names(cls) -> tuple[str, ...]:
        """Return PVL-Delta parameter names in optimizer-array order.

        Returns:
            `learning_rate`, `outcome_sensitivity`, `loss_aversion`, and
            `response_consistency`, in that order.
        """

        return PVL_DELTA_PARAMETER_NAMES

    @property
    def parameter_bounds(self) -> ParameterBounds:
        """Return the numerical bounds used for PVL-Delta optimization.

        The learning-rate endpoints use the project's small open-bound approximation;
        the remaining bounds come directly from the model constants.

        Returns:
            Bounds aligned with the PVL-Delta parameter order.
        """

        return PVL_DELTA_PARAMETER_BOUNDS

    def negative_log_likelihood(
        self,
        parameters: Float1DArray,
        data: SubjectData,
    ) -> float:
        """Compute the PVL-Delta negative log-likelihood for one subject.

        On each trial, response consistency determines softmax sensitivity,
        the observed choice contributes its negative log-probability, the net
        outcome is transformed by outcome sensitivity and loss aversion, and
        only the chosen deck expectancy is updated.

        Args:
            parameters: Learning rate, outcome sensitivity, loss aversion, and
                response consistency in optimizer order.
            data: Validated subject choices and outcomes.

        Returns:
            Negative log-likelihood of the observed choice sequence, or
            positive infinity when the parameter vector is outside bounds.
        """

        parameter_array = self.validate_parameters(parameters)

        if not self.parameters_within_bounds(parameter_array):
            return float("inf")

        learning_rate = float(parameter_array[0])
        outcome_sensitivity = float(parameter_array[1])
        loss_aversion = float(parameter_array[2])
        response_consistency = float(parameter_array[3])

        theta = (3.0**response_consistency) - 1.0

        expectancies = np.zeros(N_IGT_DECKS, dtype=np.float64)
        scaled_outcomes = data.outcomes / self._payoff_scale

        negative_log_likelihood = 0.0

        for choice, outcome in zip(
            cast(Iterable[np.int64], data.choices),
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
    ) -> Float2DArray:
        """Return the precomputed Sobol optimizer starting points.

        The subject data argument is accepted to satisfy the common
        [ComputationalModel][igt.models.base.ComputationalModel] interface;
        PVL-Delta starts depend only on parameter bounds and Sobol settings.

        Args:
            data: Subject data required by the shared model interface.

        Returns:
            A copy of the Sobol starting-point matrix.
        """

        _ = data
        return self._starts.copy()
