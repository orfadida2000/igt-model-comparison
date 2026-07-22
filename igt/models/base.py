"""Shared interfaces and data structures for computational models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from igt.typing import FloatArray, IntArray, ParameterBounds


@dataclass(frozen=True, slots=True)
class SubjectData:
    """Trial-level Iowa Gambling Task data for one subject.

    Choices use the dataset's deck coding:

    - 1: deck A
    - 2: deck B
    - 3: deck C
    - 4: deck D

    Wins are expected to be non-negative and losses are expected to be
    non-positive.
    """

    choices: IntArray
    wins: FloatArray
    losses: FloatArray

    def __post_init__(self) -> None:
        """Validate the subject arrays."""

        if self.choices.ndim != 1:
            raise ValueError("choices must be a one-dimensional array.")

        if self.wins.ndim != 1:
            raise ValueError("wins must be a one-dimensional array.")

        if self.losses.ndim != 1:
            raise ValueError("losses must be a one-dimensional array.")

        n_trials = self.choices.size

        if n_trials == 0:
            raise ValueError("SubjectData must contain at least one trial.")

        if self.wins.size != n_trials or self.losses.size != n_trials:
            raise ValueError("choices, wins, and losses must contain the same number of trials.")

        if not np.isfinite(self.wins).all():
            raise ValueError("wins contains missing or non-finite values.")

        if not np.isfinite(self.losses).all():
            raise ValueError("losses contains missing or non-finite values.")

        valid_choices = np.isin(
            self.choices,
            np.array([1, 2, 3, 4], dtype=np.int64),
        )

        if not valid_choices.all():
            invalid_choices = np.unique(self.choices[~valid_choices]).tolist()

            raise ValueError(f"choices contains invalid deck codes: {invalid_choices}")

        if (self.wins < 0.0).any():
            raise ValueError("wins must contain only non-negative values.")

        if (self.losses > 0.0).any():
            raise ValueError("losses must contain only non-positive values.")

    @property
    def outcomes(self) -> FloatArray:
        """Return the net outcome for every trial."""

        return self.wins + self.losses

    @property
    def n_trials(self) -> int:
        """Return the number of trials."""

        return int(self.choices.size)


class ComputationalModel(ABC):
    """Common interface implemented by every computational model."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model's stable programmatic name."""

    @property
    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """Return parameter names in optimizer-array order."""

    @property
    @abstractmethod
    def parameter_bounds(self) -> ParameterBounds:
        """Return parameter bounds in optimizer-array order."""

    @abstractmethod
    def negative_log_likelihood(
        self,
        parameters: FloatArray,
        data: SubjectData,
    ) -> float:
        """Return the negative log-likelihood for one subject."""

    @abstractmethod
    def starting_points(
        self,
        data: SubjectData,
    ) -> FloatArray:
        """Return local-optimizer starts.

        The returned array must have shape:

            (number_of_starts, number_of_parameters)
        """

    @property
    def n_parameters(self) -> int:
        """Return the number of free model parameters."""

        return len(self.parameter_names)

    def validate_parameters(
        self,
        parameters: FloatArray,
    ) -> FloatArray:
        """Validate and return one optimizer parameter vector."""

        parameter_array = np.asarray(
            parameters,
            dtype=np.float64,
        )

        expected_shape = (self.n_parameters,)

        if parameter_array.shape != expected_shape:
            raise ValueError(
                "Expected a parameter vector with shape "
                f"{expected_shape}, got {parameter_array.shape}."
            )

        if not np.isfinite(parameter_array).all():
            raise ValueError("Model parameters must contain only finite values.")

        return parameter_array

    def parameters_within_bounds(
        self,
        parameters: FloatArray,
    ) -> bool:
        """Return whether every parameter is inside its model bound."""

        parameter_array = self.validate_parameters(parameters)

        for value, (lower_bound, upper_bound) in zip(
            parameter_array,
            self.parameter_bounds,
            strict=True,
        ):
            if value < lower_bound or value > upper_bound:
                return False

        return True
