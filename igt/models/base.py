"""Shared interfaces and data structures for computational models."""

from abc import ABC, abstractmethod

import numpy as np

from igt.typing import FloatArray, ParameterBounds

from .typing import SubjectData


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

    @property
    def _parameter_name_to_bound_map(self) -> dict[str, tuple[float, float]]:
        """Return a mapping from parameter names to their bounds."""

        return dict(zip(self.parameter_names, self.parameter_bounds, strict=True))

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
