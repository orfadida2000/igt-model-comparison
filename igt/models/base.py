"""Shared interfaces and data structures for computational models."""

from abc import ABC, abstractmethod

import numpy as np

from igt.typing import FloatArray, ParameterBounds

from .typing import SubjectData


class ComputationalModel(ABC):
    """Common interface implemented by every computational model."""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return the model's stable programmatic name."""

    @property
    def name(self) -> str:
        """Return the model's stable programmatic name."""

        return self.get_name()

    @classmethod
    @abstractmethod
    def get_parameter_names(cls) -> tuple[str, ...]:
        """Return parameter names in optimizer-array order."""

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return parameter names in optimizer-array order."""

        return self.get_parameter_names()

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

    @classmethod
    def get_n_parameters(cls) -> int:
        """Return the number of free model parameters."""

        return len(cls.get_parameter_names())

    @property
    def n_parameters(self) -> int:
        """Return the number of free model parameters."""

        return self.get_n_parameters()

    @property
    def _parameter_name_to_bound_map(self) -> dict[str, tuple[float, float]]:
        """Return a mapping from parameter names to their bounds."""

        return dict(zip(self.parameter_names, self.parameter_bounds, strict=True))

    @classmethod
    def validate_parameters(
        cls,
        parameters: FloatArray,
    ) -> FloatArray:
        """Validate and return one optimizer parameter vector."""

        parameter_array = np.asarray(
            parameters,
            dtype=np.float64,
        )

        expected_shape = (cls.get_n_parameters(),)

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
