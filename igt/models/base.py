"""Abstract interface shared by computational models fitted to IGT participants.

The base class standardizes model naming, parameter metadata and bounds, likelihood
evaluation, optimizer-start generation, and reusable parameter validation so the
execution layer can fit different models through one interface.
"""

from abc import ABC, abstractmethod

import numpy as np

from igt.typing import Float1DArray, Float2DArray, ParameterBounds

from .typing import SubjectData


class ComputationalModel(ABC):
    """Abstract interface shared by computational models fitted to IGT subjects.

    Concrete models provide stable names, optimizer parameter ordering and
    bounds, a subject-level negative log-likelihood, and model-specific starting
    points. Fitting code relies on this interface rather than on model-specific
    implementations.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return the model's stable programmatic name.

        Returns:
            Canonical model name used in result tables, registries, and logs.
        """

    @property
    def name(self) -> str:
        """Return the model's stable programmatic name.

        Returns:
            Canonical model name returned by [`get_name`][igt.models.base.ComputationalModel.get_name].
        """

        return self.get_name()

    @classmethod
    @abstractmethod
    def get_parameter_names(cls) -> tuple[str, ...]:
        """Return free-parameter names in optimizer-array order.

        Returns:
            Ordered parameter names expected by likelihood, bounds, and result-record
            operations.
        """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return free-parameter names in optimizer-array order.

        Returns:
            Ordered parameter names returned by
            [`get_parameter_names`][igt.models.base.ComputationalModel.get_parameter_names].
        """

        return self.get_parameter_names()

    @property
    @abstractmethod
    def parameter_bounds(self) -> ParameterBounds:
        """Return parameter bounds in optimizer-array order.

        Returns:
            Lower and upper bounds aligned with [`parameter_names`][igt.models.base.ComputationalModel.parameter_names].
        """

    @abstractmethod
    def negative_log_likelihood(
        self,
        parameters: Float1DArray,
        data: SubjectData,
    ) -> float:
        """Evaluate the negative log-likelihood of one participant's observed choices.

        Args:
            parameters: Candidate parameter vector in `parameter_names` order.
            data: Validated subject choices and monetary outcomes.

        Returns:
            Negative log-likelihood under the candidate parameter vector.
        """

    @abstractmethod
    def starting_points(
        self,
        data: SubjectData,
    ) -> Float2DArray:
        """Generate local-optimizer starting points for one participant.

        Args:
            data: Validated subject data that may be used to choose model-specific starts.

        Returns:
            Two-dimensional matrix with one parameter vector per row and columns in
            `parameter_names` order.
        """

    @classmethod
    def get_n_parameters(cls) -> int:
        """Return the number of free parameters defined by the model class.

        Returns:
            Number of names returned by `get_parameter_names`.
        """

        return len(cls.get_parameter_names())

    @property
    def n_parameters(self) -> int:
        """Return the number of free parameters defined by the model.

        Returns:
            Number of model parameters.
        """

        return self.get_n_parameters()

    @property
    def _parameter_name_to_bound_map(self) -> dict[str, tuple[float, float]]:
        """Map each parameter name to its configured lower and upper bounds.

        Returns:
            Dictionary whose keys follow `parameter_names` and whose values are the
            corresponding entries from `parameter_bounds`.
        """

        return dict(zip(self.parameter_names, self.parameter_bounds, strict=True))

    @classmethod
    def validate_parameters(
        cls,
        parameters: Float1DArray,
    ) -> Float1DArray:
        """Validate and normalize one model parameter vector.

        Args:
            parameters: Candidate parameter vector in model-defined optimizer order.

        Returns:
            Finite one-dimensional floating-point parameter array with exactly the
            model's expected number of elements.

        Raises:
            ValueError: If the vector has the wrong shape or contains non-finite values.
        """

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
        parameters: Float1DArray,
    ) -> bool:
        """Check whether every parameter lies within the model's configured bounds.

        Args:
            parameters: Candidate parameter vector in model-defined optimizer order.

        Returns:
            `True` when every value lies inclusively between its lower and upper bound;
            otherwise `False`.

        Raises:
            ValueError: If `parameters` fails the common parameter-vector validation.
        """

        parameter_array = self.validate_parameters(parameters)

        for value, (lower_bound, upper_bound) in zip(
            parameter_array,
            self.parameter_bounds,
            strict=True,
        ):
            if value < lower_bound or value > upper_bound:
                return False

        return True
