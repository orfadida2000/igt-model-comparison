"""Configuration for result analysis and visualization."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from igt.constants.models import DEFAULT_MODEL_PARAMETER_BOUNDS
from igt.subject_selection import _validate_nonnegative_finite_float
from igt.typing import ModelParameterBounds, NamedParameterBounds, ParameterBound


def _normalize_parameter_bound(
    bound: ParameterBound,
    *,
    model_name: str,
    parameter_name: str,
) -> ParameterBound:
    """Validate and normalize one parameter-bound pair."""

    if not isinstance(bound, tuple) or len(bound) != 2:
        raise TypeError(
            "Each parameter bound must be a two-item tuple. "
            f"Model: {model_name!r}; parameter: {parameter_name!r}."
        )

    lower_bound, upper_bound = bound

    if (
        isinstance(lower_bound, (bool, np.bool_))
        or not isinstance(lower_bound, Real)
        or isinstance(upper_bound, (bool, np.bool_))
        or not isinstance(upper_bound, Real)
    ):
        raise TypeError(
            "Parameter bounds must contain real numbers. "
            f"Model: {model_name!r}; parameter: {parameter_name!r}."
        )

    normalized_lower_bound = float(lower_bound)
    normalized_upper_bound = float(upper_bound)

    if not np.isfinite(normalized_lower_bound) or not np.isfinite(normalized_upper_bound):
        raise ValueError(
            "Parameter bounds must be finite. "
            f"Model: {model_name!r}; parameter: {parameter_name!r}."
        )

    if normalized_lower_bound > normalized_upper_bound:
        raise ValueError(
            "A parameter lower bound must not exceed its upper bound. "
            f"Model: {model_name!r}; parameter: {parameter_name!r}."
        )

    return normalized_lower_bound, normalized_upper_bound


def _normalize_parameter_bounds(
    parameter_bounds: ModelParameterBounds,
) -> ModelParameterBounds:
    """Validate and freeze model-to-parameter bound mappings."""

    if not isinstance(parameter_bounds, Mapping):
        raise TypeError("parameter_bounds must be a mapping.")

    normalized_models: dict[str, NamedParameterBounds] = {}

    for raw_model_name, raw_model_bounds in parameter_bounds.items():
        if not isinstance(raw_model_name, str):
            raise TypeError("Parameter-bound model names must be strings.")

        model_name = raw_model_name.strip()

        if not model_name:
            raise ValueError("Parameter-bound model names must not be empty.")

        if not isinstance(raw_model_bounds, Mapping):
            raise TypeError(f"Parameter bounds for model {model_name!r} must be a mapping.")

        normalized_model_bounds: dict[str, ParameterBound] = {}

        for raw_parameter_name, raw_bound in raw_model_bounds.items():
            if not isinstance(raw_parameter_name, str):
                raise TypeError(f"Parameter names for model {model_name!r} must be strings.")

            parameter_name = raw_parameter_name.strip()

            if not parameter_name:
                raise ValueError(f"Parameter names for model {model_name!r} must not be empty.")

            normalized_model_bounds[parameter_name] = _normalize_parameter_bound(
                raw_bound,
                model_name=model_name,
                parameter_name=parameter_name,
            )

        if not normalized_model_bounds:
            raise ValueError(f"At least one parameter bound is required for model {model_name!r}.")

        normalized_models[model_name] = MappingProxyType(normalized_model_bounds)

    if not normalized_models:
        raise ValueError("At least one model parameter-bound mapping is required.")

    return MappingProxyType(normalized_models)


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Configuration controlling result validation, tables, and figures."""

    figure_formats: tuple[str, ...] = ("png",)
    figure_dpi: int = 300
    histogram_bins: int | str = "auto"
    confidence_level: float = 0.95
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 42
    numeric_tolerance: float = 1e-10
    boundary_tolerance: float = 1e-8
    parameter_bounds: ModelParameterBounds = DEFAULT_MODEL_PARAMETER_BOUNDS

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""

        if not isinstance(self.figure_formats, tuple):
            raise TypeError("figure_formats must be a tuple of strings.")

        if not self.figure_formats:
            raise ValueError("At least one figure format is required.")

        normalized_formats: list[str] = []

        for figure_format in self.figure_formats:
            if not isinstance(figure_format, str):
                raise TypeError("Every figure format must be a string.")

            normalized_format = figure_format.strip().lower().lstrip(".")

            if not normalized_format:
                raise ValueError("Figure formats must not be empty.")

            normalized_formats.append(normalized_format)

        if len(set(normalized_formats)) != len(normalized_formats):
            raise ValueError("Figure formats must be unique.")

        object.__setattr__(self, "figure_formats", tuple(normalized_formats))

        if isinstance(self.figure_dpi, (bool, np.bool_)) or not isinstance(
            self.figure_dpi,
            Integral,
        ):
            raise TypeError("figure_dpi must be an integer.")

        if self.figure_dpi <= 0:
            raise ValueError("figure_dpi must be positive.")

        if isinstance(self.histogram_bins, (bool, np.bool_)):
            raise TypeError("histogram_bins must be a positive integer or string.")

        if isinstance(self.histogram_bins, (int, np.integer)):
            normalized_histogram_bins = int(self.histogram_bins)

            if normalized_histogram_bins <= 0:
                raise ValueError("histogram_bins must be positive.")

            object.__setattr__(
                self,
                "histogram_bins",
                normalized_histogram_bins,
            )
        elif isinstance(self.histogram_bins, str):
            normalized_histogram_bins = self.histogram_bins.strip()

            if not normalized_histogram_bins:
                raise ValueError("histogram_bins must not be empty.")

            object.__setattr__(
                self,
                "histogram_bins",
                normalized_histogram_bins,
            )
        else:
            raise TypeError("histogram_bins must be a positive integer or string.")

        if isinstance(self.confidence_level, (bool, np.bool_)) or not isinstance(
            self.confidence_level,
            Real,
        ):
            raise TypeError("confidence_level must be a real number.")

        normalized_confidence_level = float(self.confidence_level)

        if not np.isfinite(normalized_confidence_level):
            raise ValueError("confidence_level must be finite.")

        if not 0.0 < normalized_confidence_level < 1.0:
            raise ValueError("confidence_level must be strictly between 0 and 1.")

        object.__setattr__(
            self,
            "confidence_level",
            normalized_confidence_level,
        )

        for parameter_name in ("bootstrap_resamples", "bootstrap_seed"):
            value = getattr(self, parameter_name)

            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{parameter_name} must be an integer.")

            normalized_value = int(value)

            if parameter_name == "bootstrap_resamples" and normalized_value <= 0:
                raise ValueError("bootstrap_resamples must be positive.")

            if parameter_name == "bootstrap_seed" and normalized_value < 0:
                raise ValueError("bootstrap_seed must be non-negative.")

            object.__setattr__(self, parameter_name, normalized_value)

        for parameter_name in ("numeric_tolerance", "boundary_tolerance"):
            value = getattr(self, parameter_name)

            normalized_value = _validate_nonnegative_finite_float(
                value,
                parameter_name=parameter_name,
            )

            object.__setattr__(self, parameter_name, normalized_value)

        object.__setattr__(
            self,
            "parameter_bounds",
            _normalize_parameter_bounds(self.parameter_bounds),
        )
