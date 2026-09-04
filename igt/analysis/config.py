"""Configuration objects and normalization helpers for result analysis.

This module defines [`AnalysisConfig`][igt.analysis.config.AnalysisConfig] and
[`FigureStyleConfig`][igt.analysis.config.FigureStyleConfig], which centralize
plotting, bootstrap, numerical-tolerance, and parameter-bound settings used
throughout the `igt.analysis` subpackage.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
from matplotlib.typing import RcKeyType

from igt.constants.models import DEFAULT_MODEL_PARAMETER_BOUNDS
from igt.subject_selection import (
    _validate_nonnegative_finite_float,
    _validate_positive_finite_float,
)
from igt.typing import ModelParameterBounds, NamedParameterBounds, ParameterBound

FigureSizePreset = Literal["normal", "wide", "square"]

_DEFAULT_MATPLOTLIB_RC: Mapping[RcKeyType, object] = MappingProxyType(
    {
        # Embed TrueType fonts as Type 42 in PDF output. This setting is ignored by
        # raster backends such as PNG.
        "pdf.fonttype": 42,
    }
)


def _normalize_matplotlib_rc(
    rc_parameters: Mapping[RcKeyType, object],
) -> Mapping[RcKeyType, object]:
    """Validate, merge, and freeze Matplotlib rc parameter overrides.

    Project defaults are applied first and caller-supplied values can override them.
    In particular, PDF output defaults to TrueType/Type 42 font embedding.
    """

    if not isinstance(rc_parameters, Mapping):
        raise TypeError("matplotlib_rc must be a mapping.")

    normalized_parameters = dict(_DEFAULT_MATPLOTLIB_RC)

    for raw_name, value in rc_parameters.items():
        if not isinstance(raw_name, str):
            raise TypeError("Matplotlib rc parameter names must be strings.")

        name = raw_name.strip()

        if not name:
            raise ValueError("Matplotlib rc parameter names must not be empty.")

        normalized_parameters[cast(RcKeyType, name)] = value

    return MappingProxyType(normalized_parameters)


def _normalize_parameter_bound(
    bound: ParameterBound,
    *,
    model_name: str,
    parameter_name: str,
) -> ParameterBound:
    """Validate and normalize the lower and upper bounds of one model parameter.

    Args:
        bound: Candidate `(lower, upper)` bound pair.
        model_name: Model owning the parameter, included in validation diagnostics.
        parameter_name: Parameter whose bound is being normalized.

    Returns:
        Finite floating-point lower and upper bounds.

    Raises:
        TypeError: If the bound is not a two-item tuple or either endpoint is not a
            non-Boolean real number.
        ValueError: If either endpoint is non-finite or the lower bound exceeds the
            upper bound.
    """

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
    """Validate and freeze the complete model-to-parameter bound mapping.

    Model and parameter names are stripped, every model must define at least one
    parameter bound, and each bound is normalized by
    [`_normalize_parameter_bound`][igt.analysis.config._normalize_parameter_bound].
    Nested mappings and the outer mapping are returned as read-only proxies.

    Args:
        parameter_bounds: Candidate mapping from model names to parameter-bound mappings.

    Returns:
        Immutable normalized model and parameter bound mappings.

    Raises:
        TypeError: If the outer or nested values are not mappings, names are not
            strings, or an individual bound has an invalid endpoint type.
        ValueError: If a normalized name is empty, a model defines no parameters,
            the outer mapping is empty, or an individual bound is invalid.
    """

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
class FigureStyleConfig:
    """Central physical-size and font-size defaults for analysis figures.

    All standard figure presets share one base width. Normal and wide plots use
    different base heights; square plots derive their height directly from the base
    width so they remain square if that width is changed later. A uniform scale can be
    applied per plot without changing its aspect ratio.

    The category-height settings support plots whose required vertical space depends
    directly on the number of displayed categories, such as source-study bar charts.

    Attributes:
        width_inches: Shared base width for standard figures.
        normal_height_inches: Base height for ordinary rectangular figures.
        wide_height_inches: Base height for wider-aspect figures.
        title_font_size: Matplotlib title font size in points.
        axis_label_font_size: Axis-label font size in points.
        tick_label_font_size: Tick-label font size in points.
        legend_font_size: Legend font size in points.
        annotation_font_size: Base/default text size in points for annotations and
            other plot text not covered by a more specific rc setting.
        category_height_per_item_inches: Additional vertical space allocated per
            displayed category in dynamically sized categorical plots.
        category_height_padding_inches: Fixed vertical padding added to dynamically
            sized categorical plots.
    """

    width_inches: float = 6.5
    normal_height_inches: float = 4.5
    wide_height_inches: float = 4.0
    title_font_size: float = 12.0
    axis_label_font_size: float = 10.0
    tick_label_font_size: float = 10.0
    legend_font_size: float = 10.0
    annotation_font_size: float = 10.0
    category_height_per_item_inches: float = 0.48
    category_height_padding_inches: float = 1.5

    def __post_init__(self) -> None:
        """Validate and normalize all physical-size and font-size settings."""

        for parameter_name in (
            "width_inches",
            "normal_height_inches",
            "wide_height_inches",
            "title_font_size",
            "axis_label_font_size",
            "tick_label_font_size",
            "legend_font_size",
            "annotation_font_size",
            "category_height_per_item_inches",
        ):
            object.__setattr__(
                self,
                parameter_name,
                _validate_positive_finite_float(
                    getattr(self, parameter_name),
                    parameter_name=parameter_name,
                ),
            )

        object.__setattr__(
            self,
            "category_height_padding_inches",
            _validate_nonnegative_finite_float(
                self.category_height_padding_inches,
                parameter_name="category_height_padding_inches",
            ),
        )

    def figure_size(
        self,
        preset: FigureSizePreset = "normal",
        *,
        scale: float = 1.0,
    ) -> tuple[float, float]:
        """Return a standard figure size with an optional uniform scale.

        Args:
            preset: Base aspect-ratio preset.
            scale: Uniform multiplier applied to both width and height.

        Returns:
            `(width_inches, height_inches)` for Matplotlib's `figsize` argument.

        Raises:
            ValueError: If the preset is unsupported or the scale is not positive and
                finite.
        """

        normalized_scale = _validate_positive_finite_float(
            scale,
            parameter_name="scale",
        )

        if preset == "normal":
            height = self.normal_height_inches
        elif preset == "wide":
            height = self.wide_height_inches
        elif preset == "square":
            height = self.width_inches
        else:
            raise ValueError("preset must be 'normal', 'wide', or 'square'.")

        return (
            self.width_inches * normalized_scale,
            height * normalized_scale,
        )

    def categorical_figure_size(
        self,
        n_categories: int,
        *,
        scale: float = 1.0,
    ) -> tuple[float, float]:
        """Return a dynamically sized figure for a vertical category list.

        The width stays at the shared base width while the height grows linearly with
        the number of categories, never falling below the normal base height.
        """

        if isinstance(n_categories, (bool, np.bool_)) or not isinstance(
            n_categories,
            Integral,
        ):
            raise TypeError("n_categories must be an integer.")

        normalized_n_categories = int(n_categories)

        if normalized_n_categories <= 0:
            raise ValueError("n_categories must be positive.")

        normalized_scale = _validate_positive_finite_float(
            scale,
            parameter_name="scale",
        )
        dynamic_height = max(
            self.normal_height_inches,
            self.category_height_per_item_inches * normalized_n_categories
            + self.category_height_padding_inches,
        )

        return (
            self.width_inches * normalized_scale,
            dynamic_height * normalized_scale,
        )

    def matplotlib_font_rc(self) -> Mapping[RcKeyType, object]:
        """Return Matplotlib rc settings derived from the configured font sizes."""

        return MappingProxyType(
            {
                "font.size": self.annotation_font_size,
                "axes.titlesize": self.title_font_size,
                "axes.labelsize": self.axis_label_font_size,
                "xtick.labelsize": self.tick_label_font_size,
                "ytick.labelsize": self.tick_label_font_size,
                "legend.fontsize": self.legend_font_size,
            }
        )


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Configuration for validation, statistical inference, tables, and figures.

    Attributes:
        figure_formats: File formats written for every generated figure. PNG and PDF
            are generated by default so each logical figure has an easy-view raster
            copy and a publication-quality vector copy.
        figure_dpi: DPI passed to Matplotlib `savefig`. It controls raster outputs such
            as PNG and any rasterized components embedded inside vector outputs; normal
            PDF lines, text, and axes remain vector graphics.
        figure_style: Central figure-size, dynamic-height, and font-size settings.
        matplotlib_rc: Matplotlib rc overrides applied temporarily while the standard
            figure pipeline runs. `pdf.fonttype=42` is included by default so PDF text
            is embedded as TrueType/Type 42 fonts. Raster formats ignore that setting.
        histogram_bins: Bin count or NumPy histogram strategy used by distribution
            plots.
        confidence_level: Confidence level used for bootstrap and exact binomial
            intervals.
        bootstrap_resamples: Number of BCa bootstrap resamples used for AIC and BIC
            difference estimates.
        bootstrap_seed: Seed used to make bootstrap intervals reproducible.
        numeric_tolerance: Absolute tolerance used by cross-table numerical validation.
        boundary_tolerance: Absolute tolerance used when identifying parameter estimates
            at configured bounds.
        parameter_bounds: Model-to-parameter bound mapping used by parameter summaries
            and boundary diagnostics.
    """

    figure_formats: tuple[str, ...] = ("png", "pdf")
    figure_dpi: int = 300
    figure_style: FigureStyleConfig = field(default_factory=FigureStyleConfig)
    matplotlib_rc: Mapping[RcKeyType, object] = field(default_factory=dict)
    histogram_bins: int | str = "auto"
    confidence_level: float = 0.95
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 42
    numeric_tolerance: float = 1e-10
    boundary_tolerance: float = 1e-8
    parameter_bounds: ModelParameterBounds = DEFAULT_MODEL_PARAMETER_BOUNDS

    def __post_init__(self) -> None:
        """Validate and normalize analysis configuration fields.

        Figure formats are stripped and de-dotted, integer settings are normalized from
        NumPy/Python integral values, confidence and tolerance settings are converted to
        finite floats, Matplotlib rc overrides are merged with project defaults and
        frozen, and the parameter-bound mapping is frozen after validation.
        """

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

        object.__setattr__(self, "figure_dpi", int(self.figure_dpi))

        if not isinstance(self.figure_style, FigureStyleConfig):
            raise TypeError("figure_style must be a FigureStyleConfig instance.")

        object.__setattr__(
            self,
            "matplotlib_rc",
            _normalize_matplotlib_rc(self.matplotlib_rc),
        )

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
