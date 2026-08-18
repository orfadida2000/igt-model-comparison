"""Matplotlib figure generation for model-fit and model-comparison results.

The module contains focused plotting helpers for criterion differences, model wins,
fit metrics, source-study preferences, boundary diagnostics, uniform-choice
improvement, parameter distributions, and inferential confidence intervals.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from pandas import DataFrame, Series

from igt.constants.models import (
    PVL_DELTA_MODEL_NAME,
    Q_LEARNING_MODEL_NAME,
)
from igt.constants.schema import MODEL_COLUMN, NLL_COLUMN, SOURCE_STUDY_COLUMN
from igt.typing import Float1DArray, StrPathLike
from igt.utils.io import normalize_path

from .config import AnalysisConfig


def _save_figure(
    figure: Figure,
    output_stem: StrPathLike,
    config: AnalysisConfig,
) -> tuple[Path, ...]:
    """Save and close one Matplotlib figure in every configured format.

    The parent directory of `output_stem` is created when needed. Each configured
    format is written with the configured DPI and a tight bounding box before the
    figure is closed.

    Args:
        figure: Figure to persist.
        output_stem: Destination path without a suffix.
        config: Figure formats and raster DPI settings.

    Returns:
        Paths of all figure files written, in configured format order.
    """

    output_stem = normalize_path(output_stem, parameter_name="output_stem")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for figure_format in config.figure_formats:
        output_path = output_stem.with_suffix(f".{figure_format}")
        figure.savefig(
            output_path,
            dpi=config.figure_dpi,
            bbox_inches="tight",
        )
        output_paths.append(output_path)

    plt.close(figure)
    return tuple(output_paths)


def _finite_values(series: Series, *, name: str) -> Float1DArray:
    """Convert a plotting Series to finite floating-point values.

    Non-finite values are discarded rather than plotted; at least one finite value must
    remain.

    Args:
        series: Source values to convert numerically.
        name: Human-readable value name included in an empty-result diagnostic.

    Returns:
        One-dimensional array containing the finite floating-point values.

    Raises:
        ValueError: If numeric conversion fails or no finite value remains.
    """

    values = pd.to_numeric(series, errors="raise").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        raise ValueError(f"No finite values are available for {name}.")

    return finite_values


def _single_integer_value(
    data: DataFrame,
    mask: Series,
    column_name: str,
    *,
    description: str,
) -> int:
    """Extract exactly one integer from a masked DataFrame column.

    Args:
        data: Source table.
        mask: Boolean row mask selecting the expected record.
        column_name: Integer column to extract.
        description: Human-readable description included in cardinality diagnostics.

    Returns:
        The single selected integer value.

    Raises:
        ValueError: If the mask selects anything other than exactly one value.
    """

    values = data.loc[mask, column_name].to_numpy(dtype=np.int64)

    if values.size != 1:
        raise ValueError(f"Expected exactly one {description} value, found {values.size}.")

    return int(values[0])


def plot_signed_difference_distribution(
    subject_comparison: DataFrame,
    *,
    criterion: str,
    output_stem: StrPathLike,
    config: AnalysisConfig,
) -> tuple[Path, ...]:
    """Plot the distribution of signed Q-minus-PVL AIC or BIC differences.

    The zero reference line separates values favoring PVL-Delta (positive) from values
    favoring Q-learning (negative).

    Args:
        subject_comparison: Paired participant-level comparison table.
        criterion: Criterion name, either `"aic"` or `"bic"` case-insensitively.
        output_stem: Destination path without a figure suffix.
        config: Plotting and output configuration.

    Returns:
        Paths of the saved figure files.

    Raises:
        ValueError: If the criterion is unsupported or no finite criterion differences
            are available.
    """

    normalized_criterion = criterion.lower()

    if normalized_criterion not in {"aic", "bic"}:
        raise ValueError("criterion must be 'aic' or 'bic'.")

    column_name = f"{normalized_criterion}_q_minus_pvl"
    values = _finite_values(subject_comparison[column_name], name=column_name)
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.hist(values, bins=config.histogram_bins)
    axis.axvline(0.0, linestyle="--", linewidth=1.5)
    axis.set_title(f"Distribution of signed {normalized_criterion.upper()} differences")
    axis.set_xlabel(
        f"{normalized_criterion.upper()}(Q-learning) − {normalized_criterion.upper()}(PVL-Delta)"
    )
    axis.set_ylabel("Subjects")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return _save_figure(figure, output_stem, config)


def plot_paired_metric_scatter(
    subject_comparison: DataFrame,
    *,
    metric: str,
    output_stem: StrPathLike,
    config: AnalysisConfig,
) -> tuple[Path, ...]:
    """Plot paired Q-learning and PVL-Delta values against the identity line.

    Args:
        subject_comparison: Paired participant-level comparison table.
        metric: One of negative log-likelihood, AIC, or BIC.
        output_stem: Destination path without a figure suffix.
        config: Plotting and output configuration.

    Returns:
        Paths of the saved figure files.

    Raises:
        ValueError: If `metric` is unsupported, either model has no finite values, or
            the paired value arrays have different shapes.
    """

    supported_metrics = {
        NLL_COLUMN: "Negative log-likelihood",
        "aic": "AIC",
        "bic": "BIC",
    }

    if metric not in supported_metrics:
        raise ValueError(
            f"Unsupported metric {metric!r}. Expected one of {sorted(supported_metrics)}."
        )

    q_values = _finite_values(
        subject_comparison[f"q_{metric}"],
        name=f"q_{metric}",
    )
    pvl_values = _finite_values(
        subject_comparison[f"pvl_{metric}"],
        name=f"pvl_{metric}",
    )

    if q_values.shape != pvl_values.shape:
        raise ValueError("Paired metric arrays have different shapes.")

    lower_limit = float(min(np.min(q_values), np.min(pvl_values)))
    upper_limit = float(max(np.max(q_values), np.max(pvl_values)))
    padding = max((upper_limit - lower_limit) * 0.04, 1e-9)
    plot_limits = (lower_limit - padding, upper_limit + padding)

    figure, axis = plt.subplots(figsize=(6.5, 6.5))
    axis.scatter(q_values, pvl_values, alpha=0.65, s=24)
    axis.plot(plot_limits, plot_limits, linestyle="--", linewidth=1.5)
    axis.set_xlim(plot_limits)
    axis.set_ylim(plot_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"Q-learning versus PVL-Delta {supported_metrics[metric]}")
    axis.set_xlabel(f"Q-learning {supported_metrics[metric]}")
    axis.set_ylabel(f"PVL-Delta {supported_metrics[metric]}")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return _save_figure(figure, output_stem, config)


def plot_model_win_counts(
    model_win_table: DataFrame,
    *,
    output_stem: StrPathLike,
    config: AnalysisConfig,
) -> tuple[Path, ...]:
    """Plot Q-learning and PVL-Delta AIC/BIC win counts side by side.

    Args:
        model_win_table: Descriptive criterion/model win-count table.
        output_stem: Destination path without a figure suffix.
        config: Figure-output configuration.

    Returns:
        Paths of the saved figure files.

    Raises:
        ValueError: If any criterion/model combination does not have exactly one win
            count in `model_win_table`.
    """

    criteria = ("AIC", "BIC")
    q_wins = [
        _single_integer_value(
            model_win_table,
            model_win_table["criterion"].eq(criterion)
            & model_win_table[MODEL_COLUMN].eq(Q_LEARNING_MODEL_NAME),
            "wins",
            description=f"{criterion} Q-learning win-count",
        )
        for criterion in criteria
    ]
    pvl_wins = [
        _single_integer_value(
            model_win_table,
            model_win_table["criterion"].eq(criterion)
            & model_win_table[MODEL_COLUMN].eq(PVL_DELTA_MODEL_NAME),
            "wins",
            description=f"{criterion} PVL-Delta win-count",
        )
        for criterion in criteria
    ]

    x_positions = np.arange(len(criteria), dtype=np.float64)
    bar_width = 0.36
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.bar(
        x_positions - bar_width / 2,
        q_wins,
        width=bar_width,
        label="Q-learning",
    )
    axis.bar(
        x_positions + bar_width / 2,
        pvl_wins,
        width=bar_width,
        label="PVL-Delta",
    )
    axis.set_xticks(x_positions, criteria)
    axis.set_title("Subject-level model wins")
    axis.set_xlabel("Model-selection criterion")
    axis.set_ylabel("Subjects")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return _save_figure(figure, output_stem, config)


def plot_study_preference_rates(
    study_preference: DataFrame,
    *,
    output_stem: StrPathLike,
    config: AnalysisConfig,
) -> tuple[Path, ...]:
    """Plot PVL-Delta AIC and BIC win rates by source study.

    Studies are ordered by AIC PVL-Delta win rate and labels include the number of
    eligible participants contributing to each study.

    Args:
        study_preference: Per-study preference counts, rates, and signed differences.
        output_stem: Destination path without a figure suffix.
        config: Figure-output configuration.

    Returns:
        Paths of the saved figure files.
    """

    sorted_data = study_preference.sort_values(
        by="pvl_aic_win_rate",
        ascending=True,
        kind="mergesort",
    )
    study_names = sorted_data[SOURCE_STUDY_COLUMN].astype("string").to_numpy(dtype=str)
    n_subjects = sorted_data["n_subjects"].to_numpy(dtype=np.int64)
    aic_rates = sorted_data["pvl_aic_win_rate"].to_numpy(dtype=np.float64)
    bic_rates = sorted_data["pvl_bic_win_rate"].to_numpy(dtype=np.float64)
    labels = [
        f"{study_name} (n={int(subject_count)})"
        for study_name, subject_count in zip(
            study_names,
            n_subjects,
            strict=True,
        )
    ]
    y_positions = np.arange(len(sorted_data), dtype=np.float64)
    bar_height = 0.36
    figure_height = max(5.0, 0.48 * len(sorted_data) + 1.5)
    figure, axis = plt.subplots(figsize=(9.0, figure_height))
    axis.barh(
        y_positions - bar_height / 2,
        aic_rates,
        height=bar_height,
        label="AIC",
    )
    axis.barh(
        y_positions + bar_height / 2,
        bic_rates,
        height=bar_height,
        label="BIC",
    )
    axis.set_yticks(y_positions, labels)
    axis.set_xlim(0.0, 1.0)
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set_title("PVL-Delta preference rate by source study")
    axis.set_xlabel("Subjects favoring PVL-Delta")
    axis.set_ylabel("Source study")
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    return _save_figure(figure, output_stem, config)


def plot_parameter_distribution(
    fits: DataFrame,
    *,
    model_name: str,
    parameter_name: str,
    output_stem: StrPathLike,
    config: AnalysisConfig,
) -> tuple[Path, ...]:
    """Plot the fitted distribution of one configured model parameter.

    Configured lower and upper parameter bounds are drawn as reference lines when
    available.

    Args:
        fits: Validated per-participant model-fit table.
        model_name: Canonical model whose parameter estimates are plotted.
        parameter_name: Parameter column to visualize.
        output_stem: Destination path without a figure suffix.
        config: Parameter-bound and plotting configuration.

    Returns:
        Paths of the saved figure files.
    """

    model_rows = fits.loc[fits[MODEL_COLUMN].eq(model_name)]
    values = _finite_values(
        model_rows[parameter_name],
        name=f"{model_name}.{parameter_name}",
    )
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    axis.hist(values, bins=config.histogram_bins)

    parameter_bounds = config.parameter_bounds.get(model_name, {}).get(parameter_name)

    if parameter_bounds is not None:
        lower_bound, upper_bound = parameter_bounds
        axis.axvline(
            lower_bound,
            linestyle="--",
            linewidth=1.25,
            label="Lower bound",
        )
        axis.axvline(
            upper_bound,
            linestyle=":",
            linewidth=1.25,
            label="Upper bound",
        )
        axis.legend()

    axis.set_title(f"{model_name.replace('_', ' ').title()}: {parameter_name.replace('_', ' ')}")
    axis.set_xlabel("Fitted value")
    axis.set_ylabel("Subjects")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return _save_figure(figure, output_stem, config)


def plot_boundary_rates(
    boundary_summary: DataFrame,
    *,
    output_stem: StrPathLike,
    config: AnalysisConfig,
) -> tuple[Path, ...]:
    """Plot the proportion of fits with at least one parameter on a bound.

    Args:
        boundary_summary: Model-level boundary-category count and rate table.
        output_stem: Destination path without a figure suffix.
        config: Plotting and output configuration.

    Returns:
        Paths of the saved figure files.

    Raises:
        ValueError: If a supported model does not have exactly one
            `at_least_one_any_bound` rate or that rate is non-finite.
    """

    categories = [
        "at_least_one_lower_bound",
        "at_least_one_upper_bound",
        "at_least_one_any_bound",
    ]
    display_labels = [
        "Lower bound",
        "Upper bound",
        "Any bound",
    ]
    x_positions = np.arange(len(categories), dtype=np.float64)
    bar_width = 0.36
    figure, axis = plt.subplots(figsize=(8.0, 5.0))

    for offset, model_name, display_name in (
        (-bar_width / 2, Q_LEARNING_MODEL_NAME, "Q-learning"),
        (bar_width / 2, PVL_DELTA_MODEL_NAME, "PVL-Delta"),
    ):
        model_rows = boundary_summary.loc[boundary_summary[MODEL_COLUMN].eq(model_name)].set_index(
            "category"
        )
        missing_categories = set(categories) - set(model_rows.index)

        if missing_categories:
            missing_text = ", ".join(sorted(missing_categories))
            raise ValueError(
                f"Boundary summary for model {model_name!r} is missing categories: {missing_text}"
            )

        rates = model_rows.loc[categories, "rate"].to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        if not np.isfinite(rates).all():
            raise ValueError(
                f"Boundary rates for model {model_name!r} contain missing or non-finite values."
            )

        axis.bar(
            x_positions + offset,
            rates,
            width=bar_width,
            label=display_name,
        )

    axis.set_xticks(x_positions, display_labels)
    axis.set_ylim(0.0, 1.0)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set_title("Fits with parameters at configured bounds")
    axis.set_xlabel("Boundary category")
    axis.set_ylabel("Fits")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return _save_figure(figure, output_stem, config)


def plot_uniform_improvement_distribution(
    fits: DataFrame,
    *,
    output_stem: StrPathLike,
    config: AnalysisConfig,
) -> tuple[Path, ...]:
    """Plot each model's NLL improvement over the uniform-choice baseline.

    Args:
        fits: Validated per-participant model-fit table.
        output_stem: Destination path without a figure suffix.
        config: Histogram and figure-output configuration.

    Returns:
        Paths of the saved figure files.

    Raises:
        ValueError: If either supported model has no finite uniform-choice improvement
            values to plot.
    """

    figure, axis = plt.subplots(figsize=(8.0, 5.0))

    for model_name, display_name in (
        (Q_LEARNING_MODEL_NAME, "Q-learning"),
        (PVL_DELTA_MODEL_NAME, "PVL-Delta"),
    ):
        values = _finite_values(
            fits.loc[
                fits[MODEL_COLUMN].eq(model_name),
                "nll_improvement_over_uniform",
            ],
            name=f"{model_name}.nll_improvement_over_uniform",
        )
        axis.hist(
            values,
            bins=config.histogram_bins,
            alpha=0.55,
            label=display_name,
        )

    axis.axvline(0.0, linestyle="--", linewidth=1.25)
    axis.set_title("Improvement over uniform-choice negative log-likelihood")
    axis.set_xlabel("Uniform-choice NLL − fitted-model NLL")
    axis.set_ylabel("Subjects")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return _save_figure(figure, output_stem, config)


def plot_uniform_improvement_scatter(
    subject_comparison: DataFrame,
    *,
    output_stem: StrPathLike,
    config: AnalysisConfig,
) -> tuple[Path, ...]:
    """Plot paired model improvement over the uniform-choice negative log-likelihood.

    Args:
        subject_comparison: Paired participant-level comparison table.
        output_stem: Destination path without a figure suffix.
        config: Plotting and output configuration.

    Returns:
        Paths of the saved figure files.

    Raises:
        ValueError: If either model lacks finite values or the paired arrays have
            different shapes.
    """

    q_values = _finite_values(
        subject_comparison["q_nll_improvement_over_uniform"],
        name="q_nll_improvement_over_uniform",
    )
    pvl_values = _finite_values(
        subject_comparison["pvl_nll_improvement_over_uniform"],
        name="pvl_nll_improvement_over_uniform",
    )

    if q_values.shape != pvl_values.shape:
        raise ValueError("Paired uniform-improvement arrays have different shapes.")

    lower_limit = float(min(np.min(q_values), np.min(pvl_values)))
    upper_limit = float(max(np.max(q_values), np.max(pvl_values)))
    padding = max((upper_limit - lower_limit) * 0.04, 1e-9)
    plot_limits = (lower_limit - padding, upper_limit + padding)
    figure, axis = plt.subplots(figsize=(6.5, 6.5))
    axis.scatter(q_values, pvl_values, alpha=0.65, s=24)
    axis.plot(plot_limits, plot_limits, linestyle="--", linewidth=1.5)
    axis.set_xlim(plot_limits)
    axis.set_ylim(plot_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("Improvement over uniform choice")
    axis.set_xlabel("Q-learning improvement")
    axis.set_ylabel("PVL-Delta improvement")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return _save_figure(figure, output_stem, config)



def plot_criterion_difference_confidence_intervals(
    criterion_inference: DataFrame,
    *,
    output_stem: StrPathLike,
    config: AnalysisConfig,
) -> tuple[Path, ...]:
    """Plot mean and median AIC/BIC differences with BCa bootstrap intervals.

    Point estimates and interval limits are read from the criterion-inference table;
    the zero reference line marks no Q-versus-PVL criterion difference.

    Args:
        criterion_inference: AIC/BIC population-level difference-inference table.
        output_stem: Destination path without a figure suffix.
        config: Confidence-level and figure-output configuration.

    Returns:
        Paths of the saved figure files.

    Raises:
        ValueError: If a criterion does not have exactly one inference row, an estimate
            or interval limit is non-finite, or an interval excludes its own point
            estimate.
    """

    records: list[tuple[str, float, float, float]] = []

    for criterion in ("AIC", "BIC"):
        rows = criterion_inference.loc[criterion_inference["criterion"].eq(criterion)]

        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly one {criterion} inference row, found {len(rows)}."
            )

        mean = rows["mean_difference"].to_numpy(dtype=np.float64)[0]
        mean_lower = rows["mean_ci_lower"].to_numpy(dtype=np.float64)[0]
        mean_upper = rows["mean_ci_upper"].to_numpy(dtype=np.float64)[0]
        median = rows["median_difference"].to_numpy(dtype=np.float64)[0]
        median_lower = rows["median_ci_lower"].to_numpy(dtype=np.float64)[0]
        median_upper = rows["median_ci_upper"].to_numpy(dtype=np.float64)[0]
        records.extend(
            [
                (f"{criterion} mean", mean, mean_lower, mean_upper),
                (f"{criterion} median", median, median_lower, median_upper),
            ]
        )

    labels = [record[0] for record in records]
    estimates = np.array([record[1] for record in records], dtype=np.float64)
    lower_bounds = np.array([record[2] for record in records], dtype=np.float64)
    upper_bounds = np.array([record[3] for record in records], dtype=np.float64)

    if not (
        np.isfinite(estimates).all()
        and np.isfinite(lower_bounds).all()
        and np.isfinite(upper_bounds).all()
    ):
        raise ValueError("Criterion inference contains non-finite confidence-interval values.")

    lower_errors = estimates - lower_bounds
    upper_errors = upper_bounds - estimates

    if (lower_errors < 0.0).any() or (upper_errors < 0.0).any():
        raise ValueError("Criterion confidence intervals do not contain their point estimates.")

    y_positions = np.arange(len(records), dtype=np.float64)
    confidence_percent = config.confidence_level * 100.0
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.errorbar(
        estimates,
        y_positions,
        xerr=np.vstack([lower_errors, upper_errors]),
        fmt="o",
        capsize=4,
    )
    axis.axvline(0.0, linestyle="--", linewidth=1.5)
    axis.set_yticks(y_positions, labels)
    axis.invert_yaxis()
    axis.set_title(
        f"Criterion differences with {confidence_percent:g}% bootstrap confidence intervals"
    )
    axis.set_xlabel("Q-learning − PVL-Delta (positive favors PVL-Delta)")
    axis.set_ylabel("Estimate")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    return _save_figure(figure, output_stem, config)


def plot_pvl_win_rate_confidence_intervals(
    model_win_inference: DataFrame,
    *,
    output_stem: StrPathLike,
    config: AnalysisConfig,
) -> tuple[Path, ...]:
    """Plot PVL-Delta non-tied AIC/BIC win rates with exact confidence intervals.

    The 50% reference line corresponds to the null win probability used by the exact
    binomial tests in the model-win inference table.

    Args:
        model_win_inference: AIC/BIC population-level model-win inference table.
        output_stem: Destination path without a figure suffix.
        config: Confidence-level and figure-output configuration.

    Returns:
        Paths of the saved figure files.

    Raises:
        ValueError: If required confidence-interval values are non-finite or an
            interval does not contain its point estimate.
    """

    sorted_rows = model_win_inference.set_index("criterion").loc[["AIC", "BIC"]]
    rates = sorted_rows["pvl_win_rate_non_tied"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    lower_bounds = sorted_rows["pvl_win_rate_ci_lower"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    upper_bounds = sorted_rows["pvl_win_rate_ci_upper"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    if not (
        np.isfinite(rates).all()
        and np.isfinite(lower_bounds).all()
        and np.isfinite(upper_bounds).all()
    ):
        raise ValueError("Model-win inference contains non-finite confidence-interval values.")

    lower_errors = rates - lower_bounds
    upper_errors = upper_bounds - rates

    if (lower_errors < 0.0).any() or (upper_errors < 0.0).any():
        raise ValueError("Win-rate confidence intervals do not contain their point estimates.")

    x_positions = np.arange(2, dtype=np.float64)
    confidence_percent = config.confidence_level * 100.0
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.errorbar(
        x_positions,
        rates,
        yerr=np.vstack([lower_errors, upper_errors]),
        fmt="o",
        capsize=4,
    )
    axis.axhline(0.5, linestyle="--", linewidth=1.5)
    axis.set_xticks(x_positions, ["AIC", "BIC"])
    axis.set_ylim(0.0, 1.0)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set_title(
        f"PVL-Delta win rates with {confidence_percent:g}% exact confidence intervals"
    )
    axis.set_xlabel("Model-selection criterion")
    axis.set_ylabel("PVL-Delta win rate among non-tied subjects")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return _save_figure(figure, output_stem, config)


def generate_all_figures(
    fits: DataFrame,
    subject_comparison: DataFrame,
    study_preference: DataFrame,
    boundary_summary: DataFrame,
    model_win_table: DataFrame,
    criterion_inference: DataFrame,
    model_win_inference: DataFrame,
    *,
    output_directory: StrPathLike,
    config: AnalysisConfig,
) -> tuple[Path, ...]:
    """Generate the complete standard figure set for one analysis run.

    The function creates criterion-difference histograms, paired fit-metric scatters,
    model-win and inference figures, study preference and boundary plots, uniform-choice
    diagnostics, and one parameter-distribution figure for every configured parameter
    column present in the fit table.

    Args:
        fits: Validated model-fit result table.
        subject_comparison: Paired participant-level model-comparison table.
        study_preference: Per-study model-preference summary.
        boundary_summary: Model-level parameter-boundary summary.
        model_win_table: Descriptive AIC/BIC model-win table.
        criterion_inference: Population-level AIC/BIC difference-inference table.
        model_win_inference: Population-level exact model-win inference table.
        output_directory: Directory that receives all figure files and parameter
            subdirectories.
        config: Analysis plotting and output configuration.

    Returns:
        Paths of every generated figure file, including all configured formats.
    """

    output_directory = normalize_path(output_directory, parameter_name="output_directory")
    generated_paths: list[Path] = []

    for criterion in ("aic", "bic"):
        generated_paths.extend(
            plot_signed_difference_distribution(
                subject_comparison,
                criterion=criterion,
                output_stem=(output_directory / f"signed_{criterion}_difference_distribution"),
                config=config,
            )
        )

    for metric, filename in (
        (NLL_COLUMN, "q_vs_pvl_negative_log_likelihood"),
        ("aic", "q_vs_pvl_aic"),
        ("bic", "q_vs_pvl_bic"),
    ):
        generated_paths.extend(
            plot_paired_metric_scatter(
                subject_comparison,
                metric=metric,
                output_stem=output_directory / filename,
                config=config,
            )
        )

    generated_paths.extend(
        plot_model_win_counts(
            model_win_table,
            output_stem=output_directory / "model_win_counts",
            config=config,
        )
    )
    generated_paths.extend(
        plot_criterion_difference_confidence_intervals(
            criterion_inference,
            output_stem=(output_directory / "criterion_difference_confidence_intervals"),
            config=config,
        )
    )
    generated_paths.extend(
        plot_pvl_win_rate_confidence_intervals(
            model_win_inference,
            output_stem=(output_directory / "pvl_win_rate_confidence_intervals"),
            config=config,
        )
    )
    generated_paths.extend(
        plot_study_preference_rates(
            study_preference,
            output_stem=output_directory / "study_pvl_preference_rates",
            config=config,
        )
    )
    generated_paths.extend(
        plot_boundary_rates(
            boundary_summary,
            output_stem=output_directory / "boundary_fit_rates",
            config=config,
        )
    )
    generated_paths.extend(
        plot_uniform_improvement_distribution(
            fits,
            output_stem=(output_directory / "uniform_choice_improvement_distribution"),
            config=config,
        )
    )
    generated_paths.extend(
        plot_uniform_improvement_scatter(
            subject_comparison,
            output_stem=(output_directory / "uniform_choice_improvement_scatter"),
            config=config,
        )
    )

    for model_name, parameter_bounds in config.parameter_bounds.items():
        model_rows = fits.loc[fits[MODEL_COLUMN].eq(model_name)]

        if model_rows.empty:
            continue

        for parameter_name in parameter_bounds:
            if (
                parameter_name not in model_rows.columns
                or not model_rows[parameter_name].notna().any()
            ):
                continue

            generated_paths.extend(
                plot_parameter_distribution(
                    fits,
                    model_name=model_name,
                    parameter_name=parameter_name,
                    output_stem=(
                        output_directory
                        / "parameters"
                        / model_name
                        / f"{parameter_name}_distribution"
                    ),
                    config=config,
                )
            )

    return tuple(generated_paths)
