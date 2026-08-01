"""Compare completed computational-model fits."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from igt.constants.schema import PARTICIPANT_KEY_COLUMNS
from igt.execution.typing import ModelFitResult


def fit_results_to_dataframe(
    results: Sequence[ModelFitResult],
) -> pd.DataFrame:
    """Convert model-fit results into a flat table."""

    if len(results) == 0:
        table = pd.DataFrame(columns=ModelFitResult.get_result_columns())
    else:
        table = pd.DataFrame(result.to_record() for result in results)

    return table.sort_values(
        by=["source_study", *PARTICIPANT_KEY_COLUMNS, "model"],
        ignore_index=True,
    )


def add_model_comparison_columns(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Add valid within-subject AIC/BIC differences and winner indicators.

    A subject is comparison-eligible only when every fit in the subject group
    converged, all AIC/BIC values are finite, and there is exactly one row for
    each of at least two distinct models. Ineligible rows remain in the output
    for diagnostics but receive no deltas or winner flags.
    """

    required_columns = {
        "source_study",
        "model",
        "aic",
        "bic",
        "converged",
    } | set(PARTICIPANT_KEY_COLUMNS)
    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required fit-result columns: {missing_text}")

    compared = results.copy()
    compared["_finite_comparison_metrics"] = np.isfinite(
        compared[["aic", "bic"]].to_numpy(dtype=np.float64)
    ).all(axis=1)

    grouped = compared.groupby(
        list(PARTICIPANT_KEY_COLUMNS),
        sort=False,
        dropna=False,
    )
    group_sizes = grouped["model"].transform("size")
    unique_model_counts = grouped["model"].transform("nunique")
    all_converged = grouped["converged"].transform("all")
    all_metrics_finite = grouped["_finite_comparison_metrics"].transform("all")

    compared["comparison_eligible"] = (
        all_converged
        & all_metrics_finite
        & unique_model_counts.ge(2)
        & group_sizes.eq(unique_model_counts)
    )
    compared = compared.drop(columns="_finite_comparison_metrics")

    compared["delta_aic"] = np.nan
    compared["delta_bic"] = np.nan

    eligible_mask = compared["comparison_eligible"]
    eligible = compared.loc[eligible_mask]

    if not eligible.empty:
        eligible_grouped = eligible.groupby(
            list(PARTICIPANT_KEY_COLUMNS),
            sort=False,
            dropna=False,
        )
        compared.loc[eligible_mask, "delta_aic"] = eligible["aic"] - eligible_grouped[
            "aic"
        ].transform("min")
        compared.loc[eligible_mask, "delta_bic"] = eligible["bic"] - eligible_grouped[
            "bic"
        ].transform("min")

    compared["best_aic"] = eligible_mask & compared["delta_aic"].eq(0.0)
    compared["best_bic"] = eligible_mask & compared["delta_bic"].eq(0.0)

    return compared


def summarize_model_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize convergence and valid information-criterion comparisons."""

    summary_columns = [
        "model",
        "n_fits",
        "n_converged",
        "convergence_rate",
        "n_comparisons",
        "mean_negative_log_likelihood",
        "mean_aic",
        "mean_bic",
        "aic_wins",
        "bic_wins",
    ]

    compared = add_model_comparison_columns(results)

    if compared.empty:
        return pd.DataFrame(columns=summary_columns)

    fit_counts = (
        compared.groupby("model", sort=True)
        .agg(
            n_fits=("model", "size"),
            n_converged=("converged", "sum"),
        )
        .reset_index()
    )
    fit_counts["convergence_rate"] = fit_counts["n_converged"] / fit_counts["n_fits"]

    eligible = compared.loc[compared["comparison_eligible"]]

    if eligible.empty:
        comparison_metrics = pd.DataFrame(
            columns=[
                "model",
                "n_comparisons",
                "mean_negative_log_likelihood",
                "mean_aic",
                "mean_bic",
                "aic_wins",
                "bic_wins",
            ]
        )
    else:
        comparison_metrics = (
            eligible.groupby("model", sort=True)
            .agg(
                n_comparisons=("model", "size"),
                mean_negative_log_likelihood=("negative_log_likelihood", "mean"),
                mean_aic=("aic", "mean"),
                mean_bic=("bic", "mean"),
                aic_wins=("best_aic", "sum"),
                bic_wins=("best_bic", "sum"),
            )
            .reset_index()
        )

    summary = fit_counts.merge(
        comparison_metrics,
        on="model",
        how="left",
        validate="one_to_one",
    )

    for count_column in ("n_comparisons", "aic_wins", "bic_wins"):
        summary[count_column] = summary[count_column].fillna(0).astype("int64")

    return summary.loc[:, summary_columns]
