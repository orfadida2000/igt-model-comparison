"""Utilities for comparing completed computational-model fits.

The module converts fit records into stable tables, determines within-participant
AIC/BIC comparison eligibility and winners, and summarizes convergence and model
preference by model.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from igt.constants.schema import (
    CONVERGED_COLUMN,
    MODEL_COLUMN,
    NLL_COLUMN,
    PARTICIPANT_KEY_COLUMNS,
    SOURCE_STUDY_COLUMN,
)
from igt.execution.typing import ModelFitResult


def fit_results_to_dataframe(
    results: Sequence[ModelFitResult],
) -> pd.DataFrame:
    """Convert model-fit result records into a stable flat table.

    Args:
        results: Completed per-model, per-participant fit records.

    Returns:
        A DataFrame sorted by source study, participant key, and model, with one
        row for each supplied fit result.
    """

    if len(results) == 0:
        table = pd.DataFrame(columns=ModelFitResult.get_result_columns())
    else:
        table = pd.DataFrame(result.to_record() for result in results)

    return table.sort_values(
        by=[SOURCE_STUDY_COLUMN, *PARTICIPANT_KEY_COLUMNS, MODEL_COLUMN],
        ignore_index=True,
    )


def add_model_comparison_columns(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Add within-participant information-criterion comparisons to fit results.

    A participant is comparison-eligible only when every model row converged, all
    AIC and BIC values are finite, at least two distinct models are present, and
    there is exactly one row per model. Eligible rows receive criterion deltas from
    the within-participant minimum and Boolean winner indicators; ineligible rows
    remain available for diagnostics without valid comparison values.

    Args:
        results: Per-participant model-fit table.

    Returns:
        A copy of `results` augmented with comparison eligibility, AIC/BIC deltas,
        and best-model indicators.

    Raises:
        ValueError: If a required participant, model, convergence, AIC, BIC, or
            source-study column is missing.
    """

    required_columns = {
        SOURCE_STUDY_COLUMN,
        MODEL_COLUMN,
        "aic",
        "bic",
        CONVERGED_COLUMN,
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
    group_sizes = grouped[MODEL_COLUMN].transform("size")
    unique_model_counts = grouped[MODEL_COLUMN].transform("nunique")
    all_converged = grouped[CONVERGED_COLUMN].transform("all")
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
    """Summarize convergence and valid model-comparison outcomes by model.

    Args:
        results: Per-participant model-fit table.

    Returns:
        One row per model containing fit and convergence counts, mean fit metrics,
        valid comparison counts, and AIC/BIC win counts.
    """

    summary_columns = [
        MODEL_COLUMN,
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
        compared.groupby(MODEL_COLUMN, sort=True)
        .agg(
            n_fits=(MODEL_COLUMN, "size"),
            n_converged=(CONVERGED_COLUMN, "sum"),
        )
        .reset_index()
    )
    fit_counts["convergence_rate"] = fit_counts["n_converged"] / fit_counts["n_fits"]

    eligible = compared.loc[compared["comparison_eligible"]]

    if eligible.empty:
        comparison_metrics = pd.DataFrame(
            columns=[
                MODEL_COLUMN,
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
            eligible.groupby(MODEL_COLUMN, sort=True)
            .agg(
                n_comparisons=(MODEL_COLUMN, "size"),
                mean_negative_log_likelihood=(NLL_COLUMN, "mean"),
                mean_aic=("aic", "mean"),
                mean_bic=("bic", "mean"),
                aic_wins=("best_aic", "sum"),
                bic_wins=("best_bic", "sum"),
            )
            .reset_index()
        )

    summary = fit_counts.merge(
        comparison_metrics,
        on=MODEL_COLUMN,
        how="left",
        validate="one_to_one",
    )

    for count_column in ("n_comparisons", "aic_wins", "bic_wins"):
        summary[count_column] = summary[count_column].fillna(0).astype("int64")

    return summary.loc[:, summary_columns]
