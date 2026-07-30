"""Compare completed computational-model fits."""

from collections.abc import Sequence

import pandas as pd

from igt.execution.fitting import ModelFitResult


def fit_results_to_dataframe(
    results: Sequence[ModelFitResult],
) -> pd.DataFrame:
    """Convert model-fit results into a flat table."""

    if len(results) == 0:
        table = pd.DataFrame(columns=ModelFitResult.get_result_columns())
    else:
        table = pd.DataFrame(result.to_record() for result in results)

    return table.sort_values(
        by=["n_trials", "subject_id", "model"],
        ignore_index=True,
    )


def add_model_comparison_columns(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Add within-subject AIC/BIC differences and winner indicators."""

    required_columns = {
        "n_trials",
        "subject_id",
        "model",
        "aic",
        "bic",
    }
    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required fit-result columns: {missing_text}")

    compared = results.copy()
    group_columns = ["n_trials", "subject_id"]

    compared["delta_aic"] = compared["aic"] - compared.groupby(group_columns)["aic"].transform(
        "min"
    )
    compared["delta_bic"] = compared["bic"] - compared.groupby(group_columns)["bic"].transform(
        "min"
    )
    compared["best_aic"] = compared["delta_aic"].eq(0.0)
    compared["best_bic"] = compared["delta_bic"].eq(0.0)

    return compared


def summarize_model_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize fit quality and information-criterion wins by model."""

    compared = add_model_comparison_columns(results)

    summary = (
        compared.groupby("model", sort=True)
        .agg(
            n_fits=("model", "size"),
            n_converged=("converged", "sum"),
            mean_negative_log_likelihood=("negative_log_likelihood", "mean"),
            mean_aic=("aic", "mean"),
            mean_bic=("bic", "mean"),
            aic_wins=("best_aic", "sum"),
            bic_wins=("best_bic", "sum"),
        )
        .reset_index()
    )

    summary["convergence_rate"] = summary["n_converged"] / summary["n_fits"]

    return summary
