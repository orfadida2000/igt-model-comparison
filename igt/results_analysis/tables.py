"""Derived analysis tables built from model-fit results."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame

from igt.constants.models import (
    PVL_DELTA_MODEL_NAME,
    Q_LEARNING_MODEL_NAME,
)
from igt.constants.schema import (
    MODEL_COLUMN,
    NLL_COLUMN,
    PARTICIPANT_KEY_COLUMNS,
    SOURCE_STUDY_COLUMN,
)

from .config import AnalysisConfig

METRIC_COLUMNS = (
    NLL_COLUMN,
    "aic",
    "bic",
    "nll_improvement_over_uniform",
)


def _model_prefix(model_name: str) -> str:
    """Return the stable output prefix for one supported model."""

    if model_name == Q_LEARNING_MODEL_NAME:
        return "q"

    if model_name == PVL_DELTA_MODEL_NAME:
        return "pvl"

    raise ValueError(f"Unexpected model name: {model_name!r}")


def build_subject_comparison_table(
    comparison: DataFrame,
) -> DataFrame:
    """Build one paired model-comparison row per eligible subject."""

    key_columns = list(PARTICIPANT_KEY_COLUMNS)
    common_columns = [*key_columns, SOURCE_STUDY_COLUMN]
    eligible = comparison.loc[
        comparison["comparison_eligible"]
        & comparison[MODEL_COLUMN].isin(
            [
                Q_LEARNING_MODEL_NAME,
                PVL_DELTA_MODEL_NAME,
            ]
        )
    ].copy()

    model_tables: dict[str, DataFrame] = {}

    for model_name in (
        Q_LEARNING_MODEL_NAME,
        PVL_DELTA_MODEL_NAME,
    ):
        prefix = _model_prefix(model_name)
        model_rows = eligible.loc[
            eligible[MODEL_COLUMN].eq(model_name),
            [
                *common_columns,
                *METRIC_COLUMNS,
                "best_aic",
                "best_bic",
            ],
        ].copy()
        rename_mapping = {
            column_name: f"{prefix}_{column_name}"
            for column_name in (
                *METRIC_COLUMNS,
                "best_aic",
                "best_bic",
            )
        }
        model_tables[model_name] = model_rows.rename(columns=rename_mapping)

    q_rows = model_tables[Q_LEARNING_MODEL_NAME]
    pvl_rows = model_tables[PVL_DELTA_MODEL_NAME]
    paired = q_rows.merge(
        pvl_rows,
        on=key_columns,
        how="inner",
        suffixes=("_q", "_pvl"),
        validate="one_to_one",
    )

    q_study_column = f"{SOURCE_STUDY_COLUMN}_q"
    pvl_study_column = f"{SOURCE_STUDY_COLUMN}_pvl"

    if not paired[q_study_column].equals(paired[pvl_study_column]):
        mismatch = paired.loc[
            paired[q_study_column].ne(paired[pvl_study_column]),
            [*key_columns, q_study_column, pvl_study_column],
        ]
        raise ValueError(f"Model rows disagree on source study:\n{mismatch.to_string(index=False)}")

    paired[SOURCE_STUDY_COLUMN] = paired[q_study_column]
    paired = paired.drop(columns=[q_study_column, pvl_study_column])

    paired["nll_q_minus_pvl"] = (
        paired["q_negative_log_likelihood"] - paired["pvl_negative_log_likelihood"]
    )
    paired["aic_q_minus_pvl"] = paired["q_aic"] - paired["pvl_aic"]
    paired["bic_q_minus_pvl"] = paired["q_bic"] - paired["pvl_bic"]
    paired["uniform_improvement_pvl_minus_q"] = (
        paired["pvl_nll_improvement_over_uniform"] - paired["q_nll_improvement_over_uniform"]
    )

    ordered_columns = [
        *key_columns,
        SOURCE_STUDY_COLUMN,
        "q_negative_log_likelihood",
        "pvl_negative_log_likelihood",
        "nll_q_minus_pvl",
        "q_aic",
        "pvl_aic",
        "aic_q_minus_pvl",
        "q_bic",
        "pvl_bic",
        "bic_q_minus_pvl",
        "q_nll_improvement_over_uniform",
        "pvl_nll_improvement_over_uniform",
        "uniform_improvement_pvl_minus_q",
        "q_best_aic",
        "pvl_best_aic",
        "q_best_bic",
        "pvl_best_bic",
    ]

    return paired.loc[:, ordered_columns].sort_values(
        by=key_columns,
        kind="mergesort",
        ignore_index=True,
    )


def build_study_preference_table(
    subject_comparison: DataFrame,
) -> DataFrame:
    """Summarize subject-level model preference within each source study."""

    records: list[dict[str, float | int | str]] = []

    for study_name, rows in subject_comparison.groupby(
        SOURCE_STUDY_COLUMN,
        sort=True,
        observed=True,
    ):
        n_subjects = len(rows)

        if n_subjects == 0:
            continue

        pvl_aic_wins = int(rows["pvl_best_aic"].sum())
        pvl_bic_wins = int(rows["pvl_best_bic"].sum())
        q_aic_wins = int(rows["q_best_aic"].sum())
        q_bic_wins = int(rows["q_best_bic"].sum())

        records.append(
            {
                SOURCE_STUDY_COLUMN: str(study_name),
                "n_subjects": n_subjects,
                "pvl_aic_wins": pvl_aic_wins,
                "pvl_aic_win_rate": pvl_aic_wins / n_subjects,
                "q_aic_wins": q_aic_wins,
                "pvl_bic_wins": pvl_bic_wins,
                "pvl_bic_win_rate": pvl_bic_wins / n_subjects,
                "q_bic_wins": q_bic_wins,
                "mean_aic_q_minus_pvl": float(rows["aic_q_minus_pvl"].mean()),
                "median_aic_q_minus_pvl": float(rows["aic_q_minus_pvl"].median()),
                "mean_bic_q_minus_pvl": float(rows["bic_q_minus_pvl"].mean()),
                "median_bic_q_minus_pvl": float(rows["bic_q_minus_pvl"].median()),
            }
        )

    return DataFrame(records).sort_values(
        by=["pvl_aic_win_rate", "n_subjects"],
        ascending=[False, False],
        kind="mergesort",
        ignore_index=True,
    )


def build_boundary_summary_table(fits: DataFrame) -> DataFrame:
    """Summarize fit-level parameter-boundary counts by model."""

    records: list[dict[str, float | int | str]] = []

    for model_name, rows in fits.groupby(
        MODEL_COLUMN,
        sort=True,
        observed=True,
    ):
        n_fits = len(rows)

        if n_fits == 0:
            continue

        category_masks = {
            "at_least_one_lower_bound": rows["n_parameters_at_lower_bound"].gt(0),
            "at_least_one_upper_bound": rows["n_parameters_at_upper_bound"].gt(0),
            "at_least_one_any_bound": rows["n_parameters_at_any_bound"].gt(0),
            "no_parameters_at_bound": rows["n_parameters_at_any_bound"].eq(0),
        }

        for category, mask in category_masks.items():
            count = int(mask.sum())
            records.append(
                {
                    MODEL_COLUMN: str(model_name),
                    "category": category,
                    "n_fits": n_fits,
                    "count": count,
                    "rate": count / n_fits,
                }
            )

    return DataFrame(records)


def _parameter_columns_for_model(
    fits: DataFrame,
    model_name: str,
    config: AnalysisConfig,
) -> Iterable[str]:
    """Yield configured parameter columns containing model values."""

    model_bounds = config.parameter_bounds.get(model_name, {})
    model_rows = fits.loc[fits[MODEL_COLUMN].eq(model_name)]

    for parameter_name in model_bounds:
        if parameter_name in model_rows.columns and model_rows[parameter_name].notna().any():
            yield parameter_name


def build_parameter_summary_table(
    fits: DataFrame,
    config: AnalysisConfig,
) -> DataFrame:
    """Summarize fitted parameter distributions and boundary frequencies."""

    records: list[dict[str, float | int | str]] = []

    for model_name in (
        Q_LEARNING_MODEL_NAME,
        PVL_DELTA_MODEL_NAME,
    ):
        model_rows = fits.loc[fits[MODEL_COLUMN].eq(model_name)]
        model_bounds = config.parameter_bounds.get(model_name, {})

        for parameter_name in _parameter_columns_for_model(
            fits,
            model_name,
            config,
        ):
            numeric_values = (
                pd.to_numeric(
                    model_rows[parameter_name],
                    errors="raise",
                )
                .dropna()
                .to_numpy(dtype=np.float64)
            )

            if numeric_values.size == 0:
                continue

            lower_bound, upper_bound = model_bounds[parameter_name]
            lower_mask = np.isclose(
                numeric_values,
                lower_bound,
                rtol=0.0,
                atol=config.boundary_tolerance,
            )
            upper_mask = np.isclose(
                numeric_values,
                upper_bound,
                rtol=0.0,
                atol=config.boundary_tolerance,
            )
            standard_deviation = (
                float(np.std(numeric_values, ddof=1)) if numeric_values.size > 1 else float("nan")
            )

            records.append(
                {
                    MODEL_COLUMN: model_name,
                    "parameter": parameter_name,
                    "n": int(numeric_values.size),
                    "mean": float(np.mean(numeric_values)),
                    "standard_deviation": standard_deviation,
                    "minimum": float(np.min(numeric_values)),
                    "q1": float(np.quantile(numeric_values, 0.25)),
                    "median": float(np.median(numeric_values)),
                    "q3": float(np.quantile(numeric_values, 0.75)),
                    "maximum": float(np.max(numeric_values)),
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "n_at_lower_bound": int(lower_mask.sum()),
                    "rate_at_lower_bound": float(lower_mask.mean()),
                    "n_at_upper_bound": int(upper_mask.sum()),
                    "rate_at_upper_bound": float(upper_mask.mean()),
                }
            )

    return DataFrame(records)


def build_model_win_table(
    subject_comparison: DataFrame,
) -> DataFrame:
    """Build model-win counts and rates for AIC and BIC."""

    records: list[dict[str, float | int | str]] = []
    n_subjects = len(subject_comparison)

    if n_subjects == 0:
        raise ValueError("No eligible subjects are available for model comparison.")

    for criterion in ("aic", "bic"):
        for model_name, prefix in (
            (Q_LEARNING_MODEL_NAME, "q"),
            (PVL_DELTA_MODEL_NAME, "pvl"),
        ):
            count = int(subject_comparison[f"{prefix}_best_{criterion}"].sum())
            records.append(
                {
                    "criterion": criterion.upper(),
                    MODEL_COLUMN: model_name,
                    "n_subjects": n_subjects,
                    "wins": count,
                    "win_rate": count / n_subjects,
                }
            )

    return DataFrame(records)
