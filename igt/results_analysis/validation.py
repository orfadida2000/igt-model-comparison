"""Validation for result-analysis input tables."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame

from igt.constants.models import (
    LEARNING_RATE_PARAMETER_NAME,
    PVL_DELTA_MODEL_NAME,
    Q_LEARNING_MODEL_NAME,
)
from igt.constants.schema import (
    CONVERGED_COLUMN,
    MODEL_COLUMN,
    NLL_COLUMN,
    PARTICIPANT_KEY_COLUMNS,
    SOURCE_STUDY_COLUMN,
)
from igt.subject_selection import normalize_subject_key_columns
from igt.utils.tabular import (
    normalize_boolean_series,
    normalize_integer_series,
    normalize_nonempty_string_series,
)

from .config import AnalysisConfig
from .io import ResultTables

FIT_REQUIRED_COLUMNS = {
    MODEL_COLUMN,
    SOURCE_STUDY_COLUMN,
    NLL_COLUMN,
    "log_likelihood",
    "aic",
    "bic",
    "uniform_choice_nll",
    "nll_improvement_over_uniform",
    "uniform_choice_fit",
    "n_parameters_at_lower_bound",
    "n_parameters_at_upper_bound",
    "n_parameters_at_any_bound",
    CONVERGED_COLUMN,
    LEARNING_RATE_PARAMETER_NAME,
}

COMPARISON_REQUIRED_COLUMNS = {
    "comparison_eligible",
    "delta_aic",
    "delta_bic",
    "best_aic",
    "best_bic",
}

SUMMARY_REQUIRED_COLUMNS = {
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
}

FIT_FLOAT_COLUMNS = (
    NLL_COLUMN,
    "log_likelihood",
    "aic",
    "bic",
    "uniform_choice_nll",
    "nll_improvement_over_uniform",
)

FIT_INTEGER_COLUMNS = (
    "n_parameters_at_lower_bound",
    "n_parameters_at_upper_bound",
    "n_parameters_at_any_bound",
)

FIT_BOOLEAN_COLUMNS = (
    "uniform_choice_fit",
    CONVERGED_COLUMN,
)

COMPARISON_FLOAT_COLUMNS = (
    "delta_aic",
    "delta_bic",
)

COMPARISON_BOOLEAN_COLUMNS = (
    "comparison_eligible",
    "best_aic",
    "best_bic",
)

SUMMARY_FLOAT_COLUMNS = (
    "convergence_rate",
    "mean_negative_log_likelihood",
    "mean_aic",
    "mean_bic",
)

SUMMARY_INTEGER_COLUMNS = (
    "n_fits",
    "n_converged",
    "n_comparisons",
    "aic_wins",
    "bic_wins",
)


def _require_dataframe(data: object, *, name: str) -> DataFrame:
    """Validate that a value is a pandas DataFrame."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(data).__name__}.")

    return data


def _validate_columns(
    data: DataFrame,
    required_columns: set[str],
    *,
    table_name: str,
) -> None:
    """Validate required and duplicate column names."""

    if data.columns.has_duplicates:
        duplicates = data.columns[data.columns.duplicated(keep=False)].tolist()
        raise ValueError(f"The {table_name} table contains duplicate columns: {duplicates}")

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"The {table_name} table is missing columns: {missing_text}")


def _normalize_float_columns(
    data: DataFrame,
    columns: Sequence[str],
    *,
    table_name: str,
) -> None:
    """Normalize required finite numeric columns in place."""

    for column_name in columns:
        try:
            numeric_values = pd.to_numeric(
                data[column_name],
                errors="raise",
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{table_name}.{column_name} contains nonnumeric values.") from error

        numeric_array = numeric_values.to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        if not np.isfinite(numeric_array).all():
            invalid_count = int((~np.isfinite(numeric_array)).sum())
            raise ValueError(
                f"{table_name}.{column_name} contains {invalid_count} "
                "missing or non-finite value(s)."
            )

        data[column_name] = numeric_array


def _normalize_integer_columns(
    data: DataFrame,
    columns: Sequence[str],
    *,
    table_name: str,
) -> None:
    """Normalize required integer columns in place."""

    for column_name in columns:
        data[column_name] = normalize_integer_series(
            data[column_name],
            column_name=f"{table_name}.{column_name}",
        )


def _normalize_boolean_columns(
    data: DataFrame,
    columns: Sequence[str],
    *,
    table_name: str,
) -> None:
    """Normalize required Boolean columns in place."""

    for column_name in columns:
        data[column_name] = normalize_boolean_series(
            data[column_name],
            column_name=f"{table_name}.{column_name}",
        )


def _normalize_fit_like_table(
    data: DataFrame,
    *,
    table_name: str,
    include_comparison_columns: bool,
) -> DataFrame:
    """Normalize one fit or model-comparison result table."""

    normalized = data.copy()
    normalized_keys = normalize_subject_key_columns(normalized)

    for column_name in PARTICIPANT_KEY_COLUMNS:
        normalized[column_name] = normalized_keys[column_name]

    normalized[MODEL_COLUMN] = normalize_nonempty_string_series(
        normalized[MODEL_COLUMN],
        column_name=f"{table_name}.{MODEL_COLUMN}",
    )
    normalized[SOURCE_STUDY_COLUMN] = normalize_nonempty_string_series(
        normalized[SOURCE_STUDY_COLUMN],
        column_name=f"{table_name}.{SOURCE_STUDY_COLUMN}",
    )

    _normalize_float_columns(
        normalized,
        FIT_FLOAT_COLUMNS,
        table_name=table_name,
    )
    _normalize_integer_columns(
        normalized,
        FIT_INTEGER_COLUMNS,
        table_name=table_name,
    )
    _normalize_boolean_columns(
        normalized,
        FIT_BOOLEAN_COLUMNS,
        table_name=table_name,
    )

    if include_comparison_columns:
        _normalize_float_columns(
            normalized,
            COMPARISON_FLOAT_COLUMNS,
            table_name=table_name,
        )
        _normalize_boolean_columns(
            normalized,
            COMPARISON_BOOLEAN_COLUMNS,
            table_name=table_name,
        )

    duplicate_mask = normalized.duplicated(
        subset=[*PARTICIPANT_KEY_COLUMNS, MODEL_COLUMN],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_keys = normalized.loc[
            duplicate_mask,
            [*PARTICIPANT_KEY_COLUMNS, MODEL_COLUMN],
        ]
        raise ValueError(
            f"The {table_name} table contains duplicate participant-model "
            f"keys:\n{duplicate_keys.to_string(index=False)}"
        )

    return normalized


def _normalize_summary_table(summary: DataFrame) -> DataFrame:
    """Normalize the aggregate model-summary table."""

    normalized = summary.copy()
    normalized[MODEL_COLUMN] = normalize_nonempty_string_series(
        normalized[MODEL_COLUMN],
        column_name=f"summary.{MODEL_COLUMN}",
    )
    _normalize_float_columns(
        normalized,
        SUMMARY_FLOAT_COLUMNS,
        table_name="summary",
    )
    _normalize_integer_columns(
        normalized,
        SUMMARY_INTEGER_COLUMNS,
        table_name="summary",
    )

    if normalized[MODEL_COLUMN].duplicated(keep=False).any():
        duplicate_models = sorted(
            normalized.loc[
                normalized[MODEL_COLUMN].duplicated(keep=False),
                MODEL_COLUMN,
            ]
            .unique()
            .tolist()
        )
        raise ValueError(f"The summary table contains duplicate model rows: {duplicate_models}")

    return normalized


def _validate_models(data: DataFrame, *, table_name: str) -> None:
    """Validate that exactly the two expected models are present."""

    expected_models = {
        Q_LEARNING_MODEL_NAME,
        PVL_DELTA_MODEL_NAME,
    }
    observed_models = set(data[MODEL_COLUMN].unique().tolist())

    if observed_models != expected_models:
        missing_models = sorted(expected_models - observed_models)
        unexpected_models = sorted(observed_models - expected_models)
        details: list[str] = []

        if missing_models:
            details.append(f"missing={missing_models}")

        if unexpected_models:
            details.append(f"unexpected={unexpected_models}")

        raise ValueError(f"The {table_name} table has an invalid model set: " + "; ".join(details))


def _validate_subject_model_coverage(
    data: DataFrame,
    *,
    table_name: str,
) -> None:
    """Ensure each participant has exactly one row for each model."""

    model_counts = data.groupby(
        list(PARTICIPANT_KEY_COLUMNS),
        sort=False,
        observed=True,
    )[MODEL_COLUMN].nunique()
    incomplete_mask = model_counts.ne(2)

    if incomplete_mask.any():
        incomplete_count = int(incomplete_mask.sum())
        raise ValueError(
            f"The {table_name} table is missing a model row for {incomplete_count} participant(s)."
        )


def _validate_source_study_consistency(
    data: DataFrame,
    *,
    table_name: str,
) -> None:
    """Ensure both model rows assign each participant to one study."""

    study_counts = data.groupby(
        list(PARTICIPANT_KEY_COLUMNS),
        sort=False,
        observed=True,
    )[SOURCE_STUDY_COLUMN].nunique()
    mismatch_mask = study_counts.ne(1)

    if mismatch_mask.any():
        mismatch_count = int(mismatch_mask.sum())
        raise ValueError(
            f"The {table_name} table assigns multiple source studies to "
            f"{mismatch_count} participant(s)."
        )


def _validate_fits_and_comparison_alignment(
    fits: DataFrame,
    comparison: DataFrame,
    *,
    config: AnalysisConfig,
) -> None:
    """Validate that fit values are preserved in the comparison table."""

    if len(fits) != len(comparison):
        raise ValueError(
            "Fit and comparison tables have different row counts: "
            f"fits={len(fits)}, comparison={len(comparison)}."
        )

    key_columns = [*PARTICIPANT_KEY_COLUMNS, MODEL_COLUMN]
    common_columns = [
        column_name for column_name in fits.columns if column_name in comparison.columns
    ]
    value_columns = [
        column_name for column_name in common_columns if column_name not in key_columns
    ]
    fits_indexed = fits.set_index(key_columns).sort_index()
    comparison_indexed = comparison.set_index(key_columns).sort_index()

    if not fits_indexed.index.equals(comparison_indexed.index):
        raise ValueError(
            "Fit and comparison tables do not contain the same participant-model keys."
        )

    try:
        pd.testing.assert_frame_equal(
            fits_indexed.loc[:, value_columns],
            comparison_indexed.loc[:, value_columns],
            check_dtype=False,
            check_exact=False,
            rtol=config.numeric_tolerance,
            atol=config.numeric_tolerance,
            check_categorical=False,
        )
    except AssertionError as error:
        raise ValueError(
            "Common fit columns differ between the fit and comparison tables."
        ) from error


def _validate_summary(
    comparison: DataFrame,
    summary: DataFrame,
    *,
    config: AnalysisConfig,
) -> None:
    """Validate the summary table against the comparison table."""

    eligible = comparison.loc[comparison["comparison_eligible"]]
    expected_rows: list[dict[str, float | int | str]] = []

    for model_name in (
        Q_LEARNING_MODEL_NAME,
        PVL_DELTA_MODEL_NAME,
    ):
        model_rows = comparison.loc[comparison[MODEL_COLUMN].eq(model_name)]
        model_eligible = eligible.loc[eligible[MODEL_COLUMN].eq(model_name)]
        converged = model_rows[CONVERGED_COLUMN].to_numpy(dtype=np.bool_)
        negative_log_likelihood = model_rows[NLL_COLUMN].to_numpy(dtype=np.float64)
        aic = model_rows["aic"].to_numpy(dtype=np.float64)
        bic = model_rows["bic"].to_numpy(dtype=np.float64)
        best_aic = model_eligible["best_aic"].to_numpy(dtype=np.bool_)
        best_bic = model_eligible["best_bic"].to_numpy(dtype=np.bool_)

        expected_rows.append(
            {
                MODEL_COLUMN: model_name,
                "n_fits": int(model_rows.shape[0]),
                "n_converged": int(np.count_nonzero(converged)),
                "convergence_rate": float(np.mean(converged)),
                "n_comparisons": int(model_eligible.shape[0]),
                "mean_negative_log_likelihood": float(np.mean(negative_log_likelihood)),
                "mean_aic": float(np.mean(aic)),
                "mean_bic": float(np.mean(bic)),
                "aic_wins": int(np.count_nonzero(best_aic)),
                "bic_wins": int(np.count_nonzero(best_bic)),
            }
        )

    expected = DataFrame(expected_rows).sort_values(MODEL_COLUMN).reset_index(drop=True)
    observed = summary.loc[:, expected.columns].sort_values(MODEL_COLUMN).reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(
            observed,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=config.numeric_tolerance,
            atol=config.numeric_tolerance,
        )
    except AssertionError as error:
        raise ValueError("The summary table is inconsistent with the comparison table.") from error


def validate_result_tables(
    tables: ResultTables,
    config: AnalysisConfig,
) -> ResultTables:
    """Validate and normalize all result tables."""

    fits_input = _require_dataframe(tables.fits, name="fits")
    comparison_input = _require_dataframe(
        tables.comparison,
        name="comparison",
    )
    summary_input = _require_dataframe(tables.summary, name="summary")

    fit_required_columns = {
        *FIT_REQUIRED_COLUMNS,
        *PARTICIPANT_KEY_COLUMNS,
    }
    comparison_required_columns = {
        *fit_required_columns,
        *COMPARISON_REQUIRED_COLUMNS,
    }

    _validate_columns(fits_input, fit_required_columns, table_name="fits")
    _validate_columns(
        comparison_input,
        comparison_required_columns,
        table_name="comparison",
    )
    _validate_columns(
        summary_input,
        SUMMARY_REQUIRED_COLUMNS,
        table_name="summary",
    )

    fits = _normalize_fit_like_table(
        fits_input,
        table_name="fits",
        include_comparison_columns=False,
    )
    comparison = _normalize_fit_like_table(
        comparison_input,
        table_name="comparison",
        include_comparison_columns=True,
    )
    summary = _normalize_summary_table(summary_input)

    for table_name, data in (
        ("fits", fits),
        ("comparison", comparison),
        ("summary", summary),
    ):
        _validate_models(data, table_name=table_name)

    for table_name, data in (
        ("fits", fits),
        ("comparison", comparison),
    ):
        _validate_subject_model_coverage(data, table_name=table_name)
        _validate_source_study_consistency(data, table_name=table_name)

    _validate_fits_and_comparison_alignment(
        fits,
        comparison,
        config=config,
    )
    _validate_summary(
        comparison,
        summary,
        config=config,
    )

    return ResultTables(
        fits=fits,
        comparison=comparison,
        summary=summary,
    )
