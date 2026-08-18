"""Participant-key validation, filtering, and targeted fit-result selection.

The helpers normalize compound `(n_trials, subject_id)` keys, validate model-result
tables, select participants by likelihood or Q-learning inverse-temperature criteria,
and filter long-format trial data to explicit participant subsets.
"""

from collections.abc import Iterable
from numbers import Real

import numpy as np
import pandas as pd

from igt.constants.models import INVERSE_TEMPERATURE_PARAMETER_NAME
from igt.constants.schema import (
    CONVERGED_COLUMN,
    MODEL_COLUMN,
    N_TRIALS_COLUMN,
    NLL_COLUMN,
    PARTICIPANT_KEY_COLUMNS,
)
from igt.models import ComputationalModel, QLearningModel
from igt.typing import CustomTypeError, NonEmptyUniformTuple, StrPathLike
from igt.utils.io import read_csv
from igt.utils.tabular import (
    normalize_boolean_series,
    normalize_integer_series,
    normalize_nonempty_string_series,
)

DEFAULT_INVERSE_TEMPERATURE_SELECTION_THRESHOLD = 19.5


def normalize_subject_key_columns(
    subject_keys: pd.DataFrame,
) -> pd.DataFrame:
    """Validate participant-key columns while preserving rows and order.

    Args:
        subject_keys: Table containing the participant-key columns defined
            by `PARTICIPANT_KEY_COLUMNS`.

    Returns:
        A copy containing only the participant-key columns, normalized to
        NumPy `int64` values while preserving row order and index.

    Raises:
        TypeError: If `subject_keys` is not a pandas DataFrame.
        ValueError: If a required participant-key column is missing or
            contains an invalid integer value.
    """

    if not isinstance(subject_keys, pd.DataFrame):
        raise TypeError(
            f"subject_keys must be a pandas DataFrame, got {type(subject_keys).__name__}."
        )

    key_columns = list(PARTICIPANT_KEY_COLUMNS)
    missing_columns = set(key_columns) - set(subject_keys.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Subject-key table is missing columns: {missing_text}")

    normalized = subject_keys.loc[:, key_columns].copy()

    for column_name in key_columns:
        normalized[column_name] = normalize_integer_series(
            normalized[column_name],
            column_name=column_name,
        )

    return normalized


def normalize_subject_keys(
    subject_keys: pd.DataFrame,
) -> pd.DataFrame:
    """Validate, deduplicate, and sort participant-key rows.

    Args:
        subject_keys: Table containing the participant-key columns defined
            by `PARTICIPANT_KEY_COLUMNS`.

    Returns:
        A normalized participant-key table with duplicate rows removed and
        rows sorted by `PARTICIPANT_KEY_COLUMNS`.

    Raises:
        TypeError: If `subject_keys` is not a pandas DataFrame.
        ValueError: If a required participant-key column is missing or
            contains an invalid integer value.
    """

    key_columns = list(PARTICIPANT_KEY_COLUMNS)
    normalized = normalize_subject_key_columns(subject_keys)

    return normalized.drop_duplicates().sort_values(
        by=key_columns,
        kind="mergesort",
        ignore_index=True,
    )


def _validate_model_name(model: str) -> str:
    """Validate and normalize a model name.

    Args:
        model: Model name to validate.

    Returns:
        The model name with surrounding whitespace removed.

    Raises:
        TypeError: If `model` is not a string.
        ValueError: If the normalized model name is empty.
    """

    if not isinstance(model, str):
        raise TypeError(f"model must be a string, got {type(model).__name__}.")

    model_name = model.strip()

    if not model_name:
        raise ValueError("model must not be empty.")

    return model_name


def _validate_finite_float(
    value: Real | float | int,
    *,
    parameter_name: str,
) -> float:
    """Validate a finite real-valued argument.

    Args:
        value: Value to validate.
        parameter_name: Parameter name used in validation error messages.

    Returns:
        The validated value converted to `float`.

    Raises:
        TypeError: If `value` is Boolean or is not a real number.
        ValueError: If `value` is non-finite.
    """

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, float, int)):
        raise TypeError(
            f"{parameter_name} must be a real number (float or int), got {type(value).__name__}."
        )

    try:
        parsed_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{parameter_name} cannot be interpreted as a number.") from error

    if not np.isfinite(parsed_value):
        raise ValueError(f"{parameter_name} must be finite.")

    return parsed_value


def _validate_nonnegative_finite_float(
    value: Real | float | int,
    *,
    parameter_name: str,
) -> float:
    """Validate a finite, nonnegative real-valued argument.

    Args:
        value: Value to validate.
        parameter_name: Parameter name used in validation error messages.

    Returns:
        The validated value converted to `float`.

    Raises:
        TypeError: If `value` is Boolean or is not a real number.
        ValueError: If `value` is non-finite or negative.
    """

    parsed_value = _validate_finite_float(
        value,
        parameter_name=parameter_name,
    )

    if parsed_value < 0.0:
        raise ValueError(f"{parameter_name} must be nonnegative.")

    return parsed_value


def _validate_positive_finite_float(
    value: Real | float | int,
    *,
    parameter_name: str,
) -> float:
    """Validate a finite, positive real-valued argument.

    Args:
        value: Value to validate.
        parameter_name: Parameter name used in validation error messages.

    Returns:
        The validated value converted to `float`.

    Raises:
        TypeError: If `value` is Boolean or is not a real number.
        ValueError: If `value` is non-finite or not positive.
    """

    parsed_value = _validate_nonnegative_finite_float(
        value,
        parameter_name=parameter_name,
    )

    if parsed_value == 0:
        raise ValueError(f"{parameter_name} must be positive.")

    return parsed_value


def _validate_column_names(
    columns: tuple[str, ...], excluded_columns: set[str] | None = None
) -> NonEmptyUniformTuple[str]:
    """Validate a nonempty tuple of unique, permitted column names.

    Args:
        columns: Candidate column names in the order they should be preserved.
        excluded_columns: Optional set of names that must not appear in `columns`.

    Returns:
        The validated column names as a nonempty tuple.

    Raises:
        TypeError: If `columns` is not a tuple or `excluded_columns` is not a set.
        CustomTypeError: If any element of `columns` is not a string.
        ValueError: If `columns` is empty, contains duplicates, or contains a
            disallowed name.
        RuntimeError: If iteration over a value already established to be a tuple
            fails unexpectedly.
    """

    if excluded_columns is None:
        excluded_columns = set()

    if not isinstance(excluded_columns, set):
        raise TypeError(
            f"'excluded_columns' must be a set of strings, got {type(excluded_columns).__name__}."
        )

    if not isinstance(columns, tuple):
        raise TypeError(f"'columns' must be a tuple of strings, got {type(columns).__name__}.")

    clean_columns: Iterable[str] = []

    try:
        for column in columns:
            if not isinstance(column, str):
                raise CustomTypeError(
                    f"'columns' must be a tuple of strings, but found a non-string value: {column!r} of type {type(column).__name__}."
                )

            if column in excluded_columns:
                raise ValueError(
                    f"'columns' contains a disallowed column name: {column!r}. Disallowed columns: {excluded_columns!r}."
                )

            clean_columns.append(column)

        clean_columns = tuple(clean_columns)
    except CustomTypeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"'columns' is determined to be an instance of tuple, but an error occurred while iterating over its elements: {e}"
        ) from e

    if len(clean_columns) == 0:
        raise ValueError(f"'columns' must not be an empty tuple, got {columns!r}.")

    if len(clean_columns) != len(set(clean_columns)):
        raise ValueError(f"'columns' must not contain duplicate values, got {columns!r}.")

    return clean_columns


def _prepare_fit_results_for_selection(
    fit_results: pd.DataFrame,
    comparison_columns: tuple[str, ...],
    *,
    require_converge_column: bool,
) -> pd.DataFrame:
    """Validate fit-result rows for comparison-based subject selection.

    The returned table contains only participant keys, model identity, the
    requested comparison columns, and, when requested, convergence status.
    Numeric comparison columns are converted to finite floating-point values.

    Args:
        fit_results: Per-model fit-results table.
        comparison_columns: Unique numeric columns used for model comparison.
        require_converge_column: Whether a valid `converged` column is
            required and normalized.

    Returns:
        A normalized copy containing the columns required for subject
        selection.

    Raises:
        TypeError: If an argument has an invalid container or Boolean type.
        ValueError: If a required column is missing, a comparison column is
            disallowed or nonnumeric, a value is non-finite, or duplicate
            participant-model rows are present.
    """

    if not isinstance(fit_results, pd.DataFrame):
        raise TypeError(
            f"fit_results must be a pandas DataFrame, got {type(fit_results).__name__}."
        )

    if not isinstance(require_converge_column, (bool, np.bool_)):
        raise TypeError("require_converge_column must be a Boolean value.")

    comparison_columns = _validate_column_names(
        comparison_columns,
        excluded_columns={MODEL_COLUMN, CONVERGED_COLUMN, *PARTICIPANT_KEY_COLUMNS},
    )

    relevant_columns = [*PARTICIPANT_KEY_COLUMNS, MODEL_COLUMN, *comparison_columns]

    if require_converge_column:
        relevant_columns.append(CONVERGED_COLUMN)

    missing_columns = set(relevant_columns) - set(fit_results.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Fit-results table is missing columns: {missing_text}")

    results = fit_results.loc[:, relevant_columns].copy()

    normalized_keys = normalize_subject_key_columns(results)

    for column_name in PARTICIPANT_KEY_COLUMNS:
        results[column_name] = normalized_keys[column_name]

    results[MODEL_COLUMN] = normalize_nonempty_string_series(
        results[MODEL_COLUMN],
        column_name=MODEL_COLUMN,
    )

    for comparison_column in comparison_columns:
        try:
            comparison_values = pd.to_numeric(
                results[comparison_column],
                errors="raise",
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{comparison_column} contains values that cannot be interpreted as numbers."
            ) from error

        comparison_values_array = comparison_values.to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        if not np.isfinite(comparison_values_array).all():
            invalid_count = int((~np.isfinite(comparison_values_array)).sum())
            raise ValueError(
                f"{comparison_column} contains {invalid_count} missing or non-finite value(s)."
            )

        results[comparison_column] = comparison_values_array

    if require_converge_column:
        results[CONVERGED_COLUMN] = normalize_boolean_series(
            results[CONVERGED_COLUMN],
            column_name=CONVERGED_COLUMN,
        )

    duplicate_columns = [
        *PARTICIPANT_KEY_COLUMNS,
        MODEL_COLUMN,
    ]

    duplicate_mask = results.duplicated(
        subset=duplicate_columns,
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            "Fit-results table contains "
            f"{duplicate_count} rows belonging to duplicate "
            "participant-model combinations."
        )

    return results


def _prepare_fit_results_for_nll_selection(
    fit_results: pd.DataFrame,
    *,
    require_converge_column: bool,
) -> pd.DataFrame:
    """Validate and normalize fit-result rows used for NLL comparison.

    Args:
        fit_results: Per-model fit-results table.
        require_converge_column: Whether the normalized table must include a `converged` column that can be normalized into boolean type.

    Returns:
        A copy containing normalized participant keys, model names,
        negative log-likelihood values, and optionally convergence values.

    Raises:
        TypeError: If `fit_results` is not a pandas DataFrame or
            `require_converge_column` is not Boolean.
        ValueError: If a required column is missing, a value is invalid, or
            duplicate participant-model combinations are present.
    """

    return _prepare_fit_results_for_selection(
        fit_results,
        comparison_columns=(NLL_COLUMN,),
        require_converge_column=require_converge_column,
    )


def _validate_subject_model_coverage(
    fit_results: pd.DataFrame,
    *,
    model: str,
) -> None:
    """Ensure every subject has one result for every available model.

    Args:
        fit_results: Normalized per-model fit-results table.
        model: Name of the target model that must be present.

    Raises:
        ValueError: If no models are present, the target model is absent,
            fewer than two models are present, or any subject is missing a
            result for an available model.
    """

    model_names = tuple(sorted(fit_results[MODEL_COLUMN].unique().tolist()))

    if not model_names:
        raise ValueError("Fit-results table does not contain any model results.")

    if model not in model_names:
        available_text = ", ".join(model_names)
        raise ValueError(
            f"Model {model!r} is not present in the fit-results table. "
            f"Available models: {available_text}"
        )

    if len(model_names) < 2:
        raise ValueError("At least two models are required for NLL comparison.")

    key_columns = list(PARTICIPANT_KEY_COLUMNS)

    model_counts = fit_results.groupby(
        key_columns,
        sort=False,
        observed=True,
    )[MODEL_COLUMN].nunique()

    incomplete_mask = model_counts.ne(len(model_names))

    if incomplete_mask.any():
        incomplete_count = int(incomplete_mask.sum())
        raise ValueError(
            "Fit-results table does not contain results for every model "
            f"for {incomplete_count} subject(s)."
        )


def _select_fully_converged_subject_rows(
    fit_results: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only subjects for whom every model fit converged.

    Args:
        fit_results: Normalized fit-results table containing a
            `converged` column.

    Returns:
        A copy containing all model rows for subjects whose model fits all
        converged.
    """

    key_columns = list(PARTICIPANT_KEY_COLUMNS)

    fully_converged_mask = fit_results.groupby(
        key_columns,
        sort=False,
        observed=True,
    )[CONVERGED_COLUMN].transform("all")

    return fit_results.loc[fully_converged_mask].copy()


def _empty_subject_keys_from(
    fit_results: pd.DataFrame,
) -> pd.DataFrame:
    """Create an empty normalized participant-key table.

    Args:
        fit_results: Fit-results table containing the participant-key
            columns.

    Returns:
        An empty normalized DataFrame with the columns and dtypes expected
        for participant keys.
    """

    key_columns = list(PARTICIPANT_KEY_COLUMNS)

    return normalize_subject_keys(
        fit_results.loc[
            fit_results.index[:0],
            key_columns,
        ]
    )


def select_subjects_with_target_is_uniquely_nll_best_model(
    fit_results: pd.DataFrame,
    *,
    target_model: ComputationalModel | type[ComputationalModel] | str,
    atol_per_trial: Real | float = 1e-8,
    fully_converged: bool = True,
) -> pd.DataFrame:
    """Select subjects for whom the target model is uniquely NLL-best.

    The target model must beat the best competing model by more than the
    participant-specific tolerance `n_trials * atol_per_trial`. This makes the
    comparison tolerance scale with the number of trials contributing to the
    summed negative log-likelihood.

    Args:
        fit_results: Per-model fit-results table.
        target_model: Target model instance, model class, or registered model
            name.
        atol_per_trial: Absolute NLL equality tolerance per trial.
        fully_converged: Whether to consider only subjects for whom every
            model fit converged. The `converged` column is validated in either
            case.

    Returns:
        Unique selected participant keys ordered by
        `PARTICIPANT_KEY_COLUMNS`.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument or fit-results value is invalid, model
            coverage is incomplete, or the table cannot support the
            requested comparison.
    """

    if not isinstance(fully_converged, bool):
        raise TypeError(
            f"fully_converged must be a Boolean value, got {type(fully_converged).__name__}."
        )

    target_model_name = target_model if isinstance(target_model, str) else target_model.get_name()
    target_model_name = _validate_model_name(target_model_name)

    atol_per_trial = _validate_nonnegative_finite_float(
        atol_per_trial,
        parameter_name="atol_per_trial",
    )

    results_df = _prepare_fit_results_for_nll_selection(
        fit_results,
        require_converge_column=True,
    )

    _validate_subject_model_coverage(
        results_df,
        model=target_model_name,
    )

    if fully_converged:
        results_df = _select_fully_converged_subject_rows(results_df)

    if results_df.empty:
        return _empty_subject_keys_from(results_df)

    key_columns = list(PARTICIPANT_KEY_COLUMNS)

    target_nll_col_name = "target_model_" + NLL_COLUMN.strip()
    target_model_results_df = results_df.loc[
        results_df[MODEL_COLUMN].eq(target_model_name),
        [
            *key_columns,
            NLL_COLUMN,
        ],
    ].rename(
        columns={
            NLL_COLUMN: target_nll_col_name,
        }
    )

    competing_models_results_df = results_df.loc[
        results_df[MODEL_COLUMN].ne(target_model_name),
        [
            *key_columns,
            NLL_COLUMN,
        ],
    ]

    competitor_nll_col_name = "competitor_model_" + NLL_COLUMN.strip()
    best_competitor_model_results_df = competing_models_results_df.groupby(
        key_columns,
        as_index=False,
        sort=False,
        observed=True,
    ).agg(
        **{
            competitor_nll_col_name: (
                NLL_COLUMN,
                "min",  # NLL is better when lower
            )
        }
    )

    target_vs_competitor_results_df = target_model_results_df.merge(
        best_competitor_model_results_df,
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )

    # A subject is selected if the target models's NLL value isn't considered as equal to the competitor (the best model out of the competing/remaining models) model's NLL value (i.e. the absolute difference is greater than: number of trials the subject has times the absolute tolerance per trial),
    # and the target model's NLL value is the better (smaller) out of the two. An equivalent check is to check if the target model's NLL value is better (smaller) than the competitor (best competing model) model's NLL value by a magnitude larger than: number of trials the subject has times the absolute tolerance per trial.
    # The magnitude > tolerance gives us the "not equal" part, and the target model's NLL being smaller gives us the "target is best" part.

    per_subject_nll_difference = (
        target_vs_competitor_results_df[competitor_nll_col_name]
        - target_vs_competitor_results_df[target_nll_col_name]
    )

    selected_subjects_mask = (
        per_subject_nll_difference
        > atol_per_trial * target_vs_competitor_results_df[N_TRIALS_COLUMN]
    )
    selected_subjects_keys = target_vs_competitor_results_df.loc[
        selected_subjects_mask,
        key_columns,
    ]

    return normalize_subject_keys(selected_subjects_keys)


def select_subjects_with_target_is_uniquely_nll_best_model_from_csv(
    fit_results_csv: StrPathLike,
    *,
    target_model: ComputationalModel | type[ComputationalModel] | str,
    atol_per_trial: Real | float = 1e-8,
    fully_converged: bool = True,
) -> pd.DataFrame:
    """Read fit results and select subjects for which the target is NLL-best.

    This is the CSV convenience wrapper around
    [select_subjects_with_target_is_uniquely_nll_best_model][igt.subject_selection.select_subjects_with_target_is_uniquely_nll_best_model].

    Args:
        fit_results_csv: Path to the per-model fit-results CSV file.
        target_model: Target model instance, model class, or registered model
            name.
        atol_per_trial: Absolute NLL equality tolerance per trial.
        fully_converged: Whether to consider only subjects for whom every
            model fit converged. The `converged` column is validated in either
            case.

    Returns:
        Unique participant keys ordered by `PARTICIPANT_KEY_COLUMNS`.

    Raises:
        FileNotFoundError: If `fit_results_csv` does not identify an
            existing file.
        TypeError: If an argument has an invalid type.
        ValueError: If the CSV contents or selection arguments are invalid.
    """

    fit_results = read_csv(
        fit_results_csv,
        table_name="fit-results",
    )

    return select_subjects_with_target_is_uniquely_nll_best_model(
        fit_results,
        target_model=target_model,
        atol_per_trial=atol_per_trial,
        fully_converged=fully_converged,
    )


def select_q_inverse_temperature_subject_keys(
    fit_results: pd.DataFrame,
    *,
    threshold: Real | float = DEFAULT_INVERSE_TEMPERATURE_SELECTION_THRESHOLD,
    require_convergence: bool = True,
) -> pd.DataFrame:
    """Select participant keys whose Q-learning estimate reaches a high value.

    Args:
        fit_results: Per-model fit-results table.
        threshold: Inclusive inverse-temperature threshold.
        require_convergence: Whether to exclude nonconverged Q-learning
            fits.

    Returns:
        Unique participant keys ordered by `PARTICIPANT_KEY_COLUMNS`.

    Raises:
        TypeError: If `fit_results` is not a pandas DataFrame or
            `require_convergence` is not Boolean.
        ValueError: If a required column is missing or contains invalid
            values.
    """

    if not isinstance(fit_results, pd.DataFrame):
        raise TypeError(
            f"fit_results must be a pandas DataFrame, got {type(fit_results).__name__}."
        )

    if not isinstance(require_convergence, (bool, np.bool_)):
        raise TypeError("require_convergence must be a Boolean value.")

    parsed_threshold = _validate_nonnegative_finite_float(threshold, parameter_name="threshold")

    required_columns = {
        MODEL_COLUMN,
        INVERSE_TEMPERATURE_PARAMETER_NAME,
        *PARTICIPANT_KEY_COLUMNS,
    }

    if require_convergence:
        required_columns.add(CONVERGED_COLUMN)

    missing_columns = required_columns - set(fit_results.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Fit-results table is missing columns: {missing_text}")

    normalized_models = normalize_nonempty_string_series(
        fit_results[MODEL_COLUMN],
        column_name=MODEL_COLUMN,
    )

    q_results = fit_results.loc[normalized_models.eq(QLearningModel.get_name())].copy()

    if q_results.empty:
        return normalize_subject_keys(pd.DataFrame(columns=list(PARTICIPANT_KEY_COLUMNS)))

    try:
        inverse_temperatures = pd.to_numeric(
            q_results[INVERSE_TEMPERATURE_PARAMETER_NAME],
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Q-learning inverse_temperature contains values that cannot be interpreted as numbers."
        ) from error

    inverse_temperature_array = inverse_temperatures.to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    invalid_temperature_mask = ~np.isfinite(inverse_temperature_array)

    if invalid_temperature_mask.any():
        invalid_count = int(invalid_temperature_mask.sum())
        raise ValueError(
            "Q-learning inverse_temperature contains "
            f"{invalid_count} missing or non-finite value(s)."
        )

    selected_mask = inverse_temperatures.ge(parsed_threshold)

    if require_convergence:
        converged = normalize_boolean_series(
            q_results[CONVERGED_COLUMN],
            column_name=CONVERGED_COLUMN,
        )
        selected_mask &= converged

    return normalize_subject_keys(
        q_results.loc[
            selected_mask,
            list(PARTICIPANT_KEY_COLUMNS),
        ]
    )


def select_q_inverse_temperature_subject_keys_from_csv(
    fit_results_csv: StrPathLike,
    *,
    threshold: Real | float = DEFAULT_INVERSE_TEMPERATURE_SELECTION_THRESHOLD,
    require_convergence: bool = True,
) -> pd.DataFrame:
    """Read a fit-results CSV and select Q-learning participant keys.

    Args:
        fit_results_csv: Path to the per-model fit-results CSV file.
        threshold: Inclusive inverse-temperature threshold.
        require_convergence: Whether to exclude nonconverged Q-learning
            fits.

    Returns:
        Unique participant keys ordered by `PARTICIPANT_KEY_COLUMNS`.

    Raises:
        FileNotFoundError: If `fit_results_csv` does not identify an
            existing file.
        TypeError: If an argument has an invalid type.
        ValueError: If the CSV contents or selection arguments are invalid.
    """

    fit_results = read_csv(
        fit_results_csv,
        table_name="fit-results",
    )

    return select_q_inverse_temperature_subject_keys(
        fit_results,
        threshold=threshold,
        require_convergence=require_convergence,
    )


def filter_subjects_by_keys(
    data: pd.DataFrame,
    subject_keys: pd.DataFrame,
) -> pd.DataFrame:
    """Return trial rows belonging to explicitly requested participants.

    Every requested participant key must exist in the input data. The
    original trial-table columns are preserved, and selected rows are sorted
    chronologically.

    Args:
        data: IGT trial-level dataset.
        subject_keys: Table containing the participant keys to retain.

    Returns:
        A copy containing trial rows for the requested participants, sorted
        by participant-key columns and trial number.

    Raises:
        TypeError: If `data` or `subject_keys` is not a pandas
            DataFrame.
        ValueError: If required columns are missing, participant keys are
            invalid, or requested participant keys are absent from
            `data`.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}.")

    key_columns = list(PARTICIPANT_KEY_COLUMNS)
    required_data_columns = {
        *key_columns,
        "trial",
    }

    missing_data_columns = required_data_columns - set(data.columns)

    if missing_data_columns:
        missing_text = ", ".join(sorted(missing_data_columns))
        raise ValueError(f"IGT data is missing columns: {missing_text}")

    normalized_keys = normalize_subject_keys(subject_keys)

    if normalized_keys.empty:
        return data.iloc[0:0].copy()

    normalized_data_keys = normalize_subject_key_columns(data)

    available_keys = normalized_data_keys.drop_duplicates(
        ignore_index=True,
    )

    key_validation = normalized_keys.merge(
        available_keys,
        on=key_columns,
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    missing_keys = key_validation.loc[
        key_validation["_merge"].eq("left_only"),
        key_columns,
    ]

    if not missing_keys.empty:
        raise ValueError(
            "Requested participant keys were not found in the IGT "
            "dataset:\n"
            f"{missing_keys.to_string(index=False)}"
        )

    requested_index = pd.MultiIndex.from_frame(normalized_keys)
    data_index = pd.MultiIndex.from_frame(normalized_data_keys)

    selected = data.loc[data_index.isin(requested_index)].copy()

    return selected.sort_values(
        by=[
            *key_columns,
            "trial",
        ],
        kind="mergesort",
        ignore_index=True,
    )
