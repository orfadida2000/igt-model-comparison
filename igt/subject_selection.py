"""Select and apply participant keys for targeted fitting analyses."""

from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

from igt.constants.schema import PARTICIPANT_KEY_COLUMNS
from igt.models import ComputationalModel, QLearningModel

DEFAULT_INVERSE_TEMPERATURE_SELECTION_THRESHOLD = 19.5


def _normalize_integer_series(
    series: pd.Series,
    *,
    column_name: str,
) -> pd.Series:
    """Validate and convert a series to NumPy int64 values.

    Args:
        series: Series containing values expected to represent integers.
        column_name: Column name used in validation error messages.

    Returns:
        A series containing the normalized values with NumPy ``int64``
        dtype and the same index and name as the input series.

    Raises:
        ValueError: If the series has Boolean, missing, nonnumeric,
            non-finite, non-integer, or out-of-range values.
    """

    if is_bool_dtype(series.dtype):
        raise ValueError(f"{column_name} must contain integers, not Boolean values.")

    try:
        numeric_values = pd.to_numeric(
            series,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{column_name} contains values that cannot be interpreted as numbers."
        ) from error

    if numeric_values.isna().any():
        raise ValueError(f"{column_name} contains missing values.")

    if numeric_values.empty:
        return pd.Series(
            index=series.index,
            dtype=np.int64,
            name=series.name,
        )

    int64_info = np.iinfo(np.int64)

    if is_integer_dtype(numeric_values.dtype):
        minimum = int(numeric_values.min())
        maximum = int(numeric_values.max())

        if minimum < int64_info.min or maximum > int64_info.max:
            raise ValueError(f"{column_name} contains values outside the int64 range.")

        return numeric_values.astype(np.int64)

    numeric_array = numeric_values.to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    if not np.isfinite(numeric_array).all():
        raise ValueError(f"{column_name} contains non-finite values.")

    if not np.equal(numeric_array, np.trunc(numeric_array)).all():
        raise ValueError(f"{column_name} must contain only integer values.")

    if np.any(numeric_array < -(2**63)) or np.any(numeric_array >= 2**63):
        raise ValueError(f"{column_name} contains values outside the int64 range.")

    return pd.Series(
        numeric_array.astype(np.int64),
        index=series.index,
        name=series.name,
    )


def normalize_subject_key_columns(
    subject_keys: pd.DataFrame,
) -> pd.DataFrame:
    """Validate participant-key columns while preserving rows and order.

    Args:
        subject_keys: Table containing the participant-key columns defined
            by ``PARTICIPANT_KEY_COLUMNS``.

    Returns:
        A copy containing only the participant-key columns, normalized to
        NumPy ``int64`` values while preserving row order and index.

    Raises:
        TypeError: If ``subject_keys`` is not a pandas DataFrame.
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
        normalized[column_name] = _normalize_integer_series(
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
            by ``PARTICIPANT_KEY_COLUMNS``.

    Returns:
        A normalized participant-key table with duplicate rows removed and
        rows sorted by ``PARTICIPANT_KEY_COLUMNS``.

    Raises:
        TypeError: If ``subject_keys`` is not a pandas DataFrame.
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


def _normalize_boolean_series(
    series: pd.Series,
    *,
    column_name: str,
) -> pd.Series:
    """Return a strict Boolean series parsed from common CSV representations.

    Args:
        series: Series containing Boolean values or supported textual and
            numeric Boolean representations.
        column_name: Column name used in validation error messages.

    Returns:
        A Boolean series with the same index as the input series.

    Raises:
        ValueError: If the series contains missing values or values other
            than ``True``, ``False``, ``"true"``, ``"false"``, ``1``, or
            ``0``.
    """

    if is_bool_dtype(series.dtype):
        if series.isna().any():
            raise ValueError(f"{column_name} contains missing values.")

        return series.astype(bool)

    normalized = series.astype("string").str.strip().str.lower()

    if normalized.isna().any():
        raise ValueError(f"{column_name} contains missing values.")

    parsed = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    )

    invalid_mask = parsed.isna()

    if invalid_mask.any():
        invalid_values = sorted(normalized.loc[invalid_mask].unique().tolist())

        raise ValueError(
            f"{column_name} contains values that cannot be interpreted "
            f"as booleans: {invalid_values}"
        )

    return parsed.astype(bool)


def _normalize_model_series(
    series: pd.Series,
    *,
    column_name: str = "model",
) -> pd.Series:
    """Validate and normalize a series of model names.

    Args:
        series: Series containing model names.
        column_name: Column name used in validation error messages.

    Returns:
        A string series containing model names with surrounding whitespace
        removed.

    Raises:
        ValueError: If any model name is missing or empty.
    """

    normalized = series.astype("string").str.strip()
    invalid_mask = normalized.isna() | normalized.eq("")

    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())
        raise ValueError(f"{column_name} contains {invalid_count} missing or empty value(s).")

    return normalized


def _validate_model_name(model: str) -> str:
    """Validate and normalize a model name.

    Args:
        model: Model name to validate.

    Returns:
        The model name with surrounding whitespace removed.

    Raises:
        TypeError: If ``model`` is not a string.
        ValueError: If the normalized model name is empty.
    """

    if not isinstance(model, str):
        raise TypeError(f"model must be a string, got {type(model).__name__}.")

    model_name = model.strip()

    if not model_name:
        raise ValueError("model must not be empty.")

    return model_name


def _validate_nonnegative_finite_float(
    value: Real | float,
    *,
    parameter_name: str,
) -> float:
    """Validate a finite, nonnegative real-valued argument.

    Args:
        value: Value to validate.
        parameter_name: Parameter name used in validation error messages.

    Returns:
        The validated value converted to ``float``.

    Raises:
        TypeError: If ``value`` is Boolean or is not a real number.
        ValueError: If ``value`` is non-finite or negative.
    """

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{parameter_name} must be a real number, got {type(value).__name__}.")

    parsed_value = float(value)

    if not np.isfinite(parsed_value):
        raise ValueError(f"{parameter_name} must be finite.")

    if parsed_value < 0.0:
        raise ValueError(f"{parameter_name} must be nonnegative.")

    return parsed_value


def _validate_threshold(threshold: Real | float) -> float:
    """Validate and normalize an inverse-temperature threshold.

    Args:
        threshold: Threshold value to validate.

    Returns:
        The validated threshold converted to ``float``.

    Raises:
        TypeError: If ``threshold`` is Boolean or is not a real number.
        ValueError: If ``threshold`` is non-finite or negative.
    """

    return _validate_nonnegative_finite_float(
        threshold,
        parameter_name="threshold",
    )


def _prepare_fit_results_for_nll_selection(
    fit_results: pd.DataFrame,
    *,
    require_convergence: bool,
) -> pd.DataFrame:
    """Validate and normalize fit-result rows used for NLL comparison.

    Args:
        fit_results: Per-model fit-results table.
        require_convergence: Whether the normalized table must include and
            validate the ``converged`` column.

    Returns:
        A copy containing normalized participant keys, model names,
        negative log-likelihood values, and optionally convergence values.

    Raises:
        TypeError: If ``fit_results`` is not a pandas DataFrame or
            ``require_convergence`` is not Boolean.
        ValueError: If a required column is missing, a value is invalid, or
            duplicate participant-model combinations are present.
    """

    if not isinstance(fit_results, pd.DataFrame):
        raise TypeError(
            f"fit_results must be a pandas DataFrame, got {type(fit_results).__name__}."
        )

    if not isinstance(require_convergence, (bool, np.bool_)):
        raise TypeError("require_convergence must be a Boolean value.")

    key_columns = list(PARTICIPANT_KEY_COLUMNS)

    relevant_columns = [
        *key_columns,
        "model",
        "negative_log_likelihood",
    ]

    if require_convergence:
        relevant_columns.append("converged")

    missing_columns = set(relevant_columns) - set(fit_results.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Fit-results table is missing columns: {missing_text}")

    results = fit_results.loc[:, relevant_columns].copy()

    normalized_keys = normalize_subject_key_columns(results)

    for column_name in key_columns:
        results[column_name] = normalized_keys[column_name]

    results["model"] = _normalize_model_series(
        results["model"],
    )

    try:
        nll_values = pd.to_numeric(
            results["negative_log_likelihood"],
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "negative_log_likelihood contains values that cannot be interpreted as numbers."
        ) from error

    nll_array = nll_values.to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    if not np.isfinite(nll_array).all():
        invalid_count = int((~np.isfinite(nll_array)).sum())
        raise ValueError(
            f"negative_log_likelihood contains {invalid_count} missing or non-finite value(s)."
        )

    results["negative_log_likelihood"] = nll_array

    if require_convergence:
        results["converged"] = _normalize_boolean_series(
            results["converged"],
            column_name="converged",
        )

    duplicate_columns = [
        *key_columns,
        "model",
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

    model_names = tuple(sorted(fit_results["model"].unique().tolist()))

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
    )["model"].nunique()

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
            ``converged`` column.

    Returns:
        A copy containing all model rows for subjects whose model fits all
        converged.
    """

    key_columns = list(PARTICIPANT_KEY_COLUMNS)

    fully_converged_mask = fit_results.groupby(
        key_columns,
        sort=False,
        observed=True,
    )["converged"].transform("all")

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


def read_fit_results_csv(
    fit_results_csv: Path,
) -> pd.DataFrame:
    """Read a fit-results CSV file.

    Args:
        fit_results_csv: Path to the fit-results CSV file.

    Returns:
        The parsed fit-results table.

    Raises:
        FileNotFoundError: If ``fit_results_csv`` does not identify an
            existing file.
    """

    csv_path = Path(fit_results_csv)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Fit-results CSV does not exist: {csv_path}")

    return pd.read_csv(csv_path)


def select_model_lowest_nll_subject_keys(
    fit_results: pd.DataFrame,
    *,
    model: ComputationalModel | type[ComputationalModel] | str,
    epsilon: Real | float = 1e-8,
    require_convergence: bool = True,
) -> pd.DataFrame:
    """Select subjects for whom one model has the lowest NLL.

    The requested model is selected only when its NLL is lower than the
    best competing model's NLL by more than ``epsilon``.

    Args:
        fit_results: Per-model fit-results table.
        model: Model to evaluate, either as a model class, model instance or a model name string.
        epsilon: Minimum required NLL advantage over the best competitor.
        require_convergence: Whether every model fit for a subject must
            have converged.

    Returns:
        Unique participant keys ordered by ``PARTICIPANT_KEY_COLUMNS``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument or fit-results value is invalid, model
            coverage is incomplete, or the table cannot support the
            requested comparison.
    """

    model_name = model if isinstance(model, str) else model.get_name()
    model_name = _validate_model_name(model_name)

    parsed_epsilon = _validate_nonnegative_finite_float(
        epsilon,
        parameter_name="epsilon",
    )

    results = _prepare_fit_results_for_nll_selection(
        fit_results,
        require_convergence=require_convergence,
    )

    _validate_subject_model_coverage(
        results,
        model=model_name,
    )

    if require_convergence:
        results = _select_fully_converged_subject_rows(results)

    if results.empty:
        return _empty_subject_keys_from(results)

    key_columns = list(PARTICIPANT_KEY_COLUMNS)

    target_results = results.loc[
        results["model"].eq(model_name),
        [
            *key_columns,
            "negative_log_likelihood",
        ],
    ].rename(
        columns={
            "negative_log_likelihood": "target_nll",
        }
    )

    competing_results = results.loc[
        results["model"].ne(model_name),
        [
            *key_columns,
            "negative_log_likelihood",
        ],
    ]

    best_competing_results = competing_results.groupby(
        key_columns,
        as_index=False,
        sort=False,
        observed=True,
    ).agg(
        best_competing_nll=(
            "negative_log_likelihood",
            "min",
        )
    )

    comparisons = target_results.merge(
        best_competing_results,
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )

    nll_advantage = comparisons["best_competing_nll"] - comparisons["target_nll"]

    selected_mask = nll_advantage.gt(parsed_epsilon)

    return normalize_subject_keys(
        comparisons.loc[
            selected_mask,
            key_columns,
        ]
    )


def select_model_lowest_nll_subject_keys_from_csv(
    fit_results_csv: Path,
    *,
    model: ComputationalModel | type[ComputationalModel] | str,
    epsilon: Real | float = 1e-8,
    require_convergence: bool = True,
) -> pd.DataFrame:
    """Read a fit-results CSV and select subjects won by one model.

    Args:
        fit_results_csv: Path to the per-model fit-results CSV file.
        model: Model to evaluate, either as a model class, model instance or a model name string.
        epsilon: Minimum required NLL advantage over the best competitor.
        require_convergence: Whether every model fit for a subject must
            have converged.

    Returns:
        Unique participant keys ordered by ``PARTICIPANT_KEY_COLUMNS``.

    Raises:
        FileNotFoundError: If ``fit_results_csv`` does not identify an
            existing file.
        TypeError: If an argument has an invalid type.
        ValueError: If the CSV contents or selection arguments are invalid.
    """

    fit_results = read_fit_results_csv(fit_results_csv)

    return select_model_lowest_nll_subject_keys(
        fit_results,
        model=model,
        epsilon=epsilon,
        require_convergence=require_convergence,
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
        Unique participant keys ordered by ``PARTICIPANT_KEY_COLUMNS``.

    Raises:
        TypeError: If ``fit_results`` is not a pandas DataFrame or
            ``require_convergence`` is not Boolean.
        ValueError: If a required column is missing or contains invalid
            values.
    """

    if not isinstance(fit_results, pd.DataFrame):
        raise TypeError(
            f"fit_results must be a pandas DataFrame, got {type(fit_results).__name__}."
        )

    if not isinstance(require_convergence, (bool, np.bool_)):
        raise TypeError("require_convergence must be a Boolean value.")

    parsed_threshold = _validate_threshold(threshold)

    required_columns = {
        "model",
        "inverse_temperature",
        *PARTICIPANT_KEY_COLUMNS,
    }

    if require_convergence:
        required_columns.add("converged")

    missing_columns = required_columns - set(fit_results.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Fit-results table is missing columns: {missing_text}")

    normalized_models = _normalize_model_series(
        fit_results["model"],
    )

    q_results = fit_results.loc[normalized_models.eq(QLearningModel.get_name())].copy()

    if q_results.empty:
        return normalize_subject_keys(pd.DataFrame(columns=list(PARTICIPANT_KEY_COLUMNS)))

    try:
        inverse_temperatures = pd.to_numeric(
            q_results["inverse_temperature"],
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
        converged = _normalize_boolean_series(
            q_results["converged"],
            column_name="converged",
        )
        selected_mask &= converged

    return normalize_subject_keys(
        q_results.loc[
            selected_mask,
            list(PARTICIPANT_KEY_COLUMNS),
        ]
    )


def select_q_inverse_temperature_subject_keys_from_csv(
    fit_results_csv: Path,
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
        Unique participant keys ordered by ``PARTICIPANT_KEY_COLUMNS``.

    Raises:
        FileNotFoundError: If ``fit_results_csv`` does not identify an
            existing file.
        TypeError: If an argument has an invalid type.
        ValueError: If the CSV contents or selection arguments are invalid.
    """

    fit_results = read_fit_results_csv(fit_results_csv)

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
        TypeError: If ``data`` or ``subject_keys`` is not a pandas
            DataFrame.
        ValueError: If required columns are missing, participant keys are
            invalid, or requested participant keys are absent from
            ``data``.
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
