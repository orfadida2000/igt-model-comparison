"""Select and apply participant keys for targeted fitting analyses."""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from igt.constants.schema import PARTICIPANT_KEY_COLUMNS
from igt.models.q_learning import QLearningModel

DEFAULT_INVERSE_TEMPERATURE_SELECTION_THRESHOLD = 19.5


def _validate_threshold(threshold: float) -> float:
    """Return a validated finite, non-negative threshold."""

    if isinstance(threshold, bool):
        raise TypeError("threshold must be a real number, not bool.")

    try:
        parsed_threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise TypeError("threshold must be a real number.") from exc

    if not np.isfinite(parsed_threshold):
        raise ValueError("threshold must be finite.")

    if parsed_threshold < 0.0:
        raise ValueError("threshold must be greater than or equal to zero.")

    return parsed_threshold


def _normalize_boolean_series(series: pd.Series, *, column_name: str) -> pd.Series:
    """Return a strict Boolean series parsed from common CSV representations."""

    if is_bool_dtype(series.dtype):
        if series.isna().any():
            raise ValueError(f"{column_name} contains missing values.")
        return series.astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
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
        invalid_values = sorted(normalized.loc[invalid_mask].dropna().unique().tolist())
        raise ValueError(
            f"{column_name} contains values that cannot be interpreted as booleans: "
            f"{invalid_values}"
        )

    return parsed.astype(bool)


def normalize_subject_keys(subject_keys: pd.DataFrame) -> pd.DataFrame:
    """Validate, deduplicate, and sort participant-key rows."""

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
        numeric_values = pd.to_numeric(normalized[column_name], errors="raise")

        if numeric_values.isna().any():
            raise ValueError(f"{column_name} contains missing values.")

        integer_values = numeric_values.astype(np.int64)

        if not np.array_equal(
            numeric_values.to_numpy(dtype=np.float64),
            integer_values.to_numpy(dtype=np.float64),
        ):
            raise ValueError(f"{column_name} must contain only integer values.")

        normalized[column_name] = integer_values

    return normalized.drop_duplicates().sort_values(
        by=key_columns,
        kind="mergesort",
        ignore_index=True,
    )


def select_q_inverse_temperature_subject_keys(
    fit_results: pd.DataFrame,
    *,
    threshold: float = DEFAULT_INVERSE_TEMPERATURE_SELECTION_THRESHOLD,
    require_convergence: bool = True,
) -> pd.DataFrame:
    """Select participant keys whose Q-learning estimate reaches a high value.

    Args:
        fit_results: Per-model fit-results table.
        threshold: Inclusive inverse-temperature threshold.
        require_convergence: Whether to exclude nonconverged Q-learning fits.

    Returns:
        Unique participant keys ordered by ``PARTICIPANT_KEY_COLUMNS``.
    """

    if not isinstance(fit_results, pd.DataFrame):
        raise TypeError(
            f"fit_results must be a pandas DataFrame, got {type(fit_results).__name__}."
        )

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

    q_results = fit_results.loc[
        fit_results["model"].astype("string").eq(QLearningModel.get_name())
    ].copy()

    if q_results.empty:
        return normalize_subject_keys(pd.DataFrame(columns=list(PARTICIPANT_KEY_COLUMNS)))

    inverse_temperatures = pd.to_numeric(
        q_results["inverse_temperature"],
        errors="coerce",
    )

    invalid_temperature_mask = ~np.isfinite(inverse_temperatures.to_numpy(dtype=np.float64))
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

    return normalize_subject_keys(q_results.loc[selected_mask, list(PARTICIPANT_KEY_COLUMNS)])


def select_q_inverse_temperature_subject_keys_from_csv(
    fit_results_csv: Path,
    *,
    threshold: float = DEFAULT_INVERSE_TEMPERATURE_SELECTION_THRESHOLD,
    require_convergence: bool = True,
) -> pd.DataFrame:
    """Read a fit-results CSV and select Q-learning participant keys."""

    csv_path = Path(fit_results_csv)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Fit-results CSV does not exist: {csv_path}")

    fit_results = pd.read_csv(csv_path)
    return select_q_inverse_temperature_subject_keys(
        fit_results,
        threshold=threshold,
        require_convergence=require_convergence,
    )


def filter_subjects_by_keys(
    data: pd.DataFrame,
    subject_keys: pd.DataFrame,
) -> pd.DataFrame:
    """Return trial rows belonging to the explicitly requested participants.

    Every requested key must exist in the input data. The original trial-table
    columns are preserved and the selected rows are sorted chronologically.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}.")

    key_columns = list(PARTICIPANT_KEY_COLUMNS)
    required_data_columns = {*key_columns, "trial"}
    missing_data_columns = required_data_columns - set(data.columns)

    if missing_data_columns:
        missing_text = ", ".join(sorted(missing_data_columns))
        raise ValueError(f"IGT data is missing columns: {missing_text}")

    normalized_keys = normalize_subject_keys(subject_keys)

    if normalized_keys.empty:
        return data.iloc[0:0].copy()

    available_keys = normalize_subject_keys(data.loc[:, key_columns])
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
            "Requested participant keys were not found in the IGT dataset:\n"
            f"{missing_keys.to_string(index=False)}"
        )

    requested_index = pd.MultiIndex.from_frame(normalized_keys)
    data_index = pd.MultiIndex.from_frame(data.loc[:, key_columns])
    selected = data.loc[data_index.isin(requested_index)].copy()

    return selected.sort_values(
        by=[*key_columns, "trial"],
        kind="mergesort",
        ignore_index=True,
    )
