"""General validation and normalization helpers for tabular values."""

import numpy as np
import pandas as pd
from pandas import Series
from pandas.api.types import is_bool_dtype, is_integer_dtype


def normalize_integer_series(
    series: Series,
    *,
    column_name: str,
) -> Series:
    """Validate and convert a Series to NumPy int64 values."""

    if not isinstance(series, Series):
        raise TypeError(f"series must be a pandas Series, got {type(series).__name__}.")

    if not isinstance(column_name, str):
        raise TypeError("column_name must be a string.")

    normalized_column_name = column_name.strip()

    if not normalized_column_name:
        raise ValueError("column_name must not be empty.")

    if is_bool_dtype(series.dtype):
        raise ValueError(f"{normalized_column_name} must contain integers, not Boolean values.")

    try:
        numeric_values = pd.to_numeric(
            series,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{normalized_column_name} contains values that cannot be interpreted as numbers."
        ) from error

    if numeric_values.isna().any():
        raise ValueError(f"{normalized_column_name} contains missing values.")

    if numeric_values.empty:
        return Series(
            index=series.index,
            dtype=np.int64,
            name=series.name,
        )

    if is_integer_dtype(numeric_values.dtype):
        integer_array = numeric_values.to_numpy()
        int64_info = np.iinfo(np.int64)

        if np.any(integer_array < int64_info.min) or np.any(integer_array > int64_info.max):
            raise ValueError(f"{normalized_column_name} contains values outside the int64 range.")

        return numeric_values.astype(np.int64)

    numeric_array = numeric_values.to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    if not np.isfinite(numeric_array).all():
        raise ValueError(f"{normalized_column_name} contains non-finite values.")

    if not np.equal(numeric_array, np.trunc(numeric_array)).all():
        raise ValueError(f"{normalized_column_name} must contain only integer values.")

    if np.any(numeric_array < -(2**63)) or np.any(numeric_array >= 2**63):
        raise ValueError(f"{normalized_column_name} contains values outside the int64 range.")

    return Series(
        numeric_array.astype(np.int64),
        index=series.index,
        name=series.name,
    )


def normalize_boolean_series(
    series: Series,
    *,
    column_name: str,
) -> Series:
    """Parse a strict Boolean Series from common CSV representations."""

    if not isinstance(series, Series):
        raise TypeError(f"series must be a pandas Series, got {type(series).__name__}.")

    if not isinstance(column_name, str):
        raise TypeError("column_name must be a string.")

    normalized_column_name = column_name.strip()

    if not normalized_column_name:
        raise ValueError("column_name must not be empty.")

    if is_bool_dtype(series.dtype):
        if series.isna().any():
            raise ValueError(f"{normalized_column_name} contains missing values.")

        return series.astype(bool)

    normalized = series.astype("string").str.strip().str.lower()

    if normalized.isna().any():
        raise ValueError(f"{normalized_column_name} contains missing values.")

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
            f"{normalized_column_name} contains values that cannot be "
            f"interpreted as booleans: {invalid_values}"
        )

    return parsed.astype(bool)


def normalize_nonempty_string_series(
    series: Series,
    *,
    column_name: str,
) -> Series:
    """Validate and trim a Series of required nonempty strings."""

    if not isinstance(series, Series):
        raise TypeError(f"series must be a pandas Series, got {type(series).__name__}.")

    if not isinstance(column_name, str):
        raise TypeError("column_name must be a string.")

    normalized_column_name = column_name.strip()

    if not normalized_column_name:
        raise ValueError("column_name must not be empty.")

    normalized = series.astype("string").str.strip()
    invalid_mask = normalized.isna() | normalized.eq("")

    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())
        raise ValueError(
            f"{normalized_column_name} contains {invalid_count} missing or empty value(s)."
        )

    return normalized
