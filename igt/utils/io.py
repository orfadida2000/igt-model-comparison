"""General file input and output utilities."""

from os import PathLike
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from igt.typing import LineEnding, StrPathLike


def normalize_path(
    path: StrPathLike,
    *,
    parameter_name: str = "path",
) -> Path:
    """Validate and convert a filesystem path."""

    if not isinstance(parameter_name, str):
        raise TypeError("parameter_name must be a string.")

    normalized_parameter_name = parameter_name.strip()

    if not normalized_parameter_name:
        raise ValueError("parameter_name must not be empty.")

    if not isinstance(path, (str, Path, PathLike)):
        raise TypeError(
            f"{normalized_parameter_name} must be a str, pathlib.Path, or os.PathLike, got {type(path).__name__}."
        )

    if isinstance(path, str):
        path = path.strip()

        if not path:
            raise ValueError(f"{normalized_parameter_name} must not be empty.")

    try:
        return Path(path)
    except Exception as e:
        raise ValueError(
            f"The string path-like object {normalized_parameter_name} failed to be converted to a pathlib.Path: {e}"
        ) from e


def read_csv(
    path: StrPathLike,
    *,
    encoding: str = "utf-8-sig",
    table_name: str = "CSV",
) -> DataFrame:
    """Read a CSV file into a DataFrame."""

    csv_path = normalize_path(path)

    if not isinstance(table_name, str):
        raise TypeError("table_name must be a string.")

    normalized_table_name = table_name.strip()

    if not normalized_table_name:
        raise ValueError("table_name must not be empty.")

    if not csv_path.is_file():
        raise FileNotFoundError(f"{normalized_table_name} CSV does not exist: {csv_path}")

    try:
        return pd.read_csv(
            csv_path,
            encoding=encoding,
        )
    except (
        OSError,
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        raise ValueError(f"Failed to read {normalized_table_name} CSV: {csv_path}") from error


def write_csv(
    data: DataFrame,
    path: StrPathLike,
    *,
    index: bool = False,
    encoding: str = "utf-8",
    newline: LineEnding = LineEnding.LF,
) -> None:
    """Write a DataFrame as a UTF-8 CSV with LF line endings."""

    if not isinstance(data, DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}.")

    csv_path = normalize_path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        csv_path,
        index=index,
        encoding=encoding,
        lineterminator=newline.value,
    )


def write_text(
    text: str,
    path: StrPathLike,
    *,
    encoding: str = "utf-8",
    newline: LineEnding = LineEnding.LF,
) -> None:
    """Write UTF-8 text with exactly one final LF line ending."""

    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}.")

    text_path = normalize_path(path)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(
        text.rstrip("\r\n") + newline.value,
        encoding=encoding,
        newline=newline.value,
    )
