"""General filesystem input and output utilities.

The helpers normalize path-like values, read CSV files with project-compatible
encoding behavior, and write CSV or text artifacts while creating parent directories
and applying explicit line-ending conventions.
"""

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
    """Validate a path-like value and convert it to `pathlib.Path`.

    The helper trims string paths but intentionally does not require the resulting
    path to exist.

    Args:
        path: String or path-like value to normalize.
        parameter_name: Human-readable argument name used in validation messages.

    Returns:
        The normalized `Path` object.

    Raises:
        TypeError: If `parameter_name` is not a string or `path` is not path-like.
        ValueError: If `parameter_name` or a string path is empty, or if conversion
            to `Path` fails.
    """

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
    """Read a required CSV file into a pandas DataFrame.

    The default UTF-8-with-signature encoding accepts both ordinary UTF-8 files and
    UTF-8 files containing a byte-order mark.

    Args:
        path: Path of the CSV file to read.
        encoding: Text encoding passed to `pandas.read_csv`.
        table_name: Human-readable table name used in error messages.

    Returns:
        The parsed DataFrame.

    Raises:
        TypeError: If `table_name` or the path argument has an invalid type.
        ValueError: If the table name is empty, the path cannot be normalized, or
            the CSV cannot be decoded or parsed.
        FileNotFoundError: If the normalized path is not an existing file.
    """

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
    """Write a DataFrame to CSV using an explicit line-ending policy.

    Parent directories are created automatically before the file is written.

    Args:
        data: DataFrame to serialize.
        path: Destination CSV path.
        index: Whether to include the DataFrame index.
        encoding: Text encoding used for the output file.
        newline: Line-ending sequence supplied to pandas as the CSV line terminator.

    Raises:
        TypeError: If `data` is not a DataFrame or the destination is not path-like.
        ValueError: If the destination path cannot be normalized.
    """

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
    """Write text with exactly one configured trailing line ending.

    Existing trailing carriage returns and line feeds are removed before one final
    line ending is appended. Parent directories are created automatically.

    Args:
        text: Text content to write.
        path: Destination text-file path.
        encoding: Text encoding used for the output file.
        newline: Line-ending sequence used both by `Path.write_text` and at EOF.

    Raises:
        TypeError: If `text` is not a string or the destination is not path-like.
        ValueError: If the destination path cannot be normalized.
    """

    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}.")

    text_path = normalize_path(path)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(
        text.rstrip("\r\n") + newline.value,
        encoding=encoding,
        newline=newline.value,
    )
