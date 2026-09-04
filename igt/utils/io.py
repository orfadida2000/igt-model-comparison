"""General filesystem input and output utilities.

The helpers normalize path-like values, read CSV files with project-compatible
encoding behavior, and write CSV, LaTeX-table, or text artifacts while creating
parent directories and applying explicit line-ending conventions.
"""

from collections.abc import Callable
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
            f"{normalized_parameter_name} must be a str, pathlib.Path, or os.PathLike, "
            f"got {type(path).__name__}."
        )

    if isinstance(path, str):
        path = path.strip()

        if not path:
            raise ValueError(f"{normalized_parameter_name} must not be empty.")

    try:
        return Path(path)
    except Exception as error:
        raise ValueError(
            f"The string path-like object {normalized_parameter_name} failed to be "
            f"converted to a pathlib.Path: {error}"
        ) from error


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


def write_latex_table(
    data: DataFrame,
    path: StrPathLike,
    *,
    index: bool = False,
    escape: bool = True,
    na_rep: str = "",
    float_format: str | Callable[[float], str] | None = "%.6g",
    column_format: str | None = None,
    longtable: bool | None = None,
    caption: str | tuple[str, str] | None = None,
    label: str | None = None,
    position: str | None = None,
    encoding: str = "utf-8",
    newline: LineEnding = LineEnding.LF,
) -> None:
    """Write a DataFrame as a LaTeX table.

    The table is rendered through `pandas.DataFrame.to_latex` and then written with
    the project's normal text-output conventions. Escaping is enabled by default so
    ordinary result labels such as model names containing underscores remain valid
    LaTeX. Pandas' standard output uses `booktabs` rules.

    Args:
        data: DataFrame to serialize.
        path: Destination `.tex` path.
        index: Whether to include the DataFrame index.
        escape: Whether to escape LaTeX-special characters in data and labels.
        na_rep: Representation used for missing values.
        float_format: Pandas-compatible floating-point formatter. Defaults to six
            significant digits so very small values remain visible in scientific notation.
        column_format: Optional LaTeX column-format specification.
        longtable: Whether pandas should emit a `longtable` environment.
        caption: Optional LaTeX table caption.
        label: Optional LaTeX cross-reference label.
        position: Optional LaTeX table-placement specifier.
        encoding: Text encoding used for the output file.
        newline: Line-ending policy used for the output file.

    Raises:
        TypeError: If `data` is not a DataFrame.
        ValueError: If the destination path cannot be normalized.
        ImportError: If pandas' LaTeX renderer dependency is unavailable.
    """

    if not isinstance(data, DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}.")

    latex = data.to_latex(
        index=index,
        escape=escape,
        na_rep=na_rep,
        float_format=float_format,
        column_format=column_format,
        longtable=longtable,
        caption=caption,
        label=label,
        position=position,
    )

    if not isinstance(latex, str):
        raise RuntimeError("pandas.DataFrame.to_latex did not return LaTeX text.")

    write_text(
        latex,
        path,
        encoding=encoding,
        newline=newline,
    )


def convert_csv_to_latex(
    csv_path: StrPathLike,
    latex_path: StrPathLike | None = None,
    *,
    csv_encoding: str = "utf-8-sig",
    table_name: str = "CSV",
    index: bool = False,
    escape: bool = True,
    na_rep: str = "",
    float_format: str | Callable[[float], str] | None = "%.6g",
    column_format: str | None = None,
    longtable: bool | None = None,
    caption: str | tuple[str, str] | None = None,
    label: str | None = None,
    position: str | None = None,
    latex_encoding: str = "utf-8",
    newline: LineEnding = LineEnding.LF,
) -> None:
    """Read a CSV file and write an equivalent LaTeX table.

    If `latex_path` is omitted, the CSV suffix is replaced with `.tex`. The CSV is
    loaded through [`read_csv`][igt.utils.io.read_csv] and rendering is delegated to
    [`write_latex_table`][igt.utils.io.write_latex_table], keeping DataFrame-to-LaTeX
    behavior in one source of truth.

    Args:
        csv_path: Source CSV file.
        latex_path: Optional destination `.tex` file. Defaults beside the source CSV.
        csv_encoding: Encoding used to read the CSV.
        table_name: Human-readable table name used in CSV diagnostics.
        index: Whether to include the DataFrame index in the LaTeX table.
        escape: Whether to escape LaTeX-special characters.
        na_rep: Representation used for missing values.
        float_format: Pandas-compatible floating-point formatter. Defaults to six
            significant digits so very small values remain visible in scientific notation.
        column_format: Optional LaTeX column-format specification.
        longtable: Whether pandas should emit a `longtable` environment.
        caption: Optional LaTeX table caption.
        label: Optional LaTeX cross-reference label.
        position: Optional LaTeX table-placement specifier.
        latex_encoding: Encoding used for the generated LaTeX file.
        newline: Line-ending policy used for the generated LaTeX file.
    """

    normalized_csv_path = normalize_path(csv_path, parameter_name="csv_path")
    normalized_latex_path = (
        normalized_csv_path.with_suffix(".tex")
        if latex_path is None
        else normalize_path(latex_path, parameter_name="latex_path")
    )
    data = read_csv(
        normalized_csv_path,
        encoding=csv_encoding,
        table_name=table_name,
    )
    write_latex_table(
        data,
        normalized_latex_path,
        index=index,
        escape=escape,
        na_rep=na_rep,
        float_format=float_format,
        column_format=column_format,
        longtable=longtable,
        caption=caption,
        label=label,
        position=position,
        encoding=latex_encoding,
        newline=newline,
    )
