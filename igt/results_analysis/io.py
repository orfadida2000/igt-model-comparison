"""Input helpers specific to model-result analysis."""

from dataclasses import dataclass

from pandas import DataFrame

from igt.typing import StrPathLike
from igt.utils.io import read_csv


@dataclass(frozen=True, slots=True)
class ResultTables:
    """The three result tables produced by the fitting pipeline."""

    fits: DataFrame
    comparison: DataFrame
    summary: DataFrame


def load_result_tables(
    fits_path: StrPathLike,
    comparison_path: StrPathLike,
    summary_path: StrPathLike,
) -> ResultTables:
    """Load the fit, comparison, and summary CSV files."""

    return ResultTables(
        fits=read_csv(
            fits_path,
            table_name="fit-results",
        ),
        comparison=read_csv(
            comparison_path,
            table_name="model-comparison",
        ),
        summary=read_csv(
            summary_path,
            table_name="model-summary",
        ),
    )
