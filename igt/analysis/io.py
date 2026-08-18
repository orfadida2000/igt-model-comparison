"""Input data structures and loading helpers for result analysis.

The analysis pipeline consumes the fit, comparison, and summary CSV files produced
by the fitting workflow. This module loads those files together into a
[`ResultTables`][igt.analysis.io.ResultTables] container.
"""

from dataclasses import dataclass

from pandas import DataFrame

from igt.typing import StrPathLike
from igt.utils.io import read_csv


@dataclass(frozen=True, slots=True)
class ResultTables:
    """The three mutually consistent result tables consumed by analysis.

    Attributes:
        fits: Complete per-subject, per-model fit results.
        comparison: Fit results augmented with comparison eligibility, AIC/BIC
            deltas, and winner flags.
        summary: Aggregate model-level fit and comparison summary.
    """

    fits: DataFrame
    comparison: DataFrame
    summary: DataFrame


def load_result_tables(
    fits_path: StrPathLike,
    comparison_path: StrPathLike,
    summary_path: StrPathLike,
) -> ResultTables:
    """Load the fit, comparison, and summary CSV files for one analysis run.

    Each file is read through the project's shared CSV loader; schema and cross-table
    validation are intentionally deferred to
    [`validate_result_tables`][igt.analysis.validation.validate_result_tables].

    Args:
        fits_path: Path to the complete per-participant model-fit CSV.
        comparison_path: Path to the corresponding model-comparison CSV.
        summary_path: Path to the corresponding aggregate model-summary CSV.

    Returns:
        `ResultTables` containing the three loaded DataFrames.

    Raises:
        FileNotFoundError: If any supplied CSV path does not exist.
        ValueError: If a supplied path is invalid or a CSV cannot be parsed as expected.
    """

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
