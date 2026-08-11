"""Command-line interface for result analysis."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from .config import AnalysisConfig
from .pipeline import generate_results_analysis


def _positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error

    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")

    return parsed


def _histogram_bins(value: str) -> int | str:
    """Parse a positive integer or a NumPy histogram strategy name."""

    try:
        parsed = int(value)
    except ValueError:
        normalized = value.strip()

        if not normalized:
            raise argparse.ArgumentTypeError("histogram bins must not be empty") from None

        return normalized

    if parsed <= 0:
        raise argparse.ArgumentTypeError("histogram bins must be positive")

    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate IGT model-result CSVs and generate standard analysis tables and figures."
        )
    )
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--figure-formats",
        nargs="+",
        default=["png"],
        help="One or more Matplotlib output formats, for example: png svg.",
    )
    parser.add_argument("--figure-dpi", type=_positive_int, default=300)
    parser.add_argument(
        "--histogram-bins",
        type=_histogram_bins,
        default="auto",
        help="A positive integer or a NumPy histogram strategy such as auto.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the result-analysis command-line interface."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config = AnalysisConfig(
        figure_formats=tuple(args.figure_formats),
        figure_dpi=args.figure_dpi,
        histogram_bins=args.histogram_bins,
    )
    outputs = generate_results_analysis(
        args.fits,
        args.comparison,
        args.summary,
        args.output_directory,
        config=config,
    )
    print(f"Report: {outputs.report_path}")
    print(f"Figures: {len(outputs.figure_paths)}")
    print(f"Tables: {len(outputs.table_paths)}")
    return 0
