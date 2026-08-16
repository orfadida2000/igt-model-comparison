"""Command-line interface for result analysis."""

import argparse
import logging
import time
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

from igt.analysis.config import AnalysisConfig
from igt.analysis.pipeline import generate_results_analysis
from igt.cli_parsing.parser import get_parser
from igt.cli_parsing.type_filters.factory.numeric import (
    get_type_filters_for_finite_float_with_range,
)
from igt.cli_parsing.type_filters.factory.path import (
    get_type_filters_for_existing_file_with_extensions_path,
)
from igt.cli_parsing.type_filters.presets import (
    NumericArgTypeProvider,
    PathArgTypeProvider,
    StringArgTypeProvider,
)
from igt.cli_parsing.typing import (
    ArgSpec,
)
from igt.constants.config import (
    DEFAULT_NOTIFY_FORMSUBMIT_ID,
    DEFAULT_ROOT_LOG_LEVEL,
    FILENAME_DATETIME_FMT,
    FIXED_SEED,
    USE_FIXED_NOTIFY_FORMSUBMIT_ID,
    USE_FIXED_SEED,
)
from igt.constants.path import LOGS_DIR, RESULTS_DIR
from igt.logging import application_logging_cleanup, configure_application_logging
from igt.notify.formsubmit import (
    error_email_notifier,
    send_formsubmit_email_script_success_notification,
)

LOGGER_NAME: Final[str] = "scripts.results_analysis"


def _positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error

    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")

    return parsed


def _nonnegative_int(value: str) -> int:
    """Parse a non-negative integer for argparse."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error

    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")

    return parsed


def _confidence_level(value: str) -> float:
    """Parse a confidence level strictly between zero and one."""

    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("confidence level must be a number") from error

    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("confidence level must be strictly between 0 and 1")

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
    parser.add_argument(
        "--confidence-level",
        type=_confidence_level,
        default=0.95,
        help="Confidence level for bootstrap and exact binomial intervals.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=_positive_int,
        default=10_000,
        help="Number of BCa bootstrap resamples for criterion differences.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=_nonnegative_int,
        default=42,
        help="Random seed used for reproducible bootstrap confidence intervals.",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Create the argument parser, parse the command-line arguments and return the namespace.

    Some arguments are only added to the parser if certain conditions are met,
    such as whether a fixed seed or fixed FormSubmit ID is used. The parser is configured
    with appropriate type filters, default values, and help messages for each argument.

    Returns:
        The parsed command-line arguments namespace.
    """

    arg_specs: list[ArgSpec] = [
        ArgSpec(
            name_or_flags="--fit-results",
            type_filters=get_type_filters_for_existing_file_with_extensions_path(".csv"),
            required=True,
            help="Path to the input CSV file containing the complete fit results of each subject with every model (required)",
            extra_options={
                "metavar": "FILE_PATH",
                "dest": "fit_results_path",
            },
        ),
        ArgSpec(
            name_or_flags="--comparison-results",
            type_filters=get_type_filters_for_existing_file_with_extensions_path(".csv"),
            required=True,
            help="Path to the input CSV file containing the complete model comparison results (required)",
            extra_options={
                "metavar": "FILE_PATH",
                "dest": "comparison_results_path",
            },
        ),
        ArgSpec(
            name_or_flags="--summary-results",
            type_filters=get_type_filters_for_existing_file_with_extensions_path(".csv"),
            required=True,
            help="Path to the input CSV file containing the complete summary results (required)",
            extra_options={
                "metavar": "FILE_PATH",
                "dest": "summary_results_path",
            },
        ),
        ArgSpec(
            name_or_flags=("--output-dir",),
            type_filters=(PathArgTypeProvider.DIR_PATH,),
            default=str(RESULTS_DIR / "analysis"),
            help="Directory to save the analysis results files (default: %(default)s)",
            extra_options={
                "metavar": "OUTPUT_DIR",
            },
        ),
        ArgSpec(
            name_or_flags=("--logging-dir",),
            type_filters=(PathArgTypeProvider.DIR_PATH,),
            default=str(LOGS_DIR / "analysis"),
            help="Directory to save the log files (default: %(default)s)",
            extra_options={
                "metavar": "LOGGING_DIR",
            },
        ),
        ArgSpec(
            name_or_flags=("--figure-formats",),
            type_filters=(StringArgTypeProvider.NON_WHITESPACE_STRING,),
            default=["png"],
            help="One or more Matplotlib output formats, for example: png svg (default: %(default)s)",
            extra_options={
                "metavar": "FORMATS",
                "nargs": "+",
            },
        ),
        ArgSpec(
            name_or_flags=("--figure-dpi",),
            type_filters=(NumericArgTypeProvider.POSITIVE_INTEGER,),
            default="300",
            help="DPI (dots per inch) for the output figures (default: %(default)s)",
            extra_options={
                "metavar": "DPI",
            },
        ),
        ArgSpec(
            name_or_flags=("--histogram-bins",),
            type_filters=_histogram_bins,
            default="auto",
            help="A positive integer or a NumPy histogram strategy such as auto (default: %(default)s)",
            extra_options={
                "metavar": "BINS",
            },
        ),
        ArgSpec(
            name_or_flags=("--confidence-level",),
            type_filters=get_type_filters_for_finite_float_with_range(
                0, 1, min_inclusive=False, max_inclusive=False
            ),
            default="0.95",
            help="Confidence level for bootstrap and exact binomial intervals (default: %(default)s)",
            extra_options={
                "metavar": "CONFIDENCE",
            },
        ),
        ArgSpec(
            name_or_flags=("--bootstrap-resamples",),
            type_filters=(NumericArgTypeProvider.POSITIVE_INTEGER,),
            default="10000",
            help="Number of BCa bootstrap resamples for criterion differences (default: %(default)s)",
            extra_options={
                "metavar": "RESAMPLES",
            },
        ),
    ]

    if not USE_FIXED_SEED:
        arg_specs.append(
            ArgSpec(
                name_or_flags=("--seed",),
                type_filters=(NumericArgTypeProvider.NON_NEGATIVE_INTEGER,),
                default=str(FIXED_SEED),
                help="Integer seed for reproducible bootstrap confidence intervals (default: %(default)s)",
                extra_options={
                    "metavar": "SEED",
                    "dest": "bootstrap_seed",
                },
            )
        )

    arg_specs.append(
        ArgSpec(
            name_or_flags=("--log-level",),
            type_filters=(NumericArgTypeProvider.INTEGER,),
            default=str(DEFAULT_ROOT_LOG_LEVEL),
            help="Logging level for the root logger; use negative value to disable logging (default: %(default)s)",
            extra_options={
                "metavar": "ROOT_LOG_LEVEL",
            },
        )
    )

    if not USE_FIXED_NOTIFY_FORMSUBMIT_ID:
        arg_specs.append(
            ArgSpec(
                name_or_flags=("--notify-formsubmit-id",),
                type_filters=(StringArgTypeProvider.ALPHANUMERIC_STRING,),
                default=DEFAULT_NOTIFY_FORMSUBMIT_ID,
                help="FormSubmit ID to send email notifications upon script completion (default: %(default)s)",
                extra_options={
                    "metavar": "ID",
                },
            )
        )

    (
        parser,
        resolved_info,
    ) = get_parser(
        arg_specs,
        description="Validate IGT model-result CSVs and generate standard analysis tables and figures.",
        extra_options={},
    )

    return parser.parse_args(argv)


def _normalize_args(
    args: argparse.Namespace,
) -> argparse.Namespace:
    """
    Normalize the parsed command-line arguments namespace and return a new namespace with resolved runtime values.

    Args:
        args: The parsed command-line arguments namespace.

    Returns:
        The normalized command-line arguments namespace.
    """

    normalized_args: dict[str, Any] = {
        "fit_results_path": args.fit_results_path,
        "comparison_results_path": args.comparison_results_path,
        "summary_results_path": args.summary_results_path,
        "output_dir": args.output_dir,
        "logging_dir": args.logging_dir,
        "figure_formats": tuple(args.figure_formats),
        "figure_dpi": args.figure_dpi,
        "histogram_bins": args.histogram_bins,
        "confidence_level": args.confidence_level,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": (FIXED_SEED if USE_FIXED_SEED else args.bootstrap_seed),
        "logging_disabled": args.log_level < 0,
        "log_level": None if args.log_level < 0 else args.log_level,
        "notify_formsubmit_id": (
            DEFAULT_NOTIFY_FORMSUBMIT_ID
            if USE_FIXED_NOTIFY_FORMSUBMIT_ID
            else args.notify_formsubmit_id
        ),
    }

    return argparse.Namespace(**normalized_args)


def _setup(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, argparse.Namespace, str, Path | None]:
    start_datetime_str = datetime.now().strftime(FILENAME_DATETIME_FMT)
    args = _parse_args(argv)
    normalized_args = _normalize_args(args)

    normalized_args.output_dir = Path(normalized_args.output_dir) / start_datetime_str

    logging_path = configure_application_logging(
        disabled=normalized_args.logging_disabled,
        root_logger_level=normalized_args.log_level,
        log_file_path=(
            normalized_args.logging_dir / (f"results_analysis_{start_datetime_str}.log")
        ),
    )
    return args, normalized_args, start_datetime_str, logging_path


def _run(
    *,
    normalized_args: argparse.Namespace,
    start_datetime_str: str,
    logger: logging.Logger | str = LOGGER_NAME,
) -> Sequence[str | Path]:
    logger = logging.getLogger(logger) if isinstance(logger, str) else logger

    output_paths: list[Path] = []

    config = AnalysisConfig(
        figure_formats=tuple(normalized_args.figure_formats),
        figure_dpi=normalized_args.figure_dpi,
        histogram_bins=normalized_args.histogram_bins,
        confidence_level=normalized_args.confidence_level,
        bootstrap_resamples=normalized_args.bootstrap_resamples,
        bootstrap_seed=normalized_args.bootstrap_seed,
    )

    logger.info("Generating result analysis...")
    logger.debug("Using analysis configuration: %s", config)

    outputs = generate_results_analysis(
        normalized_args.fit_results_path,
        normalized_args.comparison_results_path,
        normalized_args.summary_results_path,
        normalized_args.output_dir,
        config=config,
    )
    logger.info("Result analysis completed successfully.")
    logger.info(f"Report: {outputs.report_path}")
    logger.info(f"Figures: {len(outputs.figure_paths)}")
    logger.info(f"Tables: {len(outputs.table_paths)}")

    output_paths.append(outputs.report_path)
    output_paths.extend(outputs.figure_paths)
    output_paths.extend(outputs.table_paths)

    return output_paths


def _cleanup(
    *,
    logger: logging.Logger | str = LOGGER_NAME,
) -> None:
    """Perform any necessary cleanup after the script has finished running."""
    logger = logging.getLogger(logger) if isinstance(logger, str) else logger

    logger.info("Cleaning up application logging...")
    application_logging_cleanup()


def main(argv: Sequence[str] | None = None) -> None:
    """Main entry point for the script. Parses command-line arguments, sets up logging, and runs the analysis pipeline."""
    start_counter = time.perf_counter()

    (
        args,
        normalized_args,
        start_datetime_str,
        logging_path,
    ) = _setup(argv)

    notify_formsubmit_id: str | None = normalized_args.notify_formsubmit_id
    logger = logging.getLogger(LOGGER_NAME)

    try:
        with error_email_notifier(
            formsubmit_id=notify_formsubmit_id,
            script_name=Path(__file__).name,
            start_counter=start_counter,
        ):
            logger.info("Starting IGT result analysis...")

            logger.debug("Parsed command-line arguments: %s", args)
            logger.debug("Normalized command-line arguments: %s", normalized_args)

            normalized_args.output_dir.mkdir(parents=True, exist_ok=True)

            result_files = _run(
                normalized_args=normalized_args,
                start_datetime_str=start_datetime_str,
                logger=logger,
            )
            result_files = [Path(f) for f in result_files]

            if logging_path is not None:
                logger.info("Saved log file to: %s", logging_path)

                result_files.append(Path(logging_path))

            end_counter = time.perf_counter()
            elapsed_time = end_counter - start_counter
            elapsed_time_obj = timedelta(seconds=round(elapsed_time))

            logger.info(
                "Completed IGT result analysis in %s.",
                elapsed_time_obj,
            )

            if notify_formsubmit_id is not None:
                logger.info(
                    "Sending FormSubmit notification to %r with results files: %r as one zip attachment.",
                    notify_formsubmit_id,
                    [f.name for f in result_files],
                )

            if notify_formsubmit_id is not None:
                zip_filename = "igt_analysis_results.zip"
                email_message = f"""
The IGT result analysis has completed successfully.

The following results files have been generated and are attached in a zip file named {zip_filename!r}:
{"\n".join(f"  - {f.name}" for f in result_files)}
                """.strip()

                send_formsubmit_email_script_success_notification(
                    formsubmit_id=notify_formsubmit_id,
                    message=email_message,
                    duration_seconds=elapsed_time,
                    script_name=Path(__file__).name,
                    file_paths=result_files,
                    zip_filename=zip_filename,
                )
    finally:
        logger.info("Performing cleanup...")
        _cleanup(logger=logger)


if __name__ == "__main__":
    main()
