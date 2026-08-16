"""Command-line entry point for fitting and comparing IGT models."""

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
from igt.cli_parsing.type_filters.factory.path import (
    get_type_filters_for_existing_file_with_extensions_path,
)
from igt.cli_parsing.type_filters.presets import (
    NumericArgTypeProvider,
    PathArgTypeProvider,
    StringArgTypeProvider,
)
from igt.cli_parsing.typing import (
    ArgAction,
    ArgSpec,
)
from igt.comparison import (
    add_model_comparison_columns,
    summarize_model_comparison,
)
from igt.constants.config import (
    DEFAULT_N_PVL_STARTS,
    DEFAULT_N_Q_STARTS,
    DEFAULT_N_SUBJECTS,
    DEFAULT_N_WORKERS,
    DEFAULT_NOTIFY_FORMSUBMIT_ID,
    DEFAULT_ROOT_LOG_LEVEL,
    FILENAME_DATETIME_FMT,
    FIXED_SEED,
    SUBJECTS_AVAILABLE,
    USE_FIXED_NOTIFY_FORMSUBMIT_ID,
    USE_FIXED_SEED,
    USE_RDATA_PARENT_DIR_FOR_LOGGING,
    USE_RDATA_PARENT_DIR_FOR_OUTPUT,
)
from igt.constants.fitting import DEFAULT_MAX_ITERATIONS
from igt.constants.models import DEFAULT_MAX_INVERSE_TEMPERATURE
from igt.constants.path import IGT_DATASET_PATH, LOGS_DIR, RESULTS_DIR
from igt.constants.schema import PARTICIPANT_KEY_COLUMNS
from igt.execution.pipeline import FittingPipelineConfig, run_fitting_pipeline
from igt.logging import application_logging_cleanup, configure_application_logging
from igt.models import PVLDeltaModel, QLearningModel
from igt.notify.formsubmit import (
    error_email_notifier,
    send_formsubmit_email_script_success_notification,
)

LOGGER_NAME: Final[str] = "igt.main"


def _n_pvl_starts_power_of_two(n_starts: int) -> int:
    """Parse and validate the number of PVL-Delta Sobol starts for the argument parser."""

    if (n_starts & (n_starts - 1)) != 0:
        raise argparse.ArgumentTypeError(
            f"Invalid number of PVL-Delta Sobol starts: must be a power of two, got {n_starts}"
        )

    return n_starts


def _parse_args() -> argparse.Namespace:
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
            name_or_flags=("--rdata-path",),
            type_filters=get_type_filters_for_existing_file_with_extensions_path(
                extensions=(".rdata", ".rda")
            ),
            default=str(IGT_DATASET_PATH),
            help="Path to the input IGTdata.rdata file (default: %(default)s)",
            extra_options={
                "metavar": "FILE_PATH",
            },
        ),
        ArgSpec(
            name_or_flags=("--max-iterations",),
            type_filters=(NumericArgTypeProvider.POSITIVE_INTEGER,),
            default=str(DEFAULT_MAX_ITERATIONS),
            help="Maximum number of iterations for each parameter optimization in the fitting process; must be a positive integer (default: %(default)s)",
            extra_options={
                "metavar": "MAX_ITERATIONS",
            },
        ),
        ArgSpec(
            name_or_flags=("--q-starts",),
            type_filters=(NumericArgTypeProvider.POSITIVE_INTEGER,),
            default=str(DEFAULT_N_Q_STARTS),
            help="Maximum number of distinct grid-local-minimum starts used for the Q-learning model fitting; must be a positive integer (default: %(default)s)",
            extra_options={
                "metavar": "N_Q_STARTS",
                "dest": "n_q_starts",
            },
        ),
        ArgSpec(
            name_or_flags=("--pvl-starts",),
            type_filters=(NumericArgTypeProvider.POSITIVE_INTEGER, _n_pvl_starts_power_of_two),
            default=str(DEFAULT_N_PVL_STARTS),
            help="Number of distinct Sobol starts used for the PVL-Delta model fitting; must be a positive integer and a power of two (default: %(default)s)",
            extra_options={
                "metavar": "N_PVL_STARTS",
                "dest": "n_pvl_starts",
            },
        ),
        ArgSpec(
            name_or_flags=("--q-max-inverse-temperature",),
            type_filters=(NumericArgTypeProvider.POSITIVE_FINITE_FLOAT,),
            default=str(DEFAULT_MAX_INVERSE_TEMPERATURE),
            help=(
                "Upper bound for the Q-learning inverse temperature parameter. The default "
                "Q-learning grid automatically preserves approximately unit spacing along this dimension (default: %(default)s)"
            ),
            extra_options={
                "metavar": "MAX_INVERSE_TEMPERATURE",
            },
        ),
    ]

    if not USE_FIXED_SEED:
        arg_specs.append(
            ArgSpec(
                name_or_flags=("--seed",),
                type_filters=(NumericArgTypeProvider.INTEGER,),
                default="-1",
                help="Integer seed used by the scrambled Sobol generator; use negative value to disable seeding (default: %(default)s)",
                extra_options={
                    "metavar": "SEED",
                    "dest": "rng_seed",
                },
            )
        )

    arg_specs.extend(
        [
            ArgSpec(
                name_or_flags=("--workers",),
                type_filters=(NumericArgTypeProvider.INTEGER,),
                default=str(DEFAULT_N_WORKERS),
                help=(
                    "Number of worker processes to use for the fitting process; "
                    "use 0 for serial execution and negative value for all available CPU cores (default: %(default)s)"
                ),
                extra_options={
                    "metavar": "N_WORKERS",
                    "dest": "n_workers",
                },
            ),
            ArgSpec(
                name_or_flags=("--subjects",),
                type_filters=(NumericArgTypeProvider.INTEGER,),
                default=str(DEFAULT_N_SUBJECTS),
                help=(
                    "Number of subjects to use for the fitting process; "
                    "use negative value for all available subjects (default: %(default)s)"
                ),
                extra_options={
                    "metavar": "N_SUBJECTS",
                    "dest": "n_subjects",
                },
            ),
        ]
    )

    if not USE_RDATA_PARENT_DIR_FOR_OUTPUT:
        arg_specs.append(
            ArgSpec(
                name_or_flags=("--output-dir",),
                type_filters=(PathArgTypeProvider.DIR_PATH,),
                default=str(RESULTS_DIR / "igt_model_comparison"),
                help="Directory to save the results files (default: %(default)s)",
                extra_options={
                    "metavar": "OUTPUT_DIR",
                },
            )
        )

    if not USE_RDATA_PARENT_DIR_FOR_LOGGING:
        arg_specs.append(
            ArgSpec(
                name_or_flags=("--logging-dir",),
                type_filters=(PathArgTypeProvider.DIR_PATH,),
                default=str(LOGS_DIR / "igt_model_comparison"),
                help="Directory to save the log files (default: %(default)s)",
                extra_options={
                    "metavar": "LOGGING_DIR",
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

    arg_specs.append(
        ArgSpec(
            name_or_flags=("--no-progress",),
            action=ArgAction.STORE_TRUE,
            help="Disable progress bar display during the fitting process",
        )
    )

    arg_specs.append(
        ArgSpec(
            name_or_flags=("--analyze",),
            action=ArgAction.STORE_TRUE,
            help="Run the analysis steps on the computed results and generate summary tables, plots, and perform statistical tests",
        )
    )

    (
        parser,
        resolved_info,
    ) = get_parser(
        arg_specs,
        description="Fit the Q-learning and PVL-Delta models to the Steingroever IGT dataset and compare their performance.",
        extra_options={},
    )

    return parser.parse_args()


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
        "rdata_path": args.rdata_path,
        "max_iterations": args.max_iterations,
        "n_q_starts": args.n_q_starts,
        "n_pvl_starts": args.n_pvl_starts,
        "q_max_inverse_temperature": args.q_max_inverse_temperature,
        "rng_seed": (
            FIXED_SEED if USE_FIXED_SEED else (None if args.rng_seed < 0 else args.rng_seed)
        ),
        "n_workers": None if args.n_workers < 0 else (1 if args.n_workers == 0 else args.n_workers),
        "n_subjects": None if args.n_subjects < 0 else args.n_subjects,
        "effective_n_subjects": (
            SUBJECTS_AVAILABLE
            if (args.n_subjects < 0 or args.n_subjects > SUBJECTS_AVAILABLE)
            else args.n_subjects
        ),
        "output_dir": (
            args.output_dir
            if not USE_RDATA_PARENT_DIR_FOR_OUTPUT
            else args.rdata_path.parent / "output_files"
        ),
        "logging_dir": (
            args.logging_dir
            if not USE_RDATA_PARENT_DIR_FOR_LOGGING
            else args.rdata_path.parent / "logs"
        ),
        "logging_disabled": args.log_level < 0,
        "log_level": None if args.log_level < 0 else args.log_level,
        "notify_formsubmit_id": (
            DEFAULT_NOTIFY_FORMSUBMIT_ID
            if USE_FIXED_NOTIFY_FORMSUBMIT_ID
            else args.notify_formsubmit_id
        ),
        "no_progress": args.no_progress,
        "analyze": args.analyze,
    }

    return argparse.Namespace(**normalized_args)


def _setup() -> tuple[argparse.Namespace, argparse.Namespace, str, Path | None]:
    """Parse command-line arguments, configure logging, and return runtime values."""

    start_datetime_str = datetime.now().strftime(FILENAME_DATETIME_FMT)

    args = _parse_args()
    normalized_args = _normalize_args(args)

    normalized_args.output_dir = Path(normalized_args.output_dir) / start_datetime_str

    logging_path = configure_application_logging(
        disabled=normalized_args.logging_disabled,
        root_logger_level=normalized_args.log_level,
        log_file_path=(
            normalized_args.logging_dir
            / (
                "igt_model_comparison_"
                f"{normalized_args.effective_n_subjects}_subjects_"
                f"{start_datetime_str}.log"
            )
        ),
    )

    return args, normalized_args, start_datetime_str, logging_path


def _run(
    *,
    normalized_args: argparse.Namespace,
    start_datetime_str: str,
    logger: logging.Logger | str = LOGGER_NAME,
) -> Sequence[str | Path]:
    """Run the IGT model fitting and comparison pipeline based on command-line arguments."""

    logger = logging.getLogger(logger) if isinstance(logger, str) else logger

    models = (
        QLearningModel(
            n_starts=normalized_args.n_q_starts,
            max_inverse_temperature=normalized_args.q_max_inverse_temperature,
        ),
        PVLDeltaModel(
            n_starts=normalized_args.n_pvl_starts,
            rng=normalized_args.rng_seed,
        ),
    )

    results_table = run_fitting_pipeline(
        FittingPipelineConfig(
            rdata_path=normalized_args.rdata_path,
            models=models,
            max_iterations=normalized_args.max_iterations,
            n_workers=normalized_args.n_workers,
            show_progress=not normalized_args.no_progress,
            n_subjects=normalized_args.n_subjects,
            subject_keys=None,
        )
    )

    logger.info("Processing and saving results...")

    comparison_table = add_model_comparison_columns(results_table)
    summary_table = summarize_model_comparison(results_table)

    output_dir = normalized_args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_n_subjects = int(
        results_table.loc[:, list(PARTICIPANT_KEY_COLUMNS)].drop_duplicates().shape[0]
    )
    filename_suffix = f"{actual_n_subjects}_subjects_{start_datetime_str}"
    fits_path = output_dir / f"model_fits_{filename_suffix}.csv"
    comparison_path = output_dir / f"model_comparison_{filename_suffix}.csv"
    summary_path = output_dir / f"model_summary_{filename_suffix}.csv"

    results_table.to_csv(fits_path, index=False, lineterminator="\n")
    comparison_table.to_csv(
        comparison_path,
        index=False,
        lineterminator="\n",
    )
    summary_table.to_csv(summary_path, index=False, lineterminator="\n")

    logger.info("Saved model fits: %s", fits_path)
    logger.info("Saved model comparison: %s", comparison_path)
    logger.info("Saved model summary: %s", summary_path)

    logger.debug("Model summary:\n%s", summary_table.to_string(index=False))

    output_paths: list[Path] = [fits_path, comparison_path, summary_path]

    if normalized_args.analyze:
        logger.info("Generating result analysis...")

        config = AnalysisConfig()

        logger.debug("Using analysis configuration: %s", config)

        outputs = generate_results_analysis(
            fits_path,
            comparison_path,
            summary_path,
            normalized_args.output_dir / "analysis",
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


def main() -> None:
    """Main entry point for the IGT model fitting and comparison script."""

    start_counter = time.perf_counter()

    (
        args,
        normalized_args,
        start_datetime_str,
        logging_path,
    ) = _setup()

    notify_formsubmit_id: str | None = normalized_args.notify_formsubmit_id
    logger = logging.getLogger(LOGGER_NAME)

    try:
        with error_email_notifier(
            formsubmit_id=notify_formsubmit_id,
            script_name=Path(__file__).name,
            start_counter=start_counter,
        ):
            logger.info("Starting IGT model fitting and comparison...")
            logger.debug("Parsed command-line arguments: %s", args)
            logger.debug("Normalized command-line arguments: %s", normalized_args)

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

            logger.info("IGT model fitting and comparison completed in %s.", elapsed_time_obj)

            if notify_formsubmit_id is not None:
                logger.info(
                    "Sending FormSubmit notification to %r with results files: %r as one zip attachment.",
                    notify_formsubmit_id,
                    [f.name for f in result_files],
                )

            if notify_formsubmit_id is not None:
                zip_filename = "igt_model_comparison_output_files.zip"
                email_message = f"""
The IGT model fitting and comparison script has completed successfully.

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
