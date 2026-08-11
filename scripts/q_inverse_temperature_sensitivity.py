"""Run the targeted Q-learning inverse-temperature sensitivity analysis."""

import argparse
import logging
import time
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

from pandas import DataFrame

from igt.cli_parsing.factory.path import get_type_filters_for_existing_file_with_extensions_path
from igt.cli_parsing.parser import get_parser
from igt.cli_parsing.typing import (
    ArgAction,
    ArgSpec,
    NumericArgType,
    PathArgType,
    StringArgType,
)
from igt.constants.config import (
    DEFAULT_N_Q_STARTS,
    DEFAULT_N_WORKERS,
    DEFAULT_NOTIFY_FORMSUBMIT_ID,
    DEFAULT_ROOT_LOG_LEVEL,
    FILENAME_DATETIME_FMT,
    USE_FIXED_NOTIFY_FORMSUBMIT_ID,
    USE_RDATA_PARENT_DIR_FOR_LOGGING,
    USE_RDATA_PARENT_DIR_FOR_OUTPUT,
)
from igt.constants.fitting import DEFAULT_MAX_ITERATIONS
from igt.constants.path import IGT_DATASET_PATH, LOGS_DIR, RESULTS_DIR
from igt.execution.pipeline import FittingPipelineConfig, run_fitting_pipeline
from igt.logging import application_logging_cleanup, configure_application_logging
from igt.models import QLearningModel
from igt.notify.formsubmit import (
    error_email_notifier,
    send_formsubmit_email_script_success_notification,
)
from igt.subject_selection import (
    DEFAULT_INVERSE_TEMPERATURE_SELECTION_THRESHOLD,
    select_q_inverse_temperature_subject_keys_from_csv,
)

MAX_INVERSE_TEMPERATURES: Final[tuple[float, ...]] = (100.0,)
N_STARTS_VALUES: Final[tuple[int, ...]] = (DEFAULT_N_Q_STARTS, DEFAULT_N_Q_STARTS * 2)

LOGGER_NAME: Final[str] = "scripts.q_inverse_temperature_sensitivity"


def _existing_file_path(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def _directory_path(value: str) -> Path:
    path = Path(value)
    if path.exists() and not path.is_dir():
        raise argparse.ArgumentTypeError(f"Path exists but is not a directory: {path}")
    return path


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
            name_or_flags="fit-results-path",
            type_filters=get_type_filters_for_existing_file_with_extensions_path(".csv"),
            help="Path to the input CSV file containing previous model fitting results used for subject selection",
            extra_options={
                "metavar": "FILE_PATH",
            },
        ),
        ArgSpec(
            name_or_flags=("--selection-threshold",),
            type_filters=(NumericArgType.POSITIVE_FINITE_FLOAT,),
            default=str(DEFAULT_INVERSE_TEMPERATURE_SELECTION_THRESHOLD),
            help=(
                "Inverse-temperature threshold for selecting converged Q-learning fits. "
                "Subjects that their fitted inverse-temperature is greater than or equal to this value will be selected; "
                "must be a positive finite float (default: %(default)s)"
            ),
            extra_options={
                "metavar": "INVERSE_TEMPERATURE_THRESHOLD",
            },
        ),
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
            name_or_flags=("--workers",),
            type_filters=(NumericArgType.INTEGER,),
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
    ]

    if not USE_RDATA_PARENT_DIR_FOR_OUTPUT:
        arg_specs.append(
            ArgSpec(
                name_or_flags=("--output-dir",),
                type_filters=(PathArgType.DIR_PATH,),
                default=str(RESULTS_DIR / "q_inverse_temperature_sensitivity"),
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
                type_filters=(PathArgType.DIR_PATH,),
                default=str(LOGS_DIR / "q_inverse_temperature_sensitivity"),
                help="Directory to save the log files (default: %(default)s)",
                extra_options={
                    "metavar": "LOGGING_DIR",
                },
            )
        )

    arg_specs.append(
        ArgSpec(
            name_or_flags=("--log-level",),
            type_filters=(NumericArgType.INTEGER,),
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
                type_filters=(StringArgType.ALPHANUMERIC_STRING,),
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

    (
        parser,
        resolved_info,
    ) = get_parser(
        arg_specs,
        description="Refit capped Q-learning subjects with specified inverse-temperature maxima and number of starts.",
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
        "fit_results_path": args.fit_results_path,
        "selection_threshold": args.selection_threshold,
        "rdata_path": args.rdata_path,
        "n_workers": None if args.n_workers < 0 else (1 if args.n_workers == 0 else args.n_workers),
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
    }

    return argparse.Namespace(**normalized_args)


def _format_numeric_filename_value(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _setup() -> tuple[argparse.Namespace, argparse.Namespace, str, DataFrame, Path | None]:
    start_datetime_str = datetime.now().strftime(FILENAME_DATETIME_FMT)
    args = _parse_args()
    normalized_args = _normalize_args(args)

    normalized_args.output_dir = Path(normalized_args.output_dir) / start_datetime_str

    subject_keys = select_q_inverse_temperature_subject_keys_from_csv(
        normalized_args.fit_results_path,
        threshold=normalized_args.selection_threshold,
        require_convergence=True,
    )

    logging_path = configure_application_logging(
        disabled=normalized_args.logging_disabled,
        root_logger_level=normalized_args.log_level,
        log_file_path=(
            normalized_args.logging_dir
            / (
                f"q_inverse_temperature_sensitivity_{len(subject_keys)}_subjects_{start_datetime_str}.log"
            )
        ),
    )
    return args, normalized_args, start_datetime_str, subject_keys, logging_path


def _run(
    *,
    normalized_args: argparse.Namespace,
    start_datetime_str: str,
    subject_keys: DataFrame,
    logger: logging.Logger | str = LOGGER_NAME,
) -> Sequence[str | Path]:
    logger = logging.getLogger(logger) if isinstance(logger, str) else logger

    output_paths: list[Path] = []

    for max_inverse_temperature in MAX_INVERSE_TEMPERATURES:
        logger.info(
            "Running Q-learning fits for max inverse temperature=%g, n_starts=%r",
            max_inverse_temperature,
            N_STARTS_VALUES,
        )
        for n_starts in N_STARTS_VALUES:
            q_learning_model = QLearningModel(
                n_starts=n_starts,
                max_inverse_temperature=max_inverse_temperature,
            )

            logger.info(
                "Running Q-learning fits for max inverse temperature=%g, n_starts=%d",
                max_inverse_temperature,
                n_starts,
            )
            results_table = run_fitting_pipeline(
                FittingPipelineConfig(
                    rdata_path=normalized_args.rdata_path,
                    models=(q_learning_model,),
                    max_iterations=DEFAULT_MAX_ITERATIONS,
                    n_workers=normalized_args.n_workers,
                    show_progress=not normalized_args.no_progress,
                    n_subjects=None,
                    subject_keys=subject_keys,
                )
            )

            logger.info(
                "Completed Q-learning fits for max inverse temperature=%g, n_starts=%d",
                max_inverse_temperature,
                n_starts,
            )

            beta_label = _format_numeric_filename_value(max_inverse_temperature)
            output_path = normalized_args.output_dir / (
                "q_learning_max_inverse_temperature_"
                f"{beta_label}_{len(subject_keys)}_subjects_"
                f"{n_starts}_starts_"
                f"{start_datetime_str}.csv"
            )

            results_table.to_csv(output_path, index=False, lineterminator="\n")
            output_paths.append(output_path)
            logger.info(
                "Saved Q-learning fits for maximum inverse temperature=%g, n_starts=%d to: %s",
                max_inverse_temperature,
                n_starts,
                output_path,
            )

        logger.info(
            "Completed Q-learning fits for max inverse temperature=%g, n_starts=%r",
            max_inverse_temperature,
            N_STARTS_VALUES,
        )

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
    """Select capped subjects and run the three Q-learning sensitivity fits."""
    start_counter = time.perf_counter()

    (
        args,
        normalized_args,
        start_datetime_str,
        subject_keys,
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
            logger.info("Starting Q-learning inverse-temperature sensitivity analysis...")

            logger.debug("Parsed command-line arguments: %s", args)
            logger.debug("Normalized command-line arguments: %s", normalized_args)

            logger.info(
                "Selected %d converged Q-learning subjects with inverse temperature >= %.1f.",
                len(subject_keys),
                normalized_args.selection_threshold,
            )

            normalized_args.output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Running Q-learning fits for max inverse temperatures=%r, n_starts=%r",
                tuple(f"{val:g}" for val in MAX_INVERSE_TEMPERATURES),
                N_STARTS_VALUES,
            )

            result_files = _run(
                normalized_args=normalized_args,
                start_datetime_str=start_datetime_str,
                subject_keys=subject_keys,
                logger=logger,
            )
            result_files = [Path(f) for f in result_files]

            subject_keys_path = normalized_args.output_dir / (
                "selected_subjects_beta_ge_"
                f"{_format_numeric_filename_value(normalized_args.selection_threshold)}_"
                f"{start_datetime_str}.csv"
            )
            subject_keys.to_csv(subject_keys_path, index=False, lineterminator="\n")
            logger.info("Saved selected subject keys to: %s", subject_keys_path)
            result_files.append(subject_keys_path)

            if logging_path is not None:
                logger.info("Saved log file to: %s", logging_path)

                result_files.append(Path(logging_path))

            end_counter = time.perf_counter()
            elapsed_time = end_counter - start_counter
            elapsed_time_obj = timedelta(seconds=round(elapsed_time))

            logger.info(
                "Completed Q-learning inverse-temperature sensitivity analysis in %s.",
                elapsed_time_obj,
            )

            if notify_formsubmit_id is not None:
                logger.info(
                    "Sending FormSubmit notification to %r with results files: %r as one zip attachment.",
                    notify_formsubmit_id,
                    [f.name for f in result_files],
                )

            if notify_formsubmit_id is not None:
                zip_filename = "q_inverse_temperature_sensitivity_output_files.zip"
                email_message = f"""
The Q-learning inverse-temperature sensitivity analysis has completed successfully.

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
