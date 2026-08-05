"""Run the targeted Q-learning inverse-temperature sensitivity analysis."""

import argparse
import logging
import time
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

from pandas import DataFrame

from igt.constants.config import (
    DEFAULT_N_Q_STARTS,
    DEFAULT_N_WORKERS,
    DEFAULT_NOTIFY_FORMSUBMIT_ID,
    DEFAULT_ROOT_LOG_LEVEL,
    FILENAME_DATETIME_FMT,
)
from igt.constants.fitting import DEFAULT_MAX_ITERATIONS
from igt.constants.path import IGT_DATASET_PATH, LOGS_DIR, RESULTS_DIR
from igt.execution.pipeline import FittingPipelineConfig, run_fitting_pipeline
from igt.logging import application_logging_cleanup, configure_application_logging
from igt.models.q_learning import QLearningModel
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
    parser = argparse.ArgumentParser(
        description=(
            f"Refit capped Q-learning subjects with inverse-temperature maxima: {MAX_INVERSE_TEMPERATURES}, and for each maximum, refit with number of starts: {N_STARTS_VALUES}."
        )
    )
    parser.add_argument(
        "fit_results_path",
        type=_existing_file_path,
        help=(
            "Previous full model-fits CSV used to select converged Q-learning "
            "fits with inverse_temperature >= --selection-threshold."
        ),
    )
    parser.add_argument(
        "--selection-threshold",
        type=float,
        default=DEFAULT_INVERSE_TEMPERATURE_SELECTION_THRESHOLD,
        help=(
            "Inverse-temperature threshold for selecting converged Q-learning fits "
            f"(default: {DEFAULT_INVERSE_TEMPERATURE_SELECTION_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--rdata-path",
        type=_existing_file_path,
        default=IGT_DATASET_PATH,
        help=f"Input IGT RData file (default: {IGT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=_directory_path,
        default=(RESULTS_DIR / "q_inverse_temperature_sensitivity"),
        help=f"Sensitivity output directory (default: {RESULTS_DIR / 'q_inverse_temperature_sensitivity'}).",
    )
    parser.add_argument(
        "--logging-dir",
        type=_directory_path,
        default=(LOGS_DIR / "q_inverse_temperature_sensitivity"),
        help=f"Log directory (default: {LOGS_DIR / 'q_inverse_temperature_sensitivity'}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_N_WORKERS,
        help=(
            "Worker processes; use 0 for serial execution and a negative value "
            "for all available CPU cores."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=DEFAULT_ROOT_LOG_LEVEL,
        help="Root logging level; use a negative value to disable logging.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the subject progress bar.",
    )

    return parser.parse_args()


def _normalize_args(args: argparse.Namespace) -> dict[str, Any]:
    normalized_args: dict[str, Any] = {
        "fit_results_path": args.fit_results_path,
        "selection_threshold": args.selection_threshold
        if args.selection_threshold >= 0
        else DEFAULT_INVERSE_TEMPERATURE_SELECTION_THRESHOLD,
        "rdata_path": args.rdata_path,
        "n_workers": (None if args.workers < 0 else (1 if args.workers == 0 else args.workers)),
        "output_dir": args.output_dir or args.rdata_path.parent / "output_files",
        "logging_dir": args.logging_dir or args.rdata_path.parent / "logs",
        "logging_disabled": bool(args.log_level < 0),
        "log_level": None if args.log_level < 0 else args.log_level,
        "no_progress": args.no_progress,
        "notify_formsubmit_id": DEFAULT_NOTIFY_FORMSUBMIT_ID,
    }

    return normalized_args


def _format_numeric_filename_value(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _setup() -> tuple[argparse.Namespace, dict[str, Any], str, DataFrame, Path | None]:
    start_datetime_str = datetime.now().strftime(FILENAME_DATETIME_FMT)
    args = _parse_args()
    normalized_args = _normalize_args(args)

    normalized_args["output_dir"] = Path(normalized_args["output_dir"]) / start_datetime_str

    subject_keys = select_q_inverse_temperature_subject_keys_from_csv(
        normalized_args["fit_results_path"],
        threshold=normalized_args["selection_threshold"],
        require_convergence=True,
    )

    logging_path = configure_application_logging(
        disabled=normalized_args["logging_disabled"],
        root_level=normalized_args["log_level"],
        log_file_path=(
            normalized_args["logging_dir"]
            / (
                f"q_inverse_temperature_sensitivity_{len(subject_keys)}_subjects_{start_datetime_str}.log"
            )
        ),
    )
    return args, normalized_args, start_datetime_str, subject_keys, logging_path


def _run(
    *,
    normalized_args: dict[str, Any],
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
                    rdata_path=normalized_args["rdata_path"],
                    models=(q_learning_model,),
                    max_iterations=DEFAULT_MAX_ITERATIONS,
                    n_workers=normalized_args["n_workers"],
                    show_progress=not normalized_args["no_progress"],
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
            output_path = normalized_args["output_dir"] / (
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

    notify_formsubmit_id: str | None = normalized_args["notify_formsubmit_id"]

    with error_email_notifier(
        formsubmit_id=notify_formsubmit_id,
        script_name=Path(__file__).name,
        start_counter=start_counter,
    ):
        logger = logging.getLogger(LOGGER_NAME)
        logger.info("Starting Q-learning inverse-temperature sensitivity analysis...")

        logger.debug("Parsed command-line arguments: %r", vars(args))
        logger.debug("Normalized command-line arguments: %r", normalized_args)

        logger.info(
            "Selected %d converged Q-learning subjects with inverse temperature >= %.1f.",
            len(subject_keys),
            normalized_args["selection_threshold"],
        )

        normalized_args["output_dir"].mkdir(parents=True, exist_ok=True)

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

        subject_keys_path = normalized_args["output_dir"] / (
            "selected_subjects_beta_ge_"
            f"{_format_numeric_filename_value(normalized_args['selection_threshold'])}_"
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
            "Completed Q-learning inverse-temperature sensitivity analysis in %s.", elapsed_time_obj
        )

        if notify_formsubmit_id is not None:
            logger.info(
                "Sending FormSubmit notification to %r with results files: %r as one zip attachment.",
                notify_formsubmit_id,
                [f.name for f in result_files],
            )

        logger.info("Performing cleanup...")
        _cleanup(logger=logger)

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


if __name__ == "__main__":
    main()
