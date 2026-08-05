"""Command-line entry point for fitting and comparing IGT models."""

import argparse
import logging
import time
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

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
    USE_DEFAULT_NOTIFY_FORMSUBMIT_ID,
    USE_FIXED_SEED,
)
from igt.constants.fitting import DEFAULT_MAX_ITERATIONS
from igt.constants.models import DEFAULT_MAX_INVERSE_TEMPERATURE
from igt.constants.path import IGT_DATASET_PATH, LOGS_DIR, RESULTS_DIR
from igt.constants.schema import PARTICIPANT_KEY_COLUMNS
from igt.execution.pipeline import FittingPipelineConfig, run_fitting_pipeline
from igt.logging import application_logging_cleanup, configure_application_logging
from igt.models.pvl_delta import PVLDeltaModel
from igt.models.q_learning import QLearningModel
from igt.notify.formsubmit import (
    error_email_notifier,
    send_formsubmit_email_script_success_notification,
)
from igt.parser import get_parser

LOGGER_NAME: Final[str] = "igt.main"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return resolved runtime values."""

    parser = get_parser(
        default_rdata_path=IGT_DATASET_PATH,
        default_max_iterations=DEFAULT_MAX_ITERATIONS,
        default_n_q_starts=DEFAULT_N_Q_STARTS,
        default_n_pvl_starts=DEFAULT_N_PVL_STARTS,
        default_q_max_inverse_temperature=DEFAULT_MAX_INVERSE_TEMPERATURE,
        default_seed=None if USE_FIXED_SEED else -1,
        default_n_workers=DEFAULT_N_WORKERS,
        default_n_subjects=DEFAULT_N_SUBJECTS,
        default_output_dir=RESULTS_DIR / "igt_model_comparison",
        default_logging_dir=LOGS_DIR / "igt_model_comparison",
        default_log_level=DEFAULT_ROOT_LOG_LEVEL,
        default_notify_formsubmit_id=None
        if USE_DEFAULT_NOTIFY_FORMSUBMIT_ID
        else DEFAULT_NOTIFY_FORMSUBMIT_ID,
    )

    return parser.parse_args()


def _normalize_args(args: argparse.Namespace) -> dict[str, Any]:
    """Normalize command-line arguments and return a dictionary of runtime values."""

    normalized_args: dict[str, Any] = {
        "rdata_path": args.rdata_path,
        "max_iterations": args.max_iterations,
        "n_q_starts": args.q_starts,
        "n_pvl_starts": args.pvl_starts,
        "q_max_inverse_temperature": args.q_max_inverse_temperature,
        "rng_seed": (FIXED_SEED if USE_FIXED_SEED else (None if args.seed < 0 else args.seed)),
        "n_workers": (None if args.workers < 0 else (1 if args.workers == 0 else args.workers)),
        "n_subjects": None if args.subjects < 0 else args.subjects,
        "effective_n_subjects": (SUBJECTS_AVAILABLE if args.subjects < 0 else args.subjects),
        "output_dir": args.output_dir or args.rdata_path.parent / "output_files",
        "logging_dir": args.logging_dir or args.rdata_path.parent / "logs",
        "logging_disabled": bool(args.log_level < 0),
        "log_level": None if args.log_level < 0 else args.log_level,
        "no_progress": args.no_progress,
        "notify_formsubmit_id": DEFAULT_NOTIFY_FORMSUBMIT_ID
        if USE_DEFAULT_NOTIFY_FORMSUBMIT_ID
        else (None if not args.notify_formsubmit_id else args.notify_formsubmit_id),
    }

    return normalized_args


def _setup() -> tuple[argparse.Namespace, dict[str, Any], str, Path | None]:
    """Parse command-line arguments, configure logging, and return runtime values."""

    start_datetime_str = datetime.now().strftime(FILENAME_DATETIME_FMT)
    args = _parse_args()
    normalized_args = _normalize_args(args)

    normalized_args["output_dir"] = Path(normalized_args["output_dir"]) / start_datetime_str

    logging_path = configure_application_logging(
        disabled=normalized_args["logging_disabled"],
        root_level=normalized_args["log_level"],
        log_file_path=(
            normalized_args["logging_dir"]
            / (
                "igt_model_comparison_"
                f"{normalized_args['effective_n_subjects']}_subjects_"
                f"{start_datetime_str}.log"
            )
        ),
    )

    return args, normalized_args, start_datetime_str, logging_path


def _run(
    *,
    normalized_args: dict[str, Any],
    start_datetime_str: str,
    logger: logging.Logger | str = LOGGER_NAME,
) -> Sequence[str | Path]:
    """Run the IGT model fitting and comparison pipeline based on command-line arguments."""

    logger = logging.getLogger(logger) if isinstance(logger, str) else logger

    models = (
        QLearningModel(
            n_starts=normalized_args["n_q_starts"],
            max_inverse_temperature=normalized_args["q_max_inverse_temperature"],
        ),
        PVLDeltaModel(
            n_starts=normalized_args["n_pvl_starts"],
            rng=normalized_args["rng_seed"],
        ),
    )

    results_table = run_fitting_pipeline(
        FittingPipelineConfig(
            rdata_path=normalized_args["rdata_path"],
            models=models,
            max_iterations=normalized_args["max_iterations"],
            n_workers=normalized_args["n_workers"],
            show_progress=not normalized_args["no_progress"],
            n_subjects=normalized_args["n_subjects"],
            subject_keys=None,
        )
    )

    logger.info("Processing and saving results...")

    comparison_table = add_model_comparison_columns(results_table)
    summary_table = summarize_model_comparison(results_table)

    output_dir = normalized_args["output_dir"]
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

    return [fits_path, comparison_path, summary_path]


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

    notify_formsubmit_id: str | None = normalized_args["notify_formsubmit_id"]

    with error_email_notifier(
        formsubmit_id=notify_formsubmit_id,
        script_name=Path(__file__).name,
        start_counter=start_counter,
    ):
        logger = logging.getLogger(LOGGER_NAME)
        logger.info("Starting IGT model fitting and comparison...")
        logger.debug("Parsed command-line arguments: %r", vars(args))
        logger.debug("Normalized command-line arguments: %r", normalized_args)

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

        logger.info("Performing cleanup...")
        _cleanup(logger=logger)

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


if __name__ == "__main__":
    main()
