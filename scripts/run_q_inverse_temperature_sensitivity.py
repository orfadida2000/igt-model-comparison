"""Run the targeted Q-learning inverse-temperature sensitivity analysis."""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Final

from igt.constants.config import (
    DEFAULT_N_Q_STARTS,
    DEFAULT_N_WORKERS,
    DEFAULT_ROOT_LOG_LEVEL,
    FILENAME_DATETIME_FMT,
)
from igt.constants.fitting import DEFAULT_MAX_ITERATIONS
from igt.constants.path import IGT_DATASET_PATH, LOGS_DIR, RESULTS_DIR
from igt.execution.pipeline import FittingPipelineConfig, run_fitting_pipeline
from igt.logging_setup import configure_application_logging
from igt.models.q_learning import QLearningModel
from igt.subject_selection import (
    DEFAULT_INVERSE_TEMPERATURE_SELECTION_THRESHOLD,
    select_q_inverse_temperature_subject_keys_from_csv,
)

MAX_INVERSE_TEMPERATURES: Final[tuple[float, ...]] = (20.0, 50.0, 100.0)
DEFAULT_OUTPUT_DIR: Final[Path] = RESULTS_DIR / "q_inverse_temperature_sensitivity"


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
            "Refit capped Q-learning subjects with inverse-temperature maxima 20, 50, and 100."
        )
    )
    parser.add_argument(
        "fit_results_path",
        type=_existing_file_path,
        help=(
            "Previous full model-fits CSV used to select converged Q-learning "
            "fits with inverse_temperature >= 19.5."
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
        default=DEFAULT_OUTPUT_DIR,
        help=f"Sensitivity output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--logging-dir",
        type=_directory_path,
        default=LOGS_DIR,
        help=f"Log directory (default: {LOGS_DIR}).",
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


def _normalize_n_workers(value: int) -> int | None:
    if value < 0:
        return None
    if value == 0:
        return 1
    return value


def _format_numeric_filename_value(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def main() -> None:
    """Select capped subjects and run the three Q-learning sensitivity fits."""

    args = _parse_args()
    datetime_str = datetime.now().strftime(FILENAME_DATETIME_FMT)
    subject_keys = select_q_inverse_temperature_subject_keys_from_csv(
        args.fit_results_path,
        threshold=DEFAULT_INVERSE_TEMPERATURE_SELECTION_THRESHOLD,
        require_convergence=True,
    )
    n_selected_subjects = len(subject_keys)

    logging_path = configure_application_logging(
        disabled=args.log_level < 0,
        root_level=None if args.log_level < 0 else args.log_level,
        log_file_path=(
            args.logging_dir
            / (
                "q_inverse_temperature_sensitivity_"
                f"{n_selected_subjects}_subjects_{datetime_str}.log"
            )
        ),
    )

    logger = logging.getLogger("igt.q_inverse_temperature_sensitivity")
    logger.info("Starting Q-learning inverse-temperature sensitivity analysis...")
    logger.info(
        "Selected %d converged Q-learning subjects with inverse temperature >= %.1f.",
        n_selected_subjects,
        DEFAULT_INVERSE_TEMPERATURE_SELECTION_THRESHOLD,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_keys_path = args.output_dir / (
        "selected_subjects_beta_ge_"
        f"{_format_numeric_filename_value(DEFAULT_INVERSE_TEMPERATURE_SELECTION_THRESHOLD)}_"
        f"{datetime_str}.csv"
    )

    output_paths: list[Path] = []

    for max_inverse_temperature in MAX_INVERSE_TEMPERATURES:
        model = QLearningModel(
            n_starts=DEFAULT_N_Q_STARTS,
            max_inverse_temperature=max_inverse_temperature,
        )
        results_table = run_fitting_pipeline(
            FittingPipelineConfig(
                rdata_path=args.rdata_path,
                models=(model,),
                max_iterations=DEFAULT_MAX_ITERATIONS,
                n_workers=_normalize_n_workers(args.workers),
                show_progress=not args.no_progress,
                n_subjects=None,
                subject_keys=subject_keys,
            )
        )

        beta_label = _format_numeric_filename_value(max_inverse_temperature)
        output_path = args.output_dir / (
            "q_learning_max_inverse_temperature_"
            f"{beta_label}_{n_selected_subjects}_subjects_"
            f"{datetime_str}.csv"
        )
        results_table.to_csv(output_path, index=False, lineterminator="\n")
        output_paths.append(output_path)
        logger.info(
            "Saved Q-learning fits for maximum inverse temperature %g: %s",
            max_inverse_temperature,
            output_path,
        )

    subject_keys.to_csv(subject_keys_path, index=False, lineterminator="\n")
    logger.info("Saved selected subject keys: %s", subject_keys_path)

    if logging_path is not None:
        logger.info("Saved log file: %s", logging_path)

    logger.info("Completed sensitivity outputs: %r", output_paths)


if __name__ == "__main__":
    main()
