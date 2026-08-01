"""Command-line entry point for fitting and comparing IGT models."""

import argparse
import logging
from datetime import datetime
from typing import Any

from igt.comparison import (
    add_model_comparison_columns,
    summarize_model_comparison,
)
from igt.constants.config import (
    DEFAULT_N_PVL_STARTS,
    DEFAULT_N_Q_STARTS,
    DEFAULT_N_SUBJECTS,
    DEFAULT_N_WORKERS,
    DEFAULT_ROOT_LOG_LEVEL,
    FILENAME_DATETIME_FMT,
    FIXED_SEED,
    SUBJECTS_AVAILABLE,
    USE_FIXED_SEED,
)
from igt.constants.fitting import DEFAULT_MAX_ITERATIONS
from igt.constants.models import DEFAULT_MAX_INVERSE_TEMPERATURE
from igt.constants.path import IGT_DATASET_PATH, LOGS_DIR, RESULTS_DIR
from igt.constants.schema import PARTICIPANT_KEY_COLUMNS
from igt.execution.pipeline import FittingPipelineConfig, run_fitting_pipeline
from igt.logging_setup import configure_application_logging
from igt.models.pvl_delta import PVLDeltaModel
from igt.models.q_learning import QLearningModel
from igt.parser import get_parser


def parse_args() -> tuple[argparse.Namespace, dict[str, Any]]:
    """Parse command-line arguments and return resolved runtime values."""

    # A negative user-provided seed requests a generator without a fixed seed.
    if USE_FIXED_SEED:
        default_seed_for_parser = None
    else:
        default_seed_for_parser = -1

    parser = get_parser(
        default_rdata_path=IGT_DATASET_PATH,
        default_max_iterations=DEFAULT_MAX_ITERATIONS,
        default_n_q_starts=DEFAULT_N_Q_STARTS,
        default_n_pvl_starts=DEFAULT_N_PVL_STARTS,
        default_q_max_inverse_temperature=DEFAULT_MAX_INVERSE_TEMPERATURE,
        default_seed=default_seed_for_parser,
        default_n_workers=DEFAULT_N_WORKERS,
        default_n_subjects=DEFAULT_N_SUBJECTS,
        default_output_dir=RESULTS_DIR,
        default_logging_dir=LOGS_DIR,
        default_log_level=DEFAULT_ROOT_LOG_LEVEL,
    )

    args = parser.parse_args()

    normalized_args: dict[str, Any] = {
        "rdata_path": args.rdata_path,
        "max_iterations": args.max_iterations,
        "n_q_starts": args.q_starts,
        "n_pvl_starts": args.pvl_starts,
        "q_max_inverse_temperature": args.q_max_inverse_temperature,
        "rng_seed": (
            FIXED_SEED
            if USE_FIXED_SEED
            else (None if args.seed < 0 else args.seed)
        ),
        "n_workers": (
            None
            if args.workers < 0
            else (1 if args.workers == 0 else args.workers)
        ),
        "n_subjects": None if args.subjects < 0 else args.subjects,
        "effective_n_subjects": (
            SUBJECTS_AVAILABLE if args.subjects < 0 else args.subjects
        ),
        "output_dir": args.output_dir or args.rdata_path.parent,
        "logging_dir": args.logging_dir or args.rdata_path.parent,
        "logging_disabled": bool(args.log_level < 0),
        "log_level": None if args.log_level < 0 else args.log_level,
        "no_progress": args.no_progress,
    }

    return args, normalized_args


def main() -> None:
    """Load the dataset, fit both models, compare them, and save CSV outputs."""

    args, normalized_args = parse_args()
    datetime_str = datetime.now().strftime(FILENAME_DATETIME_FMT)
    logging_path = configure_application_logging(
        disabled=normalized_args["logging_disabled"],
        root_level=normalized_args["log_level"],
        log_file_path=(
            normalized_args["logging_dir"]
            / (
                "igt_model_comparison_"
                f"{normalized_args['effective_n_subjects']}_subjects_"
                f"{datetime_str}.log"
            )
        ),
    )

    logger = logging.getLogger("igt.main")
    logger.info("Starting IGT model fitting and comparison...")
    logger.debug("Parsed command-line arguments: %r", vars(args))
    logger.debug("Normalized command-line arguments: %r", normalized_args)

    models = (
        QLearningModel(
            n_starts=normalized_args["n_q_starts"],
            max_inverse_temperature=normalized_args[
                "q_max_inverse_temperature"
            ],
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
        results_table.loc[:, list(PARTICIPANT_KEY_COLUMNS)]
        .drop_duplicates()
        .shape[0]
    )
    filename_suffix = f"{actual_n_subjects}_subjects_{datetime_str}"
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

    if logging_path is not None:
        logger.info("Saved log file: %s", logging_path)

    logger.debug("Model summary:\n%s", summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
