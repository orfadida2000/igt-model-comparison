"""Command-line entry point for fitting and comparing IGT models."""

import argparse
import logging
from datetime import datetime
from typing import Any

from igt.comparison import (
    add_model_comparison_columns,
    fit_results_to_dataframe,
    summarize_model_comparison,
)
from igt.constants.config import (
    DATETIME_FORMAT,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_N_PVL_STARTS,
    DEFAULT_N_Q_STARTS,
    DEFAULT_N_SUBJECTS,
    DEFAULT_N_WORKERS,
    DEFAULT_ROOT_LOG_LEVEL,
    FILE_LOG_LEVEL,
    FILENAME_DATETIME_FMT,
    FIXED_SEED,
    LOG_FORMAT,
    SUBJECTS_AVAILABLE,
    TERMINAL_LOG_LEVEL,
    USE_FIXED_SEED,
)
from igt.constants.path import IGT_DATASET_PATH, LOGS_DIR, RESULTS_DIR
from igt.execution.manager import fit_all_subjects
from igt.models.pvl_delta import PVLDeltaModel
from igt.models.q_learning import QLearningModel
from igt.parser import get_parser
from igt.rdata_preprocessing import load_igt_long_table
from igt.typing import (
    BaseLogHandlerConfig,
    FileLogHandlerConfig,
    NullLogHandlerConfig,
    StandardOutput,
    TerminalLogHandlerConfig,
)


def parse_args() -> tuple[argparse.Namespace, dict[str, Any]]:
    """Parse command-line arguments."""

    # negative seed value means that an rng will be created without a fixed seed.
    if USE_FIXED_SEED:
        default_seed_for_parser = None  # the seed arg is excluded from the parser
    else:
        default_seed_for_parser = -1  # the seed arg is included in the parser, with default -1

    parser = get_parser(
        default_rdata_path=IGT_DATASET_PATH,
        default_max_iterations=DEFAULT_MAX_ITERATIONS,
        default_n_q_starts=DEFAULT_N_Q_STARTS,
        default_n_pvl_starts=DEFAULT_N_PVL_STARTS,
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
        "rng_seed": FIXED_SEED if USE_FIXED_SEED else (None if args.seed < 0 else args.seed),
        "n_workers": None if args.workers < 0 else (1 if args.workers == 0 else args.workers),
        "n_subjects": None if args.subjects < 0 else args.subjects,
        "effective_n_subjects": SUBJECTS_AVAILABLE if args.subjects < 0 else args.subjects,
        "output_dir": args.output_dir or args.rdata_path.parent,
        "logging_dir": args.logging_dir or args.rdata_path.parent,
        "logging_disabled": bool(args.log_level < 0),
        "log_level": None if args.log_level < 0 else args.log_level,
        "no_progress": args.no_progress,
    }

    return args, normalized_args


def config_root_logger(
    level: int | None = None, handler_configs: list[BaseLogHandlerConfig] | None = None
) -> None:
    handler_configs = handler_configs or []
    handler_configs = [
        handler_config
        for handler_config in handler_configs
        if isinstance(handler_config, BaseLogHandlerConfig)
        and not isinstance(handler_config, NullLogHandlerConfig)
    ]
    handler_configs = handler_configs or [NullLogHandlerConfig()]

    root_logger = logging.getLogger()

    if level is not None:
        root_logger.setLevel(level)

    root_logger.handlers.clear()
    for handler_config in handler_configs:
        handler = handler_config.create_handler()
        root_logger.addHandler(handler)


def main() -> None:
    """Load the dataset, fit both models, compare them, and save CSV outputs."""

    args, normalized_args = parse_args()

    handler_configs: list[BaseLogHandlerConfig] = []

    datetime_str = datetime.now().strftime(FILENAME_DATETIME_FMT)

    if not normalized_args["logging_disabled"]:
        handler_configs.append(
            TerminalLogHandlerConfig(
                level=TERMINAL_LOG_LEVEL,
                log_format=LOG_FORMAT,
                datetime_format=DATETIME_FORMAT,
                stream=StandardOutput.STDERR,
            )
        )

        logging_path = (
            normalized_args["logging_dir"]
            / f"igt_model_comparison_{normalized_args['effective_n_subjects']}_subjects_{datetime_str}.log"
        )
        handler_configs.append(
            FileLogHandlerConfig(
                level=FILE_LOG_LEVEL,
                log_format=LOG_FORMAT,
                datetime_format=DATETIME_FORMAT,
                file_path=logging_path,
            )
        )

        config_root_logger(level=normalized_args["log_level"], handler_configs=handler_configs)
    else:
        logging_path = None
        config_root_logger(handler_configs=handler_configs)

    logger = logging.getLogger("igt.main")

    logger.info("Starting IGT model fitting and comparison...")
    logger.debug("Parsed command-line arguments: %r", vars(args))
    logger.debug("Normalized command-line arguments: %r", normalized_args)

    logger.info("Loading IGT dataset from: %s", normalized_args["rdata_path"])
    data = load_igt_long_table(normalized_args["rdata_path"])

    models = (
        QLearningModel(
            n_starts=normalized_args["n_q_starts"],
        ),
        PVLDeltaModel(
            n_starts=normalized_args["n_pvl_starts"],
            rng=normalized_args["rng_seed"],
        ),
    )

    logger.info("Fitting IGT models...")
    fit_results = fit_all_subjects(
        data,
        models,
        optimizer_options={"maxiter": normalized_args["max_iterations"]},
        show_progress=not normalized_args["no_progress"],
        n_workers=normalized_args["n_workers"],
        n_subjects=normalized_args["n_subjects"],
    )
    logger.info("Model fitting completed.")

    logger.info("Processing and saving results...")
    results_table = fit_results_to_dataframe(fit_results)
    comparison_table = add_model_comparison_columns(results_table)
    summary_table = summarize_model_comparison(results_table)

    output_dir = normalized_args["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    filename_suffix = f"{len(fit_results)}_subjects_{datetime_str}"
    fits_path = output_dir / f"model_fits_{filename_suffix}.csv"
    comparison_path = output_dir / f"model_comparison_{filename_suffix}.csv"
    summary_path = output_dir / f"model_summary_{filename_suffix}.csv"

    results_table.to_csv(
        fits_path,
        index=False,
        lineterminator="\n",
    )
    comparison_table.to_csv(
        comparison_path,
        index=False,
        lineterminator="\n",
    )
    summary_table.to_csv(
        summary_path,
        index=False,
        lineterminator="\n",
    )

    logger.info("Saved model fits: %s", fits_path)
    logger.info("Saved model comparison: %s", comparison_path)
    logger.info("Saved model summary: %s", summary_path)

    if logging_path is not None:
        logger.info("Saved log file: %s", logging_path)

    logger.debug("Model summary:\n%s", summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
