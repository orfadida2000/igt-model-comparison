"""Command-line entry point for fitting and comparing IGT models."""

import argparse
from pathlib import Path

from igt.comparison import (
    add_model_comparison_columns,
    fit_results_to_dataframe,
    summarize_model_comparison,
)
from igt.constants.path import IGT_DATASET_PATH, RESULTS_DIR
from igt.execution import fit_all_subjects
from igt.models.pvl_delta import PVLDeltaModel
from igt.models.q_learning import QLearningModel
from igt.parser import get_parser
from igt.rdata_preprocessing import load_igt_long_table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = get_parser(
        default_rdata_path=IGT_DATASET_PATH,
        default_output_dir=RESULTS_DIR,
        default_n_q_starts=5,
        default_n_pvl_starts=32,
        default_rng=None,
        default_max_iterations=1_000,
        default_n_workers=1,
        default_n_subjects=None,
    )

    return parser.parse_args()


def main() -> None:
    """Load the dataset, fit both models, compare them, and save CSV outputs."""

    args = parse_args()

    data = load_igt_long_table(args.rdata_path)

    models = (
        QLearningModel(
            n_starts=args.q_starts,
        ),
        PVLDeltaModel(
            n_starts=args.pvl_starts,
            rng=args.rng,
        ),
    )

    fit_results = fit_all_subjects(
        data,
        models,
        optimizer_options={"maxiter": args.max_iterations},
        show_progress=not args.no_progress,
        n_workers=args.workers,
        n_subjects=args.subjects,
    )

    results_table = fit_results_to_dataframe(fit_results)
    comparison_table = add_model_comparison_columns(results_table)
    summary_table = summarize_model_comparison(results_table)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fits_path = output_dir / "model_fits.csv"
    comparison_path = output_dir / "model_comparison.csv"
    summary_path = output_dir / "model_summary.csv"

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

    print(f"Saved model fits: {fits_path}")
    print(f"Saved model comparison: {comparison_path}")
    print(f"Saved model summary: {summary_path}")
    print("\nModel summary:")
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
