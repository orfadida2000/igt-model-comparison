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
from igt.rdata_preprocessing import load_igt_long_table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Fit Q-learning and PVL-Delta to the Steingroever IGT dataset."
    )

    parser.add_argument(
        "--rdata-path",
        type=Path,
        default=IGT_DATASET_PATH,
        help="Path to the input IGTdata.rdata file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory in which result CSV files are written.",
    )
    parser.add_argument(
        "--pvl-starts",
        type=int,
        default=32,
        help="Number of Sobol starts for PVL-Delta; must be a power of two.",
    )
    parser.add_argument(
        "--rng",
        type=int,
        default=42,
        help="Integer RNG seed used by the scrambled Sobol generator.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1_000,
        help="Maximum L-BFGS-B iterations per optimization run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=("Number of subject-fitting worker processes. Use 1 for serial execution."),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the subject progress bar.",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        default=None,
        help="Number of subjects to fit (default: all subjects).",
    )

    return parser.parse_args()


def main() -> None:
    """Load the dataset, fit both models, compare them, and save CSV outputs."""

    args = parse_args()

    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be greater than zero.")

    if args.workers <= 0:
        raise ValueError("--workers must be greater than zero.")

    data = load_igt_long_table(args.rdata_path)

    models = (
        QLearningModel(),
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
