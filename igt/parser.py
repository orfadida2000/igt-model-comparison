import argparse
from pathlib import Path


def _n_q_starts_type(value: str) -> int:
    """Parse and validate the number of Q-learning grid starts."""

    try:
        n_starts = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid number of Q-learning grid starts: {value}"
        ) from exc

    return max(1, n_starts)


def _n_pvl_starts_type(value: str) -> int:
    def closest_power_of_2(n: int) -> int:
        if n <= 0:
            return 1

        prev_pow = 1 << (n.bit_length() - 1)

        # If it's already a perfect power of 2, just return it
        if n == prev_pow:
            return n

        next_pow = prev_pow << 1

        # Compare the distances
        return prev_pow if (n - prev_pow) < (next_pow - n) else next_pow

    """Parse and validate the number of PVL-Delta Sobol starts."""

    try:
        n_starts = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid number of PVL-Delta Sobol starts: {value}"
        ) from exc

    n_starts = closest_power_of_2(n_starts)

    return n_starts


def _non_negative_int_type(value: str) -> int:
    """Parse and validate a non-negative integer."""

    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid non-negative integer: {value}") from exc

    if n < 0:
        raise argparse.ArgumentTypeError(
            f"Non-negative integer must be greater than or equal to zero: {value}"
        )

    return n


def _positive_int_type(value: str) -> int:
    """Parse and validate a positive integer."""

    n = _non_negative_int_type(value)

    if n == 0:
        raise argparse.ArgumentTypeError(f"Positive integer must be greater than zero: {value}")

    return n


def _rdata_path_type(value: str) -> Path:
    """Parse and validate the RData file path."""

    path = Path(value)

    if not path.is_file():
        raise argparse.ArgumentTypeError(f"RData file does not exist: {value}")

    ext = path.suffix.lower()

    if ext not in {".rdata", ".rda"}:
        raise argparse.ArgumentTypeError(
            f"RData file must have a .rdata or .rda extension: {value}"
        )

    return path


def _output_dir_type(value: str) -> Path:
    """Parse and validate the output directory path."""

    path = Path(value)

    if path.exists() and not path.is_dir():
        raise argparse.ArgumentTypeError(f"Output path exists but is not a directory: {value}")

    return path


def get_parser(
    default_rdata_path: Path,
    default_output_dir: Path,
    default_n_q_starts: int,
    default_n_pvl_starts: int,
    default_rng: int | None,
    default_max_iterations: int,
    default_n_workers: int,
    default_n_subjects: int | None,
) -> argparse.ArgumentParser:
    """Get the argument parser."""

    def validate_default_args():
        if not default_rdata_path.is_file():
            raise ValueError(f"default_rdata_path does not exist: {default_rdata_path}")
        if default_rdata_path.suffix.lower() not in {".rdata", ".rda"}:
            raise ValueError(
                f"default_rdata_path must have a .rdata or .rda extension: {default_rdata_path}"
            )

        if default_output_dir.exists() and not default_output_dir.is_dir():
            raise ValueError(
                f"default_output_dir exists but is not a directory: {default_output_dir}"
            )

        if default_n_q_starts < 1:
            raise ValueError(f"default_n_q_starts must be at least 1: {default_n_q_starts}")

        if default_n_pvl_starts < 1:
            raise ValueError(f"default_n_pvl_starts must be at least 1: {default_n_pvl_starts}")
        if (default_n_pvl_starts & (default_n_pvl_starts - 1)) != 0:
            raise ValueError(f"default_n_pvl_starts must be a power of two: {default_n_pvl_starts}")

        if default_rng is not None and default_rng < 0:
            raise ValueError(f"default_rng must be non-negative or None: {default_rng}")

        if default_max_iterations <= 0:
            raise ValueError(
                f"default_max_iterations must be greater than zero: {default_max_iterations}"
            )

        if default_n_workers <= 0:
            raise ValueError(f"default_n_workers must be greater than zero: {default_n_workers}")

        if default_n_subjects is not None and default_n_subjects <= 0:
            raise ValueError(
                f"default_n_subjects must be greater than zero or None: {default_n_subjects}"
            )

    validate_default_args()

    parser = argparse.ArgumentParser(
        description="Fit Q-learning and PVL-Delta to the Steingroever IGT dataset."
    )

    parser.add_argument(
        "--rdata-path",
        type=_rdata_path_type,
        default=default_rdata_path,
        help="Path to the input IGTdata.rdata file.",
    )
    parser.add_argument(
        "--output-dir",
        type=_output_dir_type,
        default=default_output_dir,
        help="Directory in which result CSV files are written.",
    )
    parser.add_argument(
        "--q-starts",
        type=_n_q_starts_type,
        default=default_n_q_starts,
        help="Number of grid starts for Q-learning; must be at least 1.",
    )
    parser.add_argument(
        "--pvl-starts",
        type=_n_pvl_starts_type,
        default=default_n_pvl_starts,
        help="Number of Sobol starts for PVL-Delta; must be a power of two.",
    )
    parser.add_argument(
        "--rng",
        type=_non_negative_int_type,
        default=default_rng,
        help="Integer RNG seed used by the scrambled Sobol generator.",
    )
    parser.add_argument(
        "--max-iterations",
        type=_positive_int_type,
        default=default_max_iterations,
        help="Maximum L-BFGS-B iterations per optimization run.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int_type,
        default=default_n_workers,
        help=("Number of subject-fitting worker processes. Use 1 for serial execution."),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the subject progress bar.",
    )
    parser.add_argument(
        "--subjects",
        type=_positive_int_type,
        default=default_n_subjects,
        help="Number of subjects to fit (default: all subjects).",
    )

    return parser
