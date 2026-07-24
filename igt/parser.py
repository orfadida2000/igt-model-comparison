import argparse
from pathlib import Path


def _validate_n_q_starts(value: str | int, is_default: bool, err_class: type[BaseException]) -> int:
    """Parse and validate the maximum number of Q-learning grid-local-minimum starts."""

    if is_default:
        invalid_msg_prefix = "Invalid default number of Q-learning grid starts"
    else:
        invalid_msg_prefix = "Invalid number of Q-learning grid starts"

    if isinstance(value, int):
        if isinstance(value, bool):
            raise err_class(
                f"{invalid_msg_prefix}: expected a string or integer, got {type(value).__name__}"
            )

        n_starts = value
    else:
        if not isinstance(value, str):
            raise err_class(
                f"{invalid_msg_prefix}: expected a string or integer, got {type(value).__name__}"
            )

        try:
            n_starts = int(value)
        except ValueError as exc:
            raise err_class(
                f"{invalid_msg_prefix}: the string '{value}' is not a valid integer"
            ) from exc

    if n_starts < 1:
        raise err_class(f"{invalid_msg_prefix}: must be at least 1, got {n_starts}")

    return n_starts


def _n_q_starts_type(value: str) -> int:
    """Parse the maximum number of Q-learning grid-local-minimum starts for the argument parser."""

    return _validate_n_q_starts(value, is_default=False, err_class=argparse.ArgumentTypeError)


def _validate_n_pvl_starts(
    value: str | int, is_default: bool, err_class: type[BaseException]
) -> int:
    """Parse and validate the number of PVL-Delta Sobol starts."""

    if is_default:
        invalid_msg_prefix = "Invalid default number of PVL-Delta Sobol starts"
    else:
        invalid_msg_prefix = "Invalid number of PVL-Delta Sobol starts"

    if isinstance(value, int):
        if isinstance(value, bool):
            raise err_class(
                f"{invalid_msg_prefix}: expected a string or integer, got {type(value).__name__}"
            )

        n_starts = value
    else:
        if not isinstance(value, str):
            raise err_class(
                f"{invalid_msg_prefix}: expected a string or integer, got {type(value).__name__}"
            )

        try:
            n_starts = int(value)
        except ValueError as exc:
            raise err_class(
                f"{invalid_msg_prefix}: the string '{value}' is not a valid integer"
            ) from exc

    if n_starts < 1:
        raise err_class(f"{invalid_msg_prefix}: must be at least 1, got {n_starts}")

    if (n_starts & (n_starts - 1)) != 0:
        raise err_class(f"{invalid_msg_prefix}: must be a power of two, got {n_starts}")

    return n_starts


def _n_pvl_starts_type(value: str) -> int:
    """Parse and validate the number of PVL-Delta Sobol starts for the argument parser."""

    return _validate_n_pvl_starts(value, is_default=False, err_class=argparse.ArgumentTypeError)


def _validate_non_negative_int(value: str | int, label: str, err_class: type[BaseException]) -> int:
    """Parse and validate a non-negative integer."""

    if isinstance(value, int):
        if isinstance(value, bool):
            raise err_class(
                f"Invalid {label.lower()}: expected a string or integer, got {type(value).__name__}"
            )

        n = value
    else:
        if not isinstance(value, str):
            raise err_class(
                f"Invalid {label.lower()}: expected a string or integer, got {type(value).__name__}"
            )
        try:
            n = int(value)
        except ValueError as exc:
            raise err_class(f"Invalid {label.lower()}: {value}") from exc

    if n < 0:
        raise err_class(f"{label} must be greater than or equal to zero: {value}")

    return n


def _non_negative_int_type(value: str) -> int:
    """Parse and validate a non-negative integer for the argument parser."""

    return _validate_non_negative_int(
        value, label="Non-negative integer", err_class=argparse.ArgumentTypeError
    )


def _validate_positive_int(value: str | int, label: str, err_class: type[BaseException]) -> int:
    """Parse and validate a positive integer."""

    n = _validate_non_negative_int(value, label=label, err_class=err_class)

    if n == 0:
        raise err_class(f"{label} must be greater than zero, got {value}")

    return n


def _positive_int_type(value: str) -> int:
    """Parse and validate a positive integer for the argument parser."""

    return _validate_positive_int(
        value, label="Positive integer", err_class=argparse.ArgumentTypeError
    )


def _validate_rdata_path(
    value: str | Path, is_default: bool, err_class: type[BaseException]
) -> Path:
    """Parse and validate the RData file path."""

    if is_default:
        invalid_msg_prefix = "Invalid default RData file path"
    else:
        invalid_msg_prefix = "Invalid RData file path"

    if not isinstance(value, (str, Path)):
        raise err_class(
            f"{invalid_msg_prefix}: expected a string or Path, got {type(value).__name__}"
        )

    path = Path(value)

    if not path.is_file():
        raise err_class(f"{invalid_msg_prefix}: RData file does not exist: {value}")

    ext = path.suffix.lower()

    if ext not in {".rdata", ".rda"}:
        raise err_class(
            f"{invalid_msg_prefix}: RData file must have a .rdata or .rda extension, got {ext}"
        )

    return path


def _rdata_path_type(value: str) -> Path:
    """Parse and validate the RData file path for the argument parser."""

    return _validate_rdata_path(value, is_default=False, err_class=argparse.ArgumentTypeError)


def _validate_output_dir(
    value: str | Path, is_default: bool, err_class: type[BaseException]
) -> Path:
    """Parse and validate the output directory path."""

    if is_default:
        invalid_msg_prefix = "Invalid default output directory path"
    else:
        invalid_msg_prefix = "Invalid output directory path"

    if not isinstance(value, (str, Path)):
        raise err_class(
            f"{invalid_msg_prefix}: expected a string or Path, got {type(value).__name__}"
        )

    path = Path(value)

    if path.exists() and not path.is_dir():
        raise err_class(f"{invalid_msg_prefix}: Output path exists but is not a directory: {value}")

    return path


def _output_dir_type(value: str) -> Path:
    """Parse and validate the output directory path for the argument parser."""

    return _validate_output_dir(value, is_default=False, err_class=argparse.ArgumentTypeError)


def get_parser(
    default_rdata_path: str | Path,
    default_output_dir: str | Path,
    default_n_q_starts: int,
    default_n_pvl_starts: int,
    default_rng: int | None,
    default_max_iterations: int,
    default_n_workers: int,
    default_n_subjects: int | None,
) -> argparse.ArgumentParser:
    """Get the argument parser."""

    def validate_default_args() -> None:
        _validate_rdata_path(default_rdata_path, is_default=True, err_class=ValueError)

        _validate_output_dir(default_output_dir, is_default=True, err_class=ValueError)

        _validate_n_q_starts(default_n_q_starts, is_default=True, err_class=ValueError)

        _validate_n_pvl_starts(default_n_pvl_starts, is_default=True, err_class=ValueError)

        if default_rng is not None:
            _validate_non_negative_int(default_rng, label="Default RNG seed", err_class=ValueError)

        _validate_positive_int(
            default_max_iterations, label="Default maximum iterations", err_class=ValueError
        )

        _validate_positive_int(
            default_n_workers, label="Default number of workers", err_class=ValueError
        )

        if default_n_subjects is not None:
            _validate_positive_int(
                default_n_subjects, label="Default number of subjects", err_class=ValueError
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
        help=(
            "Maximum number of distinct grid-local-minimum starts for "
            "Q-learning; must be at least 1."
        ),
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
