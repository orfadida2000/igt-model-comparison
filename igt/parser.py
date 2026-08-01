import argparse
from math import isfinite
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


def _validate_int(value: str | int, label: str, err_class: type[BaseException]) -> int:
    """Parse and validate an integer."""

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

    return n


def _int_type(value: str) -> int:
    """Parse and validate a integer for the argument parser."""

    return _validate_int(value, label="Integer", err_class=argparse.ArgumentTypeError)


def _validate_non_negative_int(value: str | int, label: str, err_class: type[BaseException]) -> int:
    """Parse and validate a non-negative integer."""

    n = _validate_int(value, label=label, err_class=err_class)

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


def _validate_positive_float(
    value: str | float,
    label: str,
    err_class: type[BaseException],
) -> float:
    """Parse and validate a finite positive floating-point value."""

    if isinstance(value, bool):
        raise err_class(
            f"Invalid {label.lower()}: expected a string or number, got bool"
        )

    try:
        parsed_value = float(value)
    except (TypeError, ValueError) as exc:
        raise err_class(f"Invalid {label.lower()}: {value}") from exc

    if not isfinite(parsed_value):
        raise err_class(f"{label} must be finite, got {value}")

    if parsed_value <= 0.0:
        raise err_class(f"{label} must be greater than zero, got {value}")

    return parsed_value


def _positive_float_type(value: str) -> float:
    """Parse and validate a positive float for the argument parser."""

    return _validate_positive_float(
        value, label="Positive floating-point value", err_class=argparse.ArgumentTypeError
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


def _validate_dir_path(value: str | Path, label: str, err_class: type[BaseException]) -> Path:
    """Parse and validate a directory path."""

    if not isinstance(value, (str, Path)):
        raise err_class(
            f"Invalid {label.lower()}: expected a string or Path, got {type(value).__name__}"
        )

    path = Path(value)

    if path.exists() and not path.is_dir():
        raise err_class(f"Invalid {label.lower()}: path exists but is not a directory: {path}")

    return path


def _dir_path_type(value: str) -> Path:
    """Parse and validate a directory path for the argument parser."""

    return _validate_dir_path(value, label="Directory path", err_class=argparse.ArgumentTypeError)


def get_parser(
    *,
    default_rdata_path: str | Path | None = None,
    default_max_iterations: int = 1000,
    default_n_q_starts: int = 1,
    default_n_pvl_starts: int = 1,
    default_q_max_inverse_temperature: float = 20.0,
    default_seed: int | None = -1,
    default_n_workers: int = 0,
    default_n_subjects: int = -1,
    default_output_dir: str | Path | None = None,
    default_logging_dir: str | Path | None = None,
    default_log_level: int = -1,
) -> argparse.ArgumentParser:
    """Get the argument parser."""
    parser = argparse.ArgumentParser(
        description="Fit Q-learning and PVL-Delta to the Steingroever IGT dataset."
    )

    if default_rdata_path is None:
        parser.add_argument(
            "rdata-path",
            type=_rdata_path_type,
            help="Path to the input IGTdata.rdata file (required).",
        )
    else:
        default_rdata_path = _validate_rdata_path(
            default_rdata_path, is_default=True, err_class=ValueError
        )
        parser.add_argument(
            "--rdata-path",
            type=_rdata_path_type,
            default=default_rdata_path,
            help=f"Path to the input IGTdata.rdata file (default: {default_rdata_path}).",
        )

    _validate_positive_int(
        default_max_iterations, label="Default maximum iterations", err_class=ValueError
    )
    parser.add_argument(
        "--max-iterations",
        type=_positive_int_type,
        default=default_max_iterations,
        help=f"Maximum L-BFGS-B iterations per optimization; must be a positive integer (default: {default_max_iterations}).",
    )

    _validate_n_q_starts(default_n_q_starts, is_default=True, err_class=ValueError)
    parser.add_argument(
        "--q-starts",
        type=_n_q_starts_type,
        default=default_n_q_starts,
        help=(
            "Maximum number of distinct grid-local-minimum starts for "
            f"Q-learning; must be a positive integer (default: {default_n_q_starts})."
        ),
    )

    _validate_n_pvl_starts(default_n_pvl_starts, is_default=True, err_class=ValueError)
    parser.add_argument(
        "--pvl-starts",
        type=_n_pvl_starts_type,
        default=default_n_pvl_starts,
        help=(
            "Number of Sobol starts for "
            f"PVL-Delta; must be an integer power of 2 (default: {default_n_pvl_starts})."
        ),
    )

    _validate_positive_float(
        default_q_max_inverse_temperature,
        label="Default Q-learning maximum inverse temperature",
        err_class=ValueError,
    )
    parser.add_argument(
        "--q-max-inverse-temperature",
        type=_positive_float_type,
        default=default_q_max_inverse_temperature,
        help=(
            "Upper bound for the Q-learning inverse temperature. The default "
            "Q-learning grid automatically preserves approximately unit spacing "
            f"along this dimension (default: {default_q_max_inverse_temperature})."
        ),
    )

    if default_seed is not None:
        _validate_int(default_seed, label="Default seed", err_class=ValueError)
        parser.add_argument(
            "--seed",
            type=_int_type,
            default=default_seed,
            help=f"Integer seed used by the scrambled Sobol generator; use negative value for no fixed seed (default: {default_seed}).",
        )
    # when default_seed is None, we don't add a --seed argument, since we use a fixed one in that case.

    _validate_int(default_n_workers, label="Default number of workers", err_class=ValueError)
    parser.add_argument(
        "--workers",
        type=_int_type,
        default=default_n_workers,
        help=f"Number of subject-fitting worker processes; use 0 for serial execution and negative value for all available CPU cores (default: {default_n_workers}).",
    )

    _validate_int(default_n_subjects, label="Default number of subjects", err_class=ValueError)
    parser.add_argument(
        "--subjects",
        type=_int_type,
        default=default_n_subjects,
        help=f"Number of subjects to fit; use negative value to fit all subjects (default: {default_n_subjects}).",
    )

    if default_output_dir is None:
        parser.add_argument(
            "--output-dir",
            type=_dir_path_type,
            help="Directory in which result CSV files are written (default: directory of the RData file).",
        )
    else:
        default_output_dir = _validate_dir_path(
            default_output_dir, label="Default output directory", err_class=ValueError
        )
        parser.add_argument(
            "--output-dir",
            type=_dir_path_type,
            default=default_output_dir,
            help=f"Directory in which result CSV files are written (default: {default_output_dir}).",
        )

    if default_logging_dir is None:
        parser.add_argument(
            "--logging-dir",
            type=_dir_path_type,
            help="Directory in which log files are written (default: directory of the RData file).",
        )
    else:
        default_logging_dir = _validate_dir_path(
            default_logging_dir, label="Default logging directory", err_class=ValueError
        )
        parser.add_argument(
            "--logging-dir",
            type=_dir_path_type,
            default=default_logging_dir,
            help=f"Directory in which log files are written (default: {default_logging_dir}).",
        )

    _validate_int(default_log_level, label="Default log level", err_class=ValueError)
    parser.add_argument(
        "--log-level",
        type=int,
        default=default_log_level,
        help=f"Logging level of the root logger; use negative value to disable logging (default: {default_log_level}).",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar shown during fitting.",
    )

    return parser
