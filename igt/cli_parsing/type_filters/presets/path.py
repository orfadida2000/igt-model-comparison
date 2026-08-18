"""Preset filesystem-path validators and argparse-compatible type filters.

The presets cover existing files, existing directories, extension-constrained files,
and normalized path inputs. `PathArgTypeProvider` exposes the common combinations to
the command-line type-filter registry.
"""

import argparse
from pathlib import Path

from igt.cli_parsing.type_filters.core.definitions import TypeFilterChainDefinition
from igt.cli_parsing.type_filters.core.registry import TypeFilterRegistry
from igt.typing import StrPathLike
from igt.utils.io import normalize_path


def validate_dir_path(value: StrPathLike, label: str, err_class: type[BaseException]) -> Path:
    """Parse and validate a directory path.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated dir path.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    try:
        path = normalize_path(value, parameter_name=label.lower().replace(" ", "_"))
    except Exception as e:
        raise err_class(f"Invalid {label.lower()}: {e}") from None

    if path.exists() and not path.is_dir():
        raise err_class(f"Invalid {label.lower()}: path exists but is not a directory: {path}")

    return path


def dir_path_type(value: str) -> Path:
    """Parse and validate a directory path for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated dir path value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_dir_path(value, label="Directory path", err_class=argparse.ArgumentTypeError)


def validate_existing_dir_path(
    value: StrPathLike, label: str, err_class: type[BaseException]
) -> Path:
    """Parse and validate an existing directory path.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated existing dir path.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    path = validate_dir_path(value, label=label, err_class=err_class)

    if not path.exists():
        raise err_class(f"Invalid {label.lower()}: path does not exist: {path}")

    return path


def existing_dir_path_type(value: str) -> Path:
    """Parse and validate an existing directory path for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated existing dir path value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_existing_dir_path(
        value, label="Existing directory path", err_class=argparse.ArgumentTypeError
    )


def validate_file_path(value: StrPathLike, label: str, err_class: type[BaseException]) -> Path:
    """Parse and validate a file path.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated file path.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    try:
        path = normalize_path(value, parameter_name=label.lower().replace(" ", "_"))
    except Exception as e:
        raise err_class(f"Invalid {label.lower()}: {e}") from None

    if path.exists() and not path.is_file():
        raise err_class(f"Invalid {label.lower()}: path exists but is not a file: {path}")

    return path


def file_path_type(value: str) -> Path:
    """Parse and validate a file path for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated file path value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_file_path(value, label="File path", err_class=argparse.ArgumentTypeError)


def validate_existing_file_path(
    value: StrPathLike, label: str, err_class: type[BaseException]
) -> Path:
    """Parse and validate an existing file path.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated existing file path.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    path = validate_file_path(value, label=label, err_class=err_class)

    if not path.exists():
        raise err_class(f"Invalid {label.lower()}: path does not exist: {path}")

    return path


def existing_file_path_type(value: str) -> Path:
    """Parse and validate an existing file path for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated existing file path value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_existing_file_path(
        value, label="Existing file path", err_class=argparse.ArgumentTypeError
    )


class PathArgTypeProvider(TypeFilterChainDefinition):
    """Named filesystem-path type-filter chains registered for command-line parsing.

    Members cover directory and file paths, including variants that require the
    target to already exist. Each member is resolved by
    [`TypeFilterRegistry`][igt.cli_parsing.type_filters.core.registry.TypeFilterRegistry].
    """

    FILE_PATH = (file_path_type,)
    EXISTING_FILE_PATH = (existing_file_path_type,)
    DIR_PATH = (dir_path_type,)
    EXISTING_DIR_PATH = (existing_dir_path_type,)


TypeFilterRegistry.register_provider(PathArgTypeProvider)
