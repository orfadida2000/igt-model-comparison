import argparse
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from igt.typing import NonEmptyMixedTuple, StrPathLike
from igt.utils.io import normalize_path


def validate_dir_path(value: StrPathLike, label: str, err_class: type[BaseException]) -> Path:
    """Parse and validate a directory path."""

    try:
        path = normalize_path(value, parameter_name=label.lower().replace(" ", "_"))
    except Exception as e:
        raise err_class(f"Invalid {label.lower()}: {e}") from None

    if path.exists() and not path.is_dir():
        raise err_class(f"Invalid {label.lower()}: path exists but is not a directory: {path}")

    return path


def dir_path_type(value: str) -> Path:
    """Parse and validate a directory path for the argument parser."""

    return validate_dir_path(value, label="Directory path", err_class=argparse.ArgumentTypeError)


def validate_existing_dir_path(
    value: StrPathLike, label: str, err_class: type[BaseException]
) -> Path:
    """Parse and validate an existing directory path."""

    path = validate_dir_path(value, label=label, err_class=err_class)

    if not path.exists():
        raise err_class(f"Invalid {label.lower()}: path does not exist: {path}")

    return path


def existing_dir_path_type(value: str) -> Path:
    """Parse and validate an existing directory path for the argument parser."""

    return validate_existing_dir_path(
        value, label="Existing directory path", err_class=argparse.ArgumentTypeError
    )


def validate_file_path(value: StrPathLike, label: str, err_class: type[BaseException]) -> Path:
    """Parse and validate a file path."""

    try:
        path = normalize_path(value, parameter_name=label.lower().replace(" ", "_"))
    except Exception as e:
        raise err_class(f"Invalid {label.lower()}: {e}") from None

    if path.exists() and not path.is_file():
        raise err_class(f"Invalid {label.lower()}: path exists but is not a file: {path}")

    return path


def file_path_type(value: str) -> Path:
    """Parse and validate a file path for the argument parser."""

    return validate_file_path(value, label="File path", err_class=argparse.ArgumentTypeError)


def validate_existing_file_path(
    value: StrPathLike, label: str, err_class: type[BaseException]
) -> Path:
    """Parse and validate an existing file path."""

    path = validate_file_path(value, label=label, err_class=err_class)

    if not path.exists():
        raise err_class(f"Invalid {label.lower()}: path does not exist: {path}")

    return path


def existing_file_path_type(value: str) -> Path:
    """Parse and validate an existing file path for the argument parser."""

    return validate_existing_file_path(
        value, label="Existing file path", err_class=argparse.ArgumentTypeError
    )


class PathArgType(Enum):
    """Enumeration of argument types for command-line argument parsing."""

    FILE_PATH = "file path"
    EXISTING_FILE_PATH = "existing file path"
    DIR_PATH = "directory path"
    EXISTING_DIR_PATH = "existing directory path"


ARG_TYPE_CALLABLE_MAP: Final[
    Mapping[
        PathArgType,
        Callable[[str], Any] | NonEmptyMixedTuple[Callable[[str], Any], Callable[[Any], Any]],
    ]
] = MappingProxyType(
    {
        PathArgType.FILE_PATH: file_path_type,
        PathArgType.EXISTING_FILE_PATH: existing_file_path_type,
        PathArgType.DIR_PATH: dir_path_type,
        PathArgType.EXISTING_DIR_PATH: existing_dir_path_type,
    }
)
