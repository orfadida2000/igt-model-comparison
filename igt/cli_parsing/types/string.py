import argparse
from collections.abc import Callable, Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from igt.typing import NonEmptyMixedTuple


def validate_non_empty_string(value: str, label: str, err_class: type[BaseException]) -> str:
    """Parse and validate a non-empty string."""

    if not isinstance(value, str):
        raise err_class(f"Invalid {label.lower()}: expected a string, got {type(value).__name__}")

    if not value:
        raise err_class(f"{label} must not be empty.")

    return value


def non_empty_string_type(value: str) -> str:
    """Parse and validate a non-empty string for the argument parser."""

    return validate_non_empty_string(
        value, label="Non-empty string", err_class=argparse.ArgumentTypeError
    )


def validate_non_whitespace_string(value: str, label: str, err_class: type[BaseException]) -> str:
    """Parse and validate a non-whitespace string."""

    if not isinstance(value, str):
        raise err_class(f"Invalid {label.lower()}: expected a string, got {type(value).__name__}")

    value = value.strip()

    if not value:
        raise err_class(f"{label} must not be empty or whitespace only.")

    return value


def non_whitespace_string_type(value: str) -> str:
    """Parse and validate a non-whitespace string for the argument parser."""

    return validate_non_whitespace_string(
        value, label="Non-whitespace string", err_class=argparse.ArgumentTypeError
    )


def validate_alphanumeric_string(value: str, label: str, err_class: type[BaseException]) -> str:
    """Parse and validate an alphanumeric string."""

    if not isinstance(value, str):
        raise err_class(f"Invalid {label.lower()}: expected a string, got {type(value).__name__}")

    value = value.strip()

    if not value.isalnum():
        raise err_class(f"{label} must be alphanumeric.")

    return value


def alphanumeric_string_type(value: str) -> str:
    """Parse and validate an alphanumeric string for the argument parser."""

    return validate_alphanumeric_string(
        value, label="Alphanumeric string", err_class=argparse.ArgumentTypeError
    )


class StringArgType(Enum):
    """Enumeration of argument types for command-line argument parsing."""

    NON_WHITESPACE_STRING = "non-whitespace string"
    NON_EMPTY_STRING = "non-empty string"
    ALPHANUMERIC_STRING = "alphanumeric string"


ARG_TYPE_CALLABLE_MAP: Final[
    Mapping[
        StringArgType,
        Callable[[str], Any] | NonEmptyMixedTuple[Callable[[str], Any], Callable[[Any], Any]],
    ]
] = MappingProxyType(
    {
        StringArgType.NON_WHITESPACE_STRING: non_whitespace_string_type,
        StringArgType.NON_EMPTY_STRING: non_empty_string_type,
        StringArgType.ALPHANUMERIC_STRING: alphanumeric_string_type,
    }
)
