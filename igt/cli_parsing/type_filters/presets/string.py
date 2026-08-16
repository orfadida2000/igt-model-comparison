import argparse

from igt.cli_parsing.type_filters.core.definitions import (
    TypeFilterChainDefinition,
)
from igt.cli_parsing.type_filters.core.registry import TypeFilterRegistry


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


class StringArgTypeProvider(TypeFilterChainDefinition):
    """Enumeration of argument types for command-line argument parsing."""

    NON_WHITESPACE_STRING = (non_whitespace_string_type,)
    NON_EMPTY_STRING = (non_empty_string_type,)
    ALPHANUMERIC_STRING = (alphanumeric_string_type,)


TypeFilterRegistry.register_provider(StringArgTypeProvider)
