"""Factories for string command-line validation and normalization.

The module provides reusable filters for string conversion, whitespace handling,
nonempty constraints, and other string-specific checks used by project CLI presets.
"""

import argparse
import re

from igt.cli_parsing.type_filters.core.definitions import (
    TypeFilter,
)


def _filter_string_by_regex(arg_value: str, regex_pattern: str, *, flags: int = 0) -> str:
    """Validate a string with `re.search`.

    The regular expression is used as supplied, so callers should include
    anchors when full-string matching is required.

    Args:
        arg_value: String value to validate.
        regex_pattern: Pattern searched within the value.
        flags: Regular-expression flags passed to `re.search`.

    Returns:
        The unchanged string when the pattern matches.

    Raises:
        ArgumentTypeError: If the pattern is not found.
        TypeError: If the argument value is not a string.
    """

    if not isinstance(arg_value, str):
        raise TypeError(f"Expected a string argument value, got {type(arg_value).__name__}")

    if not re.search(
        regex_pattern,
        arg_value,
        flags=flags,
    ):
        raise argparse.ArgumentTypeError(
            f"Invalid argument value {arg_value!r}, the value must match the regex pattern: {regex_pattern!r}"
        )

    return arg_value


def get_type_filter_for_string_matching_regex(regex_pattern: str, *, flags: int = 0) -> TypeFilter:
    """Create an argparse-compatible regular-expression string filter.

    Args:
        regex_pattern: Pattern searched within each candidate value.
        flags: Regular-expression flags passed to `re.search`.

    Returns:
        A callable that returns matching strings unchanged and rejects
        non-matching values.

    Raises:
        TypeError: If the pattern or flags have invalid types.
    """

    if not isinstance(regex_pattern, str):
        raise TypeError(f"Expected a string regex pattern, got {type(regex_pattern).__name__}")

    if not isinstance(flags, int):
        raise TypeError(f"Expected an integer for flags, got {type(flags).__name__}")

    if isinstance(flags, bool):
        raise TypeError(
            f"Expected an integer for flags, got {type(flags).__name__} (bool is not allowed)"
        )

    def _filter(arg_value: str) -> str:
        """Apply the configured regular-expression string filter.

        Args:
            arg_value: String value to validate.

        Returns:
            The validated string.

        Raises:
            ArgumentTypeError: If the configured pattern does not match.
        """
        return _filter_string_by_regex(arg_value, regex_pattern, flags=flags)

    return _filter
