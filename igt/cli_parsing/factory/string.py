import argparse
import re
from collections.abc import Callable


def _filter_string_by_regex(arg_value: str, regex_pattern: str, *, flags: int = 0) -> str:
    """
    Filter a string argument value by a regular expression pattern. Raises an ArgumentTypeError if the pattern isn't found in the argument value.

    This function uses the re.search() method to check if the provided regex pattern is found anywhere in the argument value.
    The pattern is taken as is, so any anchoring (like ^ or $) should be included in the pattern if needed.

    Args:
        arg_value: The string argument value to be filtered.
        regex_pattern: The regular expression pattern to match against the argument value.
        flags: The flags to pass to the re.search() function.

    Returns:
        The filtered string argument value if it matches the regex pattern.

    Raises:
        - argparse.ArgumentTypeError: If the regex pattern is not found in the argument value.
        - TypeError: If the argument value is not a string.
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


def get_type_filter_for_string_matching_regex(
    regex_pattern: str, *, flags: int = 0
) -> Callable[[str], str]:
    """
    Returns a function that can be used as a type filter for argparse arguments. The returned function will filter string argument values by the provided regex pattern.

    The filter function will use the re.search() method to check if the provided regex pattern is found anywhere in the argument value.
    The pattern is taken as is, so any anchoring (like ^ or $) should be included in the pattern if needed.

    Args:
        regex_pattern: The regular expression pattern to match against the argument value.
        flags: The flags to pass to the re.search() function.

    Returns:
        A filter function that takes a string argument value and raises an ArgumentTypeError if the value does not match the regex pattern, otherwise returns the value.

    Raises:
        TypeError: If the regex_pattern is not a string or if the flags are not an integer.
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
        return _filter_string_by_regex(arg_value, regex_pattern, flags=flags)

    return _filter
