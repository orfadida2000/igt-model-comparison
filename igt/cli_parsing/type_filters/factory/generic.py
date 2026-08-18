"""Generic factories shared by specialized command-line type filters.

These helpers wrap validation callables, normalize exception handling, and compose
multiple filters into argparse-compatible parsing stages without depending on a
specific value domain.
"""

import argparse
from collections.abc import Callable, Iterable
from typing import Any


def _filter_arg_by_exclusion_values[T: Any](arg_value: T, exclusion_values: tuple[T, ...]) -> T:

    """Reject a value when it belongs to an exclusion set.

    Args:
        arg_value: Value produced by an earlier parsing stage.
        exclusion_values: Values that are not permitted.

    Returns:
        The unchanged value when it is allowed.

    Raises:
        ArgumentTypeError: If the value is excluded.
    """
    if arg_value in exclusion_values:
        raise argparse.ArgumentTypeError(
            f"Invalid argument value {arg_value!r}, the value must not be one of the following: {exclusion_values!r}"
        )

    return arg_value


def get_type_filter_for_exclusion_values(
    exclusion_values: Iterable[Any],
) -> Callable[[Any], Any]:
    """Create an argparse-compatible filter that rejects excluded values.

    Args:
        exclusion_values: Values that should be rejected. The iterable is
            materialized once when the filter is created.

    Returns:
        A callable that returns allowed values unchanged and raises
        `ArgumentTypeError` for excluded values.

    Raises:
        TypeError: If the exclusion collection is not an acceptable iterable.
    """
    if not isinstance(exclusion_values, Iterable) or isinstance(
        exclusion_values, (str, bytes, bytearray)
    ):
        raise TypeError(
            f"Expected an iterable (but not a string, bytes, or bytearray) of exclusion values, got {type(exclusion_values).__name__}"
        )

    try:
        exclusion_values = tuple(exclusion_values)
    except TypeError as e:
        raise TypeError(
            f"Expected an iterable of exclusion values, but the provided iterable could not be converted to a tuple: {e}"
        ) from e

    def _filter(arg_value: Any) -> Any:
        """Apply the configured exclusion-value check.

        Args:
            arg_value: Value to validate.

        Returns:
            The unchanged value when it is not excluded.

        Raises:
            ArgumentTypeError: If the value matches an excluded value.
        """
        return _filter_arg_by_exclusion_values(arg_value, exclusion_values)

    return _filter
