import argparse
from collections.abc import Callable, Iterable
from typing import Any


def _filter_arg_by_exclusion_values[T: Any](arg_value: T, exclusion_values: tuple[T, ...]) -> T:

    if arg_value in exclusion_values:
        raise argparse.ArgumentTypeError(
            f"Invalid argument value {arg_value!r}, the value must not be one of the following: {exclusion_values!r}"
        )

    return arg_value


def get_type_filter_for_exclusion_values(
    exclusion_values: Iterable[Any],
) -> Callable[[Any], Any]:
    """
    Returns a function that can be used as a type filter for argparse arguments. The returned function will filter argument values by the provided exclusion values.

    Args:
        exclusion_values: An iterable (not a string, bytes, or bytearray) of values to exclude.

    Returns:
        A filter function that takes an argument value and raises an ArgumentTypeError if the value is in the exclusion values, otherwise returns the value.

    Raises:
        TypeError: If the exclusion_values is not an iterable, is a string, bytes, or bytearray, or if it could not be converted to a tuple.
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
        return _filter_arg_by_exclusion_values(arg_value, exclusion_values)

    return _filter
