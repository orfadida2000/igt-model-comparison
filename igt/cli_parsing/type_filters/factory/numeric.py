import argparse

from igt.cli_parsing.type_filters.core.definitions import (
    GenericTypeFilter,
    TypeFilter,
)
from igt.cli_parsing.type_filters.presets.numeric import NumericArgTypeProvider
from igt.typing import PrimitiveNumber


def _filter_number_by_range[Number: (int, float)](
    n: Number,
    min_value: PrimitiveNumber | None = None,
    max_value: PrimitiveNumber | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> Number:
    # assumes that a float is not NaN.

    if min_value is not None:
        if n < min_value:
            raise argparse.ArgumentTypeError(
                f"Invalid number: must not be smaller than {min_value}, got {n}"
            )

        if not min_inclusive and n == min_value:
            raise argparse.ArgumentTypeError(
                f"Invalid number: must be greater than {min_value}, got {n}"
            )

    if max_value is not None:
        if n > max_value:
            raise argparse.ArgumentTypeError(
                f"Invalid number: must not be greater than {max_value}, got {n}"
            )

        if not max_inclusive and n == max_value:
            raise argparse.ArgumentTypeError(
                f"Invalid number: must be smaller than {max_value}, got {n}"
            )

    return n


def get_type_filters_for_number_with_range(
    min_value: PrimitiveNumber | None = None,
    max_value: PrimitiveNumber | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> tuple[
    NumericArgTypeProvider | TypeFilter,
    *tuple[NumericArgTypeProvider | GenericTypeFilter, ...],
]:
    """
    Returns a tuple of type filters for numbers within the given range.

    Args:
        min_value: The minimum value (inclusive or exclusive).
        max_value: The maximum value (inclusive or exclusive).
        min_inclusive: Whether the minimum value is inclusive.
        max_inclusive: Whether the maximum value is inclusive.

    Returns:
        A tuple containing the type filters for numbers within the specified range.
    """

    return (
        NumericArgTypeProvider.NUMBER,
        lambda n: _filter_number_by_range(n, min_value, max_value, min_inclusive, max_inclusive),
    )


def get_type_filters_for_finite_number_with_range(
    min_value: PrimitiveNumber | None = None,
    max_value: PrimitiveNumber | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> tuple[
    NumericArgTypeProvider | TypeFilter,
    *tuple[NumericArgTypeProvider | GenericTypeFilter, ...],
]:
    """
    Returns a tuple of type filters for finite numbers within the given range.

    Args:
        min_value: The minimum value (inclusive or exclusive).
        max_value: The maximum value (inclusive or exclusive).
        min_inclusive: Whether the minimum value is inclusive.
        max_inclusive: Whether the maximum value is inclusive.

    Returns:
        A tuple containing the type filters for finite numbers within the specified range.
    """

    def _filter_finite_number[Number: (int, float)](n: Number) -> Number:
        return _filter_number_by_range(n, min_value, max_value, min_inclusive, max_inclusive)

    return (
        NumericArgTypeProvider.FINITE_NUMBER,
        _filter_finite_number,
    )


def get_type_filters_for_integer_with_range(
    min_value: PrimitiveNumber | None = None,
    max_value: PrimitiveNumber | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> tuple[
    NumericArgTypeProvider | TypeFilter,
    *tuple[NumericArgTypeProvider | GenericTypeFilter, ...],
]:
    """
    Returns a tuple of type filters for integers within the given range.

    Args:
        min_value: The minimum value (inclusive or exclusive).
        max_value: The maximum value (inclusive or exclusive).
        min_inclusive: Whether the minimum value is inclusive.
        max_inclusive: Whether the maximum value is inclusive.

    Returns:
        A tuple containing the type filters for integers within the specified range.
    """

    return (
        NumericArgTypeProvider.INTEGER,
        lambda n: _filter_number_by_range(n, min_value, max_value, min_inclusive, max_inclusive),
    )


def get_type_filters_for_float_with_range(
    min_value: PrimitiveNumber | None = None,
    max_value: PrimitiveNumber | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> tuple[
    NumericArgTypeProvider | TypeFilter,
    *tuple[NumericArgTypeProvider | GenericTypeFilter, ...],
]:
    """
    Returns a tuple of type filters for floating-points within the given range.

    Args:
        min_value: The minimum value (inclusive or exclusive).
        max_value: The maximum value (inclusive or exclusive).
        min_inclusive: Whether the minimum value is inclusive.
        max_inclusive: Whether the maximum value is inclusive.

    Returns:
        A tuple containing the type filters for floating-points within the specified range.
    """

    return (
        NumericArgTypeProvider.FLOAT,
        lambda n: _filter_number_by_range(n, min_value, max_value, min_inclusive, max_inclusive),
    )


def get_type_filters_for_finite_float_with_range(
    min_value: PrimitiveNumber | None = None,
    max_value: PrimitiveNumber | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> tuple[
    NumericArgTypeProvider | TypeFilter,
    *tuple[NumericArgTypeProvider | GenericTypeFilter, ...],
]:
    """
    Returns a tuple of type filters for finite floating-points within the given range.

    Args:
        min_value: The minimum value (inclusive or exclusive).
        max_value: The maximum value (inclusive or exclusive).
        min_inclusive: Whether the minimum value is inclusive.
        max_inclusive: Whether the maximum value is inclusive.

    Returns:
        A tuple containing the type filters for finite floating-points within the specified range.
    """

    return (
        NumericArgTypeProvider.FINITE_FLOAT,
        lambda n: _filter_number_by_range(n, min_value, max_value, min_inclusive, max_inclusive),
    )


def get_type_filters_for_number_with_lower_bound(
    min_value: PrimitiveNumber | None = None,
    min_inclusive: bool = True,
) -> tuple[
    NumericArgTypeProvider | TypeFilter,
    *tuple[NumericArgTypeProvider | GenericTypeFilter, ...],
]:
    """
    Returns a tuple of type filters for numbers within a lower bound range.

    Args:
        min_value: The minimum value (inclusive or exclusive).
        min_inclusive: Whether the minimum value is inclusive.

    Returns:
        A tuple containing the type filters for numbers within the specified range.
    """

    return get_type_filters_for_number_with_range(
        min_value=min_value,
        max_value=None,
        min_inclusive=min_inclusive,
    )


def get_type_filters_for_finite_number_with_lower_bound(
    min_value: PrimitiveNumber | None = None,
    min_inclusive: bool = True,
) -> tuple[
    NumericArgTypeProvider | TypeFilter,
    *tuple[NumericArgTypeProvider | GenericTypeFilter, ...],
]:
    """
    Returns a tuple of type filters for finite numbers within a lower bound range.

    Args:
        min_value: The minimum value (inclusive or exclusive).
        min_inclusive: Whether the minimum value is inclusive.

    Returns:
        A tuple containing the type filters for finite numbers within the specified range.
    """

    return get_type_filters_for_finite_number_with_range(
        min_value=min_value,
        max_value=None,
        min_inclusive=min_inclusive,
    )


def get_type_filters_for_integer_with_lower_bound(
    min_value: PrimitiveNumber | None = None,
    min_inclusive: bool = True,
) -> tuple[
    NumericArgTypeProvider | TypeFilter,
    *tuple[NumericArgTypeProvider | GenericTypeFilter, ...],
]:
    """
    Returns a tuple of type filters for integers within a lower bound range.

    Args:
        min_value: The minimum value (inclusive or exclusive).
        min_inclusive: Whether the minimum value is inclusive.

    Returns:
        A tuple containing the type filters for integers within the specified range.
    """

    return get_type_filters_for_integer_with_range(
        min_value=min_value,
        max_value=None,
        min_inclusive=min_inclusive,
    )


def get_type_filters_for_float_with_lower_bound(
    min_value: PrimitiveNumber | None = None,
    min_inclusive: bool = True,
) -> tuple[
    NumericArgTypeProvider | TypeFilter,
    *tuple[NumericArgTypeProvider | GenericTypeFilter, ...],
]:
    """
    Returns a tuple of type filters for floating-points within a lower bound range.

    Args:
        min_value: The minimum value (inclusive or exclusive).
        min_inclusive: Whether the minimum value is inclusive.

    Returns:
        A tuple containing the type filters for floating-points within the specified range.
    """

    return get_type_filters_for_float_with_range(
        min_value=min_value,
        max_value=None,
        min_inclusive=min_inclusive,
    )


def get_type_filters_for_finite_float_with_lower_bound(
    min_value: PrimitiveNumber | None = None,
    min_inclusive: bool = True,
) -> tuple[
    NumericArgTypeProvider | TypeFilter,
    *tuple[NumericArgTypeProvider | GenericTypeFilter, ...],
]:
    """
    Returns a tuple of type filters for finite floating-points within a lower bound range.

    Args:
        min_value: The minimum value (inclusive or exclusive).
        min_inclusive: Whether the minimum value is inclusive.

    Returns:
        A tuple containing the type filters for finite floating-points within the specified range.
    """

    return get_type_filters_for_finite_float_with_range(
        min_value=min_value,
        max_value=None,
        min_inclusive=min_inclusive,
    )


def get_type_filters_for_number_with_upper_bound(
    max_value: PrimitiveNumber | None = None,
    max_inclusive: bool = True,
) -> tuple[
    NumericArgTypeProvider | TypeFilter,
    *tuple[NumericArgTypeProvider | GenericTypeFilter, ...],
]:
    """
    Returns a tuple of type filters for numbers within an upper bound range.

    Args:
        max_value: The maximum value (inclusive or exclusive).
        max_inclusive: Whether the maximum value is inclusive.

    Returns:
        A tuple containing the type filters for numbers within the specified range.
    """

    return get_type_filters_for_number_with_range(
        min_value=None,
        max_value=max_value,
        max_inclusive=max_inclusive,
    )


def get_type_filters_for_finite_number_with_upper_bound(
    max_value: PrimitiveNumber | None = None,
    max_inclusive: bool = True,
) -> tuple[
    NumericArgTypeProvider | TypeFilter,
    *tuple[NumericArgTypeProvider | GenericTypeFilter, ...],
]:
    """
    Returns a tuple of type filters for finite numbers within an upper bound range.

    Args:
        max_value: The maximum value (inclusive or exclusive).
        max_inclusive: Whether the maximum value is inclusive.

    Returns:
        A tuple containing the type filters for finite numbers within the specified range.
    """

    return get_type_filters_for_finite_number_with_range(
        min_value=None,
        max_value=max_value,
        max_inclusive=max_inclusive,
    )


def get_type_filters_for_integer_with_upper_bound(
    max_value: PrimitiveNumber | None = None,
    max_inclusive: bool = True,
) -> tuple[
    NumericArgTypeProvider | TypeFilter,
    *tuple[NumericArgTypeProvider | GenericTypeFilter, ...],
]:
    """
    Returns a tuple of type filters for integers within an upper bound range.

    Args:
        max_value: The maximum value (inclusive or exclusive).
        max_inclusive: Whether the maximum value is inclusive.

    Returns:
        A tuple containing the type filters for integers within the specified range.
    """

    return get_type_filters_for_integer_with_range(
        min_value=None,
        max_value=max_value,
        max_inclusive=max_inclusive,
    )


def get_type_filters_for_float_with_upper_bound(
    max_value: PrimitiveNumber | None = None,
    max_inclusive: bool = True,
) -> tuple[
    NumericArgTypeProvider | TypeFilter,
    *tuple[NumericArgTypeProvider | GenericTypeFilter, ...],
]:
    """
    Returns a tuple of type filters for floating-points within an upper bound range.

    Args:
        max_value: The maximum value (inclusive or exclusive).
        max_inclusive: Whether the maximum value is inclusive.

    Returns:
        A tuple containing the type filters for floating-points within the specified range.
    """

    return get_type_filters_for_float_with_range(
        min_value=None,
        max_value=max_value,
        max_inclusive=max_inclusive,
    )


def get_type_filters_for_finite_float_with_upper_bound(
    max_value: PrimitiveNumber | None = None,
    max_inclusive: bool = True,
) -> tuple[
    NumericArgTypeProvider | TypeFilter,
    *tuple[NumericArgTypeProvider | GenericTypeFilter, ...],
]:
    """
    Returns a tuple of type filters for finite floating-points within an upper bound range.

    Args:
        max_value: The maximum value (inclusive or exclusive).
        max_inclusive: Whether the maximum value is inclusive.

    Returns:
        A tuple containing the type filters for finite floating-points within the specified range.
    """

    return get_type_filters_for_finite_float_with_range(
        min_value=None,
        max_value=max_value,
        max_inclusive=max_inclusive,
    )
