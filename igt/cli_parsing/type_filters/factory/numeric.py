"""Factories for numeric command-line parsing and validation.

The module builds integer and floating-point filters with configurable sign,
finiteness, bound, and range constraints. Specialized presets reuse these factories
to expose consistent numeric argument semantics.
"""

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

    """Validate a numeric value against optional lower and upper bounds.

    Args:
        n: Numeric value to validate.
        min_value: Optional lower bound.
        max_value: Optional upper bound.
        min_inclusive: Whether the lower bound includes equality.
        max_inclusive: Whether the upper bound includes equality.

    Returns:
        The unchanged numeric value when it satisfies the requested bounds.

    Raises:
        ArgumentTypeError: If the value lies outside the requested range.
    """
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
    """Create a filter chain for a number constrained by lower and upper bounds.

    Args:
        min_value: Lower bound.
        min_inclusive: Whether equality at the lower bound is allowed.
        max_value: Upper bound.
        max_inclusive: Whether equality at the upper bound is allowed.

    Returns:
        A callable filter chain that parses a number and enforces the configured bounds.
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
    """Create a filter chain for a finite number constrained by lower and upper bounds.

    Args:
        min_value: Lower bound.
        min_inclusive: Whether equality at the lower bound is allowed.
        max_value: Upper bound.
        max_inclusive: Whether equality at the upper bound is allowed.

    Returns:
        A callable filter chain that parses a finite number and enforces the configured bounds.
    """

    def _filter_finite_number[Number: (int, float)](n: Number) -> Number:
        """Require a numeric value to be finite.

        Args:
            n: Numeric value to validate.

        Returns:
            The unchanged value when finite.

        Raises:
            ArgumentTypeError: If the value is infinite.
        """
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
    """Create a filter chain for a integer constrained by lower and upper bounds.

    Args:
        min_value: Lower bound.
        min_inclusive: Whether equality at the lower bound is allowed.
        max_value: Upper bound.
        max_inclusive: Whether equality at the upper bound is allowed.

    Returns:
        A callable filter chain that parses a integer and enforces the configured bounds.
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
    """Create a filter chain for a floating-point value constrained by lower and upper bounds.

    Args:
        min_value: Lower bound.
        min_inclusive: Whether equality at the lower bound is allowed.
        max_value: Upper bound.
        max_inclusive: Whether equality at the upper bound is allowed.

    Returns:
        A callable filter chain that parses a floating-point value and enforces the configured bounds.
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
    """Create a filter chain for a finite floating-point value constrained by lower and upper bounds.

    Args:
        min_value: Lower bound.
        min_inclusive: Whether equality at the lower bound is allowed.
        max_value: Upper bound.
        max_inclusive: Whether equality at the upper bound is allowed.

    Returns:
        A callable filter chain that parses a finite floating-point value and enforces the configured bounds.
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
    """Create a filter chain for a number constrained by a lower bound.

    Args:
        min_value: Lower bound.
        min_inclusive: Whether equality at the lower bound is allowed.

    Returns:
        A callable filter chain that parses a number and enforces the configured bounds.
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
    """Create a filter chain for a finite number constrained by a lower bound.

    Args:
        min_value: Lower bound.
        min_inclusive: Whether equality at the lower bound is allowed.

    Returns:
        A callable filter chain that parses a finite number and enforces the configured bounds.
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
    """Create a filter chain for a integer constrained by a lower bound.

    Args:
        min_value: Lower bound.
        min_inclusive: Whether equality at the lower bound is allowed.

    Returns:
        A callable filter chain that parses a integer and enforces the configured bounds.
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
    """Create a filter chain for a floating-point value constrained by a lower bound.

    Args:
        min_value: Lower bound.
        min_inclusive: Whether equality at the lower bound is allowed.

    Returns:
        A callable filter chain that parses a floating-point value and enforces the configured bounds.
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
    """Create a filter chain for a finite floating-point value constrained by a lower bound.

    Args:
        min_value: Lower bound.
        min_inclusive: Whether equality at the lower bound is allowed.

    Returns:
        A callable filter chain that parses a finite floating-point value and enforces the configured bounds.
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
    """Create a filter chain for a number constrained by an upper bound.

    Args:
        max_value: Upper bound.
        max_inclusive: Whether equality at the upper bound is allowed.

    Returns:
        A callable filter chain that parses a number and enforces the configured bounds.
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
    """Create a filter chain for a finite number constrained by an upper bound.

    Args:
        max_value: Upper bound.
        max_inclusive: Whether equality at the upper bound is allowed.

    Returns:
        A callable filter chain that parses a finite number and enforces the configured bounds.
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
    """Create a filter chain for a integer constrained by an upper bound.

    Args:
        max_value: Upper bound.
        max_inclusive: Whether equality at the upper bound is allowed.

    Returns:
        A callable filter chain that parses a integer and enforces the configured bounds.
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
    """Create a filter chain for a floating-point value constrained by an upper bound.

    Args:
        max_value: Upper bound.
        max_inclusive: Whether equality at the upper bound is allowed.

    Returns:
        A callable filter chain that parses a floating-point value and enforces the configured bounds.
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
    """Create a filter chain for a finite floating-point value constrained by an upper bound.

    Args:
        max_value: Upper bound.
        max_inclusive: Whether equality at the upper bound is allowed.

    Returns:
        A callable filter chain that parses a finite floating-point value and enforces the configured bounds.
    """

    return get_type_filters_for_finite_float_with_range(
        min_value=None,
        max_value=max_value,
        max_inclusive=max_inclusive,
    )
