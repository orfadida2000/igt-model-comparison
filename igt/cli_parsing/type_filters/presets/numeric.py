"""Preset numeric validators and argparse-compatible type filters.

This module defines reusable integer, floating-point, and general-number validation
functions together with `NumericArgTypeProvider`, whose members register common
numeric parsing chains for declarative CLI arguments.
"""

import argparse
from math import isfinite, isnan

from igt.cli_parsing.type_filters.core.definitions import (
    TypeFilterChainDefinition,
)
from igt.cli_parsing.type_filters.core.registry import TypeFilterRegistry
from igt.typing import PrimitiveNumber


def validate_int(value: str | int, label: str, err_class: type[BaseException]) -> int:
    """Parse and validate an integer.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated int.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    if isinstance(value, int):
        if isinstance(value, bool):
            raise err_class(
                f"Invalid {label.lower()}: expected a string or integer, got {type(value).__name__}"
            )

        n = value
    else:
        if not isinstance(value, str):
            raise err_class(
                f"Invalid {label.lower()}: expected a string or integer, got {type(value).__name__}"
            )
        try:
            n = int(value)
        except ValueError as exc:
            raise err_class(f"Invalid {label.lower()}: {value}") from exc

    return n


def int_type(value: str) -> int:
    """Parse an integer for argparse.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated int value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_int(value, label="Integer", err_class=argparse.ArgumentTypeError)


def validate_non_negative_int(value: str | int, label: str, err_class: type[BaseException]) -> int:
    """Parse and validate a non-negative integer.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated non negative int.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    n = validate_int(value, label=label, err_class=err_class)

    if n < 0:
        raise err_class(f"{label} must be greater than or equal to zero: {value}")

    return n


def non_negative_int_type(value: str) -> int:
    """Parse and validate a non-negative integer for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated non negative int value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_non_negative_int(
        value, label="Non-negative integer", err_class=argparse.ArgumentTypeError
    )


def validate_positive_int(value: str | int, label: str, err_class: type[BaseException]) -> int:
    """Parse and validate a positive integer.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated positive int.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    n = validate_non_negative_int(value, label=label, err_class=err_class)

    if n == 0:
        raise err_class(f"{label} must be greater than zero, got {value}")

    return n


def positive_int_type(value: str) -> int:
    """Parse and validate a positive integer for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated positive int value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_positive_int(
        value, label="Positive integer", err_class=argparse.ArgumentTypeError
    )


def validate_non_positive_int(value: str | int, label: str, err_class: type[BaseException]) -> int:
    """Parse and validate a non-positive integer.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated non positive int.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    n = validate_int(value, label=label, err_class=err_class)

    if n > 0:
        raise err_class(f"{label} must be less than or equal to zero: {value}")

    return n


def non_positive_int_type(value: str) -> int:
    """Parse and validate a non-positive integer for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated non positive int value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_non_positive_int(
        value, label="Non-positive integer", err_class=argparse.ArgumentTypeError
    )


def validate_negative_int(value: str | int, label: str, err_class: type[BaseException]) -> int:
    """Parse and validate a negative integer.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated negative int.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    n = validate_non_positive_int(value, label=label, err_class=err_class)

    if n == 0:
        raise err_class(f"{label} must be less than zero, got {value}")

    return n


def negative_int_type(value: str) -> int:
    """Parse and validate a negative integer for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated negative int value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_negative_int(
        value, label="Negative integer", err_class=argparse.ArgumentTypeError
    )


def validate_float(
    value: str | PrimitiveNumber, label: str, err_class: type[BaseException]
) -> float:
    """Parse a floating-point value while rejecting booleans and NaN.

    Args:
        value: Raw or already numeric value to parse.
        label: Human-readable value name used in error messages.
        err_class: Exception class used for validation failures.

    Returns:
        The parsed floating-point value.

    Raises:
        err_class: If the value cannot be converted or is NaN.
    """
    if not isinstance(value, (str, int, float)):
        raise err_class(
            f"Invalid {label.lower()}: expected a string or number, got {type(value).__name__}"
        )

    if isinstance(value, bool):
        raise err_class(f"Invalid {label.lower()}: expected a string or number, got bool")

    try:
        parsed_value = float(value)
    except (ValueError, OverflowError) as exc:
        raise err_class(f"Invalid {label.lower()}: {value}") from exc

    if isnan(parsed_value):
        raise err_class(f"{label} must not be NaN.")

    return parsed_value


def float_type(value: str) -> float:
    """Parse and validate a float for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated float value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_float(value, label="Floating-point value", err_class=argparse.ArgumentTypeError)


def validate_finite_float(
    value: str | PrimitiveNumber, label: str, err_class: type[BaseException]
) -> float:
    """Parse and validate a finite floating-point value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated finite float.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    parsed_value = validate_float(value, label=label, err_class=err_class)

    if not isfinite(parsed_value):
        raise err_class(f"{label} must be finite, got {value}")

    return parsed_value


def finite_float_type(value: str) -> float:
    """Parse and validate a finite float for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated finite float value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_finite_float(
        value, label="Finite floating-point value", err_class=argparse.ArgumentTypeError
    )


def validate_non_negative_float(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> float:
    """Parse and validate a non-negative floating-point value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated non negative float.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    parsed_value = validate_float(value, label=label, err_class=err_class)

    if parsed_value < 0.0:
        raise err_class(f"{label} must be greater than or equal to zero, got {value}")

    return parsed_value


def non_negative_float_type(value: str) -> float:
    """Parse and validate a non-negative floating-point value for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated non negative float value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_non_negative_float(
        value, label="Non-negative floating-point value", err_class=argparse.ArgumentTypeError
    )


def validate_positive_float(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> float:
    """Parse and validate a positive floating-point value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated positive float.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    parsed_value = validate_non_negative_float(value, label=label, err_class=err_class)

    if parsed_value == 0.0:
        raise err_class(f"{label} must be greater than zero, got {value}")

    return parsed_value


def positive_float_type(value: str) -> float:
    """Parse and validate a positive floating-point value for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated positive float value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_positive_float(
        value, label="Positive floating-point value", err_class=argparse.ArgumentTypeError
    )


def validate_non_positive_float(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> float:
    """Parse and validate a non-positive floating-point value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated non positive float.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    parsed_value = validate_float(value, label=label, err_class=err_class)

    if parsed_value > 0.0:
        raise err_class(f"{label} must be less than or equal to zero, got {value}")

    return parsed_value


def non_positive_float_type(value: str) -> float:
    """Parse and validate a non-positive floating-point value for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated non positive float value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_non_positive_float(
        value, label="Non-positive floating-point value", err_class=argparse.ArgumentTypeError
    )


def validate_negative_float(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> float:
    """Parse and validate a negative floating-point value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated negative float.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    parsed_value = validate_non_positive_float(value, label=label, err_class=err_class)

    if parsed_value == 0.0:
        raise err_class(f"{label} must be less than zero, got {value}")

    return parsed_value


def negative_float_type(value: str) -> float:
    """Parse and validate a negative floating-point value for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated negative float value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_negative_float(
        value, label="Negative floating-point value", err_class=argparse.ArgumentTypeError
    )


def validate_non_negative_finite_float(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> float:
    """Parse and validate a finite non-negative floating-point value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated non negative finite float.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    parsed_value = validate_finite_float(value, label=label, err_class=err_class)

    if parsed_value < 0.0:
        raise err_class(f"{label} must be greater than or equal to zero, got {value}")

    return parsed_value


def non_negative_finite_float_type(value: str) -> float:
    """Parse and validate a finite non-negative floating-point value for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated non negative finite float value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_non_negative_finite_float(
        value,
        label="Non-negative finite floating-point value",
        err_class=argparse.ArgumentTypeError,
    )


def validate_positive_finite_float(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> float:
    """Parse and validate a finite positive floating-point value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated positive finite float.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    parsed_value = validate_non_negative_finite_float(value, label=label, err_class=err_class)

    if parsed_value == 0.0:
        raise err_class(f"{label} must be greater than zero, got {value}")

    return parsed_value


def positive_finite_float_type(value: str) -> float:
    """Parse and validate a finite positive floating-point value for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated positive finite float value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_positive_finite_float(
        value, label="Positive finite floating-point value", err_class=argparse.ArgumentTypeError
    )


def validate_non_positive_finite_float(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> float:
    """Parse and validate a finite non-positive floating-point value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated non positive finite float.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    parsed_value = validate_finite_float(value, label=label, err_class=err_class)

    if parsed_value > 0.0:
        raise err_class(f"{label} must be less than or equal to zero, got {value}")

    return parsed_value


def non_positive_finite_float_type(value: str) -> float:
    """Parse and validate a finite non-positive floating-point value for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated non positive finite float value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_non_positive_finite_float(
        value,
        label="Non-positive finite floating-point value",
        err_class=argparse.ArgumentTypeError,
    )


def validate_negative_finite_float(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> float:
    """Parse and validate a finite negative floating-point value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated negative finite float.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    parsed_value = validate_non_positive_finite_float(value, label=label, err_class=err_class)

    if parsed_value == 0.0:
        raise err_class(f"{label} must be less than zero, got {value}")

    return parsed_value


def negative_finite_float_type(value: str) -> float:
    """Parse and validate a finite negative floating-point value for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated negative finite float value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_negative_finite_float(
        value, label="Negative finite floating-point value", err_class=argparse.ArgumentTypeError
    )


def validate_number(
    value: str | PrimitiveNumber, label: str, err_class: type[BaseException]
) -> PrimitiveNumber:
    """Parse a value as an integer when possible, otherwise as a float.

    Args:
        value: Raw or already numeric value to parse.
        label: Human-readable value name used in error messages.
        err_class: Exception class used for validation failures.

    Returns:
        The parsed integer or floating-point value.

    Raises:
        err_class: If neither integer nor floating-point parsing succeeds.
    """
    clean_error_messages: list[str] = []

    if isinstance(value, (str, int)):
        try:
            return validate_int(value, label=label, err_class=err_class)
        except err_class as e:
            err_msg = str(e)
            clean_err_msg = err_msg.split(":", 1)[-1].strip() if err_msg else err_msg
            clean_error_messages.append(clean_err_msg)

    try:
        return validate_float(value, label=label, err_class=err_class)
    except err_class as e:
        if not clean_error_messages:
            raise

        err_msg = str(e)
        clean_err_msg = err_msg.split(":", 1)[-1].strip() if err_msg else err_msg
        clean_error_messages.append(clean_err_msg)

        error_messages = "; ".join(str(e) for err_msg in clean_error_messages if err_msg)

        raise err_class(f"Invalid {label.lower()}: {error_messages}") from None


def number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a number for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated number value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_number(value, label="Number value", err_class=argparse.ArgumentTypeError)


def validate_finite_number(
    value: str | PrimitiveNumber, label: str, err_class: type[BaseException]
) -> PrimitiveNumber:
    """Parse and validate a finite number.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated finite number.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    number = validate_number(value, label=label, err_class=err_class)

    if isinstance(number, int):
        return number

    if not isfinite(number):
        raise err_class(f"{label} must be finite, got {value}")

    return number


def finite_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a finite number for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated finite number value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_finite_number(
        value, label="Finite number value", err_class=argparse.ArgumentTypeError
    )


def validate_non_negative_number(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> PrimitiveNumber:
    """Parse and validate a non-negative number value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated non negative number.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    number = validate_number(value, label=label, err_class=err_class)

    if number < 0.0:
        raise err_class(f"{label} must be greater than or equal to zero, got {value}")

    return number


def non_negative_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a non-negative number value for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated non negative number value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_non_negative_number(
        value, label="Non-negative number value", err_class=argparse.ArgumentTypeError
    )


def validate_positive_number(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> PrimitiveNumber:
    """Parse and validate a positive number value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated positive number.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    number = validate_non_negative_number(value, label=label, err_class=err_class)

    if number == 0.0:
        raise err_class(f"{label} must be greater than zero, got {value}")

    return number


def positive_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a positive number value for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated positive number value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_positive_number(
        value, label="Positive number value", err_class=argparse.ArgumentTypeError
    )


def validate_non_positive_number(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> PrimitiveNumber:
    """Parse and validate a non-positive number value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated non positive number.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    number = validate_number(value, label=label, err_class=err_class)

    if number > 0.0:
        raise err_class(f"{label} must be less than or equal to zero, got {value}")

    return number


def non_positive_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a non-positive number value for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated non positive number value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_non_positive_number(
        value, label="Non-positive number value", err_class=argparse.ArgumentTypeError
    )


def validate_negative_number(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> PrimitiveNumber:
    """Parse and validate a negative number value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated negative number.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    number = validate_non_positive_number(value, label=label, err_class=err_class)

    if number == 0.0:
        raise err_class(f"{label} must be less than zero, got {value}")

    return number


def negative_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a negative number value for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated negative number value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_negative_number(
        value, label="Negative number value", err_class=argparse.ArgumentTypeError
    )


def validate_non_negative_finite_number(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> PrimitiveNumber:
    """Parse and validate a finite non-negative number value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated non negative finite number.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    number = validate_finite_number(value, label=label, err_class=err_class)

    if number < 0.0:
        raise err_class(f"{label} must be greater than or equal to zero, got {value}")

    return number


def non_negative_finite_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a finite non-negative number value for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated non negative finite number value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_non_negative_finite_number(
        value,
        label="Non-negative finite number value",
        err_class=argparse.ArgumentTypeError,
    )


def validate_positive_finite_number(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> PrimitiveNumber:
    """Parse and validate a finite positive number value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated positive finite number.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    number = validate_non_negative_finite_number(value, label=label, err_class=err_class)

    if number == 0.0:
        raise err_class(f"{label} must be greater than zero, got {value}")

    return number


def positive_finite_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a finite positive number value for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated positive finite number value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_positive_finite_float(
        value, label="Positive finite number value", err_class=argparse.ArgumentTypeError
    )


def validate_non_positive_finite_number(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> PrimitiveNumber:
    """Parse and validate a finite non-positive number value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated non positive finite number.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    number = validate_finite_number(value, label=label, err_class=err_class)

    if number > 0.0:
        raise err_class(f"{label} must be less than or equal to zero, got {value}")

    return number


def non_positive_finite_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a finite non-positive number value for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated non positive finite number value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_non_positive_finite_number(
        value,
        label="Non-positive finite number value",
        err_class=argparse.ArgumentTypeError,
    )


def validate_negative_finite_number(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> PrimitiveNumber:
    """Parse and validate a finite negative number value.

    Args:
        value: Raw string or already-parsed value to validate.
        label: Human-readable label used in validation error messages.
        err_class: Exception class used when validation fails.

    Returns:
        The parsed and validated negative finite number.

    Raises:
        BaseException: The exception class supplied by `err_class` if validation fails.
    """

    number = validate_non_positive_finite_number(value, label=label, err_class=err_class)

    if number == 0.0:
        raise err_class(f"{label} must be less than zero, got {value}")

    return number


def negative_finite_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a finite negative number value for the argument parser.

    Args:
        value: Raw command-line string supplied by argparse.

    Returns:
        The parsed and validated negative finite number value.

    Raises:
        argparse.ArgumentTypeError: If the command-line value does not satisfy the required constraints.
    """

    return validate_negative_finite_number(
        value, label="Negative finite number value", err_class=argparse.ArgumentTypeError
    )


class NumericArgTypeProvider(TypeFilterChainDefinition):
    """Named numeric type-filter chains registered for command-line parsing.

    Each enum member resolves through
    [`TypeFilterRegistry`][igt.cli_parsing.type_filters.core.registry.TypeFilterRegistry]
    to an argparse-compatible parser for an integer, floating-point value, or
    number with the member's sign and finiteness constraints.
    """

    NUMBER = (number_type,)
    NON_NEGATIVE_NUMBER = (non_negative_number_type,)
    POSITIVE_NUMBER = (positive_number_type,)
    NON_POSITIVE_NUMBER = (non_positive_number_type,)
    NEGATIVE_NUMBER = (negative_number_type,)
    # break
    FINITE_NUMBER = (finite_number_type,)
    NON_NEGATIVE_FINITE_NUMBER = (non_negative_finite_number_type,)
    POSITIVE_FINITE_NUMBER = (positive_finite_number_type,)
    NON_POSITIVE_FINITE_NUMBER = (non_positive_finite_number_type,)
    NEGATIVE_FINITE_NUMBER = (negative_finite_number_type,)
    # break
    INTEGER = (int_type,)
    NON_NEGATIVE_INTEGER = (non_negative_int_type,)
    POSITIVE_INTEGER = (positive_int_type,)
    NON_POSITIVE_INTEGER = (non_positive_int_type,)
    NEGATIVE_INTEGER = (negative_int_type,)
    # break
    FLOAT = (float_type,)
    NON_NEGATIVE_FLOAT = (non_negative_float_type,)
    POSITIVE_FLOAT = (positive_float_type,)
    NON_POSITIVE_FLOAT = (non_positive_float_type,)
    NEGATIVE_FLOAT = (negative_float_type,)
    # break
    FINITE_FLOAT = (finite_float_type,)
    NON_NEGATIVE_FINITE_FLOAT = (non_negative_finite_float_type,)
    POSITIVE_FINITE_FLOAT = (positive_finite_float_type,)
    NON_POSITIVE_FINITE_FLOAT = (non_positive_finite_float_type,)
    NEGATIVE_FINITE_FLOAT = (negative_finite_float_type,)


TypeFilterRegistry.register_provider(NumericArgTypeProvider)
