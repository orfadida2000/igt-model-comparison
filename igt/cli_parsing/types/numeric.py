import argparse
from collections.abc import Callable, Mapping
from enum import Enum
from math import isfinite, isnan
from types import MappingProxyType
from typing import Any, Final

from igt.typing import NonEmptyMixedTuple, PrimitiveNumber


def validate_int(value: str | int, label: str, err_class: type[BaseException]) -> int:
    """Parse and validate an integer."""

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
    """Parse and validate a integer for the argument parser."""

    return validate_int(value, label="Integer", err_class=argparse.ArgumentTypeError)


def validate_non_negative_int(value: str | int, label: str, err_class: type[BaseException]) -> int:
    """Parse and validate a non-negative integer."""

    n = validate_int(value, label=label, err_class=err_class)

    if n < 0:
        raise err_class(f"{label} must be greater than or equal to zero: {value}")

    return n


def non_negative_int_type(value: str) -> int:
    """Parse and validate a non-negative integer for the argument parser."""

    return validate_non_negative_int(
        value, label="Non-negative integer", err_class=argparse.ArgumentTypeError
    )


def validate_positive_int(value: str | int, label: str, err_class: type[BaseException]) -> int:
    """Parse and validate a positive integer."""

    n = validate_non_negative_int(value, label=label, err_class=err_class)

    if n == 0:
        raise err_class(f"{label} must be greater than zero, got {value}")

    return n


def positive_int_type(value: str) -> int:
    """Parse and validate a positive integer for the argument parser."""

    return validate_positive_int(
        value, label="Positive integer", err_class=argparse.ArgumentTypeError
    )


def validate_non_positive_int(value: str | int, label: str, err_class: type[BaseException]) -> int:
    """Parse and validate a non-positive integer."""

    n = validate_int(value, label=label, err_class=err_class)

    if n > 0:
        raise err_class(f"{label} must be less than or equal to zero: {value}")

    return n


def non_positive_int_type(value: str) -> int:
    """Parse and validate a non-positive integer for the argument parser."""

    return validate_non_positive_int(
        value, label="Non-positive integer", err_class=argparse.ArgumentTypeError
    )


def validate_negative_int(value: str | int, label: str, err_class: type[BaseException]) -> int:
    """Parse and validate a negative integer."""

    n = validate_non_positive_int(value, label=label, err_class=err_class)

    if n == 0:
        raise err_class(f"{label} must be less than zero, got {value}")

    return n


def negative_int_type(value: str) -> int:
    """Parse and validate a negative integer for the argument parser."""

    return validate_negative_int(
        value, label="Negative integer", err_class=argparse.ArgumentTypeError
    )


def validate_float(
    value: str | PrimitiveNumber, label: str, err_class: type[BaseException]
) -> float:
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
    """Parse and validate a float for the argument parser."""

    return validate_float(value, label="Floating-point value", err_class=argparse.ArgumentTypeError)


def validate_finite_float(
    value: str | PrimitiveNumber, label: str, err_class: type[BaseException]
) -> float:
    """Parse and validate a finite floating-point value."""

    parsed_value = validate_float(value, label=label, err_class=err_class)

    if not isfinite(parsed_value):
        raise err_class(f"{label} must be finite, got {value}")

    return parsed_value


def finite_float_type(value: str) -> float:
    """Parse and validate a finite float for the argument parser."""

    return validate_finite_float(
        value, label="Finite floating-point value", err_class=argparse.ArgumentTypeError
    )


def validate_non_negative_float(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> float:
    """Parse and validate a non-negative floating-point value."""

    parsed_value = validate_float(value, label=label, err_class=err_class)

    if parsed_value < 0.0:
        raise err_class(f"{label} must be greater than or equal to zero, got {value}")

    return parsed_value


def non_negative_float_type(value: str) -> float:
    """Parse and validate a non-negative floating-point value for the argument parser."""

    return validate_non_negative_float(
        value, label="Non-negative floating-point value", err_class=argparse.ArgumentTypeError
    )


def validate_positive_float(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> float:
    """Parse and validate a positive floating-point value."""

    parsed_value = validate_non_negative_float(value, label=label, err_class=err_class)

    if parsed_value == 0.0:
        raise err_class(f"{label} must be greater than zero, got {value}")

    return parsed_value


def positive_float_type(value: str) -> float:
    """Parse and validate a positive floating-point value for the argument parser."""

    return validate_positive_float(
        value, label="Positive floating-point value", err_class=argparse.ArgumentTypeError
    )


def validate_non_positive_float(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> float:
    """Parse and validate a non-positive floating-point value."""

    parsed_value = validate_float(value, label=label, err_class=err_class)

    if parsed_value > 0.0:
        raise err_class(f"{label} must be less than or equal to zero, got {value}")

    return parsed_value


def non_positive_float_type(value: str) -> float:
    """Parse and validate a non-positive floating-point value for the argument parser."""

    return validate_non_positive_float(
        value, label="Non-positive floating-point value", err_class=argparse.ArgumentTypeError
    )


def validate_negative_float(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> float:
    """Parse and validate a negative floating-point value."""

    parsed_value = validate_non_positive_float(value, label=label, err_class=err_class)

    if parsed_value == 0.0:
        raise err_class(f"{label} must be less than zero, got {value}")

    return parsed_value


def negative_float_type(value: str) -> float:
    """Parse and validate a negative floating-point value for the argument parser."""

    return validate_negative_float(
        value, label="Negative floating-point value", err_class=argparse.ArgumentTypeError
    )


def validate_non_negative_finite_float(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> float:
    """Parse and validate a finite non-negative floating-point value."""

    parsed_value = validate_finite_float(value, label=label, err_class=err_class)

    if parsed_value < 0.0:
        raise err_class(f"{label} must be greater than or equal to zero, got {value}")

    return parsed_value


def non_negative_finite_float_type(value: str) -> float:
    """Parse and validate a finite non-negative floating-point value for the argument parser."""

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
    """Parse and validate a finite positive floating-point value."""

    parsed_value = validate_non_negative_finite_float(value, label=label, err_class=err_class)

    if parsed_value == 0.0:
        raise err_class(f"{label} must be greater than zero, got {value}")

    return parsed_value


def positive_finite_float_type(value: str) -> float:
    """Parse and validate a finite positive floating-point value for the argument parser."""

    return validate_positive_finite_float(
        value, label="Positive finite floating-point value", err_class=argparse.ArgumentTypeError
    )


def validate_non_positive_finite_float(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> float:
    """Parse and validate a finite non-positive floating-point value."""

    parsed_value = validate_finite_float(value, label=label, err_class=err_class)

    if parsed_value > 0.0:
        raise err_class(f"{label} must be less than or equal to zero, got {value}")

    return parsed_value


def non_positive_finite_float_type(value: str) -> float:
    """Parse and validate a finite non-positive floating-point value for the argument parser."""

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
    """Parse and validate a finite negative floating-point value."""

    parsed_value = validate_non_positive_finite_float(value, label=label, err_class=err_class)

    if parsed_value == 0.0:
        raise err_class(f"{label} must be less than zero, got {value}")

    return parsed_value


def negative_finite_float_type(value: str) -> float:
    """Parse and validate a finite negative floating-point value for the argument parser."""

    return validate_negative_finite_float(
        value, label="Negative finite floating-point value", err_class=argparse.ArgumentTypeError
    )


def validate_number(
    value: str | PrimitiveNumber, label: str, err_class: type[BaseException]
) -> PrimitiveNumber:
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
    """Parse and validate a number for the argument parser."""

    return validate_number(value, label="Number value", err_class=argparse.ArgumentTypeError)


def validate_finite_number(
    value: str | PrimitiveNumber, label: str, err_class: type[BaseException]
) -> PrimitiveNumber:
    """Parse and validate a finite number."""

    number = validate_number(value, label=label, err_class=err_class)

    if isinstance(number, int):
        return number

    if not isfinite(number):
        raise err_class(f"{label} must be finite, got {value}")

    return number


def finite_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a finite number for the argument parser."""

    return validate_finite_number(
        value, label="Finite number value", err_class=argparse.ArgumentTypeError
    )


def validate_non_negative_number(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> PrimitiveNumber:
    """Parse and validate a non-negative number value."""

    number = validate_number(value, label=label, err_class=err_class)

    if number < 0.0:
        raise err_class(f"{label} must be greater than or equal to zero, got {value}")

    return number


def non_negative_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a non-negative number value for the argument parser."""

    return validate_non_negative_number(
        value, label="Non-negative number value", err_class=argparse.ArgumentTypeError
    )


def validate_positive_number(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> PrimitiveNumber:
    """Parse and validate a positive number value."""

    number = validate_non_negative_number(value, label=label, err_class=err_class)

    if number == 0.0:
        raise err_class(f"{label} must be greater than zero, got {value}")

    return number


def positive_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a positive number value for the argument parser."""

    return validate_positive_number(
        value, label="Positive number value", err_class=argparse.ArgumentTypeError
    )


def validate_non_positive_number(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> PrimitiveNumber:
    """Parse and validate a non-positive number value."""

    number = validate_number(value, label=label, err_class=err_class)

    if number > 0.0:
        raise err_class(f"{label} must be less than or equal to zero, got {value}")

    return number


def non_positive_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a non-positive number value for the argument parser."""

    return validate_non_positive_number(
        value, label="Non-positive number value", err_class=argparse.ArgumentTypeError
    )


def validate_negative_number(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> PrimitiveNumber:
    """Parse and validate a negative number value."""

    number = validate_non_positive_number(value, label=label, err_class=err_class)

    if number == 0.0:
        raise err_class(f"{label} must be less than zero, got {value}")

    return number


def negative_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a negative number value for the argument parser."""

    return validate_negative_number(
        value, label="Negative number value", err_class=argparse.ArgumentTypeError
    )


def validate_non_negative_finite_number(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> PrimitiveNumber:
    """Parse and validate a finite non-negative number value."""

    number = validate_finite_number(value, label=label, err_class=err_class)

    if number < 0.0:
        raise err_class(f"{label} must be greater than or equal to zero, got {value}")

    return number


def non_negative_finite_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a finite non-negative number value for the argument parser."""

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
    """Parse and validate a finite positive number value."""

    number = validate_non_negative_finite_number(value, label=label, err_class=err_class)

    if number == 0.0:
        raise err_class(f"{label} must be greater than zero, got {value}")

    return number


def positive_finite_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a finite positive number value for the argument parser."""

    return validate_positive_finite_float(
        value, label="Positive finite number value", err_class=argparse.ArgumentTypeError
    )


def validate_non_positive_finite_number(
    value: str | PrimitiveNumber,
    label: str,
    err_class: type[BaseException],
) -> PrimitiveNumber:
    """Parse and validate a finite non-positive number value."""

    number = validate_finite_number(value, label=label, err_class=err_class)

    if number > 0.0:
        raise err_class(f"{label} must be less than or equal to zero, got {value}")

    return number


def non_positive_finite_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a finite non-positive number value for the argument parser."""

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
    """Parse and validate a finite negative number value."""

    number = validate_non_positive_finite_number(value, label=label, err_class=err_class)

    if number == 0.0:
        raise err_class(f"{label} must be less than zero, got {value}")

    return number


def negative_finite_number_type(value: str) -> PrimitiveNumber:
    """Parse and validate a finite negative number value for the argument parser."""

    return validate_negative_finite_number(
        value, label="Negative finite number value", err_class=argparse.ArgumentTypeError
    )


class NumericArgType(Enum):
    """Enumeration of argument types for command-line argument parsing."""

    NUMBER = "number"
    NON_NEGATIVE_NUMBER = "non-negative number"
    POSITIVE_NUMBER = "positive number"
    NON_POSITIVE_NUMBER = "non-positive number"
    NEGATIVE_NUMBER = "negative number"

    FINITE_NUMBER = "finite number"
    NON_NEGATIVE_FINITE_NUMBER = "non-negative finite number"
    POSITIVE_FINITE_NUMBER = "positive finite number"
    NON_POSITIVE_FINITE_NUMBER = "non-positive finite number"
    NEGATIVE_FINITE_NUMBER = "negative finite number"

    INTEGER = "integer"
    NON_NEGATIVE_INTEGER = "non-negative integer"
    POSITIVE_INTEGER = "positive integer"
    NON_POSITIVE_INTEGER = "non-positive integer"
    NEGATIVE_INTEGER = "negative integer"

    FLOAT = "float"
    NON_NEGATIVE_FLOAT = "non-negative float"
    POSITIVE_FLOAT = "positive float"
    NON_POSITIVE_FLOAT = "non-positive float"
    NEGATIVE_FLOAT = "negative float"

    FINITE_FLOAT = "finite float"
    NON_NEGATIVE_FINITE_FLOAT = "non-negative finite float"
    POSITIVE_FINITE_FLOAT = "positive finite float"
    NON_POSITIVE_FINITE_FLOAT = "non-positive finite float"
    NEGATIVE_FINITE_FLOAT = "negative finite float"


ARG_TYPE_CALLABLE_MAP: Final[
    Mapping[
        NumericArgType,
        Callable[[str], Any] | NonEmptyMixedTuple[Callable[[str], Any], Callable[[Any], Any]],
    ]
] = MappingProxyType(
    {
        NumericArgType.NUMBER: number_type,
        NumericArgType.NON_NEGATIVE_NUMBER: non_negative_number_type,
        NumericArgType.POSITIVE_NUMBER: positive_number_type,
        NumericArgType.NON_POSITIVE_NUMBER: non_positive_number_type,
        NumericArgType.NEGATIVE_NUMBER: negative_number_type,
        # break
        NumericArgType.FINITE_NUMBER: finite_number_type,
        NumericArgType.NON_NEGATIVE_FINITE_NUMBER: non_negative_finite_number_type,
        NumericArgType.POSITIVE_FINITE_NUMBER: positive_finite_number_type,
        NumericArgType.NON_POSITIVE_FINITE_NUMBER: non_positive_finite_number_type,
        NumericArgType.NEGATIVE_FINITE_NUMBER: negative_finite_number_type,
        # break
        NumericArgType.INTEGER: int_type,
        NumericArgType.NON_NEGATIVE_INTEGER: non_negative_int_type,
        NumericArgType.POSITIVE_INTEGER: positive_int_type,
        NumericArgType.NON_POSITIVE_INTEGER: non_positive_int_type,
        NumericArgType.NEGATIVE_INTEGER: negative_int_type,
        # break
        NumericArgType.FLOAT: float_type,
        NumericArgType.NON_NEGATIVE_FLOAT: non_negative_float_type,
        NumericArgType.POSITIVE_FLOAT: positive_float_type,
        NumericArgType.NON_POSITIVE_FLOAT: non_positive_float_type,
        NumericArgType.NEGATIVE_FLOAT: negative_float_type,
        # break
        NumericArgType.FINITE_FLOAT: finite_float_type,
        NumericArgType.NON_NEGATIVE_FINITE_FLOAT: non_negative_finite_float_type,
        NumericArgType.POSITIVE_FINITE_FLOAT: positive_finite_float_type,
        NumericArgType.NON_POSITIVE_FINITE_FLOAT: non_positive_finite_float_type,
        NumericArgType.NEGATIVE_FINITE_FLOAT: negative_finite_float_type,
    }
)
