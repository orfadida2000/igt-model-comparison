"""Stable constants for model definitions and parameter initialization."""

from types import MappingProxyType
from typing import Final

from igt.typing import ModelParameterBounds, ParameterBounds

N_IGT_DECKS: Final[int] = 4
PAYOFF_SCALE: Final[float] = 100.0

OPEN_BOUND_EPSILON: Final[float] = 1e-6

MIN_LEARNING_RATE: Final[float] = 0.0
MAX_LEARNING_RATE: Final[float] = 1.0

MIN_OUTCOME_SENSITIVITY: Final[float] = OPEN_BOUND_EPSILON
MAX_OUTCOME_SENSITIVITY: Final[float] = 2.0

MIN_LOSS_AVERSION: Final[float] = OPEN_BOUND_EPSILON
MAX_LOSS_AVERSION: Final[float] = 10.0

MIN_RESPONSE_CONSISTENCY: Final[float] = 0.0
MAX_RESPONSE_CONSISTENCY: Final[float] = 5.0

MIN_INVERSE_TEMPERATURE: Final[float] = 0.0
DEFAULT_MAX_INVERSE_TEMPERATURE: Final[float] = 100.0

DEFAULT_Q_LEARNING_RATE_GRID_SIZE: Final[int] = 31
DEFAULT_Q_INVERSE_TEMPERATURE_GRID_STEP: Final[float] = 1.0

Q_LEARNING_MODEL_NAME: Final[str] = "q_learning"
PVL_DELTA_MODEL_NAME: Final[str] = "pvl_delta"

LEARNING_RATE_PARAMETER_NAME: Final[str] = "learning_rate"
INVERSE_TEMPERATURE_PARAMETER_NAME: Final[str] = "inverse_temperature"
OUTCOME_SENSITIVITY_PARAMETER_NAME: Final[str] = "outcome_sensitivity"
LOSS_AVERSION_PARAMETER_NAME: Final[str] = "loss_aversion"
RESPONSE_CONSISTENCY_PARAMETER_NAME: Final[str] = "response_consistency"

Q_LEARNING_PARAMETER_NAMES: Final[tuple[str, ...]] = (
    LEARNING_RATE_PARAMETER_NAME,
    INVERSE_TEMPERATURE_PARAMETER_NAME,
)

PVL_DELTA_PARAMETER_NAMES: Final[tuple[str, ...]] = (
    LEARNING_RATE_PARAMETER_NAME,
    OUTCOME_SENSITIVITY_PARAMETER_NAME,
    LOSS_AVERSION_PARAMETER_NAME,
    RESPONSE_CONSISTENCY_PARAMETER_NAME,
)

DEFAULT_Q_LEARNING_PARAMETER_BOUNDS: Final[ParameterBounds] = (
    (MIN_LEARNING_RATE, MAX_LEARNING_RATE),
    (
        MIN_INVERSE_TEMPERATURE,
        DEFAULT_MAX_INVERSE_TEMPERATURE,
    ),
)

PVL_DELTA_PARAMETER_BOUNDS: Final[ParameterBounds] = (
    (
        MIN_LEARNING_RATE + OPEN_BOUND_EPSILON,
        MAX_LEARNING_RATE - OPEN_BOUND_EPSILON,
    ),
    (
        MIN_OUTCOME_SENSITIVITY,
        MAX_OUTCOME_SENSITIVITY,
    ),
    (
        MIN_LOSS_AVERSION,
        MAX_LOSS_AVERSION,
    ),
    (
        MIN_RESPONSE_CONSISTENCY,
        MAX_RESPONSE_CONSISTENCY,
    ),
)

DEFAULT_MODEL_PARAMETER_BOUNDS: Final[ModelParameterBounds] = MappingProxyType(
    {
        Q_LEARNING_MODEL_NAME: MappingProxyType(
            dict(
                zip(
                    Q_LEARNING_PARAMETER_NAMES,
                    DEFAULT_Q_LEARNING_PARAMETER_BOUNDS,
                    strict=True,
                )
            )
        ),
        PVL_DELTA_MODEL_NAME: MappingProxyType(
            dict(
                zip(
                    PVL_DELTA_PARAMETER_NAMES,
                    PVL_DELTA_PARAMETER_BOUNDS,
                    strict=True,
                )
            )
        ),
    }
)
