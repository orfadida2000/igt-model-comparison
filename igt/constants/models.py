"""Stable constants for model definitions and parameter initialization."""

from typing import Final

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
