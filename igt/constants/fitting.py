"""Stable defaults and numerical tolerances for model fitting.

This module centralizes optimizer iteration limits, starting-point counts, and
floating-point tolerances used when assessing fitted parameters and optimizer
results.
"""

from typing import Final

DEFAULT_FIT_METHOD: Final[str] = "L-BFGS-B"
DEFAULT_MAX_ITERATIONS: Final[int] = 1_000

UNIFORM_CHOICE_NLL_ABSOLUTE_TOLERANCE: Final[float] = 1e-8
PARAMETER_BOUNDARY_ABSOLUTE_TOLERANCE_FACTOR: Final[float] = 1e-8
