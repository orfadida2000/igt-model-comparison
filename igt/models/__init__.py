"""Computational models used in the Iowa Gambling Task comparison.

The package exposes the shared `ComputationalModel` interface together with the
project's two fitted models: `QLearningModel` and `PVLDeltaModel`. Subject-level input
validation lives in `igt.models.typing`.
"""

from .base import ComputationalModel
from .pvl_delta import PVLDeltaModel
from .q_learning import QLearningModel

__all__ = [
    "ComputationalModel",
    "QLearningModel",
    "PVLDeltaModel",
]
