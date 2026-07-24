from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

type IntArray = NDArray[np.int64]
type FloatArray = NDArray[np.float64]
type IntegerArray = NDArray[np.integer[Any]]
type FloatingArray = NDArray[np.floating[Any]]
type ParameterBounds = Sequence[tuple[float, float]]
