from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

type IntArray = NDArray[np.int64]
type FloatArray = NDArray[np.float64]
type ParameterBounds = Sequence[tuple[float, float]]
