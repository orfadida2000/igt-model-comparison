"""Named command-line type-filter presets used by project entry points.

Numeric, path, and string presets expose stable enum members that resolve through the
global type-filter registry, allowing argument specifications to refer to validation
behavior declaratively.
"""

from .numeric import NumericArgTypeProvider
from .path import PathArgTypeProvider
from .string import StringArgTypeProvider

__all__ = [
    "NumericArgTypeProvider",
    "PathArgTypeProvider",
    "StringArgTypeProvider",
]
