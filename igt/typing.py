import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray

# For strict homogeneous tuples (all elements same type)
type NonEmptyUniformTuple[T] = tuple[T, *tuple[T, ...]]

# For mixed tuples (first type vs rest type)
type NonEmptyMixedTuple[T, S] = tuple[T, *tuple[S, ...]]

type PrimitiveNumber = int | float

type IntArray = NDArray[np.int64]
type FloatArray = NDArray[np.float64]
type IntegerArray = NDArray[np.integer[Any]]
type FloatingArray = NDArray[np.floating[Any]]

type Int1DArray = np.ndarray[
    tuple[int],
    np.dtype[np.int64],
]
type Float1DArray = np.ndarray[
    tuple[int],
    np.dtype[np.float64],
]
type Integer1DArray = np.ndarray[
    tuple[int],
    np.dtype[np.integer[Any]],
]
type Floating1DArray = np.ndarray[
    tuple[int],
    np.dtype[np.floating[Any]],
]

type Int2DArray = np.ndarray[
    tuple[int, int],
    np.dtype[np.int64],
]
type Float2DArray = np.ndarray[
    tuple[int, int],
    np.dtype[np.float64],
]
type Integer2DArray = np.ndarray[
    tuple[int, int],
    np.dtype[np.integer[Any]],
]
type Floating2DArray = np.ndarray[
    tuple[int, int],
    np.dtype[np.floating[Any]],
]


type StrPathLike = str | Path | PathLike[str]


class LineEnding(Enum):
    """Enum for line endings."""

    LF = "\n"
    CRLF = "\r\n"


type ParameterBound = tuple[float, float]
type ParameterBounds = Sequence[ParameterBound]
type NamedParameterBounds = Mapping[str, ParameterBound]
type ModelParameterBounds = Mapping[str, NamedParameterBounds]


# type StrPathLike = str | Path | PathLike[str]


class StandardOutput(Enum):
    """Enum for standard output streams."""

    STDOUT = "stdout"
    STDERR = "stderr"


@dataclass(kw_only=True, frozen=True)
class BaseLogHandlerConfig:
    """Configuration for a base log handler."""

    level: int | None = None
    log_format: str | None = None
    datetime_format: str | None = None

    def __post_init__(self) -> None:
        if type(self) is BaseLogHandlerConfig:
            raise TypeError(
                "BaseLogHandlerConfig is an abstract class and cannot be instantiated directly."
            )

        if self.level is not None and not isinstance(self.level, int):
            raise TypeError(f"'level' must be an int or None, got {type(self.level).__name__}")

        if self.log_format is not None and not isinstance(self.log_format, str):
            raise TypeError(
                f"'log_format' must be a str or None, got {type(self.log_format).__name__}"
            )

        if self.datetime_format is not None and not isinstance(self.datetime_format, str):
            raise TypeError(
                f"'datetime_format' must be a str or None, got {type(self.datetime_format).__name__}"
            )

    def _create_formatter(self) -> logging.Formatter:
        """Create a logging formatter based on the configuration."""
        return logging.Formatter(self.log_format, datefmt=self.datetime_format)

    def _create_handler(self) -> logging.Handler:
        """Create a logging handler based on the configuration."""
        raise NotImplementedError("Subclasses must implement this method.")

    def create_handler(self) -> logging.Handler:
        """Create a configured logging handler."""
        handler = self._create_handler()

        if self.level is not None:
            handler.setLevel(self.level)

        handler.setFormatter(self._create_formatter())

        return handler


@dataclass(kw_only=True, frozen=True)
class NullLogHandlerConfig(BaseLogHandlerConfig):
    """Configuration for a null log handler."""

    _null_handler: ClassVar[logging.NullHandler] = logging.NullHandler()

    def _create_handler(self) -> logging.Handler:
        """Create a null log handler."""
        return self._null_handler


@dataclass(kw_only=True, frozen=True)
class TerminalLogHandlerConfig(BaseLogHandlerConfig):
    """Configuration for a terminal log handler."""

    stream: StandardOutput = StandardOutput.STDERR

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.stream, StandardOutput):
            raise TypeError(
                f"'stream' must be an instance of StandardOutput, got {type(self.stream).__name__}"
            )

    def _create_handler(self) -> logging.Handler:
        """Create a terminal log handler based on the configuration."""
        if self.stream == StandardOutput.STDOUT:
            return logging.StreamHandler(sys.stdout)
        elif self.stream == StandardOutput.STDERR:
            return logging.StreamHandler(sys.stderr)
        else:
            raise ValueError(f"Unsupported stream: {self.stream}")


@dataclass(kw_only=True, frozen=True)
class FileLogHandlerConfig(BaseLogHandlerConfig):
    """Configuration for a file log handler."""

    file_path: StrPathLike
    mode: str = "a"
    encoding: str | None = "utf-8"
    delay: bool = False
    errors: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        file_path = self.file_path
        if not isinstance(file_path, (str, Path, PathLike)):
            raise TypeError(
                f"'file_path' must be a str or pathlib.Path or os.PathLike, got {type(file_path).__name__}"
            )
        if isinstance(file_path, str):
            file_path = file_path.strip()

            if not file_path:
                raise ValueError("'file_path' must not be empty or whitespace only.")

        try:
            file_path = Path(file_path)
        except Exception as e:
            raise ValueError(
                f"The string path-like object 'file_path' failed to be converted to a pathlib.Path: {e}"
            ) from e

        object.__setattr__(self, "file_path", file_path)

        if not isinstance(self.mode, str):
            raise TypeError(f"'mode' must be a str, got {type(self.mode).__name__}")

        if self.encoding is not None and not isinstance(self.encoding, str):
            raise TypeError(f"'encoding' must be a str or None, got {type(self.encoding).__name__}")

        if not isinstance(self.delay, bool):
            raise TypeError(f"'delay' must be a bool, got {type(self.delay).__name__}")

        if self.errors is not None and not isinstance(self.errors, str):
            raise TypeError(f"'errors' must be a str or None, got {type(self.errors).__name__}")

    def _create_handler(self) -> logging.Handler:
        """Create a file log handler based on the configuration."""
        file_path = cast(Path, self.file_path)
        if file_path.exists() and not file_path.is_file():
            raise ValueError(f"The specified path '{file_path}' exists and is not a file.")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        return logging.FileHandler(
            self.file_path,
            mode=self.mode,
            encoding=self.encoding,
            delay=self.delay,
            errors=self.errors,
        )


class CustomTypeError(TypeError):
    """Custom TypeError class for easy identification of type errors raised explicitly."""

    pass
