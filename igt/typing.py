"""Shared type aliases, enums, and logging configuration structures.

The module centralizes NumPy array shape aliases, path and parameter-bound types,
line-ending and output-stream enums, and immutable handler configurations used by the
logging and execution infrastructure.
"""

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
    """Supported line-ending sequences for project-owned text output.

    Attributes:
        LF: Unix-style line-feed sequence.
        CRLF: Windows-style carriage-return plus line-feed sequence.
    """

    LF = "\n"
    CRLF = "\r\n"


type ParameterBound = tuple[float, float]
type ParameterBounds = Sequence[ParameterBound]
type NamedParameterBounds = Mapping[str, ParameterBound]
type ModelParameterBounds = Mapping[str, NamedParameterBounds]


# type StrPathLike = str | Path | PathLike[str]


class StandardOutput(Enum):
    """Standard process streams accepted by terminal logging configuration.

    Attributes:
        STDOUT: Standard output stream.
        STDERR: Standard error stream.
    """

    STDOUT = "stdout"
    STDERR = "stderr"


@dataclass(kw_only=True, frozen=True)
class BaseLogHandlerConfig:
    """Base configuration shared by project logging handlers.

    Subclasses implement [`_create_handler`][igt.typing.BaseLogHandlerConfig._create_handler]
    to construct a concrete handler, while [`create_handler`][igt.typing.BaseLogHandlerConfig.create_handler]
    applies the optional level and formatter consistently.

    Attributes:
        level: Optional logging threshold applied to the created handler.
        log_format: Optional formatter template.
        datetime_format: Optional datetime format used by the formatter.
    """

    level: int | None = None
    log_format: str | None = None
    datetime_format: str | None = None

    def __post_init__(self) -> None:
        """Validate common logging-handler configuration fields.

        Raises:
            TypeError: If the abstract base configuration is instantiated
                directly or a configured field has an invalid type.
        """
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
        """Create the formatter configured for a log handler.

        Returns:
            Formatter using the configured message and datetime format strings.
        """
        return logging.Formatter(self.log_format, datefmt=self.datetime_format)

    def _create_handler(self) -> logging.Handler:
        """Create the concrete logging handler for this configuration.

        Subclasses must override this factory method; common level and formatter setup is
        performed later by [`create_handler`][igt.typing.BaseLogHandlerConfig.create_handler].

        Returns:
            Newly created logging handler.

        Raises:
            NotImplementedError: Always, when the base implementation is called directly.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def create_handler(self) -> logging.Handler:
        """Create and fully configure a logging handler.

        The subclass-specific handler is created first, then the optional handler level and
        the formatter produced by `_create_formatter` are applied consistently.

        Returns:
            Configured logging handler ready to attach to a logger.
        """
        handler = self._create_handler()

        if self.level is not None:
            handler.setLevel(self.level)

        handler.setFormatter(self._create_formatter())

        return handler


@dataclass(kw_only=True, frozen=True)
class NullLogHandlerConfig(BaseLogHandlerConfig):
    """Configuration for a reusable null logging handler.

    The configuration returns the shared `logging.NullHandler` instance and is
    useful when a logger should intentionally discard records.
    """

    _null_handler: ClassVar[logging.NullHandler] = logging.NullHandler()

    def _create_handler(self) -> logging.Handler:
        """Return the shared null handler used to discard log records.

        Returns:
            Reusable `logging.NullHandler` instance owned by the configuration class.
        """
        return self._null_handler


@dataclass(kw_only=True, frozen=True)
class TerminalLogHandlerConfig(BaseLogHandlerConfig):
    """Configuration for a stream handler targeting standard output or error.

    Attributes:
        stream: Standard stream to which log records are written.
    """

    stream: StandardOutput = StandardOutput.STDERR

    def __post_init__(self) -> None:
        """Validate terminal-handler configuration.

        Raises:
            TypeError: If the output stream or inherited configuration fields
                are invalid.
        """
        super().__post_init__()

        if not isinstance(self.stream, StandardOutput):
            raise TypeError(
                f"'stream' must be an instance of StandardOutput, got {type(self.stream).__name__}"
            )

    def _create_handler(self) -> logging.Handler:
        """Create a stream handler for the configured standard process stream.

        Returns:
            Stream handler targeting `sys.stdout` or `sys.stderr` according to `stream`.

        Raises:
            ValueError: If `stream` contains an unsupported enum value despite prior
                configuration validation.
        """
        if self.stream == StandardOutput.STDOUT:
            return logging.StreamHandler(sys.stdout)
        elif self.stream == StandardOutput.STDERR:
            return logging.StreamHandler(sys.stderr)
        else:
            raise ValueError(f"Unsupported stream: {self.stream}")


@dataclass(kw_only=True, frozen=True)
class FileLogHandlerConfig(BaseLogHandlerConfig):
    """Configuration for a file-backed logging handler.

    Attributes:
        file_path: Destination log-file path.
        mode: File opening mode passed to the logging handler.
        encoding: Text encoding used for the log file.
        delay: Whether file creation is deferred until the first emitted record.
        errors: Text encoding error-handling policy.
    """

    file_path: StrPathLike
    mode: str = "a"
    encoding: str | None = "utf-8"
    delay: bool = False
    errors: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize file-handler configuration.

        The path is converted to `Path` after rejecting empty strings. File mode, encoding,
        delay, and encoding-error settings are type-checked after the common handler
        configuration fields are validated.

        Raises:
            TypeError: If the path or any file-handler option has an unsupported type.
            ValueError: If the path string is empty or cannot be converted to a `Path`.
        """
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
        """Create the configured file-backed logging handler.

        The destination parent directory is created automatically. An already-existing
        filesystem object is accepted only when it is a regular file.

        Returns:
            Configured `logging.FileHandler` for the normalized destination path.

        Raises:
            ValueError: If the configured destination already exists but is not a file.
        """
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
    """Marker subclass for type errors raised explicitly by project validation code.

    The dedicated subclass makes it possible for callers to distinguish deliberate
    argument-type validation failures from unrelated `TypeError` exceptions.
    """

    pass
