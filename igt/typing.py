import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

type IntArray = NDArray[np.int64]
type FloatArray = NDArray[np.float64]
type IntegerArray = NDArray[np.integer[Any]]
type FloatingArray = NDArray[np.floating[Any]]
type ParameterBounds = Sequence[tuple[float, float]]


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

    file_path: str | Path
    mode: str = "a"
    encoding: str | None = "utf-8"
    delay: bool = False
    errors: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.file_path, (str, Path)):
            raise TypeError(
                f"'file_path' must be a str or Path, got {type(self.file_path).__name__}"
            )
        if not str(self.file_path).strip():
            raise ValueError("'file_path' cannot be an empty string or whitespace")

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
        file_path = Path(self.file_path)
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
