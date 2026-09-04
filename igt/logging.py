"""Shared root-logger configuration for project entry points.

The helpers configure terminal, file, or null handlers from typed handler
configuration objects, capture the resulting application logging state, and provide a
matching cleanup operation for scripts and the main fitting workflow.
"""

import logging
from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import MappingProxyType

from igt.constants.config import (
    DATETIME_FORMAT,
    FILE_LOG_LEVEL,
    LOG_FORMAT,
    TERMINAL_LOG_LEVEL,
)
from igt.typing import (
    BaseLogHandlerConfig,
    FileLogHandlerConfig,
    NullLogHandlerConfig,
    StandardOutput,
    StrPathLike,
    TerminalLogHandlerConfig,
)
from igt.utils.io import normalize_path


@dataclass(slots=True, frozen=True)
class LoggingState:
    """Encapsulates the current state of the root logger and any additional loggers.

    Attributes:
        root_logger_level: The root logger's level when the instance was created.
        additional_logger_level_map: Mapping of additional logger names to their respective levels when the instance was created.
    """

    root_logger_level: int = field(init=False)
    additional_logger_level_map: Mapping[str, int] = field(init=False, default_factory=dict)
    additional_loggers: InitVar[Iterable[str | logging.Logger] | None] = None

    def __post_init__(self, additional_loggers: Iterable[str | logging.Logger] | None) -> None:
        additional_loggers = additional_loggers or []

        try:
            additional_loggers = tuple(additional_loggers)
        except TypeError as e:
            raise ValueError(
                "additional_loggers could not be converted to a tuple; ensure it is an iterable of logger names or Logger instances."
            ) from e

        additional_logger_level_map = {
            logger.name if isinstance(logger, logging.Logger) else logger: logger.level
            if isinstance(logger, logging.Logger)
            else logging.getLogger(logger).level
            for logger in additional_loggers
        }

        object.__setattr__(
            self, "additional_logger_level_map", MappingProxyType(additional_logger_level_map)
        )

        object.__setattr__(self, "root_logger_level", logging.getLogger().level)


def configure_root_logger(
    *,
    level: int | None = None,
    handler_configs: list[BaseLogHandlerConfig] | None = None,
) -> None:
    """Replace root-logger handlers with handlers created from project configurations.

    If no effective non-null handler configuration is supplied, a null handler is
    installed so library logging remains silent. Existing root handlers are removed
    and closed before the new handlers are attached.

    Args:
        level: Optional root-logger level to set before replacing handlers.
        handler_configs: Optional handler configurations used to construct the new
            root handlers.
    """

    handler_configs = handler_configs or []
    effective_configs = [
        handler_config
        for handler_config in handler_configs
        if isinstance(handler_config, BaseLogHandlerConfig)
        and not isinstance(handler_config, NullLogHandlerConfig)
    ]

    if not effective_configs:
        effective_configs = [NullLogHandlerConfig()]

    root_logger = logging.getLogger()

    if level is not None:
        root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    for handler_config in effective_configs:
        root_logger.addHandler(handler_config.create_handler())


def configure_application_logging(
    *,
    disabled: bool,
    root_logger_level: int | None,
    terminal_handler_level: int = TERMINAL_LOG_LEVEL,
    file_handler_level: int = FILE_LOG_LEVEL,
    log_format: str = LOG_FORMAT,
    datetime_format: str = DATETIME_FORMAT,
    terminal_stream: StandardOutput = StandardOutput.STDERR,
    log_file_path: StrPathLike | None = None,
    additional_logger_levels: Mapping[str | logging.Logger, int] | None = None,
) -> Path | None:
    """Configure terminal and optional file logging for a top-level workflow.

    When logging is disabled, the root logger is reset to a null handler. Otherwise
    a terminal handler is always configured and a file handler is added when a log
    path is supplied.

    Args:
        disabled: Whether application logging should be disabled.
        root_logger_level: Root-logger level used when logging is enabled.
        terminal_handler_level: Minimum level emitted by the terminal handler.
        file_handler_level: Minimum level emitted by the optional file handler.
        log_format: Format string used by configured handlers.
        datetime_format: Datetime format used by configured handlers.
        terminal_stream: Standard stream targeted by the terminal handler.
        log_file_path: Optional destination of the file handler.
        additional_logger_levels: Optional mapping of logger names or logger instances to levels used to
            configure additional loggers.

    Returns:
        The normalized log-file path when file logging is enabled; otherwise `None`.

    Raises:
        ValueError: If the log path cannot be normalized.
    """

    additional_logger_levels = additional_logger_levels or {}

    for logger, new_level in additional_logger_levels.items():
        logger = logging.getLogger(logger) if isinstance(logger, str) else logger
        logger.setLevel(new_level)

    if disabled:
        configure_root_logger()
        return None

    handler_configs: list[BaseLogHandlerConfig] = [
        TerminalLogHandlerConfig(
            level=terminal_handler_level,
            log_format=log_format,
            datetime_format=datetime_format,
            stream=terminal_stream,
        ),
    ]

    if log_file_path is not None:
        log_file_path = normalize_path(log_file_path)

        handler_configs.append(
            FileLogHandlerConfig(
                level=file_handler_level,
                log_format=log_format,
                datetime_format=datetime_format,
                file_path=log_file_path,
            )
        )

    configure_root_logger(
        level=root_logger_level,
        handler_configs=handler_configs,
    )

    return log_file_path


def application_logging_cleanup(
    restore_logging_state: LoggingState | None = None,
) -> None:
    """Reset application logging state.

    The function restores the logging state either to the state captured in a `LoggingState` instance or to a default state when `restore_logging_state` isn't provided.
    In addition, it remove and closes any handlers that were added to the root logger and add a single null handler instead.

    Args:
        restore_logging_state: Optional logging state to restore.
    """

    original_root_logger_level = (
        logging.WARNING
        if restore_logging_state is None
        else restore_logging_state.root_logger_level
    )
    original_additional_logger_levels = (
        {} if restore_logging_state is None else restore_logging_state.additional_logger_level_map
    )

    for logger_name, original_level in original_additional_logger_levels.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(original_level)

    configure_root_logger(level=original_root_logger_level)
