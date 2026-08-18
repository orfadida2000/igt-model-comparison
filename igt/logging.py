"""Shared root-logger configuration for project entry points.

The helpers configure terminal, file, or null handlers from typed handler
configuration objects, capture the resulting application logging state, and provide a
matching cleanup operation for scripts and the main fitting workflow.
"""

import logging
from pathlib import Path

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

    Returns:
        The normalized log-file path when file logging is enabled; otherwise `None`.

    Raises:
        ValueError: If logging is enabled without a root logger level or if the log
            path cannot be normalized.
    """

    if disabled:
        configure_root_logger()
        return None

    if root_logger_level is None:
        raise ValueError("root_logger_level must be provided when logging is enabled.")

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


def application_logging_cleanup() -> None:
    """Reset application logging to a minimal default state.

    The root logger is restored to the standard warning level with a null handler,
    which closes any terminal or file handlers created for the completed workflow.
    """

    default_root_logger_level = logging.WARNING

    configure_root_logger(level=default_root_logger_level)
