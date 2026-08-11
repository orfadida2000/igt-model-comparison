"""Shared root-logger configuration for application entry points."""

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
    """Replace the root logger's handlers with configured handlers.

    A null handler is installed when no effective handler configuration is
    supplied. Entry-point modules should call this function before starting
    work that may create worker processes.
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
    """Configure terminal and file logging for a top-level application.

    Args:
        disabled: Whether all effective logging handlers should be disabled.
        root_logger_level: Root logger level when logging is enabled.
        terminal_handler_level: Level for the terminal handler when logging is enabled.
        file_handler_level: Level for the file handler when logging is enabled.
        log_format: Log message format for all handlers when logging is enabled.
        datetime_format: Datetime format for all handlers when logging is enabled.
        terminal_stream: Destination for the terminal handler when logging is enabled.
        log_file_path: Destination for the file handler when logging is enabled.

    Returns:
        The configured log-file path, or `None` when logging is disabled or no log file path is provided.
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
    """Perform any necessary cleanup for application logging.

    This function should be called at the end of an application's execution
    to ensure that all logging resources are properly released.
    """

    default_root_logger_level = logging.WARNING

    configure_root_logger(level=default_root_logger_level)
