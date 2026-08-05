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
    TerminalLogHandlerConfig,
)


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
    root_level: int | None,
    log_file_path: Path,
) -> Path | None:
    """Configure terminal and file logging for a top-level application.

    Args:
        disabled: Whether all effective logging handlers should be disabled.
        root_level: Root logger level when logging is enabled.
        log_file_path: Destination for the file handler when logging is enabled.

    Returns:
        The configured log-file path, or ``None`` when logging is disabled.
    """

    if disabled:
        configure_root_logger()
        return None

    if root_level is None:
        raise ValueError("root_level must be provided when logging is enabled.")

    path = Path(log_file_path)
    handler_configs: list[BaseLogHandlerConfig] = [
        TerminalLogHandlerConfig(
            level=TERMINAL_LOG_LEVEL,
            log_format=LOG_FORMAT,
            datetime_format=DATETIME_FORMAT,
            stream=StandardOutput.STDERR,
        ),
        FileLogHandlerConfig(
            level=FILE_LOG_LEVEL,
            log_format=LOG_FORMAT,
            datetime_format=DATETIME_FORMAT,
            file_path=path,
        ),
    ]

    configure_root_logger(
        level=root_level,
        handler_configs=handler_configs,
    )

    return path


def application_logging_cleanup() -> None:
    """Perform any necessary cleanup for application logging.

    This function should be called at the end of an application's execution
    to ensure that all logging resources are properly released.
    """

    default_root_logger_level = logging.WARNING

    configure_root_logger(level=default_root_logger_level)
