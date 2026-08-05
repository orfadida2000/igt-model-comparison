"""Application, execution, and logging defaults."""

import logging
from typing import Final

SUBJECTS_AVAILABLE: Final[int] = 617
FIXED_SEED: Final[int] = 42
DEFAULT_N_Q_STARTS: Final[int] = 10
DEFAULT_N_PVL_STARTS: Final[int] = 32
DEFAULT_N_WORKERS: Final[int] = 0  # Use Sequential Execution by default
DEFAULT_N_SUBJECTS: Final[int] = -1  # Use all available subjects by default
DEFAULT_ROOT_LOG_LEVEL: Final[int] = logging.DEBUG
DEFAULT_NOTIFY_FORMSUBMIT_ID: Final[str | None] = None

USE_FIXED_SEED: Final[bool] = True
USE_DEFAULT_NOTIFY_FORMSUBMIT_ID: Final[bool] = True

FILENAME_DATETIME_FMT: Final[str] = "%Y-%m-%d_%H-%M-%S"
DATETIME_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT: Final[str] = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
TERMINAL_LOG_LEVEL: Final[int] = logging.INFO
FILE_LOG_LEVEL: Final[int] = logging.DEBUG
