from pathlib import Path
from typing import Final

TOP_LEVEL_PKG_DIR: Final[Path] = Path(__file__).parent.parent
PROJECT_ROOT_DIR: Final[Path] = TOP_LEVEL_PKG_DIR.parent
ASSETS_DIR: Final[Path] = PROJECT_ROOT_DIR / "assets"
DATA_DIR: Final[Path] = ASSETS_DIR / "data"
FIGURES_DIR: Final[Path] = ASSETS_DIR / "figures"
RESULTS_DIR: Final[Path] = ASSETS_DIR / "results"
LOGS_DIR: Final[Path] = ASSETS_DIR / "logs"
IGT_DATASET_PATH: Final[Path] = DATA_DIR / "IGTdata.rdata"
