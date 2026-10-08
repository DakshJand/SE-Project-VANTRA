"""Runtime configuration for VANTRA backend."""
from pathlib import Path

from pydantic_settings import BaseSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"


class Settings(BaseSettings):
    database_url: str = "postgresql://vantra:vantra@localhost:5432/vantra"
    image_dir: Path = IMAGE_DIR
    ocr_model_dir: Path = DATA_DIR / "models"
    # Matcher tuning
    max_ocr_conf_gap: float = 0.15          # OCR-conf-aware fuzzy threshold half-width
    appearance_margin: float = 0.06         # ambiguous if top-2 within this margin
    hop_skip_multiplier: float = 1.9        # time-window widening across 2 hops
    # Alerting
    blacklist_fuzzy_ed: int = 1             # max edit distance for fuzzy blacklist hit
    convoy_window_s: int = 180              # same-camera arrival window for convoy
    convoy_min_cameras: int = 2             # cameras needed to confirm convoy
    soft_history_min: int = 8               # min history to run per-plate soft anomaly
    # Security (demo default; override with VANTRA_ADMIN_KEY in production)
    admin_api_key: str = "vantra-admin"

    class Config:
        env_prefix = "VANTRA_"


settings = Settings()
