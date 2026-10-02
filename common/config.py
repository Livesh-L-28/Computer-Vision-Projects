"""
Centralized Configuration & Styling for Computer Vision Intelligence Suite.
"""

from dataclasses import dataclass
from pathlib import Path

# Base Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create standard directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Color Palettes (BGR for OpenCV) ──────────────────────────────────────────
class Colors:
    # Neon highlights
    NEON_RED = (40, 40, 255)
    NEON_CYAN = (255, 255, 40)
    NEON_GREEN = (57, 255, 20)
    NEON_YELLOW = (40, 255, 255)
    NEON_ORANGE = (0, 140, 255)
    NEON_PURPLE = (220, 60, 220)
    NEON_PINK = (255, 0, 255)
    NEON_WHITE = (255, 255, 255)

    # HUD & Panels
    PANEL_BG = (14, 10, 8)
    BORDER_CYAN = (255, 220, 0)
    TEXT_DIM = (140, 140, 160)
    TEXT_BRIGHT = (255, 255, 255)

    # Threat levels
    THREAT_LOW = (80, 220, 80)
    THREAT_MED = (0, 165, 255)
    THREAT_HIGH = (40, 40, 255)

    # Airport zones
    ZONE_RUNWAY_ACTIVE = (40, 40, 220)
    ZONE_TAXIWAY = (220, 200, 40)
    ZONE_SAFE = (60, 200, 60)
    ZONE_RESTRICTED = (0, 140, 255)


@dataclass
class AirCombatConfig:
    model_path: str = "yolov8s.pt"
    confidence_threshold: float = 0.25
    detect_every: int = 1
    tracker_max_age: int = 20
    tracker_iou_thresh: float = 0.2
    history_len: int = 30
    threat_high_thresh: float = 66.0
    threat_med_thresh: float = 33.0
    output_dir: Path = OUTPUTS_DIR / "air_combat"


@dataclass
class AirportRunwayConfig:
    model_path: str = "yolov8s.pt"
    confidence_threshold: float = 0.25
    detect_every: int = 2
    px_per_meter: float = 8.0
    tracker_max_disappeared: int = 25
    history_len: int = 40
    output_dir: Path = OUTPUTS_DIR / "airport_runway"


@dataclass
class ANPRConfig:
    vehicle_model_path: str = "yolov8n.pt"
    plate_model_name: str = "license_plate_detector.pt"
    plate_model_url: str = (
        "https://github.com/Muhammad-Zeerak-Khan/"
        "Automatic-License-Plate-Recognition-using-YOLOv8/"
        "raw/main/license_plate_detector.pt"
    )
    plate_conf: float = 0.25
    vehicle_conf: float = 0.35
    detect_every: int = 2
    half_res: bool = False
    max_ocr_per_frame: int = 3
    ocr_recheck_interval: int = 45
    gui_width: int = 1280
    gui_height: int = 720
    output_dir: Path = OUTPUTS_DIR / "license_plate"
