"""Project paths and UI constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "pose_landmarker_full.task"
VIDEO_DIR = ROOT_DIR / "videos"
OUTPUT_DIR = ROOT_DIR / "output"
LOG_DIR = ROOT_DIR / "logs"
APP_VERSION = "1.1.0"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

COLORS = {
    "bg": "#111315",
    "surface": "#191c1f",
    "surface_alt": "#202428",
    "border": "#30363b",
    "text": "#f3f5f6",
    "muted": "#9ba3a9",
    "accent": "#68d5ed",
    "accent_hover": "#8be3f5",
    "green": "#46d483",
    "red": "#ff6b6b",
    "chart_grid": "#353b40",
}
