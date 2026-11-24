"""Application logging setup."""

import logging

from config.constants import LOG_DIR


def configure_logging() -> None:
    """Log to the console and a UTF-8 rotating-by-session file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "industrial_pose_analytics.log", encoding="utf-8"),
        ],
    )
