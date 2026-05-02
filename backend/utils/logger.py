"""
============================================================
MedAI — Centralized Logger
============================================================
Provides a consistent logging setup across all modules.
Uses loguru for rich, structured logs with file rotation.
============================================================
"""

import sys
import os
from pathlib import Path
from loguru import logger

# ── Log directory ─────────────────────────────────────────
LOG_DIR = Path("backend/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "medai.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def setup_logger():
    """
    Configure loguru logger with:
      - Console output (colored)
      - File output (rotating, 10MB max, 7 days retention)
    """
    # Remove default handler
    logger.remove()

    # Console handler — human-readable, colored
    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format=(
            "<green>{time:HH:mm:ss}</green> │ "
            "<level>{level: <8}</level> │ "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> │ "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — detailed, rotating
    logger.add(
        str(LOG_FILE),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} │ {level: <8} │ {name}:{function}:{line} │ {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        encoding="utf-8",
    )

    return logger


# Initialize on import
setup_logger()

# Re-export logger for use in other modules
__all__ = ["logger"]