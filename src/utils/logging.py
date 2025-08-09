from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, lvl, logging.INFO)
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(numeric_level)
        return
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


__all__ = ["setup_logging"]
