import logging
import os
# Suppress the "Consider using the pymupdf_layout package" advertisement.
# Must be set before PyMuPDF's first import (it checks once on first use).
os.environ.setdefault("PYMUPDF_SUGGEST_LAYOUT_ANALYZER", "0")
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging() -> None:
    """Configure application-wide logging with rotation and console output.

    File handler: INFO level, 5MB max per file, keeps 3 backups.
    Console handler: WARNING for third-party libs, INFO for our modules.
    """
    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    log_path = log_dir / "app.log"
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Rotating file handler: 5 MB per file, keep 3 backups
    fh = RotatingFileHandler(
        str(log_path), encoding="utf-8",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)

    # Console: INFO for our modules, WARNING for third-party libs
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    ch.addFilter(lambda r: r.name.startswith("agent_file_create") or r.name == "__main__")
    root.addHandler(ch)

    # Silence noisy third-party libs
    for lib in ("urllib3", "httpx", "fastapi", "uvicorn", "langchain", "jieba", "fitz", "pymupdf"):
        logging.getLogger(lib).setLevel(logging.WARNING)
