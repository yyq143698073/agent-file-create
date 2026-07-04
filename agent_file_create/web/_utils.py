"""Shared utilities for web routes — path helpers, file sanitization, quality comparison.

Extracted from web/server.py to keep route files focused on HTTP handling.
"""

import logging
import os
import re
import threading
from pathlib import Path

_MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "3"))
_task_semaphore = threading.BoundedSemaphore(_MAX_CONCURRENT_TASKS)

# ── Upload security ──────────────────────────────────────────────────────────

# Allowed file extensions for upload
_ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls",
    ".pptx", ".ppt", ".txt", ".md", ".csv",
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif",
}

# Allowed MIME types (checked against Content-Type header)
_ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/msword",                                                         # .doc
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",          # .xlsx
    "application/vnd.ms-excel",                                                   # .xls
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "application/vnd.ms-powerpoint",                                              # .ppt
    "text/plain",
    "text/markdown",
    "text/csv",
    "image/png",
    "image/jpeg",
    "image/bmp",
    "image/tiff",
}

# Max upload size per file (50 MB)
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50")) * 1024 * 1024


def validate_upload_file(filename: str, content_type: str | None = None, file_size: int = 0) -> str | None:
    """Validate an uploaded file. Returns error message or None if valid.

    Checks:
        1. Filename is not empty
        2. Extension is in the allowed list
        3. MIME type is in the allowed list (if provided)
        4. File size is within the limit
    """
    if not filename or not filename.strip():
        return "文件名为空"

    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return f"不支持的文件类型: {ext}（支持: {', '.join(sorted(_ALLOWED_EXTENSIONS))}）"

    if content_type and content_type not in _ALLOWED_MIME_TYPES:
        # Be lenient: some clients send generic MIME types
        if content_type not in ("application/octet-stream", "multipart/form-data"):
            logger.debug("upload_mime_warn file=%s mime=%s", filename, content_type)

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        return f"文件过大: {file_size / 1024 / 1024:.1f}MB（限制: {MAX_UPLOAD_SIZE_BYTES / 1024 / 1024:.0f}MB）"

    return None


logger = logging.getLogger(__name__)


def get_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def html_dir() -> Path:
    return get_base_dir() / "html"


def result_dir() -> Path:
    return get_base_dir() / "result"


def sanitize_filename(name: str) -> str:
    n = (name or "").strip()
    n = n.replace("\\", "/").split("/")[-1]
    n = re.sub(r"[^0-9A-Za-z一-鿿._-]+", "_", n)
    n = n.strip("._")
    return n or "upload"


def better_quality(a: dict, b: dict) -> bool:
    try:
        af = int(a.get("filled_fields") or 0)
    except Exception:
        af = 0
    try:
        bf = int(b.get("filled_fields") or 0)
    except Exception:
        bf = 0
    if bf != af:
        return bf > af
    try:
        ar = float(a.get("field_ratio") or 0.0)
    except Exception:
        ar = 0.0
    try:
        br = float(b.get("field_ratio") or 0.0)
    except Exception:
        br = 0.0
    return br >= ar


def ingest_files_to_kb(kb_name: str, file_paths: list[str]) -> None:
    """Ingest saved files into a knowledge base (runs in background thread)."""
    _log = logging.getLogger(__name__)
    from agent_file_create.rag.kb import KnowledgeBase
    kb = KnowledgeBase()
    kb_name = (kb_name or "").strip() or "default"
    ok = 0
    for fp in file_paths:
        try:
            r = kb.ingest_file(kb=kb_name, file_path=fp)
            if r.get("ok"):
                ok += 1
            else:
                _log.warning("kb_ingest_failed file=%s err=%s", fp, r.get("error", ""))
        except Exception as e:
            _log.warning("kb_ingest_failed file=%s err=%s", fp, str(e)[:200])
    _log.info("kb_ingest_done kb=%s ok=%d/%d", kb_name, ok, len(file_paths))
