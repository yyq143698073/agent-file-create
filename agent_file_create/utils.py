import base64
import logging
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def image_to_base64(image_path: str) -> str:
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def retry_call(
    fn: Callable,
    *args: Any,
    max_retries: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
    **kwargs: Any,
) -> Any:
    """Call ``fn(*args, **kwargs)`` with exponential backoff on exception."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                wait = delay * (backoff ** attempt)
                logger.warning(
                    "retry %d/%d for %s: %s, waiting %.1fs",
                    attempt + 1,
                    max_retries,
                    getattr(fn, "__name__", str(fn)),
                    exc,
                    wait,
                )
                time.sleep(wait)
    raise last_error  # type: ignore[misc]


def safe_json(obj: Any, max_len: int = 2000) -> str:
    import json

    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    s = s.strip()
    if len(s) > max_len:
        return s[:max_len] + "…"
    return s
