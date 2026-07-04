import base64
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

from agent_file_create.errors import LLMCallError

logger = logging.getLogger(__name__)


def split_questions(text: str) -> list[str]:
    """Parse LLM-generated clarification text into a list of individual questions.

    Handles numbered lists (1. / 1) / 1、), bullet points (- / *), and
    lettered options (A. / A) / A、). Returns deduplicated questions, max 6.
    """
    out: list[str] = []
    cur = ""
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            if cur:
                out.append(cur)
                cur = ""
            continue
        is_option = bool(re.match(r"^[A-Z][.)、\s]", s))
        if is_option and cur:
            cur += "\n" + s
            continue
        if cur:
            out.append(cur)
        s = re.sub(r"^[0-9]+[.)、\s]+", "", s).strip()
        s = re.sub(r"^[-*]\s+", "", s).strip()
        if s:
            cur = s[:240]
        if len(out) >= 6:
            break
    if cur and len(out) < 6:
        out.append(cur)
    if out:
        return out
    s = str(text or "").strip()
    return [s[:240]] if s else []


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
    """Call ``fn(*args, **kwargs)`` with exponential backoff on exception.

    Raises LLMCallError if all retries are exhausted, so callers can catch
    it uniformly and decide whether to use fallback content or abort.
    """
    fn_name = getattr(fn, "__name__", str(fn))
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
                    fn_name,
                    exc,
                    wait,
                )
                time.sleep(wait)
    raise LLMCallError(
        f"{fn_name} failed after {max_retries} retries: {last_error}",
        attempt=max_retries,
    ) from last_error


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
