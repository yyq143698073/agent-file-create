import base64
from pathlib import Path
from typing import Any


def image_to_base64(image_path: str) -> str:
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


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
