import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from agent_file_create.config import EMBED_API_ENDPOINT, EMBED_API_KEY, EMBED_API_STYLE, EMBED_MODEL_NAME, OLLAMA_HOST, OPENAI_API_ENDPOINT, OPENAI_API_KEY


def _normalize_openai_base_url(endpoint: str) -> str:
    u = (endpoint or "").strip()
    if not u:
        return ""
    u = u.rstrip("/")
    if u.endswith("/embeddings"):
        u = u[: -len("/embeddings")].rstrip("/")
    if u.endswith("/v1"):
        return u
    if "/v1/" in u:
        i = u.find("/v1/")
        return u[: i + 3]
    return u


def _post_json(url: str, payload: dict, headers: dict | None = None, *, timeout_s: int = 60) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    h = {"Content-Type": "application/json; charset=utf-8", "Connection": "close"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=int(timeout_s)) as resp:
            raw = resp.read() or b""
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="ignore")[:500]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    try:
        obj = json.loads(raw.decode("utf-8"))
        return obj if isinstance(obj, dict) else {"raw": obj}
    except Exception:
        return {"raw_text": raw.decode("utf-8", errors="ignore")[:2000]}


def _as_float_list(x) -> list[float]:
    if not isinstance(x, list):
        return []
    out: list[float] = []
    for v in x:
        try:
            out.append(float(v))
        except Exception:
            return []
    return out


def _embed_ollama_one(text: str, *, model: str, base: str, timeout_s: int) -> list[float]:
    """Single-text fallback using /api/embeddings endpoint — avoids batch NaN issue.
    Returns empty list on persistent failure so caller can substitute a zero vector."""
    url = base + "/api/embeddings"
    for attempt in range(3):
        t = text[:2000] if attempt > 0 else text
        payload = {"model": model, "prompt": t}
        try:
            obj = _post_json(url, payload, timeout_s=timeout_s)
            emb = _as_float_list(obj.get("embedding"))
            if emb:
                return emb
        except Exception:
            pass
        if attempt < 2:
            time.sleep(1.0)
    return []  # empty = caller handles via zero-vector fallback


def _embed_ollama(texts: list[str], *, model: str, endpoint: str, timeout_s: int) -> list[list[float]]:
    base = (endpoint or "").strip() or (OLLAMA_HOST or "").strip()
    base = base.rstrip("/")
    # Use Ollama's OpenAI-compatible batch endpoint — handles multiple texts in one request
    url = base + "/v1/embeddings"
    payload = {"model": model, "input": texts}
    try:
        obj = _post_json(url, payload, timeout_s=timeout_s)
        data = obj.get("data")
        if not isinstance(data, list):
            raise RuntimeError("ollama v1/embeddings 返回格式错误")
        # Sort by index to preserve input order
        data.sort(key=lambda it: it.get("index", 0) if isinstance(it, dict) else 0)
        out: list[list[float]] = []
        for it in data:
            if not isinstance(it, dict):
                continue
            emb = _as_float_list(it.get("embedding"))
            if emb:
                out.append(emb)
        if len(out) != len(texts):
            raise RuntimeError(f"ollama embeddings 返回数量不匹配: got {len(out)}, expected {len(texts)}")
        return out
    except RuntimeError as e:
        err_msg = str(e)
        # bge-m3 NaN bug: certain texts cause "json: unsupported value: NaN"
        # Fall back to /api/embeddings (single-prompt endpoint) one at a time
        if "NaN" in err_msg or len(texts) == 1:
            results: list[list[float]] = []
            for t in texts:
                results.append(_embed_ollama_one(t, model=model, base=base, timeout_s=timeout_s))
            return results
        raise


def _embed_openai(texts: list[str], *, model: str, endpoint: str, api_key: str, timeout_s: int) -> list[list[float]]:
    base = _normalize_openai_base_url(endpoint)
    if not base:
        raise RuntimeError("EMBED_API_ENDPOINT 为空，无法使用 openai embedding")
    url = base.rstrip("/") + "/v1/embeddings"
    key = (api_key or "").strip() or (OPENAI_API_KEY or "").strip()
    if not key:
        raise RuntimeError("EMBED_API_KEY 为空")
    payload = {"model": model, "input": texts}
    obj = _post_json(url, payload, headers={"Authorization": "Bearer " + key}, timeout_s=timeout_s)
    data = obj.get("data")
    if not isinstance(data, list):
        raise RuntimeError("openai embeddings 返回格式错误")
    out: list[list[float]] = []
    for it in data:
        if not isinstance(it, dict):
            continue
        emb = _as_float_list(it.get("embedding"))
        if emb:
            out.append(emb)
    if len(out) != len(texts):
        raise RuntimeError("openai embeddings 返回数量不匹配")
    return out


def embed_texts(texts: Iterable[str], *, timeout_s: int = 60, max_batch: int = 32) -> list[list[float]]:
    xs = [str(x or "").strip() for x in (texts or [])]
    xs = [x for x in xs if x]
    if not xs:
        return []

    style = (EMBED_API_STYLE or "").strip().lower()
    endpoint = (EMBED_API_ENDPOINT or "").strip()
    model = (EMBED_MODEL_NAME or "").strip()
    if not style:
        style = "ollama"
    if not model:
        model = "nomic-embed-text"

    # Fallback: if ollama fails and an OpenAI-compatible endpoint is configured, try it
    fallback_embed_endpoint = ""
    raw_oai = (OPENAI_API_ENDPOINT or "").strip()
    if raw_oai and ("/v1/" in raw_oai or raw_oai.endswith("/v1")):
        # Derive embeddings base URL from chat endpoint
        base = _normalize_openai_base_url(raw_oai)
        if base:
            fallback_embed_endpoint = base

    out: list[list[float]] = []
    batch = max(1, int(max_batch or 0))
    for i in range(0, len(xs), batch):
        part = xs[i : i + batch]
        last_err = ""
        for attempt in range(3):
            try:
                if style == "openai":
                    vecs = _embed_openai(part, model=model, endpoint=endpoint, api_key=EMBED_API_KEY, timeout_s=timeout_s)
                else:
                    vecs = _embed_ollama(part, model=model, endpoint=endpoint, timeout_s=timeout_s)
                out.extend(vecs)
                last_err = ""
                break
            except Exception as e:
                last_err = str(e)[:240]
                # Fallback: try OpenAI-compatible embedding provider
                if style == "ollama" and attempt >= 1 and fallback_embed_endpoint:
                    try:
                        vecs = _embed_openai(
                            part, model="text-embedding-3-small",
                            endpoint=fallback_embed_endpoint, api_key=OPENAI_API_KEY,
                            timeout_s=timeout_s,
                        )
                        style = "openai"
                        endpoint = fallback_embed_endpoint
                        model = "text-embedding-3-small"
                        out.extend(vecs)
                        last_err = ""
                        break
                    except Exception:
                        pass
            if last_err and attempt < 2:
                time.sleep(2.0 + 2.0 * attempt)  # longer backoff: 2s, 4s, then one-by-one
        if last_err:
            import logging
            logging.getLogger(__name__).warning("embed_batch_failed batch=%d err=%s, retrying one-by-one", i, last_err)
            # Retry each text individually with health-aware recovery
            for single in part:
                single_ok = False
                for s_attempt in range(4):
                    try:
                        if style == "openai":
                            vecs = _embed_openai([single], model=model, endpoint=endpoint, api_key=EMBED_API_KEY, timeout_s=timeout_s)
                        else:
                            vecs = _embed_ollama([single], model=model, endpoint=endpoint, timeout_s=timeout_s)
                        out.extend(vecs)
                        single_ok = True
                        break
                    except Exception:
                        time.sleep(3.0 + 2.0 * s_attempt)  # 3s, 5s, 7s, 9s — wait for Ollama recovery
                if not single_ok:
                    logger.warning(
                        "embed_ollama_empty_vector text_len=%d — chunk will be invisible to vector search",
                        len(single),
                    )
                    out.append([])  # empty vector as last resort
        # Small gap between batches to let Ollama breathe
        if i + batch < len(xs):
            time.sleep(0.5)

    if not out:
        return [[] for _ in xs]
    return out


def check_embed_health() -> dict:
    """Check embedding model connectivity. Returns {'ok': True, 'dim': N} or {'ok': False, 'error': ...}."""
    try:
        vecs = embed_texts(["health check"], timeout_s=15, max_batch=1)
        if vecs and isinstance(vecs[0], list) and len(vecs[0]) > 0:
            return {"ok": True, "dim": len(vecs[0])}
        return {"ok": False, "error": "empty_embedding"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:240]}
