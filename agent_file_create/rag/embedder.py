import json
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from agent_file_create.config import EMBED_API_ENDPOINT, EMBED_API_KEY, EMBED_API_STYLE, EMBED_MODEL_NAME, OLLAMA_HOST, OPENAI_API_ENDPOINT, OPENAI_API_KEY

logger = logging.getLogger(__name__)

# ── HTTP connection pool (urllib3 if available, else stdlib fallback) ──
_HTTP_POOL = None  # lazy-init


def _get_http_pool():
    """Return a persistent connection pool, creating it on first call."""
    global _HTTP_POOL
    if _HTTP_POOL is not None:
        return _HTTP_POOL
    try:
        import urllib3
        _HTTP_POOL = urllib3.PoolManager(
            maxsize=16,
            block=False,
            timeout=urllib3.Timeout(connect=10.0, read=120.0),
            retries=urllib3.Retry(3, backoff_factor=0.5),
        )
        logger.debug("embedder: using urllib3 PoolManager (maxsize=16)")
    except ImportError:
        _HTTP_POOL = False  # sentinel — fall back to stdlib
        logger.debug("embedder: urllib3 not available, using stdlib urllib")
    return _HTTP_POOL


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
    """POST JSON to *url*, returning the parsed response dict.

    Uses a persistent urllib3 connection pool when available (avoids TCP
    handshake overhead on repeated calls); falls back to stdlib otherwise.
    """
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    h = {"Content-Type": "application/json; charset=utf-8"}
    if headers:
        h.update(headers)

    pool = _get_http_pool()
    if pool is not False:
        import urllib3
        try:
            resp = pool.request(
                "POST",
                url,
                body=data,
                headers=h,
                timeout=urllib3.Timeout(connect=10.0, read=float(timeout_s)),
            )
            if resp.status >= 400:
                body_snippet = (resp.data or b"").decode("utf-8", errors="ignore")[:500]
                raise RuntimeError(f"HTTP {resp.status}: {body_snippet}")
            raw = resp.data or b""
        except urllib3.exceptions.HTTPError as e:
            raise RuntimeError(str(e)[:500]) from e
    else:
        # stdlib fallback
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


def _embed_single_batch(
    part: list[str],
    *,
    batch_idx: int,
    style: str,
    endpoint: str,
    model: str,
    timeout_s: int,
    fallback_embed_endpoint: str,
    embed_api_key: str,
) -> list[list[float]]:
    """Process one batch with retry + fallback logic.  Returns vectors for *part*.

    This is a pure function (no mutable shared state) so it can be called from
    multiple threads concurrently.
    """
    last_err = ""
    for attempt in range(3):
        try:
            if style == "openai":
                return _embed_openai(part, model=model, endpoint=endpoint, api_key=embed_api_key, timeout_s=timeout_s)
            else:
                return _embed_ollama(part, model=model, endpoint=endpoint, timeout_s=timeout_s)
        except Exception as e:
            last_err = str(e)[:240]
            # Fallback: try OpenAI-compatible embedding provider
            if style == "ollama" and attempt >= 1 and fallback_embed_endpoint:
                try:
                    return _embed_openai(
                        part, model="text-embedding-3-small",
                        endpoint=fallback_embed_endpoint, api_key=OPENAI_API_KEY,
                        timeout_s=timeout_s,
                    )
                except Exception:
                    pass
        if last_err and attempt < 2:
            time.sleep(2.0 + 2.0 * attempt)

    # All batch attempts failed — retry one-by-one
    logger.warning("embed_batch_failed batch=%d err=%s, retrying one-by-one", batch_idx, last_err)
    results: list[list[float]] = []
    for single in part:
        single_ok = False
        for s_attempt in range(4):
            try:
                if style == "openai":
                    vecs = _embed_openai([single], model=model, endpoint=endpoint, api_key=embed_api_key, timeout_s=timeout_s)
                else:
                    vecs = _embed_ollama([single], model=model, endpoint=endpoint, timeout_s=timeout_s)
                results.extend(vecs)
                single_ok = True
                break
            except Exception:
                time.sleep(3.0 + 2.0 * s_attempt)
        if not single_ok:
            logger.warning(
                "embed_ollama_empty_vector text_len=%d — chunk will be invisible to vector search",
                len(single),
            )
            results.append([])
    return results


def embed_texts(texts: Iterable[str], *, timeout_s: int = 60, max_batch: int = 32,
                parallel_workers: int = 4) -> list[list[float]]:
    """Batch-embed *texts*, sending independent batches in parallel.

    Args:
        texts: Input strings to embed.
        timeout_s: Per-request timeout in seconds.
        max_batch: Maximum texts per API call.
        parallel_workers: Number of concurrent batch workers (1 = sequential).
    """
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
        base = _normalize_openai_base_url(raw_oai)
        if base:
            fallback_embed_endpoint = base

    embed_api_key = EMBED_API_KEY

    batch = max(1, int(max_batch or 0))
    batches = [xs[i : i + batch] for i in range(0, len(xs), batch)]

    # ── Single batch or sequential mode: process inline ────────────────
    if len(batches) <= 1 or parallel_workers <= 1:
        out: list[list[float]] = []
        for idx, part in enumerate(batches):
            out.extend(_embed_single_batch(
                part, batch_idx=idx, style=style, endpoint=endpoint,
                model=model, timeout_s=timeout_s,
                fallback_embed_endpoint=fallback_embed_endpoint,
                embed_api_key=embed_api_key,
            ))
            if idx + 1 < len(batches):
                time.sleep(0.5)
        if not out:
            return [[] for _ in xs]
        return out

    # ── Parallel mode: send batches concurrently ───────────────────────
    out: list[list[float]] = []
    with ThreadPoolExecutor(max_workers=min(parallel_workers, len(batches))) as executor:
        futures: dict = {}
        for idx, part in enumerate(batches):
            future = executor.submit(
                _embed_single_batch,
                part,
                batch_idx=idx,
                style=style,
                endpoint=endpoint,
                model=model,
                timeout_s=timeout_s,
                fallback_embed_endpoint=fallback_embed_endpoint,
                embed_api_key=embed_api_key,
            )
            futures[future] = idx

        # Collect in submission order
        results_by_idx: dict[int, list[list[float]]] = {}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results_by_idx[idx] = future.result() or []
            except Exception:
                logger.warning("embed_parallel_batch_failed batch=%d", idx)
                results_by_idx[idx] = [[] for _ in batches[idx]]

        for idx in sorted(results_by_idx):
            out.extend(results_by_idx[idx])

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
