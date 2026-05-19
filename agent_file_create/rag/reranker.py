import logging
from typing import Optional

from agent_file_create.config import (
    CONTENT_API_ENDPOINT,
    CONTENT_API_KEY,
    CONTENT_API_STYLE,
    CONTENT_MODEL_NAME,
    RERANK_ENABLED,
    RERANK_FINAL_K,
    RERANK_MODEL,
    RERANK_TOP_K,
)
from agent_file_create.llm_client import call_llm
from agent_file_create.rag.store import Hit

logger = logging.getLogger(__name__)


def _cross_encoder_rerank(query: str, hits: list[Hit], model_name: str, top_k: int) -> list[Hit]:
    try:
        from FlagEmbedding import FlagReranker
    except ImportError:
        logger.info("FlagEmbedding not installed, falling back to LLM reranker")
        return _llm_rerank(query, hits, top_k)

    try:
        reranker = FlagReranker(model_name, use_fp16=True)
        pairs = [(query, h.content) for h in hits]
        scores = reranker.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [float(scores)] * len(pairs)
    except Exception as e:
        logger.warning(f"cross_encoder_rerank_failed err={str(e)[:160]}, falling back to LLM reranker")
        return _llm_rerank(query, hits, top_k)

    for h, s in zip(hits, scores):
        h.meta["rerank_score"] = float(s)
    hits.sort(key=lambda h: float(h.meta.get("rerank_score", 0)), reverse=True)
    return hits[:top_k]


def _llm_rerank(query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    if not hits:
        return hits
    if len(hits) <= top_k:
        return hits

    items = []
    for i, h in enumerate(hits):
        body = (h.content or "").strip()
        if len(body) > 400:
            body = body[:400] + "..."
        items.append(f"[{i}] {body}")

    prompt = (
        "你是一个检索结果排序助手。请根据用户问题，对候选文本片段按相关性从高到低排序。\n"
        "只输出排序后的编号列表（如 3,0,5,1,2），不要输出任何其他文字。\n\n"
        f"用户问题：{query[:300]}\n\n"
        "候选片段：\n"
        + "\n\n".join(items)
        + "\n\n排序结果："
    )

    try:
        raw = call_llm(
            prompt,
            timeout_s=30,
            temperature=0.0,
            num_predict=120,
            system="你只输出逗号分隔的数字编号。",
            api_style=CONTENT_API_STYLE,
            api_endpoint=CONTENT_API_ENDPOINT,
            api_key=CONTENT_API_KEY,
            model_name=CONTENT_MODEL_NAME,
        )
    except Exception as e:
        logger.warning(f"llm_rerank_failed err={str(e)[:160]}, using original order")
        return hits[:top_k]

    text = (raw or "").strip()
    indices = []
    for part in text.replace(" ", "").replace("[", "").replace("]", "").split(","):
        try:
            idx = int(part)
            if 0 <= idx < len(hits) and idx not in indices:
                indices.append(idx)
        except ValueError:
            continue

    if not indices:
        return hits[:top_k]

    current = {id(h): h for h in hits}
    unranked = [h for h in hits if id(h) not in {id(current.get(i, hits[0])) for i in range(len(hits))}]
    del current

    reranked = []
    for idx in indices:
        if idx < len(hits):
            reranked.append(hits[idx])
    for h in hits:
        if h not in reranked:
            reranked.append(h)

    for i, h in enumerate(reranked):
        h.meta["rerank_score"] = float(len(reranked) - i)
    return reranked[:top_k]


def rerank(query: str, hits: list[Hit], *, top_k: Optional[int] = None) -> list[Hit]:
    if not RERANK_ENABLED:
        return hits[: (top_k or RERANK_FINAL_K)]
    if not hits:
        return hits

    cand = hits[: max(RERANK_TOP_K, len(hits))]
    final_k = top_k if top_k is not None else RERANK_FINAL_K

    model = RERANK_MODEL.strip()
    if model and model.lower() not in {"none", "false", "off", "llm"}:
        return _cross_encoder_rerank(query, cand, model, final_k)
    return _llm_rerank(query, cand, final_k)
