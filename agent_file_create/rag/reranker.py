"""Reranking layer: cross-encoder, listwise LLM, cascade, sliding-window.

Optimizations:
- Cross-encoder: parallel batch scoring via ThreadPoolExecutor
- Listwise LLM (RankGPT-style): sliding window for large candidate sets,
  diversity-aware prompts with doc/section metadata
- Cascade: coarse (cross-encoder) → fine (LLM listwise) progressive filtering
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _cross_encoder_rerank(query: str, hits: list[Hit], model_name: str,
                          top_k: int, *, batch_size: int = 8) -> list[Hit]:
    """Cross-encoder rerank with parallel batch scoring."""
    try:
        from FlagEmbedding import FlagReranker
    except ImportError:
        logger.info("FlagEmbedding not installed, falling back to LLM reranker")
        return _llm_listwise_rerank(query, hits, top_k)

    try:
        reranker = FlagReranker(model_name, use_fp16=True)
        pairs = [(query, h.content) for h in hits]

        if len(pairs) <= batch_size:
            scores = reranker.compute_score(pairs, normalize=True)
            if not isinstance(scores, list):
                scores = [float(scores)] * len(pairs)
        else:
            # Parallel batch scoring
            batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
            scores = [0.0] * len(pairs)
            max_w = min(len(batches), 6)

            def _score_batch(batch: list[tuple[str, str]], offset: int) -> tuple[int, list[float]]:
                b_scores = reranker.compute_score(batch, normalize=True)
                if not isinstance(b_scores, list):
                    b_scores = [float(b_scores)] * len(batch)
                return offset, [float(s) for s in b_scores]

            with ThreadPoolExecutor(max_workers=max_w) as ex:
                futures = {}
                offset = 0
                for b in batches:
                    futures[ex.submit(_score_batch, b, offset)] = len(b)
                    offset += len(b)
                for future in as_completed(futures):
                    off, b_scores = future.result()
                    for j, s in enumerate(b_scores):
                        scores[off + j] = s
    except Exception as e:
        logger.warning(f"cross_encoder_rerank_failed err={str(e)[:160]}, falling back to LLM reranker")
        return _llm_listwise_rerank(query, hits, top_k)

    for h, s in zip(hits, scores):
        h.meta["rerank_score"] = float(s)
    hits.sort(key=lambda h: float(h.meta.get("rerank_score", 0)), reverse=True)
    return hits[:top_k]


def _sliding_window_listwise(query: str, hits: list[Hit], top_k: int,
                              window_size: int = 10, step: int = 6) -> list[Hit]:
    """Sliding window listwise LLM rerank for large candidate sets.

    Uses the same sliding-window technique as RankGPT: process the candidate
    list in overlapping windows from back to front, so the highest-ranked
    items gradually bubble to the top.
    """
    if len(hits) <= window_size:
        return _llm_listwise_core(query, hits, top_k)

    # Process windows back-to-front (RankGPT strategy)
    ranked = list(hits)
    for start in range(len(ranked) - window_size, -step, -step):
        start = max(0, start)
        window = ranked[start:start + window_size]
        reranked_window = _llm_listwise_core(query, window, len(window))
        ranked[start:start + window_size] = reranked_window

    return ranked[:top_k]


def _llm_listwise_core(query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    """Core listwise LLM rerank — ranks all candidates in one pass with diversity awareness."""
    if not hits:
        return hits
    if len(hits) <= 1:
        return hits[:top_k]

    items_text: list[str] = []
    for i, h in enumerate(hits):
        body = (h.content or "").strip()
        if len(body) > 500:
            body = body[:500] + "..."
        doc_id = str(h.doc_id or "")[:40]
        section = str(h.section_path or "")[:40]
        meta_str = f"[doc={doc_id}, section={section}]"
        items_text.append(f"[{i}] {meta_str}\n    {body}")

    prompt = (
        "你是一个检索结果排序助手。请根据用户问题，对候选文本片段按相关性从高到低排序。\n\n"
        "排序原则：\n"
        "1) 与问题直接相关的片段排前面\n"
        "2) 同一个文档的片段应适当分散，优先覆盖不同文档和章节角度\n"
        "3) 返回覆盖不同角度的多样结果，而非重复同一内容\n\n"
        f"用户问题：{query[:300]}\n\n"
        "候选片段：\n"
        + "\n\n".join(items_text)
        + "\n\n请输出排序后的编号列表（以逗号分隔，如 3,0,5,1,2），不要输出任何其他文字。"
    )

    try:
        raw = call_llm(
            prompt,
            timeout_s=40,
            temperature=0.0,
            num_predict=200,
            system="你是一个专业的检索排序助手。只输出逗号分隔的数字编号，不输出其他内容。",
            api_style=CONTENT_API_STYLE,
            api_endpoint=CONTENT_API_ENDPOINT,
            api_key=CONTENT_API_KEY,
            model_name=CONTENT_MODEL_NAME,
        )
    except Exception as e:
        logger.warning(f"llm_listwise_failed err={str(e)[:160]}, using original order")
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

    # Build result: ranked indices first, then unranked appended
    reranked = [hits[idx] for idx in indices]
    for h in hits:
        if h not in reranked:
            reranked.append(h)

    for i, h in enumerate(reranked):
        h.meta["rerank_score"] = float(len(reranked) - i)
    return reranked[:top_k]


def _llm_listwise_rerank(query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    """Listwise LLM rerank with sliding window for large sets (>15 candidates)."""
    if len(hits) <= 15:
        return _llm_listwise_core(query, hits, top_k)
    else:
        return _sliding_window_listwise(query, hits, top_k, window_size=12, step=7)


def _cascade_rerank(query: str, hits: list[Hit], top_k: int,
                    ce_model: str, ce_intermediate: int = 20) -> list[Hit]:
    """Cascade/Progressive Reranking: coarse → fine pipeline.

    Stage 1 (coarse): Cross-encoder reduces candidate pool to ce_intermediate
    Stage 2 (fine): LLM listwise ranks the reduced pool with diversity awareness

    This gives speed (cross-encoder on large set) + quality (LLM on small set).
    """
    if not hits:
        return hits

    # Stage 1: cross-encoder coarse filtering
    intermediate = min(ce_intermediate, len(hits))
    stage1 = _cross_encoder_rerank(query, hits, ce_model, intermediate)

    # Stage 2: LLM listwise fine ranking with diversity
    return _llm_listwise_rerank(query, stage1, top_k)


def _dedup_before_rerank(hits: list[Hit]) -> list[Hit]:
    """Remove near-duplicate chunks before reranking to avoid wasted computation."""
    seen: set[str] = set()
    out: list[Hit] = []
    for h in hits:
        # Use first 80 chars as lightweight fingerprint
        fp = str(h.content or "")[:80].strip()
        if fp and fp in seen:
            continue
        if fp:
            seen.add(fp)
        out.append(h)
    return out


def _score_norm(hits: list[Hit]) -> None:
    """Min-max normalize scores in-place so downstream consumers see consistent ranges."""
    if len(hits) < 2:
        return
    vals = [float(h.score) for h in hits]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return
    for h in hits:
        h.score = (float(h.score) - lo) / (hi - lo)


def rerank(query: str, hits: list[Hit], *, top_k: Optional[int] = None,
           mode: str = "auto") -> list[Hit]:
    """Rerank hits by relevance to query.

    Args:
        query: The user query string.
        hits: Candidate hits to rerank.
        top_k: Final number of hits to return.
        mode: Reranking strategy:
            - "auto" (default): cascade if RERANK_MODEL is a cross-encoder, listwise otherwise
            - "cross_encoder": cross-encoder only
            - "listwise": LLM listwise only
            - "cascade": progressive cross-encoder → LLM listwise
            - "none": skip reranking, just truncate
    """
    if not RERANK_ENABLED or mode == "none":
        return hits[: (top_k or RERANK_FINAL_K)]
    if not hits:
        return hits

    # Deduplicate near-identical chunks
    cand = _dedup_before_rerank(hits)
    cand = cand[: max(RERANK_TOP_K, len(cand))]
    final_k = top_k if top_k is not None else RERANK_FINAL_K

    if final_k >= len(cand):
        _score_norm(cand)
        return cand

    model = RERANK_MODEL.strip()
    has_ce = model and model.lower() not in {"none", "false", "off", "llm"}

    # ── Determine effective mode ──
    effective = mode
    if effective == "auto":
        # Prefer cross-encoder (fast, consistent) over LLM listwise (slow, no benefit on homogeneous text)
        if has_ce:
            effective = "cross_encoder"
        else:
            effective = "listwise"

    # ── Execute ──
    if effective in ("cross_encoder", "cascade") and has_ce:
        result = _cross_encoder_rerank(query, cand, model, final_k)
    elif effective == "listwise":
        result = _llm_listwise_rerank(query, cand, final_k)
    else:
        result = _llm_listwise_rerank(query, cand, final_k)

    _score_norm(result)
    return result
