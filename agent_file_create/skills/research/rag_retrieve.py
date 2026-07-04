"""RAG retrieve skill — searches the internal knowledge base.

Supports multi-KB parallel retrieval, document filtering, and optional time range.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time

from agent_file_create.skills.base import SkillResult, SkillMeta, skill

logger = logging.getLogger(__name__)


async def _search_single_kb(
    kb_obj,
    kb_name: str,
    query: str,
    top_k: int,
    search_method: str,
    doc_filter: list[str] | None,
) -> list:
    """Search a single KB and return hits. Returns empty list on failure."""
    try:
        K = max(10, top_k * 2)
        if search_method == "search":
            hits = kb_obj.search(kb=kb_name, query=query, top_k=K)
        elif search_method == "search_auto":
            hits = kb_obj.search_auto(kb=kb_name, query=query, top_k=K)
        elif search_method == "search_hyde":
            hits = kb_obj.search_hyde(kb=kb_name, query=query, top_k=K)
        elif search_method == "search_expanded":
            hits = kb_obj.search_expanded(kb=kb_name, query=query, top_k=K)
        else:
            hits = kb_obj.search_adaptive(kb=kb_name, query=query, top_k=K)

        # Apply document filter if specified
        if doc_filter and hits:
            doc_set = set(str(d).strip() for d in doc_filter if d)
            if doc_set:
                hits = [h for h in hits if str(h.doc_id or "") in doc_set]

        return hits
    except Exception as e:
        logger.debug("rag_retrieve kb_search_failed kb=%s err=%s", kb_name, e)
        return []


async def _execute_rag_retrieve(
    query: str,
    kb: str = "default",
    kb_names: list[str] | None = None,
    top_k: int = 5,
    search_method: str = "search_adaptive",
    doc_filter: list[str] | None = None,
    time_range: dict | None = None,
    **kwargs,
) -> SkillResult:
    """Search the internal RAG knowledge base, optionally across multiple KBs.

    Args:
        query: Search query string.
        kb: Primary KB name (used when kb_names is empty).
        kb_names: Optional list of KB names for parallel multi-KB search.
        top_k: Number of result chunks to return after reranking.
        search_method: One of search, search_adaptive, search_auto, search_hyde, search_expanded.
        doc_filter: Optional list of document IDs to restrict search to.
        time_range: Optional dict with 'start' and 'end' timestamps (unix seconds).

    Returns:
        SkillResult with structured hit data and a human-readable summary.
    """
    if not query or not query.strip():
        return SkillResult(success=False, error="检索问题为空")

    # Resolve KB list
    kb_list: list[str] = []
    if kb_names and isinstance(kb_names, list):
        kb_list = [str(k).strip() for k in kb_names if str(k).strip()]
    if not kb_list:
        kb_list = [str(kb or "default").strip()]

    try:
        from agent_file_create.rag.kb import KnowledgeBase
        from agent_file_create.rag.reranker import rerank

        kb_obj = KnowledgeBase()
        method = str(search_method or "search_adaptive").strip()

        # ── Parallel multi-KB search ──────────────────────────────────────
        t0 = _time.perf_counter()
        if len(kb_list) == 1:
            all_hits = await _search_single_kb(
                kb_obj, kb_list[0], query, top_k, method, doc_filter,
            )
        else:
            tasks = [
                _search_single_kb(kb_obj, kbn, query, top_k, method, doc_filter)
                for kbn in kb_list
            ]
            hit_lists = await asyncio.gather(*tasks)
            all_hits = []
            seen_doc_ids: set[str] = set()
            for hits in hit_lists:
                for h in hits:
                    did = str(h.doc_id or "")
                    if did not in seen_doc_ids:
                        seen_doc_ids.add(did)
                        all_hits.append(h)
            # Re-rank merged results by score
            all_hits.sort(key=lambda x: float(x.score or 0), reverse=True)
            all_hits = all_hits[:max(10, top_k * 2)]

        elapsed = _time.perf_counter() - t0

        # ── Time range filter ─────────────────────────────────────────────
        if time_range and isinstance(time_range, dict) and all_hits:
            try:
                start_ts = float(time_range.get("start") or 0)
                end_ts = float(time_range.get("end") or _time.time() + 86400 * 365)
                # Only filter if at least one bound is meaningful
                if start_ts > 0 or end_ts < _time.time() + 86400 * 365:
                    filtered = []
                    for h in all_hits:
                        hit_ts = float(getattr(h, "updated_at", 0) or 0)
                        if hit_ts == 0 or (start_ts <= hit_ts <= end_ts):
                            filtered.append(h)
                    if filtered:
                        all_hits = filtered
            except Exception:
                pass  # Time filtering is best-effort

        # ── Rerank ────────────────────────────────────────────────────────
        if all_hits:
            try:
                all_hits = rerank(query, all_hits, top_k=top_k)
            except Exception:
                all_hits = all_hits[:top_k]

        if not all_hits:
            kb_desc = "、".join(kb_list)
            return SkillResult(
                success=True,
                summary=f"知识库「{kb_desc}」中未找到与「{query}」相关的内容。",
                data={"query": query, "kb": kb_list, "hits": [], "count": 0,
                      "elapsed_ms": int(elapsed * 1000)},
            )

        # ── Build grouped summary ──────────────────────────────────────────
        groups: dict[str, list] = {}
        for h in all_hits:
            doc = str(h.doc_id or "未知文档")[:80]
            groups.setdefault(doc, []).append(h)

        lines = [
            f"知识库检索: {query}",
            f"搜索范围: {', '.join(kb_list)}",
            f"命中 {len(all_hits)} 个相关片段 (检索 {int(elapsed * 1000)}ms):\n",
        ]
        for doc_name, doc_hits in groups.items():
            lines.append(f"### [{doc_name}] ({len(doc_hits)} 段)")
            for i, h in enumerate(doc_hits, 1):
                section = str(h.section_path or "")[:60]
                snippet = (h.content or "")[:300].replace("\n", " ")
                src = f"§{section}" if section else ""
                lines.append(f"  {i}. {src}")
                lines.append(f"     {snippet}...")
            lines.append("")

        return SkillResult(
            success=True,
            summary="\n".join(lines),
            data={
                "query": query,
                "kb": kb_list,
                "search_method": method,
                "hits": [
                    {
                        "doc_id": h.doc_id,
                        "section_path": h.section_path,
                        "score": float(h.score),
                        "snippet": (h.content or "")[:300],
                        "source": getattr(h, "source", ""),
                    }
                    for h in all_hits
                ],
                "count": len(all_hits),
                "elapsed_ms": int(elapsed * 1000),
                "groups": {k: len(v) for k, v in groups.items()},
            },
            tokens_used=sum(len(h.content or "") for h in all_hits) // 3,
        )
    except Exception as exc:
        return SkillResult(
            success=False,
            error=f"RAG检索失败: {str(exc)[:200]}",
        )


SKILL_META = skill(
    name="rag_retrieve",
    description="从内部知识库检索相关文档和资料，支持多知识库并行检索、文档过滤",
    category="research",
    stage="both",
    parameters={
        "query": {"type": "string", "description": "检索查询语句"},
        "kb": {"type": "string", "description": "主知识库名称", "default": "default"},
        "kb_names": {
            "type": "array",
            "description": "多个知识库名称列表，为空则使用 kb 参数",
            "default": None,
        },
        "top_k": {"type": "integer", "description": "返回片段数量", "default": 5},
        "search_method": {
            "type": "string",
            "description": "检索方法: search, search_adaptive, search_auto, search_hyde, search_expanded",
            "default": "search_adaptive",
        },
        "doc_filter": {
            "type": "array",
            "description": "按文档ID过滤，只检索指定文档",
            "default": None,
        },
        "time_range": {
            "type": "object",
            "description": "时间范围过滤: {'start': unix_ts, 'end': unix_ts}",
            "default": None,
        },
    },
    timeout_s=60,
    max_retries=1,
)(_execute_rag_retrieve)
