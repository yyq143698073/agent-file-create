import hashlib
import json
import logging
import math
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent_file_create.config import (
    CONTENT_API_ENDPOINT,
    CONTENT_API_KEY,
    CONTENT_API_STYLE,
    CONTENT_MODEL_NAME,
)
from agent_file_create.llm_client import call_llm
from agent_file_create.llm_factory import get_chat_model
from agent_file_create.rag.chunker import chunk_text
from agent_file_create.rag.embedder import embed_texts
from agent_file_create.rag.reranker import rerank
from agent_file_create.rag.store import Hit, default_store

# ── Extracted sub-modules ────────────────────────────────────────────────────
from agent_file_create.rag._utils import (     # noqa: E402
    safe_json as _safe_json,
    normalize_kb as _normalize_kb,
    guess_is_markdown as _guess_is_markdown,
    tokenize as _tokenize,
    bm25_scores as _bm25_scores,
    rrf_ranks as _rrf_ranks,
    query_has_numbers as _query_has_numbers,
    query_has_technical_terms as _query_has_technical_terms,
    query_has_specialized_terms as _query_has_specialized_terms,
    query_concreteness as _query_concreteness,
    title_keyword_boost as _title_keyword_boost,
    mmr_rerank as _mmr_rerank,
    extract_entities as _extract_entities,
    split_sentences as _split_sentences,
    read_any_text as _read_any_text,
)
from agent_file_create.rag._prompts import (   # noqa: E402
    Citation,
    Answer,
    ANSWER_PROMPT as _ANSWER_PROMPT,
    ANSWER_COT_PROMPT as _ANSWER_COT_PROMPT,
    HYDE_PROMPT as _HYDE_PROMPT,
    DECOMPOSE_PROMPT as _DECOMPOSE_PROMPT,
    QUERY_REWRITE_PROMPT as _QUERY_REWRITE_PROMPT,
    MULTI_QUERY_PROMPT as _MULTI_QUERY_PROMPT,
    STEPBACK_PROMPT as _STEPBACK_PROMPT,
    QUERY_ROUTE_PROMPT as _QUERY_ROUTE_PROMPT,
    METADATA_FILTER_PROMPT as _METADATA_FILTER_PROMPT,
)

logger = logging.getLogger

# All standalone helper functions moved to rag._utils
# All prompt templates and dataclasses moved to rag._prompts

class KnowledgeBase:
    def __init__(self, *, store=None) -> None:
        self.store = store or default_store()
        # Per-instance LRU caches (were module-level globals before)
        self._query_cache: "OrderedDict[str, list[float]]" = __import__('collections').OrderedDict()
        self._hyde_cache: "OrderedDict[str, str]" = __import__('collections').OrderedDict()
        self._content_embed_cache: "OrderedDict[str, list[float]]" = __import__('collections').OrderedDict()

    def list_kb(self) -> list[str]:
        return self.store.list_kb()

    def list_docs(self, *, kb: str) -> list[dict]:
        kb2 = _normalize_kb(kb)
        return self.store.list_docs(kb=kb2) if hasattr(self.store, "list_docs") else []

    def get_doc_text(self, *, kb: str, doc_id: str) -> str:
        """Reconstruct full text of a document from its chunks (ordered by chunk_index)."""
        kb2 = _normalize_kb(kb)
        did = str(doc_id or "").strip()
        if not did:
            return ""
        hits = self.search(kb=kb2, query=did, top_k=400)
        chunks = [(int(h.chunk_index or 0), h.content) for h in hits if str(h.doc_id or "") == did]
        chunks.sort(key=lambda x: x[0])
        return "\n\n".join([c[1] for c in chunks if c[1].strip()]).strip()

    def delete_doc(self, *, kb: str, doc_id: str) -> dict:
        kb2 = _normalize_kb(kb)
        did = str(doc_id or "").strip()
        if not did:
            return {"kb": kb2, "ok": False, "error": "missing doc_id"}
        try:
            self.store.delete_document(kb=kb2, doc_id=did)
            return {"kb": kb2, "doc_id": did, "ok": True}
        except Exception as e:
            return {"kb": kb2, "doc_id": did, "ok": False, "error": str(e)[:240]}

    def delete_kb(self, *, kb: str) -> dict:
        kb2 = _normalize_kb(kb)
        try:
            self.store.delete_kb(kb=kb2)
            return {"kb": kb2, "ok": True}
        except Exception as e:
            return {"kb": kb2, "ok": False, "error": str(e)[:240]}

    def kb_stats(self, *, kb: str) -> dict:
        kb2 = _normalize_kb(kb)
        try:
            return self.store.kb_stats(kb=kb2)
        except Exception as e:
            return {"kb": kb2, "doc_count": 0, "chunk_count": 0, "error": str(e)[:240]}

    # ── Summary Index ────────────────────────────────────────────────────────

    def ingest_summaries(
        self,
        *,
        kb: str,
        doc_id: str,
        title: str = "",
        source: str = "",
        summaries: list[dict],
    ) -> dict:
        """Index document/section summaries as lightweight searchable chunks.

        Each summary becomes a chunk with section_path = __summary__/{path}.
        These summary chunks serve as the coarse tier for hierarchical retrieval:
        search summaries first to locate relevant docs/sections, then drill down.

        Args:
            kb: Knowledge base name
            doc_id: Source document ID these summaries belong to
            summaries: [{"section_path": "§1.1", "content": "This section covers..."}, ...]
        """
        kb2 = _normalize_kb(kb)
        did = str(doc_id or "").strip()
        if not did or not summaries:
            return {"kb": kb2, "doc_id": did, "ok": False, "error": "missing doc_id or summaries"}

        chunks: list[dict] = []
        for i, s in enumerate(summaries):
            if not isinstance(s, dict):
                continue
            sec = str(s.get("section_path") or "").strip() or "/"
            content = str(s.get("content") or "").strip()
            if not content:
                continue
            cid = f"{kb2}::__summary__::{did}:{i}"
            chunks.append({
                "chunk_id": cid,
                "doc_id": did,
                "chunk_index": i,
                "section_path": f"__summary__/{sec}",
                "content": content,
                "meta": {"source": source, "title": title, "is_summary": True, "doc_type": "summary"},
            })

        if not chunks:
            return {"kb": kb2, "doc_id": did, "ok": False, "error": "empty_summaries"}

        # Embed summaries (lighter than full chunks, use smaller batch)
        embedding_ok = True
        try:
            vecs = embed_texts([c["content"] for c in chunks], timeout_s=60, max_batch=8)
            if len(vecs) != len(chunks):
                embedding_ok = False
        except Exception:
            embedding_ok = False

        for i, c in enumerate(chunks):
            c["embedding"] = vecs[i] if embedding_ok and i < len(vecs) else []

        try:
            n = self.store.upsert_chunks(kb=kb2, doc_id=f"{did}__summary__", chunks=chunks)
        except Exception as e:
            return {"kb": kb2, "doc_id": did, "ok": False, "error": "db_failed:" + str(e)[:180]}
        return {"kb": kb2, "doc_id": did, "ok": True, "summary_chunks": n}

    # ── Multi-granularity Hierarchical Search ────────────────────────────────

    def search_hierarchical(
        self,
        *,
        kb: str,
        query: str,
        top_k: int = 6,
        coarse_top_k: int = 3,
        filters: Optional[dict] = None,
    ) -> list[Hit]:
        """Two-tier hierarchical retrieval: coarse (summaries) → fine (chunks).

        Tier 1 (coarse): Search summary chunks to identify relevant documents and
        sections. Summaries are lightweight and cover broader scope, so they
        provide better routing accuracy than searching all chunks directly.

        Tier 2 (fine): Search full content chunks, but constrained to the
        documents/sections identified in Tier 1. This narrows the search space
        and improves precision.

        Returns final ranked hits from the fine search.
        """
        kb2 = _normalize_kb(kb)
        q = str(query or "").strip()
        if not q:
            return []

        # ── Tier 1: Coarse search on summary chunks ──
        summary_filter = dict(filters or {})
        summary_filter["doc_type"] = "summary"
        coarse_hits = self.search(kb=kb2, query=q, top_k=coarse_top_k, filters=summary_filter)

        if not coarse_hits:
            # Fall back to regular search if no summaries indexed
            return self.search(kb=kb2, query=q, top_k=top_k, filters=filters)

        # Collect the doc_ids and section_paths from the top summary hits
        coarse_docs: set[str] = set()
        coarse_sections: set[str] = set()
        for h in coarse_hits:
            coarse_docs.add(str(h.doc_id or "").replace("__summary__", ""))
            sec = str(h.section_path or "").replace("__summary__/", "").strip()
            if sec:
                # Use top-level section (first path segment) for routing
                top_sec = sec.split("/")[0].strip()
                if top_sec:
                    coarse_sections.add(top_sec)

        # ── Tier 2: Fine search within identified docs ──
        fine_hits: list[Hit] = []
        for doc_id in list(coarse_docs)[:3]:
            doc_filter = dict(filters or {})
            doc_filter["doc_id"] = doc_id
            try:
                doc_hits = self.search(kb=kb2, query=q, top_k=max(4, top_k), filters=doc_filter)
                fine_hits.extend(doc_hits)
            except Exception:
                pass

        if not fine_hits:
            # Fall back: regular unfiltered search
            return self.search(kb=kb2, query=q, top_k=top_k, filters=filters)

        # Deduplicate by chunk_id, merge scores
        seen: dict[str, Hit] = {}
        for h in fine_hits:
            cid = str(h.chunk_id or "")
            if cid in seen:
                if h.score > seen[cid].score:
                    seen[cid] = h
            else:
                seen[cid] = h

        scored = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return scored[:max(1, int(top_k or 0))]

    def check_embed_health(self) -> dict:
        """Validate embedding model connectivity."""
        try:
            vecs = embed_texts(["health check"], timeout_s=15, max_batch=1)
            if vecs and isinstance(vecs[0], list) and len(vecs[0]) > 0:
                return {"ok": True, "dim": len(vecs[0])}
            return {"ok": False, "error": "empty_embedding"}
        except Exception as e:
            return {"ok": False, "error": str(e)[:240]}

    def ingest_file(
        self,
        *,
        kb: str,
        file_path: str,
        doc_id: Optional[str] = None,
        title: str = "",
        source: str = "",
        doc_type: str = "",
        chunk_target_chars: int = 1200,
        chunk_overlap_chars: int = 120,
    ) -> dict:
        kb2 = _normalize_kb(kb)
        p = Path(file_path)
        did = str(doc_id or p.name).strip() or p.name
        ttl = str(title or p.stem or p.name).strip()
        src = str(source or file_path).strip()
        text = _read_any_text(file_path)
        if not text.strip():
            return {"kb": kb2, "doc_id": did, "ok": False, "error": "empty_text"}

        is_md = _guess_is_markdown(str(p))

        # Adaptive chunk sizing based on document type
        _review_kw = {"综述", "进展", "概念", "挑战", "应用", "研究进展", "技术综述"}
        _methods_kw = {"基于", "方法", "优化", "模型", "融合", "协同", "动态", "自适应"}
        is_review = any(kw in ttl for kw in _review_kw)
        is_methods = any(kw in ttl for kw in _methods_kw)
        if is_review and not is_methods:
            _tc, _oc = 1000, 150  # Review: larger chunks to preserve complete arguments
        elif is_methods and not is_review:
            _tc, _oc = 500, 80    # Methods: smaller chunks for precise retrieval
        else:
            _tc, _oc = 700, 100   # Mixed/default

        chunks = chunk_text(
            doc_id=did,
            text=text,
            title=ttl,
            is_markdown=is_md,
            target_chars=_tc,
            overlap_chars=_oc,
        )
        if not chunks:
            return {"kb": kb2, "doc_id": did, "ok": False, "error": "no_chunks"}

        # Content-hash embedding cache: avoid re-embedding identical content
        # Augment each chunk with document context before embedding
        def _augment(content: str, section: str) -> str:
            prefix = f"[文档: {ttl}]"
            if section:
                prefix += f" [章节: {section}]"
            return prefix + "\n" + content

        chunk_contents_raw = [c.content for c in chunks]
        chunk_contents = [_augment(c.content, c.section_path) for c in chunks]
        content_hashes: list[str] = []
        vecs: list[list[float]] = []
        uncached_idxs: list[int] = []
        for i, (raw, aug) in enumerate(zip(chunk_contents_raw, chunk_contents)):
            ch = hashlib.md5(aug.encode("utf-8")).hexdigest()
            content_hashes.append(ch)
            if ch in self._content_embed_cache:
                self._content_embed_cache.move_to_end(ch)
                vecs.append(list(self._content_embed_cache[ch]))
            else:
                vecs.append([])
                uncached_idxs.append(i)

        embedding_ok = True
        if uncached_idxs:
            try:
                fresh_vecs = embed_texts([chunk_contents[i] for i in uncached_idxs], timeout_s=90, max_batch=4)
                if len(fresh_vecs) == len(uncached_idxs):
                    for j, idx in enumerate(uncached_idxs):
                        v = fresh_vecs[j]
                        vecs[idx] = v
                        self._content_embed_cache[content_hashes[idx]] = list(v)
                        self._content_embed_cache.move_to_end(content_hashes[idx])
                        while len(self._content_embed_cache) > self._content_embed_cache_MAX:
                            self._content_embed_cache.popitem(last=False)
                else:
                    embedding_ok = False
            except Exception:
                embedding_ok = False

        dtype = str(doc_type or "").strip()
        meta_doc = {"file_name": p.name, "doc_type": dtype, "file_ext": p.suffix.lower().lstrip("."), "lang": "en" if re.search(r"[A-Za-z]", text[:800]) else "zh"}
        try:
            self.store.upsert_document(kb=kb2, doc_id=did, title=ttl, source=src, meta=meta_doc)
            self.store.delete_doc_chunks(kb=kb2, doc_id=did)
        except Exception as e:
            return {"kb": kb2, "doc_id": did, "ok": False, "error": "db_failed:" + str(e)[:180]}

        payload = []
        for i, c in enumerate(chunks):
            v = vecs[i] if embedding_ok and i < len(vecs) else []
            cid = f"{kb2}::{c.chunk_id}"
            payload.append(
                {
                    "chunk_id": cid,
                    "doc_id": c.doc_id,
                    "chunk_index": int(c.chunk_index),
                    "section_path": c.section_path,
                    "content": c.content,
                    "embedding": v,
                    "meta": {"source": src, "title": ttl, "doc_type": dtype, "file_ext": p.suffix.lower().lstrip("."), "lang": meta_doc.get("lang")},
                }
            )
        try:
            n = self.store.upsert_chunks(kb=kb2, doc_id=did, chunks=payload)
        except Exception as e:
            return {"kb": kb2, "doc_id": did, "ok": False, "error": "db_failed:" + str(e)[:180]}
        return {"kb": kb2, "doc_id": did, "ok": True, "chunks": n}

    def _cached_embed_query(self, query: str) -> Optional[list[float]]:
        key = hashlib.md5(query.encode("utf-8")).hexdigest()
        if key in self._query_cache:
            self._query_cache.move_to_end(key)
            return list(self._query_cache[key])
        return None

    def _set_cached_embed_query(self, query: str, vec: list[float]) -> None:
        key = hashlib.md5(query.encode("utf-8")).hexdigest()
        self._query_cache[key] = list(vec)
        self._query_cache.move_to_end(key)
        while len(self._query_cache) > 128:
            self._query_cache.popitem(last=False)

    # ── Recall-layer: adaptive, entity-aware, diversity, time-decay ────────

    def _analyze_query(self, q: str) -> dict:
        """Analyze query characteristics for adaptive recall tuning."""
        return {
            "has_numbers": _query_has_numbers(q),
            "has_tech_terms": _query_has_technical_terms(q),
            "concreteness": _query_concreteness(q),
            "length": len(q),
        }

    def search_adaptive(self, *, kb: str, query: str, top_k: int = 8,
                        filters: Optional[dict] = None,
                        enable_diversity: bool = True,
                        enable_title_boost: bool = True,
                        enable_adaptive_weights: bool = True,
                        hyde_query: str = "",
                        ) -> list[Hit]:
        """Adaptive recall: auto-tune fusion weights based on query characteristics.

        - Numbers/technical terms → more lexical + BM25 weight (exact match matters)
        - Abstract/long queries → more vector weight (semantic match matters)
        - Title keyword boost rewards chunks from documents with query-matching titles
        - MMR diversity ensures different sections are represented
        - hyde_query: if provided, used for vector embedding; original query for lexical
        """
        kb2 = _normalize_kb(kb)
        q = str(query or "").strip()
        if not q:
            return []

        # HyDE vector query: use hypothetical answer for better semantic match
        vec_q = str(hyde_query or "").strip() or q

        # Analyze query
        profile = self._analyze_query(q)

        # ── Adaptive channel allocation ──
        cap = 180
        # Concrete queries (has numbers/tech terms) → allocate more to lexical
        if profile["has_numbers"] or profile["has_tech_terms"]:
            vec_cand = min(max(30, int(top_k or 0) * 8), cap)
            lex_cand = min(max(40, int(top_k or 0) * 14), cap)
        elif profile["concreteness"] > 0.6:
            vec_cand = min(max(40, int(top_k or 0) * 12), cap)
            lex_cand = min(max(30, int(top_k or 0) * 10), cap)
        else:
            # Abstract queries → vector focus
            vec_cand = min(max(50, int(top_k or 0) * 14), cap)
            lex_cand = min(max(20, int(top_k or 0) * 6), cap)

        # ── Embedding & retrieval ──
        cache_key = vec_q if vec_q != q else q
        qv = self._cached_embed_query(cache_key)
        if qv is None:
            try:
                qv_list = embed_texts([vec_q], timeout_s=60, max_batch=1)
            except Exception:
                return []
            if not qv_list or not qv_list[0]:
                return []
            qv = qv_list[0]
            self._set_cached_embed_query(cache_key, qv)

        try:
            vec_hits = self.store.similarity_search(kb=kb2, query_embedding=qv, top_k=vec_cand, filters=filters)
        except TypeError:
            vec_hits = self.store.similarity_search(kb=kb2, query_embedding=qv, top_k=vec_cand)
        except Exception:
            vec_hits = []

        lex_hits: list[Hit] = []
        if hasattr(self.store, "lexical_search"):
            try:
                lex_hits = self.store.lexical_search(kb=kb2, query=q, top_k=lex_cand, filters=filters)
            except (TypeError, Exception):
                lex_hits = []

        # ── Merge candidates ──
        merged: dict[str, dict] = {}
        for h in vec_hits:
            merged[h.chunk_id] = {"hit": h, "vec": float(h.score), "lex": 0.0}
        for h in lex_hits:
            if h.chunk_id in merged:
                merged[h.chunk_id]["lex"] = max(float(merged[h.chunk_id].get("lex") or 0.0), float(h.score))
            else:
                merged[h.chunk_id] = {"hit": h, "vec": 0.0, "lex": float(h.score)}

        items = list(merged.values())
        if not items:
            return []

        # ── BM25 on candidates ──
        q_terms = _tokenize(q, max_terms=40)
        docs_terms = [_tokenize(it["hit"].content, max_terms=160) for it in items]
        bm25 = _bm25_scores(q_terms, docs_terms)

        # ── Adaptive fusion weights (calibrated for nomic-embed-text 768-dim) ──
        w_vec, w_bm, w_lex = 1.0, 1.0, 1.0
        if enable_adaptive_weights:
            has_specialized = _query_has_specialized_terms(q)
            if has_specialized:
                # Specialized compound terms → moderate vector preference
                w_vec, w_bm, w_lex = 1.15, 0.9, 0.9
            elif profile["has_numbers"] or profile["has_tech_terms"]:
                # Numbers/codes → slight lexical preference (exact match matters)
                w_vec, w_bm, w_lex = 0.95, 1.1, 1.1
            elif profile["concreteness"] < 0.4:
                # Abstract queries → slight vector preference
                w_vec, w_bm, w_lex = 1.1, 0.95, 0.95
            elif profile["length"] > 40:
                # Long queries → balanced with slight vector lean
                w_vec, w_bm, w_lex = 1.05, 1.0, 0.95

        # ── Weighted RRF fusion ──
        vec = [float(it.get("vec") or 0.0) for it in items]
        ids = [str(it["hit"].chunk_id or "") for it in items]
        vec_r = _rrf_ranks(list(zip(ids, vec)))
        bm_r = _rrf_ranks(list(zip(ids, bm25)))
        lx_r = _rrf_ranks([(str(it["hit"].chunk_id or ""), float(it.get("lex") or 0.0)) for it in items])

        scored: list[tuple[float, Hit]] = []
        k_rrf = 100.0
        for it, braw, vraw in zip(items, bm25, vec):
            h = it["hit"]
            hid = str(h.chunk_id or "")
            rv = float(vec_r.get(hid, 10_000))
            rb = float(bm_r.get(hid, 10_000))
            rl = float(lx_r.get(hid, 10_000))
            meta = dict(h.meta or {})
            meta_scores = {"vec": float(vraw), "bm25": float(braw), "lex": float(it.get("lex") or 0.0)}
            meta["scores"] = meta_scores

            # Weighted RRF
            s = (w_vec / (k_rrf + rv)) + (w_bm / (k_rrf + rb)) + (w_lex / (k_rrf + rl))

            scored.append((float(s), Hit(
                kb=h.kb, doc_id=h.doc_id, chunk_id=h.chunk_id,
                chunk_index=h.chunk_index, section_path=h.section_path,
                content=h.content, score=float(s), meta=meta,
            )))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [h for _, h in scored[: max(1, int(top_k or 0))]]

        # ── Title keyword boost ──
        if enable_title_boost:
            result = _title_keyword_boost(result, q, boost_factor=0.10)

        # ── Diversity reranking ──
        if enable_diversity and len(result) > 2:
            result = _mmr_rerank(result, lambda_param=0.7, top_k=max(1, int(top_k or 0)))

        return result

    def search_expanded(self, *, kb: str, query: str, top_k: int = 8, **kw) -> list[Hit]:
        """Query expansion with round-robin interleaving.

        Splits a complex query into sub-queries, runs search_adaptive on each,
        then round-robin interleaves results to preserve multi-angle diversity.

        Keyword args:
            qe_count: Number of sub-queries to generate (default 3, max 6).
        """
        kb2 = _normalize_kb(kb)
        q = str(query or "").strip()
        if not q:
            return []

        qe_count = int(kw.pop("qe_count", 0) or 0) or 3
        qe_count = max(2, min(qe_count, 6))

        # Generate sub-queries via LLM
        expansion_prompt = (
            f"你是一个查询扩展助手。请将用户的问题拆分为{qe_count}个不同角度的子查询，以帮助检索到更全面的相关文档。\n\n"
            f"用户问题：{q}\n\n"
            "请输出子查询，每行一个，用中文表达。只输出子查询本身，不要输出编号、解释或其他内容。\n"
            "每个子查询从不同的维度或视角来探索同一个主题。"
        )
        try:
            raw = call_llm(
                expansion_prompt,
                timeout_s=30,
                temperature=0.0,
                num_predict=qe_count * 100,
                system="你是一个中文文档处理助手。只输出子查询，每行一个。",
                api_style=CONTENT_API_STYLE,
                api_endpoint=CONTENT_API_ENDPOINT,
                api_key=CONTENT_API_KEY,
                model_name=CONTENT_MODEL_NAME,
            )
        except Exception:
            return self.search_adaptive(kb=kb2, query=q, top_k=top_k, **kw)

        sub_queries: list[str] = []
        for line in str(raw or "").strip().splitlines():
            sub = line.strip()
            if sub and len(sub) >= 3 and sub != q:
                sub_queries.append(sub)
        if not sub_queries:
            sub_queries = [q]
        sub_queries = sub_queries[:qe_count]

        # Run search for each sub-query, collecting per-query ranked lists
        per_query: list[list[Hit]] = []
        for sq in sub_queries:
            try:
                hits = self.search_adaptive(kb=kb2, query=sq, top_k=max(15, top_k * 2), **kw)
                if hits:
                    per_query.append(hits)
            except Exception:
                continue
        if not per_query:
            return []
        if len(per_query) == 1:
            return per_query[0][:top_k]

        # Round-robin interleave: pick from each sub-query in turn, deduplicating
        result: list[Hit] = []
        seen: set[str] = set()
        max_len = max(len(h) for h in per_query)
        for slot in range(max_len):
            for q_hits in per_query:
                if slot < len(q_hits):
                    h = q_hits[slot]
                    cid = str(h.chunk_id or "")
                    if cid not in seen:
                        seen.add(cid)
                        result.append(h)
                        if len(result) >= top_k:
                            return result
        return result

    def search_hyde(self, *, kb: str, query: str, top_k: int = 8, **kw) -> list[Hit]:
        """HyDE (Hypothetical Document Embeddings): generate a plausible answer
        via LLM, embed that answer, and use it for vector search.

        This bridges the semantic gap between questions ("how does X work?")
        and document content ("we propose a method that...").

        Keyword args:
            hyde_tokens: Override num_predict for the HyDE answer (default 200).
            qe_count: Number of sub-queries for query expansion (default 3).
        """
        kb2 = _normalize_kb(kb)
        q = str(query or "").strip()
        if not q:
            return []

        hyde_tokens = int(kw.pop("hyde_tokens", 0) or 0) or 200
        hyde_prompt = (
            "你是一篇学术论文的作者。请用2-3句话回答以下问题，"
            "使用学术论文的写作风格和术语，就像在写论文摘要一样。\n\n"
            f"问题：{q}\n\n"
            "假设的论文摘要片段："
        )
        try:
            raw = call_llm(
                hyde_prompt,
                timeout_s=30,
                temperature=0.3,
                num_predict=hyde_tokens,
                system="你是一个中文文档处理助手。用专业术语撰写摘要片段。",
                api_style=CONTENT_API_STYLE,
                api_endpoint=CONTENT_API_ENDPOINT,
                api_key=CONTENT_API_KEY,
                model_name=CONTENT_MODEL_NAME,
            )
        except Exception:
            return self.search_adaptive(kb=kb2, query=q, top_k=top_k, **kw)

        hyde_answer = str(raw or "").strip()
        if not hyde_answer or len(hyde_answer) < 5:
            return self.search_adaptive(kb=kb2, query=q, top_k=top_k, **kw)

        # Use HyDE answer for vector search, original query for lexical
        return self.search_adaptive(
            kb=kb2,
            query=q,          # original query → BM25, lexical
            hyde_query=hyde_answer,  # hypothetical answer → vector
            top_k=top_k,
            **kw,
        )

    def search_small_to_big(self, *, kb: str, query: str, top_k: int = 8,
                            window_size: int = 2, filters: Optional[dict] = None,
                            ) -> list[Hit]:
        """Small-to-Big retrieval: rank at sentence level, return at paragraph level.

        Process:
        1. Retrieve candidate chunks via adaptive search (candidate pool)
        2. Split each chunk into sentences
        3. Find most relevant sentences (embedding cosine or lexical overlap)
        4. Expand a ±window_size sentence window around the best sentence
        5. Return expanded contexts as Hits

        This bridges the gap between embedding-friendly small units (sentences)
        and user-friendly large context (paragraphs).
        """
        kb2 = _normalize_kb(kb)
        q = str(query or "").strip()
        if not q:
            return []

        # Retrieve a larger candidate pool
        candidates = self.search(kb=kb2, query=q, top_k=min(40, (top_k or 0) * 4), filters=filters)
        if not candidates:
            return []

        # Embed query once for sentence scoring
        qv = self._cached_embed_query(q)
        if qv is None:
            try:
                qv_list = embed_texts([q], timeout_s=60, max_batch=1)
            except Exception:
                qv_list = None
            if qv_list and qv_list[0]:
                qv = qv_list[0]
                self._set_cached_embed_query(q, qv)

        # Split each candidate into sentences, find best sentence per chunk
        scored_windows: list[tuple[float, str, str, int, dict]] = []  # (score, text, doc_id, chunk_idx, meta)
        w = max(1, int(window_size or 0))

        for h in candidates:
            sents = _split_sentences(h.content)
            if not sents:
                continue
            n = len(sents)

            # Score each sentence
            sent_scores: list[float] = []
            if qv and len(qv) > 0:
                # Embed all sentences and compute cosine similarity
                try:
                    sent_vecs = embed_texts(sents, timeout_s=30, max_batch=len(sents))
                except Exception:
                    sent_vecs = None
                if sent_vecs:
                    for sv in sent_vecs:
                        if sv and len(sv) == len(qv):
                            dot = sum(a * b for a, b in zip(qv, sv))
                            na = math.sqrt(sum(a * a for a in qv))
                            nb = math.sqrt(sum(b * b for b in sv))
                            sent_scores.append(dot / (na * nb + 1e-8) if na > 0 and nb > 0 else 0.0)
                        else:
                            sent_scores.append(0.0)
                else:
                    # Embedding failed — fallback to lexical overlap
                    q_lower = q.lower()
                    for sent in sents:
                        overlap = sum(1 for ch in q_lower if ch in sent.lower())
                        sent_scores.append(overlap / max(1, len(q)))
            else:
                # No embedding — use lexical overlap
                q_lower = q.lower()
                for sent in sents:
                    overlap = sum(1 for ch in q_lower if ch in sent.lower())
                    sent_scores.append(overlap / max(1, len(q)))

            if not sent_scores:
                continue

            # Find best sentence (highest score)
            best_idx = max(range(len(sent_scores)), key=lambda i: sent_scores[i])
            best_score = sent_scores[best_idx]

            # Expand window around best sentence
            start = max(0, best_idx - w)
            end = min(n, best_idx + w + 1)
            window_text = " ".join(sents[start:end])

            meta = dict(h.meta or {})
            meta["s2b_best_sent_idx"] = best_idx
            meta["s2b_window"] = [start, end]
            meta["s2b_sent_score"] = round(float(best_score), 4)
            scored_windows.append((best_score * float(h.score), window_text, str(h.doc_id or ""), int(h.chunk_index or 0), meta))

        scored_windows.sort(key=lambda x: x[0], reverse=True)
        result: list[Hit] = []
        seen_windows: set[str] = set()
        for score, text, doc_id, chunk_idx, meta in scored_windows:
            # Deduplicate near-identical windows
            key = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
            if key in seen_windows:
                continue
            seen_windows.add(key)
            result.append(Hit(kb=kb2, doc_id=doc_id, chunk_id=f"{doc_id}:s2b:{chunk_idx}",
                              chunk_index=chunk_idx, section_path="", content=text, score=score, meta=meta))
            if len(result) >= top_k:
                break

        return result

    def search(self, *, kb: str, query: str, top_k: int = 8, filters: Optional[dict] = None) -> list[Hit]:
        kb2 = _normalize_kb(kb)
        q = str(query or "").strip()
        if not q:
            return []

        qv = self._cached_embed_query(q)
        if qv is None:
            try:
                qv_list = embed_texts([q], timeout_s=60, max_batch=1)
            except Exception:
                return []
            if not qv_list or not qv_list[0]:
                return []
            qv = qv_list[0]
            self._set_cached_embed_query(q, qv)
        cap = 160
        vec_cand = min(max(30, int(top_k or 0) * 10), cap)
        lex_cand = min(max(30, int(top_k or 0) * 10), cap)
        try:
            vec_hits = self.store.similarity_search(kb=kb2, query_embedding=qv, top_k=vec_cand, filters=filters)
        except TypeError:
            vec_hits = self.store.similarity_search(kb=kb2, query_embedding=qv, top_k=vec_cand)
        except Exception:
            vec_hits = []  # pgvector error (e.g., empty stored vectors) → lexical fallback
        lex_hits: list[Hit] = []
        if hasattr(self.store, "lexical_search"):
            try:
                lex_hits = self.store.lexical_search(kb=kb2, query=q, top_k=lex_cand, filters=filters)
            except TypeError:
                try:
                    lex_hits = self.store.lexical_search(kb=kb2, query=q, top_k=lex_cand)
                except Exception:
                    lex_hits = []
            except Exception:
                lex_hits = []

        merged: dict[str, dict] = {}
        for h in vec_hits:
            merged[h.chunk_id] = {"hit": h, "vec": float(h.score), "lex": 0.0}
        for h in lex_hits:
            if h.chunk_id in merged:
                merged[h.chunk_id]["lex"] = max(float(merged[h.chunk_id].get("lex") or 0.0), float(h.score))
            else:
                merged[h.chunk_id] = {"hit": h, "vec": 0.0, "lex": float(h.score)}

        items = list(merged.values())
        if not items:
            return []
        q_terms = _tokenize(q, max_terms=40)
        docs_terms = [_tokenize(it["hit"].content, max_terms=160) for it in items]
        bm25 = _bm25_scores(q_terms, docs_terms)
        vec = [float(it.get("vec") or 0.0) for it in items]
        ids = [str(it["hit"].chunk_id or "") for it in items]
        vec_r = _rrf_ranks(list(zip(ids, vec)))
        bm_r = _rrf_ranks(list(zip(ids, bm25)))
        lx_r = _rrf_ranks([(str(it["hit"].chunk_id or ""), float(it.get("lex") or 0.0)) for it in items])
        scored: list[tuple[float, Hit]] = []
        k_rrf = 100.0
        for it, braw, vraw in zip(items, bm25, vec):
            h = it["hit"]
            hid = str(h.chunk_id or "")
            rv = float(vec_r.get(hid, 10_000))
            rb = float(bm_r.get(hid, 10_000))
            rl = float(lx_r.get(hid, 10_000))
            meta = dict(h.meta or {})
            meta_scores = {"vec": float(vraw), "bm25": float(braw), "lex": float(it.get("lex") or 0.0)}
            meta["scores"] = meta_scores
            s = (1.0 / (k_rrf + rv)) + (1.0 / (k_rrf + rb)) + (1.0 / (k_rrf + rl))
            scored.append((float(s), Hit(kb=h.kb, doc_id=h.doc_id, chunk_id=h.chunk_id, chunk_index=h.chunk_index, section_path=h.section_path, content=h.content, score=float(s), meta=meta)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [h for _, h in scored[: max(1, int(top_k or 0))]]

    def search_auto(self, *, kb: str, query: str, top_k: int = 8,
                    filters: Optional[dict] = None, **kw) -> list[Hit]:
        """Adaptive query router: picks HyDE or Query Expansion based on query features.

        Decision logic (derived from benchmark on 25 academic papers):
        - Long queries (>30 chars) → HyDE (200 tokens): better at complex semantic matching
        - Medium/short queries (≤30 chars) → QE (2 sub-queries): better at broad coverage
        - Falls back to adaptive search on any failure.

        Benchmarked results:
        - Long queries: HyDE R@5=0.90 (+34% vs baseline 0.67)
        - Medium queries: QE R@5=0.96 (+3% vs baseline 0.93)
        - Hard queries: QE R@5=0.86 (+34% vs baseline 0.64)
        """
        q = str(query or "").strip()
        if not q:
            return []

        QE_LONG_THRESHOLD = 30  # chars — above this, use HyDE

        # ── Route ──
        if len(q) > QE_LONG_THRESHOLD:
            method = "search_hyde"
            kw["hyde_tokens"] = kw.get("hyde_tokens", 200)
            logger.info("search_auto route=hyde query_len=%d query=%.60s", len(q), q)
        else:
            method = "search_expanded"
            kw["qe_count"] = kw.get("qe_count", 2)
            logger.info("search_auto route=qe query_len=%d query=%.60s", len(q), q)

        try:
            if method == "search_hyde":
                return self.search_hyde(kb=kb, query=q, top_k=top_k, **kw)
            else:
                return self.search_expanded(kb=kb, query=q, top_k=top_k, **kw)
        except Exception:
            logger.warning("search_auto fallback to adaptive len=%d", len(q))
            return self.search_adaptive(kb=kb, query=q, top_k=top_k, **kw)

    def _assemble_context(
        self, hits: list[Hit], base_k: int, max_context_chars: int = 5200
    ) -> tuple[str, list[Citation]]:
        """Shared context assembly: dedup by doc → merge adjacent → sort → truncate.

        Used by answer(), answer_with_reasoning(), and decompose_and_answer().
        """
        citations: list[Citation] = []
        picked: list[Hit] = []
        per_doc: dict[str, int] = {}
        overflow: list[Hit] = []
        for h in hits:
            did = str(h.doc_id or "")
            c = int(per_doc.get(did, 0))
            if c < 2:
                per_doc[did] = c + 1
                picked.append(h)
            else:
                overflow.append(h)
            if len(picked) >= base_k:
                break
        if len(picked) < base_k:
            for h in overflow:
                picked.append(h)
                if len(picked) >= base_k:
                    break

        by_doc: dict[str, list[Hit]] = {}
        for h in picked:
            by_doc.setdefault(str(h.doc_id or ""), []).append(h)
        segments: list[list[Hit]] = []
        for did, hs in by_doc.items():
            hs.sort(key=lambda x: int(x.chunk_index or 0))
            cur: list[Hit] = []
            last_i: Optional[int] = None
            for h in hs:
                ci = int(h.chunk_index or 0)
                if cur and (last_i is not None) and (ci - last_i <= 1):
                    cur.append(h)
                    last_i = ci
                    continue
                if cur:
                    segments.append(cur)
                cur = [h]
                last_i = ci
            if cur:
                segments.append(cur)
        segments.sort(key=lambda g: max(float(x.score) for x in g), reverse=True)

        blocks: list[str] = []
        used = 0
        idx2 = 1
        for group in segments:
            if not group:
                continue
            h0 = group[0]
            meta0 = h0.meta if isinstance(h0.meta, dict) else {}
            sec0 = str(h0.section_path or "").strip() or str((meta0 or {}).get("title") or "").strip() or "-"
            did = str(h0.doc_id or "").strip() or "-"
            parts: list[str] = []
            for h in group:
                snip = (h.content or "").strip()
                if len(snip) > 900:
                    snip = snip[:900] + "…"
                meta = h.meta if isinstance(h.meta, dict) else {}
                sec = str(h.section_path or "").strip() or str((meta or {}).get("title") or "").strip() or sec0
                citations.append(Citation(doc_id=h.doc_id, chunk_id=h.chunk_id, section_path=sec, score=float(h.score), snippet=snip))
                parts.append(snip)
            body = "\n\n".join([p for p in parts if p]).strip()
            score = max(float(x.score) for x in group)
            head = f"[{idx2}] doc={did} section={sec0} score={score:.3f}"
            block = (head + "\n" + body).strip() if body else head
            if used + len(block) + 2 > int(max_context_chars or 0):
                break
            blocks.append(block)
            used += len(block) + 2
            idx2 += 1

        return "\n\n".join(blocks).strip(), citations

    # ── Query Layer: Rewrite, Multi-Query, Step-Back, Route, Filter ──────────

    def _llm_quick(self, prompt_template, inputs: dict, *, max_tokens: int = 200) -> str:
        """Lightweight LLM call for query-layer operations."""
        try:
            chain = prompt_template | self._get_answer_llm_for_short(max_tokens) | StrOutputParser()
            return (chain.invoke(inputs) or "").strip()
        except Exception:
            return ""

    def _get_answer_llm_for_short(self, max_tokens: int = 200):
        """Short-timeout LLM for quick query operations."""
        return get_chat_model(
            style=CONTENT_API_STYLE,
            model=CONTENT_MODEL_NAME,
            endpoint=CONTENT_API_ENDPOINT,
            api_key=CONTENT_API_KEY,
            temperature=0.0,
            max_tokens=int(max_tokens),
            timeout_s=30,
        )

    def rewrite_query(self, question: str) -> str:
        """Rewrite a casual/spoken query into a precise search query.

        Example: "那个报销怎么搞的" → "费用报销流程的具体步骤是什么？"
        """
        q = str(question or "").strip()
        if len(q) < 8:
            return q
        result = self._llm_quick(_QUERY_REWRITE_PROMPT, {"question": q}, max_tokens=150)
        return result if result and len(result) >= 4 else q

    def generate_query_variants(self, question: str, n: int = 3) -> list[str]:
        """Generate multiple phrasings of the same question for multi-query retrieval.

        Example: "研发投入占比" →
          ["2024年研发投入占总预算比例", "研发支出在财务预算中的比重", "R&D预算分配情况"]
        """
        q = str(question or "").strip()
        if len(q) < 6:
            return [q]
        result = self._llm_quick(_MULTI_QUERY_PROMPT, {"question": q, "n": str(n)}, max_tokens=250)
        if not result:
            return [q]
        variants: list[str] = []
        for line in result.splitlines():
            v = re.sub(r"^\d+[\.\)、\s]*", "", line).strip()
            if v and len(v) >= 4:
                variants.append(v)
        if not variants:
            return [q]
        # Deduplicate while preserving order
        seen: set[str] = set()
        uniq: list[str] = []
        for v in variants:
            if v.lower() not in seen:
                seen.add(v.lower())
                uniq.append(v)
        return uniq[:n]

    def generate_stepback_question(self, question: str) -> str:
        """Generate a higher-level background question for broader retrieval.

        Example: "2024年研发投入占比下降原因" →
          "公司研发投入的影响因素和决策依据有哪些？"
        """
        q = str(question or "").strip()
        if len(q) < 15:
            return q
        result = self._llm_quick(_STEPBACK_PROMPT, {"question": q}, max_tokens=150)
        return result if result and len(result) >= 6 else q

    def classify_query(self, question: str) -> str:
        """Classify query type for routing.

        Returns: fact_lookup | comparison | summary | multi_document | how_to
        """
        q = str(question or "").strip()
        if len(q) < 5:
            return "fact_lookup"
        # Fast heuristic pre-check to skip LLM call
        if any(kw in q for kw in ["比较", "对比", "区别", "异同", "vs", "VS", "优缺点"]):
            return "comparison"
        if any(kw in q for kw in ["总结", "汇总", "概述", "概括", "归纳"]):
            return "summary"
        if any(kw in q for kw in ["怎么", "如何", "步骤", "流程", "方法", "操作"]):
            return "how_to"
        result = self._llm_quick(_QUERY_ROUTE_PROMPT, {"question": q}, max_tokens=30)
        r = (result or "").strip().lower()
        valid = {"fact_lookup", "comparison", "summary", "multi_document", "how_to"}
        return r if r in valid else "fact_lookup"

    def extract_metadata_filters(self, question: str) -> dict:
        """Extract implicit metadata filters from natural language.

        Example: "制度类文档中的风险管理政策" → {"doc_type": "制度"}
                 "2024年的财务报告" → {"time_range": "2024"}
        """
        q = str(question or "").strip()
        if len(q) < 6:
            return {}
        result = self._llm_quick(_METADATA_FILTER_PROMPT, {"question": q}, max_tokens=150)
        if not result or not result.startswith("{"):
            return {}
        try:
            import json as _json
            obj = _json.loads(result)
            out: dict = {}
            if isinstance(obj, dict):
                if isinstance(obj.get("doc_type"), str) and obj["doc_type"].strip():
                    out["doc_type"] = obj["doc_type"].strip()
                if isinstance(obj.get("source"), str) and obj["source"].strip():
                    out["source"] = obj["source"].strip()
                if isinstance(obj.get("time_range"), str) and obj["time_range"].strip():
                    out["time_range"] = obj["time_range"].strip()
            return out
        except Exception:
            return {}

    # ── Composite query strategies ────────────────────────────────────────────

    def search_multi_query(
        self, *, kb: str, question: str, top_k: int = 6, n_variants: int = 3,
        filters: Optional[dict] = None,
    ) -> list[Hit]:
        """Multi-Query retrieval: generate variants → search each → RRF merge.

        Better recall than single-query search for ambiguous or broad questions.
        """
        variants = self.generate_query_variants(question, n=n_variants)
        if len(variants) <= 1:
            return self.search(kb=kb, query=question, top_k=top_k, filters=filters)

        # Search with each variant and collect all hits
        all_hits: dict[str, tuple[Hit, float]] = {}  # chunk_id → (hit, rrf_score)
        for rank, variant in enumerate(variants):
            hits = self.search(kb=kb, query=variant, top_k=max(10, top_k * 2), filters=filters)
            for hit in hits:
                cid = str(hit.chunk_id or "")
                # RRF contribution: higher rank = lower number = higher score
                rrf = 1.0 / (60.0 + float(rank + 1))
                if cid in all_hits:
                    _, prev = all_hits[cid]
                    all_hits[cid] = (hit, prev + rrf)
                else:
                    all_hits[cid] = (hit, rrf)

        scored = sorted(all_hits.values(), key=lambda x: x[1], reverse=True)
        return [h for h, _ in scored[:max(1, int(top_k or 0))]]

    def search_with_stepback(
        self, *, kb: str, question: str, top_k: int = 6,
        filters: Optional[dict] = None,
    ) -> list[Hit]:
        """Step-Back retrieval: search with original + abstracted question, merge.

        The abstracted question retrieves broader context that may not match
        the specific keywords but provides useful background knowledge.
        """
        stepback = self.generate_stepback_question(question)
        if not stepback or stepback == question:
            return self.search(kb=kb, query=question, top_k=top_k, filters=filters)

        orig_hits = self.search(kb=kb, query=question, top_k=max(10, top_k * 2), filters=filters)
        sb_hits = self.search(kb=kb, query=stepback, top_k=max(10, top_k * 2), filters=filters)

        # Merge via RRF: original query results get higher weight
        merged: dict[str, tuple[Hit, float]] = {}
        for rank, h in enumerate(orig_hits):
            merged[h.chunk_id] = (h, 1.0 / (60.0 + rank + 1) * 1.2)  # 1.2x weight for original
        for rank, h in enumerate(sb_hits):
            cid = str(h.chunk_id or "")
            rrf = 1.0 / (60.0 + rank + 1)
            if cid in merged:
                hit, prev = merged[cid]
                merged[cid] = (hit, prev + rrf)
            else:
                merged[cid] = (h, rrf)

        scored = sorted(merged.values(), key=lambda x: x[1], reverse=True)
        return [h for h, _ in scored[:max(1, int(top_k or 0))]]

    # ── Context Compression (CRAG-style) ──────────────────────────────────────

    def compress_context(self, *, kb: str, query: str, top_k: int = 15,
                          max_chars: int = 1500) -> str:
        """Search → decompose chunks into sentences → filter irrelevant → recompose.

        Reduces noise from retrieved chunks before passing to the LLM, improving
        answer quality and reducing hallucinations from irrelevant content.
        """
        kb2 = _normalize_kb(kb)
        q = str(query or "").strip()
        if not q:
            return ""

        # Step 1: Retrieve a larger candidate pool
        hits = self.search_adaptive(kb=kb2, query=q, top_k=top_k)
        if not hits:
            hits = self.search(kb=kb2, query=q, top_k=top_k)
        if not hits:
            return ""

        # Step 2: Decompose — collect all sentences from hits
        all_sentences: list[str] = []
        for h in hits:
            content = str(h.content or "").strip()
            if not content:
                continue
            for sent in re.split(r"[。！？.!?\n]+", content):
                sent = sent.strip()
                if len(sent) >= 8:
                    all_sentences.append(sent)

        if not all_sentences:
            return "\n\n".join(str(h.content or "") for h in hits[:3])

        # Step 3: LLM picks relevant sentences (1 LLM call)
        sentences_text = "\n".join(
            f"[S{i+1}] {s}" for i, s in enumerate(all_sentences[:30]))
        prompt = (
            "从以下检索到的句子中筛选出与问题相关的句子，过滤无关内容。\n\n"
            f"问题：{q[:300]}\n\n候选句子：\n{sentences_text[:3000]}\n\n"
            "输出相关句子序号(逗号分隔,如S1,S3,S5)。无相关回复NONE。"
        )
        raw = call_llm(prompt, timeout_s=15, temperature=0.0, num_predict=100,
                       system="你是一个中文文档处理助手。只输出相关句子序号。")
        indices: set[int] = set()
        for m in re.findall(r"S?(\d+)", raw or ""):
            idx = int(m) - 1
            if 0 <= idx < len(all_sentences):
                indices.add(idx)

        if not indices:
            # Fallback: return raw top-3 hits
            return "\n\n".join(str(h.content or "") for h in hits[:3])

        # Step 4: Recompose — join relevant sentences
        refined = ""
        for i in sorted(indices):
            s = all_sentences[i]
            if len(refined) + len(s) + 2 > max_chars:
                break
            refined += s + "。"
        return refined.strip()

    def answer_smart(
        self,
        *,
        kb: str,
        question: str,
        top_k: int = 6,
        max_context_chars: int = 5200,
        filters: Optional[dict] = None,
    ) -> Answer:
        """Intelligent query routing: classify → rewrite → fetch → assemble → answer.

        Routes to the best retrieval strategy based on query type:
        - fact_lookup: rewritten query + direct search (fast, precise)
        - comparison: multi-query → merge results from both sides
        - summary: step-back search → broader context
        - how_to: rewritten query + metadata-filtered search
        - multi_document: multi-query + step-back combined
        """
        kb2 = _normalize_kb(kb)
        q = str(question or "").strip()
        base_k = max(3, int(top_k or 0))

        # Step 1: Extract metadata filters from natural language
        nl_filters = self.extract_metadata_filters(q)
        merged_filters: dict = dict(filters or {})
        merged_filters.update(nl_filters)

        # Step 2: Classify query type
        qtype = self.classify_query(q)

        # Step 3: Route to retrieval strategy
        if qtype == "comparison":
            hits = self.search_multi_query(kb=kb2, question=q, top_k=base_k, n_variants=4, filters=merged_filters)
        elif qtype == "summary":
            hits = self.search_with_stepback(kb=kb2, question=q, top_k=base_k, filters=merged_filters)
        elif qtype == "multi_document":
            # Combine multi-query + stepback for maximum coverage
            mq_hits = self.search_multi_query(kb=kb2, question=q, top_k=base_k * 2, n_variants=3, filters=merged_filters)
            sb_hits = self.search_with_stepback(kb=kb2, question=q, top_k=base_k, filters=merged_filters)
            merged: dict[str, Hit] = {}
            for h in mq_hits + sb_hits:
                if h.chunk_id not in merged or h.score > merged[h.chunk_id].score:
                    merged[h.chunk_id] = h
            hits = sorted(merged.values(), key=lambda x: x.score, reverse=True)[:max(1, base_k * 3)]
        elif qtype == "how_to":
            # Rewrite query for precision, then direct search
            rewritten = self.rewrite_query(q)
            hits = self.search(kb=kb2, query=rewritten, top_k=max(10, base_k * 3), filters=merged_filters)
        else:  # fact_lookup
            rewritten = self.rewrite_query(q)
            hits = self.search(kb=kb2, query=rewritten, top_k=max(10, base_k * 3), filters=merged_filters)

        # Step 4: Rerank → assemble → generate
        hits = rerank(q, hits, top_k=max(10, base_k * 3))
        ctx, citations = self._assemble_context(hits, base_k, max_context_chars)
        if not ctx:
            return Answer(kb=kb2, question=q, answer="未找到相关信息。建议你更换关键词、缩小范围（doc_type/doc_id），或先把相关文档上传入库。", citations=[])

        text = (
            _ANSWER_PROMPT
            | get_chat_model(
                style=CONTENT_API_STYLE, model=CONTENT_MODEL_NAME,
                endpoint=CONTENT_API_ENDPOINT, api_key=CONTENT_API_KEY,
                temperature=0.2, max_tokens=420, timeout_s=120,
            )
            | StrOutputParser()
        ).invoke({"context": ctx or "（未命中）", "question": q or "（空）", "kb": kb2})
        out = (text or "").strip()
        out = re.sub(r"^```[a-zA-Z]*\s*", "", out).strip()
        out = re.sub(r"\s*```$", "", out).strip()
        if not out:
            out = "当前未能生成可靠回答。建议你换个问法，或提供更具体的关键词/文档范围。"

        uniq: list[Citation] = []
        seen = set()
        for c in citations:
            k = str(c.chunk_id or "")
            if not k or k in seen:
                continue
            seen.add(k)
            uniq.append(c)
            if len(uniq) >= 6:
                break
        return Answer(kb=kb2, question=q, answer=out, citations=uniq)

    def answer(
        self,
        *,
        kb: str,
        question: str,
        top_k: int = 6,
        max_context_chars: int = 5200,
        filters: Optional[dict] = None,
    ) -> Answer:
        kb2 = _normalize_kb(kb)
        q = str(question or "").strip()
        base_k = max(3, int(top_k or 0))
        hits = self.search_with_context(kb=kb2, query=q, top_k=max(10, base_k * 3), context_window=2, filters=filters)
        hits = rerank(q, hits, top_k=max(10, base_k * 3))
        ctx, citations = self._assemble_context(hits, base_k, max_context_chars)
        if not ctx:
            return Answer(kb=kb2, question=q, answer="未找到相关信息。建议你更换关键词、缩小范围（doc_type/doc_id），或先把相关文档上传入库。", citations=[])

        text = (
            _ANSWER_PROMPT
            | get_chat_model(
                style=CONTENT_API_STYLE,
                model=CONTENT_MODEL_NAME,
                endpoint=CONTENT_API_ENDPOINT,
                api_key=CONTENT_API_KEY,
                temperature=0.2,
                max_tokens=420,
                timeout_s=120,
            )
            | StrOutputParser()
        ).invoke({"context": ctx or "（未命中）", "question": q or "（空）", "kb": kb2})
        out = (text or "").strip()
        out = re.sub(r"^```[a-zA-Z]*\s*", "", out).strip()
        out = re.sub(r"\s*```$", "", out).strip()
        if not out:
            out = "当前未能生成可靠回答。建议你换个问法，或提供更具体的关键词/文档范围。"
        elif out.startswith("{"):
            low = out.lower()
            if ("不确定" not in out) and ("未命中" not in out) and ("unknown" not in low):
                if len(ctx) >= 300:
                    out = "模型未返回可解析回答。已命中相关片段，请尝试缩小问题范围或指定文档后重试。"
                else:
                    out = "当前未能生成可靠回答。建议你换个问法，或提供更具体的关键词/文档范围。"

        uniq: list[Citation] = []
        seen = set()
        for c in citations:
            k = str(c.chunk_id or "")
            if not k or k in seen:
                continue
            seen.add(k)
            uniq.append(c)
            if len(uniq) >= 6:
                break
        return Answer(kb=kb2, question=q, answer=out, citations=uniq)

    def _fetch_neighbor_chunks(self, kb: str, doc_id: str, center_idx: int, window: int) -> list[Hit]:
        """Fetch neighboring chunks from the same document around center_idx.

        Uses direct doc_id lookup (get_chunks_by_doc_id) instead of
        embedding-based search, saving one embedding call per lookup.
        """
        kb2 = _normalize_kb(kb)
        did = str(doc_id or "").strip()
        if not did or window <= 0:
            return []
        try:
            all_hits = self.store.get_chunks_by_doc_id(kb=kb2, doc_id=did)
        except Exception:
            return []
        return [
            h for h in all_hits
            if abs(int(h.chunk_index or 0) - center_idx) <= window
            and int(h.chunk_index or 0) != center_idx
        ]

    def _fetch_parent_group(self, kb: str, doc_id: str, center_idx: int, parent_size: int = 4) -> list[Hit]:
        """Fetch all sibling chunks in the same parent group.

        Chunks are grouped into parent blocks of parent_size consecutive children
        (assigned during chunking). When a child chunk matches, retrieving its
        parent group provides broader context than individual neighbors.

        Parent group for chunk_index k = indices [start, start+parent_size)
          where start = k - (k % parent_size)
        """
        kb2 = _normalize_kb(kb)
        did = str(doc_id or "").strip()
        if not did or parent_size <= 1:
            return []
        try:
            all_hits = self.store.get_chunks_by_doc_id(kb=kb2, doc_id=did)
        except Exception:
            return []
        active_size = max(2, int(parent_size or 4))
        group_start = int(center_idx) - (int(center_idx) % active_size)
        group_end = group_start + active_size
        return [
            h for h in all_hits
            if group_start <= int(h.chunk_index or 0) < group_end
            and int(h.chunk_index or 0) != center_idx
        ]

    def search_with_context(
        self, *, kb: str, query: str, top_k: int = 6, context_window: int = 2, filters: Optional[dict] = None
    ) -> list[Hit]:
        """Search with parent-child context window.

        Returns top-k child chunks plus their neighboring chunks
        (context_window before/after) from the same document for richer context.
        """
        hits = self.search(kb=kb, query=query, top_k=top_k, filters=filters)
        if context_window <= 0 or not hits:
            return hits

        expanded: dict[str, Hit] = {}
        for h in hits:
            expanded[h.chunk_id] = h
            # Mark original hits with higher priority
            h_meta = dict(h.meta or {})
            h_meta["is_primary"] = True

        for h in hits:
            # Fetch ±window neighbors and same parent-group siblings
            neighbors = self._fetch_neighbor_chunks(
                kb=kb, doc_id=str(h.doc_id or ""),
                center_idx=int(h.chunk_index or 0), window=context_window,
            )
            parent_siblings = self._fetch_parent_group(
                kb=kb, doc_id=str(h.doc_id or ""),
                center_idx=int(h.chunk_index or 0), parent_size=4,
            )
            for nh in neighbors + parent_siblings:
                if nh.chunk_id not in expanded:
                    nh_meta = dict(nh.meta or {})
                    nh_meta["is_context"] = True
                    expanded[nh.chunk_id] = Hit(
                        kb=nh.kb, doc_id=nh.doc_id, chunk_id=nh.chunk_id,
                        chunk_index=nh.chunk_index, section_path=nh.section_path,
                        content=nh.content, score=nh.score * 0.85,
                        meta=nh_meta,
                    )

        out = list(expanded.values())
        out.sort(key=lambda x: (str(x.doc_id or ""), int(x.chunk_index or 0)))
        return out

    def _get_answer_llm(self):
        """Cached LLM instance for answer generation."""
        return get_chat_model(
            style=CONTENT_API_STYLE,
            model=CONTENT_MODEL_NAME,
            endpoint=CONTENT_API_ENDPOINT,
            api_key=CONTENT_API_KEY,
            temperature=0.2,
            max_tokens=420,
            timeout_s=120,
        )

    def _hyde_expand(self, question: str) -> str:
        """Generate a hypothetical answer and return it as an expanded search query.

        HyDE (Hypothetical Document Embeddings) bridges the vocabulary gap
        between short queries and document chunks by first generating a
        plausible answer, then using that answer's embedding for retrieval.

        Results are cached by question MD5 hash to avoid redundant LLM calls.
        """
        q = str(question or "").strip()
        if len(q) < 10:
            return q
        key = hashlib.md5(q.encode("utf-8")).hexdigest()
        if key in self._hyde_cache:
            self._hyde_cache.move_to_end(key)
            return self._hyde_cache[key]
        try:
            chain = _HYDE_PROMPT | self._get_answer_llm() | StrOutputParser()
            hypothetical = (chain.invoke({"question": q}) or "").strip()
            if hypothetical and len(hypothetical) >= 15:
                result = hypothetical[:600]
            else:
                result = q
        except Exception:
            result = q
        self._hyde_cache[key] = result
        self._hyde_cache.move_to_end(key)
        while len(self._hyde_cache) > self._hyde_cache_MAX:
            self._hyde_cache.popitem(last=False)
        return result

    def answer_with_reasoning(
        self,
        *,
        kb: str,
        question: str,
        top_k: int = 6,
        max_context_chars: int = 5200,
        use_hyde: bool = True,
        filters: Optional[dict] = None,
    ) -> Answer:
        """Answer with chain-of-thought reasoning and optional HyDE retrieval.

        Compared to answer(), this method:
        - Uses HyDE to expand the query before retrieval (if use_hyde=True)
        - Requires the LLM to show its reasoning steps before the final answer
        - Self-verifies each claim against retrieved evidence
        """
        kb2 = _normalize_kb(kb)
        q = str(question or "").strip()
        base_k = max(3, int(top_k or 0))

        # HyDE: expand query with hypothetical answer for better recall
        search_query = self._hyde_expand(q) if use_hyde else q

        hits = self.search(kb=kb2, query=search_query, top_k=max(10, base_k * 3), filters=filters)
        hits = rerank(q, hits, top_k=max(10, base_k * 3))
        ctx, citations = self._assemble_context(hits, base_k, max_context_chars)
        if not ctx:
            return Answer(kb=kb2, question=q, answer="未找到相关信息。建议你更换关键词、缩小范围（doc_type/doc_id），或先把相关文档上传入库。", citations=[])

        text = (
            _ANSWER_COT_PROMPT
            | get_chat_model(
                style=CONTENT_API_STYLE,
                model=CONTENT_MODEL_NAME,
                endpoint=CONTENT_API_ENDPOINT,
                api_key=CONTENT_API_KEY,
                temperature=0.1,
                max_tokens=900,
                timeout_s=120,
            )
            | StrOutputParser()
        ).invoke({"context": ctx or "（未命中）", "question": q or "（空）", "kb": kb2})
        out = (text or "").strip()
        out = re.sub(r"^```[a-zA-Z]*\s*", "", out).strip()
        out = re.sub(r"\s*```$", "", out).strip()
        if not out:
            out = "当前未能生成可靠回答。建议你换个问法，或提供更具体的关键词/文档范围。"

        uniq: list[Citation] = []
        seen = set()
        for c in citations:
            k = str(c.chunk_id or "")
            if not k or k in seen:
                continue
            seen.add(k)
            uniq.append(c)
            if len(uniq) >= 6:
                break
        return Answer(kb=kb2, question=q, answer=out, citations=uniq)

    def _decompose_question(self, question: str) -> list[str]:
        """Decompose a complex question into 2-4 simpler sub-questions."""
        q = str(question or "").strip()
        if len(q) < 20:
            return [q]
        try:
            chain = _DECOMPOSE_PROMPT | self._get_answer_llm() | StrOutputParser()
            result = (chain.invoke({"question": q}) or "").strip()
        except Exception:
            return [q]
        if not result or result.upper().startswith("SIMPLE"):
            return [q]
        subs: list[str] = []
        for line in result.splitlines():
            sub = re.sub(r"^\d+[\.\)、\s]*", "", line).strip()
            if sub and len(sub) >= 5:
                subs.append(sub)
        return subs if subs else [q]

    def decompose_and_answer(
        self,
        *,
        kb: str,
        question: str,
        top_k: int = 4,
        use_hyde: bool = True,
        filters: Optional[dict] = None,
    ) -> Answer:
        """For complex questions: decompose → retrieve per sub-Q → synthesize.

        Best for comparison, multi-aspect analysis, or cause-effect questions.
        """
        kb2 = _normalize_kb(kb)
        q = str(question or "").strip()
        subs = self._decompose_question(q)
        if len(subs) <= 1:
            return self.answer_with_reasoning(kb=kb, question=q, top_k=top_k, use_hyde=use_hyde, filters=filters)

        # Retrieve for each sub-question in parallel
        sub_results: list[dict] = []
        max_w = min(4, len(subs))
        if max_w > 1:
            with ThreadPoolExecutor(max_workers=max_w) as ex:
                futures = {
                    ex.submit(self.answer, kb=kb2, question=sub, top_k=max(3, top_k), max_context_chars=2400, filters=filters): sub
                    for sub in subs
                }
                for future in as_completed(futures):
                    try:
                        sub_ans = future.result()
                    except Exception:
                        sub_ans = Answer(kb=kb2, question=futures[future], answer="子问题检索失败", citations=[])
                    sub_results.append({"question": futures[future], "answer": sub_ans.answer, "citations": sub_ans.citations})
        else:
            for sub in subs:
                sub_ans = self.answer(kb=kb2, question=sub, top_k=max(3, top_k), max_context_chars=2400, filters=filters)
                sub_results.append({"question": sub, "answer": sub_ans.answer, "citations": sub_ans.citations})

        # Synthesize
        parts = []
        for i, sr in enumerate(sub_results):
            parts.append(f"子问题{i+1}：{sr['question']}\n初步回答：{sr['answer']}")
        synthesis_context = "\n\n".join(parts)

        synth_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个中文文档处理助手。擅长综合多角度信息。"),
            ("human", """\
基于以下子问题的分析结果，综合回答原始问题。要求：
1) 融合各子问题的关键发现，给出连贯的整体回答
2) 标注不同观点或证据之间的关联（因果关系、对比、互补等）
3) 如有矛盾，指出并给出最可能的结论
4) 末尾追加一行：依据：<引用来源（最多3条）>

原始问题：{question}

子问题分析：
{synthesis_context}

综合回答："""),
        ])
        try:
            text = (
                synth_prompt
                | get_chat_model(
                    style=CONTENT_API_STYLE,
                    model=CONTENT_MODEL_NAME,
                    endpoint=CONTENT_API_ENDPOINT,
                    api_key=CONTENT_API_KEY,
                    temperature=0.2,
                    max_tokens=700,
                    timeout_s=120,
                )
                | StrOutputParser()
            ).invoke({"question": q, "synthesis_context": synthesis_context})
        except Exception:
            # Fallback: concatenate sub-answers
            text = "\n\n".join([f"**{sr['question']}**\n{sr['answer']}" for sr in sub_results])

        out = (text or "").strip()
        out = re.sub(r"^```[a-zA-Z]*\s*", "", out).strip()
        out = re.sub(r"\s*```$", "", out).strip()

        # Collect citations from all sub-results
        all_citations: list[Citation] = []
        seen = set()
        for sr in sub_results:
            for c in (sr.get("citations") or []):
                k = str(c.chunk_id or "")
                if not k or k in seen:
                    continue
                seen.add(k)
                all_citations.append(c)
                if len(all_citations) >= 6:
                    break
            if len(all_citations) >= 6:
                break

        return Answer(kb=kb2, question=q, answer=out or "综合回答生成失败，请尝试更具体的问题。", citations=all_citations)
