"""Search mixin for KnowledgeBase — vector, lexical, HyDE, expanded, hierarchical.

Extracted from kb.py via mixin pattern. All methods reference self.* attributes
set by KnowledgeBase.__init__.
"""

import hashlib
import logging
import math
import re
from typing import Optional

from agent_file_create.config import (
    CONTENT_API_ENDPOINT,
    CONTENT_API_KEY,
    CONTENT_API_STYLE,
    CONTENT_MODEL_NAME,
    RERANK_MODEL,
)
from agent_file_create.llm_client import call_llm
from agent_file_create.rag.embedder import embed_texts
from agent_file_create.rag.store import Hit

from agent_file_create.rag._utils import (
    bm25_scores as _bm25_scores,
    compute_adaptive_rrf_k as _compute_adaptive_rrf_k,
    mmr_rerank as _mmr_rerank,
    normalize_kb as _normalize_kb,
    rrf_ranks as _rrf_ranks,
    split_sentences as _split_sentences,
    title_keyword_boost as _title_keyword_boost,
    tokenize as _tokenize,
)

logger = logging.getLogger(__name__)


class SearchMixin:
    """Retrieval methods — hybrid search, adaptive, HyDE, expanded, hierarchical."""

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
            return self.search(kb=kb2, query=q, top_k=top_k, filters=filters)

        coarse_docs: set[str] = set()
        coarse_sections: set[str] = set()
        for h in coarse_hits:
            coarse_docs.add(str(h.doc_id or "").replace("__summary__", ""))
            sec = str(h.section_path or "").replace("__summary__/", "").strip()
            if sec:
                top_sec = sec.split("/")[0].strip()
                if top_sec:
                    coarse_sections.add(top_sec)

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
            return self.search(kb=kb2, query=q, top_k=top_k, filters=filters)

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

    def search_adaptive(self, *, kb: str, query: str, top_k: int = 8,
                        filters: Optional[dict] = None,
                        enable_diversity: bool = True,
                        enable_title_boost: bool = True,
                        enable_adaptive_weights: bool = True,
                        enable_rerank: bool = False,
                        hyde_query: str = "",
                        ) -> list[Hit]:
        """Adaptive recall with optional chunk-level cross-encoder reranking.

        - Numbers/technical terms → more lexical + BM25 weight (exact match matters)
        - Abstract/long queries → more vector weight (semantic match matters)
        - Title keyword boost rewards chunks from documents with query-matching titles
        - MMR diversity ensures different sections are represented
        - enable_rerank: cross-encoder rerank top-50 RRF candidates → top_k
        - hyde_query: if provided, used for vector embedding; original query for lexical
        """
        kb2 = _normalize_kb(kb)
        q = str(query or "").strip()
        if not q:
            return []

        vec_q = str(hyde_query or "").strip() or q
        profile = self._analyze_query(q)

        cap = 240
        if profile["has_numbers"] or profile["has_tech_terms"]:
            vec_cand = min(max(40, int(top_k or 0) * 10), cap)
            lex_cand = min(max(50, int(top_k or 0) * 18), cap)
        elif profile["concreteness"] > 0.6:
            vec_cand = min(max(50, int(top_k or 0) * 15), cap)
            lex_cand = min(max(40, int(top_k or 0) * 12), cap)
        else:
            vec_cand = min(max(60, int(top_k or 0) * 18), cap)
            lex_cand = min(max(30, int(top_k or 0) * 8), cap)

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

        w_vec, w_bm, w_lex = 1.0, 1.0, 1.0
        if enable_adaptive_weights:
            has_specialized = _query_has_specialized_terms(q)  # imported below
            if has_specialized:
                w_vec, w_bm, w_lex = 1.15, 0.9, 0.9
            elif profile["has_numbers"] or profile["has_tech_terms"]:
                w_vec, w_bm, w_lex = 0.95, 1.1, 1.1
            elif profile["concreteness"] < 0.4:
                w_vec, w_bm, w_lex = 1.1, 0.95, 0.95
            elif profile["length"] > 40:
                w_vec, w_bm, w_lex = 1.05, 1.0, 0.95

        vec = [float(it.get("vec") or 0.0) for it in items]
        ids = [str(it["hit"].chunk_id or "") for it in items]
        vec_r = _rrf_ranks(list(zip(ids, vec)))
        bm_r = _rrf_ranks(list(zip(ids, bm25)))
        lx_r = _rrf_ranks([(str(it["hit"].chunk_id or ""), float(it.get("lex") or 0.0)) for it in items])

        scored: list[tuple[float, Hit]] = []
        k_rrf = _compute_adaptive_rrf_k(profile) if enable_adaptive_weights else 60.0
        for it, braw, vraw in zip(items, bm25, vec):
            h = it["hit"]
            hid = str(h.chunk_id or "")
            rv = float(vec_r.get(hid, 10_000))
            rb = float(bm_r.get(hid, 10_000))
            rl = float(lx_r.get(hid, 10_000))
            meta = dict(h.meta or {})
            meta_scores = {"vec": float(vraw), "bm25": float(braw), "lex": float(it.get("lex") or 0.0)}
            meta["scores"] = meta_scores
            meta["rrf_k"] = k_rrf
            s = (w_vec / (k_rrf + rv)) + (w_bm / (k_rrf + rb)) + (w_lex / (k_rrf + rl))
            scored.append((float(s), Hit(
                kb=h.kb, doc_id=h.doc_id, chunk_id=h.chunk_id,
                chunk_index=h.chunk_index, section_path=h.section_path,
                content=h.content, score=float(s), meta=meta,
            )))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [h for _, h in scored[: max(1, int(top_k or 0))]]

        # ── Chunk-level cross-encoder reranking (Top-50 → Top-K) ──────────
        if enable_rerank and len(scored) >= 3:
            from agent_file_create.rag.reranker import (
                _cross_encoder_rerank as _ce_rerank,
            )
            rerank_candidates = [h for _, h in scored[: max(50, int(top_k or 0) * 6)]]
            try:
                result = _ce_rerank(q, rerank_candidates, RERANK_MODEL, max(1, int(top_k or 0)))
                # Apply title boost and diversity on reranked results (lighter touch)
                if enable_title_boost:
                    result = _title_keyword_boost(result, q, boost_factor=0.08)
                return result
            except Exception:
                pass  # fall through to default path

        if enable_title_boost:
            result = _title_keyword_boost(result, q, boost_factor=0.15)

        if enable_diversity and len(result) > 2:
            result = _mmr_rerank(result, lambda_param=0.40, top_k=max(1, int(top_k or 0)))

        return result

    def search_expanded(self, *, kb: str, query: str, top_k: int = 8, **kw) -> list[Hit]:
        """Query expansion with round-robin interleaving.

        Splits a complex query into sub-queries, runs search_adaptive on each,
        then round-robin interleaves results to preserve multi-angle diversity.
        """
        kb2 = _normalize_kb(kb)
        q = str(query or "").strip()
        if not q:
            return []

        qe_count = int(kw.pop("qe_count", 0) or 0) or 3
        qe_count = max(2, min(qe_count, 6))

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

        return self.search_adaptive(
            kb=kb2,
            query=q,
            hyde_query=hyde_answer,
            top_k=top_k,
            **kw,
        )

    def search_small_to_big(self, *, kb: str, query: str, top_k: int = 8,
                            window_size: int = 2, filters: Optional[dict] = None,
                            ) -> list[Hit]:
        """Small-to-Big retrieval: rank at sentence level, return at paragraph level."""
        kb2 = _normalize_kb(kb)
        q = str(query or "").strip()
        if not q:
            return []

        candidates = self.search(kb=kb2, query=q, top_k=min(40, (top_k or 0) * 4), filters=filters)
        if not candidates:
            return []

        qv = self._cached_embed_query(q)
        if qv is None:
            try:
                qv_list = embed_texts([q], timeout_s=60, max_batch=1)
            except Exception:
                qv_list = None
            if qv_list and qv_list[0]:
                qv = qv_list[0]
                self._set_cached_embed_query(q, qv)

        scored_windows: list[tuple[float, str, str, int, dict]] = []
        w = max(1, int(window_size or 0))

        for h in candidates:
            sents = _split_sentences(h.content)
            if not sents:
                continue
            n = len(sents)

            sent_scores: list[float] = []
            if qv and len(qv) > 0:
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
                    q_lower = q.lower()
                    for sent in sents:
                        overlap = sum(1 for ch in q_lower if ch in sent.lower())
                        sent_scores.append(overlap / max(1, len(q)))
            else:
                q_lower = q.lower()
                for sent in sents:
                    overlap = sum(1 for ch in q_lower if ch in sent.lower())
                    sent_scores.append(overlap / max(1, len(q)))

            if not sent_scores:
                continue

            best_idx = max(range(len(sent_scores)), key=lambda i: sent_scores[i])
            best_score = sent_scores[best_idx]

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
        """Hybrid retrieval: vector + lexical + BM25 fused via Reciprocal Rank Fusion (RRF).

        Why hybrid? Pure vector search misses exact keyword matches (e.g. model numbers,
        error codes); pure lexical search misses semantic paraphrases. RRF combines both
        without needing per-collection calibration.
        """
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
            vec_hits = []
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
        profile = self._analyze_query(q)
        k_rrf = _compute_adaptive_rrf_k(profile)
        for it, braw, vraw in zip(items, bm25, vec):
            h = it["hit"]
            hid = str(h.chunk_id or "")
            rv = float(vec_r.get(hid, 10_000))
            rb = float(bm_r.get(hid, 10_000))
            rl = float(lx_r.get(hid, 10_000))
            meta = dict(h.meta or {})
            meta_scores = {"vec": float(vraw), "bm25": float(braw), "lex": float(it.get("lex") or 0.0)}
            meta["scores"] = meta_scores
            meta["rrf_k"] = k_rrf
            s = (1.0 / (k_rrf + rv)) + (1.0 / (k_rrf + rb)) + (1.0 / (k_rrf + rl))
            scored.append((float(s), Hit(kb=h.kb, doc_id=h.doc_id, chunk_id=h.chunk_id, chunk_index=h.chunk_index, section_path=h.section_path, content=h.content, score=float(s), meta=meta)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [h for _, h in scored[: max(1, int(top_k or 0))]]

    def search_auto(self, *, kb: str, query: str, top_k: int = 8,
                    filters: Optional[dict] = None, **kw) -> list[Hit]:
        """Adaptive query router: picks HyDE or Query Expansion based on query features."""
        q = str(query or "").strip()
        if not q:
            return []

        QE_LONG_THRESHOLD = 30

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

    def search_multi_query(
        self, *, kb: str, question: str, top_k: int = 6, n_variants: int = 3,
        filters: Optional[dict] = None,
    ) -> list[Hit]:
        """Multi-Query retrieval: generate variants → search each → RRF merge."""
        variants = self.generate_query_variants(question, n=n_variants)
        if len(variants) <= 1:
            return self.search(kb=kb, query=question, top_k=top_k, filters=filters)

        profile = self._analyze_query(question)
        mq_k = _compute_adaptive_rrf_k(profile)
        all_hits: dict[str, tuple[Hit, float]] = {}
        for rank, variant in enumerate(variants):
            hits = self.search(kb=kb, query=variant, top_k=max(10, top_k * 2), filters=filters)
            for hit in hits:
                cid = str(hit.chunk_id or "")
                rrf = 1.0 / (mq_k + float(rank + 1))
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
        """Step-Back retrieval: search with original + abstracted question, merge."""
        stepback = self.generate_stepback_question(question)
        if not stepback or stepback == question:
            return self.search(kb=kb, query=question, top_k=top_k, filters=filters)

        orig_hits = self.search(kb=kb, query=question, top_k=max(10, top_k * 2), filters=filters)
        sb_hits = self.search(kb=kb, query=stepback, top_k=max(10, top_k * 2), filters=filters)

        profile = self._analyze_query(question)
        sb_k = _compute_adaptive_rrf_k(profile)
        merged: dict[str, tuple[Hit, float]] = {}
        for rank, h in enumerate(orig_hits):
            merged[h.chunk_id] = (h, 1.0 / (sb_k + rank + 1) * 1.2)
        for rank, h in enumerate(sb_hits):
            cid = str(h.chunk_id or "")
            rrf = 1.0 / (sb_k + rank + 1)
            if cid in merged:
                hit, prev = merged[cid]
                merged[cid] = (hit, prev + rrf)
            else:
                merged[cid] = (h, rrf)

        scored = sorted(merged.values(), key=lambda x: x[1], reverse=True)
        return [h for h, _ in scored[:max(1, int(top_k or 0))]]

    def _fetch_neighbor_chunks(self, kb: str, doc_id: str, center_idx: int, window: int) -> list[Hit]:
        """Fetch neighboring chunks from the same document around center_idx."""
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
        """Fetch all sibling chunks in the same parent group."""
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
        """Search with parent-child context window."""
        hits = self.search(kb=kb, query=query, top_k=top_k, filters=filters)
        if context_window <= 0 or not hits:
            return hits

        expanded: dict[str, Hit] = {}
        for h in hits:
            expanded[h.chunk_id] = h
            h_meta = dict(h.meta or {})
            h_meta["is_primary"] = True

        for h in hits:
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


# Import needed for search_adaptive's adaptive weights
from agent_file_create.rag._utils import query_has_specialized_terms as _query_has_specialized_terms
