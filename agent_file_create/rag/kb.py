"""KnowledgeBase — hybrid vector+lexical retrieval, ingestion, question answering.

The class is composed via mixins:
  - SearchMixin   — search(), search_adaptive(), search_hyde(), etc.
  - QueryMixin    — rewrite_query(), classify_query(), _hyde_expand(), etc.
  - AnswerMixin   — answer(), answer_smart(), answer_with_reasoning(), etc.

This file keeps only the core __init__, admin/CRUD, ingestion, and embedding cache.
"""

import hashlib
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from agent_file_create.rag.chunker import chunk_text
from agent_file_create.rag.embedder import embed_texts
from agent_file_create.rag.store import default_store

from agent_file_create.rag._utils import (
    guess_is_markdown as _guess_is_markdown,
    normalize_kb as _normalize_kb,
    read_any_text as _read_any_text,
)
from agent_file_create.rag._search import SearchMixin
from agent_file_create.rag._query import QueryMixin
from agent_file_create.rag._answer import AnswerMixin

logger = logging.getLogger(__name__)


class KnowledgeBase(SearchMixin, QueryMixin, AnswerMixin):
    def __init__(self, *, store=None) -> None:
        self.store = store or default_store()
        # Per-instance LRU caches
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._hyde_cache: OrderedDict[str, str] = OrderedDict()
        self._content_embed_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._content_embed_cache_MAX = 512
        self._hyde_cache_MAX = 128  # used by QueryMixin._hyde_expand

    # ── Admin / CRUD ──────────────────────────────────────────────────────────

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
        """Index document/section summaries as lightweight searchable chunks."""
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

    # ── Embedding Health ──────────────────────────────────────────────────────

    def check_embed_health(self) -> dict:
        """Validate embedding model connectivity."""
        try:
            vecs = embed_texts(["health check"], timeout_s=15, max_batch=1)
            if vecs and isinstance(vecs[0], list) and len(vecs[0]) > 0:
                return {"ok": True, "dim": len(vecs[0])}
            return {"ok": False, "error": "empty_embedding"}
        except Exception as e:
            return {"ok": False, "error": str(e)[:240]}

    # ── File Ingestion ────────────────────────────────────────────────────────

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
        logger.info(f"kb_ingest_start kb={kb2} file={p.name} chars={len(text)}")

        # Adaptive chunk sizing based on document type
        _review_kw = {"综述", "进展", "概念", "挑战", "应用", "研究进展", "技术综述"}
        _methods_kw = {"基于", "方法", "优化", "模型", "融合", "协同", "动态", "自适应"}
        is_review = any(kw in ttl for kw in _review_kw)
        is_methods = any(kw in ttl for kw in _methods_kw)
        if is_review and not is_methods:
            _tc, _oc = 1000, 150
        elif is_methods and not is_review:
            _tc, _oc = 500, 80
        else:
            _tc, _oc = 700, 100

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

        logger.info(f"kb_chunked kb={kb2} file={p.name} chunks={len(chunks)} chunk_size={_tc}")

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
            logger.info(f"kb_embedding_start kb={kb2} file={p.name} new_chunks={len(uncached_idxs)} cached={len(chunks)-len(uncached_idxs)} total={len(chunks)}")
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
                    logger.info(f"kb_embedding_done kb={kb2} file={p.name} ok={len(uncached_idxs)} chunks")
                else:
                    embedding_ok = False
                    logger.info(f"kb_embedding_mismatch kb={kb2} file={p.name} expected={len(uncached_idxs)} got={len(fresh_vecs)}")
            except Exception as e:
                embedding_ok = False
                logger.info(f"kb_embedding_failed kb={kb2} file={p.name} err={str(e)[:120]}")

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
            logger.info(f"kb_ingest_db_failed kb={kb2} file={p.name} err={str(e)[:120]}")
            return {"kb": kb2, "doc_id": did, "ok": False, "error": "db_failed:" + str(e)[:180]}
        logger.info(f"kb_ingest_done kb={kb2} file={p.name} chunks={n} ok=True")
        return {"kb": kb2, "doc_id": did, "ok": True, "chunks": n}

    # ── Embedding Cache ───────────────────────────────────────────────────────

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
