import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
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
from agent_file_create.llm_factory import get_chat_model
from agent_file_create.rag.chunker import chunk_text
from agent_file_create.rag.embedder import embed_texts
from agent_file_create.rag.reranker import rerank
from agent_file_create.rag.store import Hit, default_store
from agent_file_create.preprocessor import extract_pdf_text_fast, read_text_file, extract_docx_structured, extract_pptx_structured, ocr_image

# Simple LRU query-embedding cache to avoid re-embedding identical queries
_QUERY_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_QUERY_CACHE_MAX = 128

_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个中文助手，只输出中文。"),
        (
            "human",
            """\
你是企业知识库问答助手。请基于给定的检索片段回答问题。
规则：
1) 只根据检索片段回答；不要编造。
2) 如果片段不足以回答，明确说不确定，并提出 1-3 个澄清问题或建议补充哪些文档。
3) 输出尽量简洁，必要时 3-6 条要点。
4) 末尾追加一行：依据：<引用编号或doc_id/section（最多3条）>；若无法定位写"依据：未命中"。

知识库：
{kb}

检索片段：
{context}

用户问题：
{question}

回答：""",
        ),
    ]
)

# ── Chain-of-thought answer prompt ─────────────────────────────────────────

_ANSWER_COT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个中文助手，擅长逐步推理和严谨分析。"),
        (
            "human",
            """\
你是企业知识库问答助手。请基于给定的检索片段，通过逐步推理回答问题。

推理步骤：
1) 问题理解：用自己的话复述问题的核心要点和隐含假设。
2) 证据梳理：从检索片段中逐条列出相关证据，标注来源编号（如 [1] [2]）。
3) 推理链条：基于证据进行逐步推理。若多个证据之间存在关联（因果、对比、递进等），请明确说明推理路径。
4) 最终回答：给出简洁的最终答案（3-6 条要点）。
5) 自我检查：逐条核查最终回答中的每个论断——是否有对应的检索片段支撑？无支撑的推断请明确标注为「（推测）」或「（材料未覆盖）」。

规则：
- 只根据检索片段回答；不要编造。
- 如果片段不足以回答，在步骤 4 明确说不确定，并在步骤 5 建议补充哪些文档。
- 末尾追加一行：依据：<引用编号（最多3条）>；若无法定位写"依据：未命中"。

知识库：
{kb}

检索片段：
{context}

用户问题：
{question}

推理过程：""",
        ),
    ]
)

# ── HyDE (Hypothetical Document Embeddings) prompt ─────────────────────────

_HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个中文技术文档撰写助手。"),
        (
            "human",
            """\
请用3-5句话编写一段可能回答以下问题的文本段落。要求：
- 使用专业、正式的语气
- 包含可能的关键术语和概念
- 模拟知识库文档的风格
- 只输出文本段落，不要解释或标注

问题：{question}

假设回答：""",
        ),
    ]
)

# ── Question decomposition prompt ──────────────────────────────────────────

_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个中文问答系统分析师。"),
        (
            "human",
            """\
判断以下问题是否需要分解为子问题来回答。如果需要，请分解为2-4个子问题，每个子问题一行。
如果问题本身很简单、不需要分解，只回复：SIMPLE

问题：{question}

分解结果：""",
        ),
    ]
)


def _safe_json(obj: Any, max_len: int) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    s = s.strip()
    if len(s) > max_len:
        return s[:max_len] + "…"
    return s


def _normalize_kb(name: str) -> str:
    n = (name or "").strip()
    n = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", n).strip("._")
    return n or "default"


def _guess_is_markdown(path: str) -> bool:
    suf = Path(path).suffix.lower()
    return suf in {".md", ".markdown"}


def _tokenize(text: str, *, max_terms: int = 80) -> list[str]:
    xs = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]{2,}", str(text or ""))
    out: list[str] = []
    for x in xs:
        k = x.lower()
        if len(k) > 32:
            k = k[:32]
        out.append(k)
        if len(out) >= int(max_terms or 0):
            break
    return out


def _bm25_scores(query_terms: list[str], docs_terms: list[list[str]]) -> list[float]:
    qs = [str(x or "").strip().lower() for x in (query_terms or []) if str(x or "").strip()]
    if not qs or not docs_terms:
        return [0.0 for _ in docs_terms]
    uniq: list[str] = []
    seen = set()
    for t in qs:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
        if len(uniq) >= 24:
            break
    if not uniq:
        return [0.0 for _ in docs_terms]

    N = float(len(docs_terms))
    dl = [float(len(x) or 1) for x in docs_terms]
    avgdl = sum(dl) / float(len(dl) or 1)
    df = {t: 0 for t in uniq}
    tfs: list[dict[str, int]] = []
    for terms in docs_terms:
        m: dict[str, int] = {}
        st = set()
        for w in terms:
            if w in df:
                m[w] = m.get(w, 0) + 1
                st.add(w)
        for w in st:
            df[w] += 1
        tfs.append(m)

    k1 = 1.2
    b = 0.75
    scores: list[float] = []
    for i, m in enumerate(tfs):
        s = 0.0
        dli = dl[i]
        for t in uniq:
            f = float(m.get(t, 0))
            if f <= 0.0:
                continue
            dft = float(df.get(t, 0))
            idf = math.log((N - dft + 0.5) / (dft + 0.5) + 1.0)
            denom = f + k1 * (1.0 - b + b * dli / float(avgdl or 1.0))
            s += idf * (f * (k1 + 1.0) / float(denom or 1.0))
        scores.append(float(s))
    return scores


def _rrf_ranks(items: list[tuple[str, float]]) -> dict[str, int]:
    xs = [(k, float(v)) for k, v in (items or []) if str(k or "").strip()]
    xs.sort(key=lambda x: x[1], reverse=True)
    out: dict[str, int] = {}
    rank = 1
    for k, _ in xs:
        if k in out:
            continue
        out[k] = rank
        rank += 1
    return out


def _read_any_text(path: str) -> str:
    p = Path(path)
    suf = p.suffix.lower().lstrip(".")
    if suf == "pdf":
        t = extract_pdf_text_fast(str(p))
        if t:
            return t.replace("\x00", "")
    # Binary / structured formats: use proper extractors
    if suf in {"xlsx", "xls"}:
        try:
            import pandas as pd
            df = pd.read_excel(str(p))
            cols = list(df.columns)
            rows = min(len(df), 200)
            lines = [f"columns={cols}", f"rows={len(df)}"]
            for _, row in df.head(rows).iterrows():
                lines.append(" | ".join([str(v) for v in row.values if str(v).strip()]))
            return "\n".join(lines).replace("\x00", "")
        except Exception:
            pass
    if suf in {"docx"}:
        try:
            t = extract_docx_structured(str(p))
            if t:
                return t.replace("\x00", "")
        except Exception:
            pass
    if suf in {"pptx", "ppt"}:
        try:
            t = extract_pptx_structured(str(p))
            if t:
                return t.replace("\x00", "")
        except Exception:
            pass
    if suf in {"png", "jpg", "jpeg", "webp", "gif", "bmp"}:
        try:
            ocr_text = ocr_image(Path(p).read_bytes())
            if ocr_text:
                return "图片OCR文字：\n" + ocr_text.replace("\x00", "").strip()
        except Exception:
            pass
        return f"[图片文件] {p.name} (无可提取文字，建议启用OCR)"

    try:
        return read_text_file(str(p)).replace("\x00", "")
    except Exception:
        try:
            data = p.read_bytes()
            return data.decode("utf-8", errors="ignore").replace("\x00", "")
        except Exception:
            return ""


@dataclass(frozen=True)
class Citation:
    doc_id: str
    chunk_id: str
    section_path: str
    score: float
    snippet: str


@dataclass(frozen=True)
class Answer:
    kb: str
    question: str
    answer: str
    citations: list[Citation]


class KnowledgeBase:
    def __init__(self, *, store=None) -> None:
        self.store = store or default_store()

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
        chunks = chunk_text(
            doc_id=did,
            text=text,
            title=ttl,
            is_markdown=is_md,
            target_chars=int(chunk_target_chars),
            overlap_chars=int(chunk_overlap_chars),
        )
        if not chunks:
            return {"kb": kb2, "doc_id": did, "ok": False, "error": "no_chunks"}

        vecs: list[list[float]] = []
        embedding_ok = True
        try:
            vecs = embed_texts([c.content for c in chunks], timeout_s=60, max_batch=24)
            if len(vecs) != len(chunks):
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

    @staticmethod
    def _cached_embed_query(query: str) -> Optional[list[float]]:
        key = hashlib.md5(query.encode("utf-8")).hexdigest()
        if key in _QUERY_CACHE:
            _QUERY_CACHE.move_to_end(key)
            return list(_QUERY_CACHE[key])
        return None

    @staticmethod
    def _set_cached_embed_query(query: str, vec: list[float]) -> None:
        key = hashlib.md5(query.encode("utf-8")).hexdigest()
        _QUERY_CACHE[key] = list(vec)
        _QUERY_CACHE.move_to_end(key)
        while len(_QUERY_CACHE) > _QUERY_CACHE_MAX:
            _QUERY_CACHE.popitem(last=False)

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
        k_rrf = 60.0
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

        ctx = "\n\n".join(blocks).strip()
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
        """Fetch neighboring chunks from the same document around center_idx."""
        kb2 = _normalize_kb(kb)
        did = str(doc_id or "").strip()
        if not did or window <= 0:
            return []
        # Fetch all chunks for this doc via a broad search, then filter by index range
        try:
            all_hits = self.search(kb=kb2, query=did, top_k=400)
        except Exception:
            return []
        return [
            h for h in all_hits
            if str(h.doc_id or "") == did
            and abs(int(h.chunk_index or 0) - center_idx) <= window
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
            neighbors = self._fetch_neighbor_chunks(
                kb=kb, doc_id=str(h.doc_id or ""),
                center_idx=int(h.chunk_index or 0), window=context_window,
            )
            for nh in neighbors:
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
        """
        q = str(question or "").strip()
        if len(q) < 10:
            return q
        try:
            chain = _HYDE_PROMPT | self._get_answer_llm() | StrOutputParser()
            hypothetical = (chain.invoke({"question": q}) or "").strip()
            if hypothetical and len(hypothetical) >= 15:
                return hypothetical[:600]
        except Exception:
            pass
        return q

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

        # Same dedup + segment assembly as answer()
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

        ctx = "\n\n".join(blocks).strip()
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

        # Retrieve for each sub-question
        sub_results: list[dict] = []
        for sub in subs:
            sub_ans = self.answer(kb=kb2, question=sub, top_k=max(3, top_k), max_context_chars=2400, filters=filters)
            sub_results.append({"question": sub, "answer": sub_ans.answer, "citations": sub_ans.citations})

        # Synthesize
        parts = []
        for i, sr in enumerate(sub_results):
            parts.append(f"子问题{i+1}：{sr['question']}\n初步回答：{sr['answer']}")
        synthesis_context = "\n\n".join(parts)

        synth_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个中文助手，擅长综合多角度信息。"),
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
