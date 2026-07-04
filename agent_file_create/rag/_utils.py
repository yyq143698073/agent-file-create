"""Utility functions extracted from kb.py — pure helpers with no class dependencies.

All functions here are stateless and can be imported freely without creating
a KnowledgeBase instance.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import jieba
jieba.setLogLevel(20)  # suppress "Building prefix dict" log noise

from agent_file_create.rag.store import Hit


# ── Generic helpers ───────────────────────────────────────────────────────────

def safe_json(obj: Any, max_len: int) -> str:
    try:
        s = __import__("json").dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    s = s.strip()
    if len(s) > max_len:
        return s[:max_len] + "…"
    return s


def normalize_kb(name: str) -> str:
    n = (name or "").strip()
    n = re.sub(r"[^0-9A-Za-z一-鿿._-]+", "_", n).strip("._")
    return n or "default"


def guess_is_markdown(path: str) -> bool:
    suf = Path(path).suffix.lower()
    return suf in {".md", ".markdown"}


# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize(text: str, *, max_terms: int = 80) -> list[str]:
    s = str(text or "").strip()
    if not s:
        return []
    tokens = jieba.lcut(s)
    out: list[str] = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        k = t.lower()
        if len(k) > 32:
            k = k[:32]
        out.append(k)
        if len(out) >= int(max_terms or 0):
            break
    return out


# ── BM25 ──────────────────────────────────────────────────────────────────────

def bm25_scores(query_terms: list[str], docs_terms: list[list[str]]) -> list[float]:
    qs = [str(x or "").strip().lower() for x in (query_terms or []) if str(x or "").strip()]
    if not qs or not docs_terms:
        return [0.0 for _ in docs_terms]
    uniq: list[str] = []
    seen: set[str] = set()
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
        st: set[str] = set()
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


# ── RRF ───────────────────────────────────────────────────────────────────────

def rrf_ranks(items: list[tuple[str, float]]) -> dict[str, int]:
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


# ── Recall-layer: query analysis helpers ──────────────────────────────────────

def query_has_numbers(q: str) -> bool:
    return bool(re.search(r"\d+", q))


def query_has_technical_terms(q: str) -> bool:
    """Heuristic: check for technical/code-like patterns that vector search may miss."""
    return bool(re.search(r"[A-Z]{2,}|\b(?:GB|TB|API|SDK|ID|URL|PDF|ISO|GDPR)\b|[A-Z]+/[A-Z]+", q))


def query_has_specialized_terms(q: str) -> bool:
    """Detect compound Chinese technical terms: 2-4 char specific nouns.

    Queries with specialized terms benefit more from vector search because
    embedding captures conceptual similarity even when exact term doesn't match.
    """
    try:
        tokens = jieba.lcut(q)
    except Exception:
        return False
    # Count 2-4 char tokens that look like compound technical terms
    # (not common stopwords/particles)
    stop_chars = set("的了吗呢吧啊嗯是都在和与或对从到用把被让给为以而因但")
    specialized = 0
    for t in tokens:
        t = t.strip()
        if 2 <= len(t) <= 6 and not any(c in stop_chars for c in t):
            specialized += 1
    return specialized >= 3  # at least 3 specialized compound terms


def query_concreteness(q: str) -> float:
    """0-1 score: how concrete/specific the query is (vs abstract/conversational)."""
    abstract_kw = ["为什么", "原因", "影响", "趋势", "发展", "前景", "意义", "作用",
                   "分析", "评估", "判断", "看法", "观点", "理解", "解释"]
    concrete_kw = ["多少", "什么时间", "在哪里", "谁", "哪个", "编号", "日期",
                   "金额", "比例", "百分比", "步骤", "流程", "定义"]
    abstract_hits = sum(1 for kw in abstract_kw if kw in q)
    concrete_hits = sum(1 for kw in concrete_kw if kw in q)
    if abstract_hits + concrete_hits == 0:
        return 0.5  # neutral
    return concrete_hits / (abstract_hits + concrete_hits)


# ── Title keyword boost ───────────────────────────────────────────────────────

def title_keyword_boost(hits: list[Hit], query: str, *, boost_factor: float = 0.10) -> list[Hit]:
    """Boost chunks whose document title contains query keywords.

    Title match is a strong relevance signal: if a query mentions "RAG超参数调优"
    and the document title is "基于贝叶斯优化的RAG系统超参数调优", that document
    should be ranked higher regardless of chunk content match.
    """
    if not hits or not query:
        return hits
    q_terms = tokenize(query, max_terms=20)
    if not q_terms:
        return hits
    # Only casefold if text contains Latin characters (noop for pure CJK)
    _needs_fold = lambda s: s.casefold() if any(c.isascii() and c.isalpha() for c in s[:200]) else s
    q_lower = _needs_fold(str(query))
    for h in hits:
        title = _needs_fold(str(h.meta.get("title") or ""))
        if not title:
            continue
        # Count query terms appearing in title (more matches = stronger signal)
        matched = sum(1 for t in q_terms if t.lower() in title)
        # Also check if the full query or a substantial substring appears
        if matched == 0 and len(query) >= 4:
            # Try bigram overlap: any 2-char sequence from query appearing in title
            for i in range(len(q_lower) - 1):
                if q_lower[i:i+2] in title:
                    matched += 0.5
        if matched > 0:
            boost = 1.0 + boost_factor * min(float(matched), 4.0)
            object.__setattr__(h, "score", float(h.score) * boost)
    # Re-sort by boosted score
    hits.sort(key=lambda x: float(x.score), reverse=True)
    return hits


# ── MMR diversity rerank ─────────────────────────────────────────────────────

def mmr_rerank(hits: list[Hit], *, lambda_param: float = 0.7, top_k: int = 8) -> list[Hit]:
    """Maximal Marginal Relevance: balance relevance with section/doc diversity.

    lambda=1.0 → pure relevance ranking; lambda=0.0 → pure diversity.
    lambda=0.7 → 70% relevance, 30% diversity.
    """
    if not hits or len(hits) <= 1:
        return hits[:top_k]

    selected: list[Hit] = [hits[0]]
    remaining = hits[1:]

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = -1.0
        for i, h in enumerate(remaining):
            # Relevance component
            rel = float(h.score)
            # Diversity component: max similarity to any already-selected hit
            max_sim = 0.0
            for s in selected:
                # Doc-level diversity: same doc → penalty
                if str(h.doc_id or "") == str(s.doc_id or ""):
                    max_sim = max(max_sim, 0.5)
                # Section-level diversity: same top-level section → small penalty
                h_sec = (str(h.section_path or "").split("/")[0] if h.section_path else "").strip()
                s_sec = (str(s.section_path or "").split("/")[0] if s.section_path else "").strip()
                if h_sec and s_sec and h_sec == s_sec:
                    max_sim = max(max_sim, 0.3)
            mmr = lambda_param * rel - (1.0 - lambda_param) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected


# ── Entity extraction ─────────────────────────────────────────────────────────

def extract_entities(q: str) -> list[str]:
    """Extract key entities from query for exact-match boosting.

    Handles Chinese text via jieba POS tagging + regex for structured patterns.
    """
    entities: list[str] = []
    seen: set[str] = set()

    # 1) Regex: dates, numbers, percentages, amounts, codes
    patterns = [
        (r"\d{4}年\d{1,2}月\d{1,2}日", "date_full"),
        (r"\d{4}年(?:\d{1,2}月)?", "date_year"),
        (r"\d+\.?\d*%", "percentage"),
        (r"(?:USD|RMB|EUR|JPY|CNY)\s*\d+\.?\d*", "currency_code"),
        (r"\d+\.?\d*\s*(?:元|万元|亿美元|万元人民币)", "currency_cn"),
        (r"[A-Z]{2,}[-一-鿿]*\d*[A-Z]*", "code"),
        (r"[A-Z]+/[A-Z]+(?:\d+\.?\d*)?", "technical_id"),
        (r"第[一二三四五六七八九十\d]+[章节条款项]", "section_ref"),
    ]
    for pat, _ in patterns:
        for m in re.finditer(pat, q):
            ent = m.group(0).strip()
            if ent and ent not in seen and len(ent) >= 2:
                entities.append(ent)
                seen.add(ent)

    # 2) Jieba: extract nouns (nr/ns/nz/nt/n - named entities, nouns)
    try:
        import jieba.posseg as pseg
        words = pseg.lcut(q)
    except Exception:
        words = [(w, "") for w in jieba.lcut(q)]
    for word, flag in words:
        w = word.strip()
        if len(w) < 2 or w in seen:
            continue
        if flag and flag[0] in ("n",):  # noun-class: n, nr, ns, nz, nt
            entities.append(w)
            seen.add(w)
        elif not flag and len(w) >= 3:  # fallback: treat longer words as potential entities
            if w not in entities:
                entities.append(w)
                seen.add(w)

    # 3) Also capture quoted strings
    for m in re.finditer(r"[""「」『』「」『』\"''](.+?)[""「」『』「」『』\"'']", q):
        ent = m.group(1).strip()
        if ent and ent not in seen and len(ent) >= 2:
            entities.append(ent)
            seen.add(ent)

    return entities


# ── Sentence splitting ────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Split Chinese/English text into sentences."""
    if not text or not str(text).strip():
        return []
    # Sentence boundary patterns for Chinese + English
    sents = re.split(r"(?<=[。！？；\n])\s*|(?<=[.!?;])\s+(?=[A-Z一-鿿])", str(text))
    out = [s.strip() for s in sents if s.strip() and len(s.strip()) >= 3]
    return out


# ── Multi-format file reader ──────────────────────────────────────────────────

def read_any_text(path: str) -> str:
    """Read text content from any supported file format (PDF/DOCX/PPTX/XLSX/image/plain)."""
    from agent_file_create.preprocessor import (
        extract_pdf_text_fast,
        extract_pdf_text_parallel,
        read_text_file,
        extract_docx_structured,
        extract_pptx_structured,
        ocr_image,
    )

    p = Path(path)
    suf = p.suffix.lower().lstrip(".")
    if suf == "pdf":
        # Use parallel extraction for large PDFs (>20 pages)
        try:
            fsize = p.stat().st_size
            if fsize > 2 * 1024 * 1024:  # >2MB → probably many pages
                t = extract_pdf_text_parallel(str(p))
            else:
                t = extract_pdf_text_fast(str(p))
        except Exception:
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
