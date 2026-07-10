"""Knowledge planner: pre-plans what each section needs and retrieves material.

Sits between Outline and Content generation — for each h2 section:
1. LLM generates a knowledge checklist (what this section needs to cover)
2. Each knowledge point is used to search the KB
3. Retrieved materials are compressed and attached to the section

Result: content generation has dedicated, pre-retrieved material per section rather
than relying on a single global dataset.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent_file_create.config import (
    CONTENT_API_ENDPOINT, CONTENT_API_KEY, CONTENT_API_STYLE, CONTENT_MODEL_NAME,
    PLANNER_MODEL_NAME,
)
from agent_file_create.llm_client import call_llm
from agent_file_create.llm_factory import get_chat_model
from agent_file_create.prompts import Citation

logger = logging.getLogger(__name__)

# Planner LLM model — uses a dedicated config if set, otherwise defaults to a
# fast model.  Knowledge checklists and sentence filtering are lightweight tasks
# that don't need the full reasoning power of the content-generation model.
_PLANNER_MODEL = (PLANNER_MODEL_NAME or "").strip() or "deepseek-v4-flash"


def _rerank_sentences(query: str, sentences: list[str], top_k: int = 8) -> list[int]:
    """Score sentences against *query* using a cross-encoder reranker.

    When ``RERANK_ENABLED`` is true and ``FlagEmbedding`` is installed this
    replaces the LLM-based sentence-filtering step with a fast, zero-token,
    local model call.  Returns sentence indices sorted by relevance (descending).

    Falls back to simple truncation when the reranker is unavailable.
    """
    from agent_file_create.config import RERANK_ENABLED, RERANK_MODEL

    if not RERANK_ENABLED or not sentences:
        return list(range(min(top_k, len(sentences))))

    try:
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker(RERANK_MODEL, use_fp16=True)
        pairs = [(query, s) for s in sentences]
        scores = reranker.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: -x[1])
        result = [idx for idx, _ in indexed[:top_k]]
        logger.debug(
            "planner_reranker query=%.40s candidates=%d selected=%d top_score=%.3f",
            query, len(sentences), len(result),
            indexed[0][1] if indexed else 0.0,
        )
        return result
    except ImportError:
        logger.debug("planner_reranker_unavailable — FlagEmbedding not installed, using LLM fallback")
    except Exception as e:
        logger.debug("planner_reranker_failed err=%s — using LLM fallback", e)

    # Fallback: return empty list — caller will use the LLM path
    return []

# ── Knowledge checklist generation prompt ────────────────────────────────────

_CHECKLIST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个中文文档处理助手。"),
    ("human", """\
你是报告撰写规划助手。以下是一个报告章节，请列出撰写该章节需要的3-5个关键知识点。

每个知识点用一句话描述，应具体可检索（包含关键实体、数字范围、概念术语）。

报告主题：{user_prompt}
章节标题：{section_title}
父章节：{parent_title}

输出：每行一个知识点，不要编号。"""),
])


def _backtrack_parent_context(kb, kb_name: str, hits: list, top_n: int = 8) -> list:
    """For top-ranked hits, retrieve full parent paragraph for richer LLM context.

    When a chunk belongs to a parent group (parent_chunk_id), expand it by
    attaching the parent's full content as ``_parent_context``.  This gives
    the downstream LLM the complete paragraph instead of a fragmented chunk.
    """
    if not hits:
        return hits

    parent_cache: dict[str, str] = {}   # parent_chunk_id → merged parent text
    expanded: list = []

    for h in hits:
        meta = dict(getattr(h, "meta", {}) or {})
        pid = str(meta.get("parent_chunk_id") or "")
        if pid and pid not in parent_cache:
            try:
                siblings = kb.store.get_parent_context(kb=kb_name, parent_chunk_id=pid)
                if siblings:
                    parent_cache[pid] = "\n\n".join(
                        getattr(s, "content", "") or "" for s in siblings
                    )
            except Exception:
                parent_cache[pid] = ""  # don't retry failures

        if pid and parent_cache.get(pid):
            # Attach parent context — Hit is a plain class, recreate
            new_meta = dict(meta)
            new_meta["_parent_context"] = parent_cache[pid]
            h = type(h)(
                kb=h.kb, doc_id=h.doc_id, chunk_id=h.chunk_id,
                chunk_index=h.chunk_index, section_path=h.section_path,
                content=h.content, score=h.score, meta=new_meta,
                parent_chunk_id=getattr(h, "parent_chunk_id", "") or pid,
            )
        expanded.append(h)

    return expanded


# ── Helpers: Content dedup ───────────────────────────────────────────────────


def _dedup_hits_by_content(hits: list, similarity_threshold: float = 0.55) -> list:
    """Drop near-duplicate chunks, keeping only the highest-score instance.

    Two chunks are considered duplicates when their content fingerprints share
    a high Jaccard similarity.  This prevents the same passage (split across
    overlapping chunks or retrieved via different queries) from occupying
    multiple slots in the top-K results.
    """
    if not hits or len(hits) <= 1:
        return hits

    import hashlib

    def _fingerprint(content: str) -> set[str]:
        import re
        text = re.sub(r"\s+", " ", str(content or "").strip().lower())
        fp: set[str] = set()
        # Character trigrams (works well for CJK + catches spelling variants)
        chars = re.sub(r"[^\w一-鿿]", "", text)
        for i in range(len(chars) - 2):
            fp.add(chars[i:i + 3])
        # Word bigrams (better for Latin text similarity)
        words = [w for w in re.split(r"\s+", text) if len(w) >= 2]
        for i in range(len(words) - 1):
            fp.add("W:" + words[i] + "|" + words[i + 1])
        return fp

    sorted_hits = sorted(hits, key=lambda h: float(getattr(h, "score", 0) or 0), reverse=True)
    kept: list = []
    kept_fps: list[set[str]] = []

    for hit in sorted_hits:
        fp = _fingerprint(getattr(hit, "content", "") or "")
        is_dup = False
        for kfp in kept_fps:
            if not fp or not kfp:
                continue
            jaccard = len(fp & kfp) / max(len(fp | kfp), 1)
            if jaccard >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(hit)
            kept_fps.append(fp)

    return kept


# ── Helpers: Query Generation ────────────────────────────────────────────────

_STOPWORDS = set("的了吗呢吧啊是都在和与或对从到用把被让给为以而因但就这那也都还没又再很更太只就可会要能说看想他她它我你".split())


def _extract_key_terms_from_body(body: str, max_terms: int = 5) -> list[str]:
    """Extract top key terms from section body text using jieba TF extraction.

    Filters out stopwords and short tokens, returning the highest-frequency
    content words as additional queries for retrieval expansion.
    """
    if not body or len(str(body or "").strip()) < 20:
        return []

    try:
        import jieba
    except Exception:
        return []

    tokens = jieba.lcut(str(body).strip())
    # Count term frequencies, excluding stopwords and short tokens
    freq: dict[str, int] = {}
    for t in tokens:
        t = t.strip()
        if len(t) < 2 or t in _STOPWORDS:
            continue
        freq[t] = freq.get(t, 0) + 1

    # Return top-N by frequency
    sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [term for term, _ in sorted_terms[:max_terms]]


def _extract_terms(text: str, min_len: int = 2, max_len: int = 10) -> list[str]:
    """Extract meaningful Chinese terms from text using jieba.
    
    Returns terms sorted by importance (longer terms first).
    """
    import jieba
    words = jieba.lcut(str(text or ""))
    result = []
    seen = set()
    for w in words:
        w = w.strip()
        if min_len <= len(w) <= max_len:
            if w not in _STOPWORDS and not re.match(r'^[\d\W]+$', w):
                if w not in seen:
                    seen.add(w)
                    result.append(w)
    result.sort(key=lambda x: (-len(x), x))
    return result


def _extract_concepts_from_title(title: str) -> list[str]:
    """Extract core concepts from section title.
    
    Example: "RAG技术背景与相关工作" → ["RAG技术", "RAG", "背景", "相关工作"]
    """
    title = str(title or "").strip()
    if not title:
        return []
    
    concepts = []
    
    # Extract English acronyms and terms
    for m in re.finditer(r'[A-Z]{2,}(?:-[A-Z]+)?', title):
        concepts.append(m.group())
    
    # Extract Chinese phrases using jieba
    terms = _extract_terms(title, min_len=2, max_len=8)
    concepts.extend(terms)
    
    # Deduplicate, preserving order
    seen = set()
    result = []
    for c in concepts:
        if c not in seen:
            seen.add(c)
            result.append(c)
    
    return result


def _generate_queries_from_knowledge_point(kp: str, section_title: str = "") -> list[str]:
    """Generate multiple queries from a single knowledge point.
    
    Returns queries in order of specificity:
    1. Complete knowledge point (for semantic search)
    2. Key phrases (3-5 words)
    3. Core concepts (1-2 words)
    """
    queries = []
    kp = str(kp or "").strip()
    
    # 1. Complete knowledge point
    if len(kp) >= 10:
        queries.append(kp)
    
    # 2. Extract key phrases
    terms = _extract_terms(kp, min_len=2, max_len=8)
    queries.extend(terms[:3])
    
    # 3. Add concepts from section title if available
    if section_title:
        title_concepts = _extract_concepts_from_title(section_title)
        # Prefer acronyms
        acronyms = [c for c in title_concepts if re.match(r'^[A-Z]{2,}$', c)]
        queries.extend(acronyms[:2])
        queries.extend([c for c in title_concepts if c not in acronyms][:2])

    # 4. Broaden retrieval: strip specific method names to generate general-concept
    #    queries that can match documents beyond the primary paper.  For example,
    #    "CKE-RAG的知识图谱冲突消融策略" → "知识图谱 冲突消融".
    _brand_terms = {
        "CKE-RAG", "CKE RAG", "BP-Cache", "BP Cache",
        "LoRAG", "Lo RAG", "Self-RAG", "Self RAG",
        "GraphRAG", "Graph RAG", "LightRAG", "Light RAG",
        "HippoRAG", "Hippo RAG",
    }
    _generic_queries: list[str] = []
    for q in queries:
        # Try stripping known brand names from the query
        stripped = q
        for _brand in _brand_terms:
            stripped = re.sub(r'\s*'.join(re.escape(c) for c in _brand), '', stripped, flags=re.IGNORECASE)
        stripped = re.sub(r'\s{2,}', ' ', stripped).strip()
        if stripped and stripped != q and len(stripped) >= 4:
            _generic_queries.append(stripped)
    # Insert generic queries after specific ones — specific matches take priority
    queries.extend(_generic_queries[:3])

    # 5. Evaluation-type queries: retrieve experiment/baseline/comparison data
    _eval_suffixes = ["对比实验", "基线比较", "性能评估", "实验结果", "消融实验", "参数设置", "baseline", "metrics"]
    for _suf in _eval_suffixes[:4]:
        if section_title:
            queries.append(f"{section_title} {_suf}"[:50])
        if len(kp) >= 6:
            queries.append(f"{kp[:40]} {_suf}"[:50])

    # Deduplicate and limit
    seen = set()
    result = []
    for q in queries:
        q = q.strip()
        if q and q not in seen and 2 <= len(q) <= 50:
            seen.add(q)
            result.append(q)

    return result[:8]


# ── Adaptive context budget ───────────────────────────────────────────────

_BASE_BUDGET: dict[str, int] = {
    "data":             2000,   # tables, numbers, metrics — need raw data
    "experiment_setup": 1800,   # methods/setup — data-adjacent but needs method context
    "analysis":         1500,   # reasoning chains — need argument structure
    "review":            800,   # background, summary — just conclusions
}

_MIN_CONTEXT_CHARS = 400
_DEFAULT_MODEL_CAP_CHARS = 32000  # conservative for most LLMs (≈128k tokens, use 0.3x = 9600)


def _get_context_budget(
    section_type: str = "review",
    target_words: int = 0,
    model_cap: int = 0,
) -> int:
    """Dynamic context budget based on section type and target document length.

    Formula: ``base * (1 + 0.5 * target_words / 8000)``
    Capped at ``min(result, model_cap * 0.3)``.

    - ``data`` gets the most (raw numbers, metrics, tables).
    - ``experiment_setup`` slightly less (method context, hyperparams).
    - ``analysis`` medium (argument chains, discussion).
    - ``review`` / background least (conclusions, summaries).
    """
    base = _BASE_BUDGET.get(section_type, 1200)

    # Continuous scaling: short docs stay near base, long docs get proportional boost
    if target_words > 0:
        scale = 1.0 + 0.5 * target_words / 8000.0
        budget = int(base * scale)
    else:
        budget = int(base)

    # Model-aware cap: never exceed 30% of model context window
    if model_cap > 0:
        cap = int(model_cap * 0.3)
    else:
        cap = int(_DEFAULT_MODEL_CAP_CHARS * 0.3)  # default: 9600

    return max(_MIN_CONTEXT_CHARS, min(budget, cap))


# ── Lightweight compression (no extra LLM) ────────────────────────────────

# _compress_hits removed — superseded by _compress_hits_annotated which adds
# citation anchoring + temporal hints. The old function is no longer called.
# Adaptive budget logic lives in _get_context_budget, shared by both paths.
# For plain-text compression (no citations), inline truncation at call sites
# uses _get_context_budget directly.

def _extract_year(hit) -> str:
    """Try to extract a publication year from a Hit's metadata.

    Checks, in order: meta dict fields, doc_id patterns, content patterns.
    Returns 4-digit year string or empty string.
    """
    import re

    meta = getattr(hit, "meta", {}) or {}

    # 1. Explicit meta fields
    for key in ("year", "pub_year", "date", "publication_year", "created"):
        val = meta.get(key)
        if val:
            m = re.search(r"(19|20)\d{2}", str(val))
            if m:
                return m.group()

    # 2. doc_id patterns: "paper_2023.pdf", "report-2024-v2"
    doc_id = str(getattr(hit, "doc_id", "") or "")
    m = re.search(r"(?:^|[^0-9])(20\d{2}|19\d{2})(?:[^0-9]|$)", doc_id)
    if m:
        return m.group(1)

    # 3. Content patterns: "Published: 2023", "2024年"
    content = str(getattr(hit, "content", "") or "")[:300]
    m = re.search(r"(?:发表于?|Published[:\s]+|出版|发表).{0,10}?(20\d{2}|19\d{2})", content)
    if m:
        return m.group(1)

    return ""


def _compress_hits_annotated(
    hits: list,
    query: str,
    max_chars: int = 1200,
    section_type: str = "review",
    target_words: int = 0,
) -> tuple[str, dict[int, Citation]]:
    """Like _compress_hits but preserves source metadata with inline citation markers.

    Uses ``【n】`` markers (NOT ``[n]`` which conflicts with Markdown link syntax).
    Each marker is prefixed directly to the sentence so LLM sentence filtering
    keeps source and content together — never separated.

    Returns:
        (annotated_text, citation_map)
        annotated_text:  each snippet prefixed with inline ``【n】`` marker + source tail
        citation_map:    {n: Citation(doc_id=..., chunk_id=..., section_path=..., snippet=...)}
    """
    budget = _get_context_budget(section_type, target_words)
    effective_max = max(max_chars, budget) if max_chars != 1200 else budget
    _data_types = {"data", "experiment_setup"}
    snippet_len = 300 if section_type in _data_types else 200
    max_hits = 12 if section_type in _data_types else 8

    parts: list[str] = []
    citation_map: dict[int, Citation] = {}
    total = 0
    counter = 0

    for h in hits[:max_hits]:
        content = str(h.content or "").strip()
        if not content:
            continue
        snippet = content[:snippet_len]
        if len(content) > snippet_len:
            snippet += "…"

        counter += 1
        doc_id = str(getattr(h, "doc_id", "") or "")
        source_name = doc_id
        if "/" in doc_id:
            source_name = doc_id.rsplit("/", 1)[-1]
        elif "\\" in doc_id:
            source_name = doc_id.rsplit("\\", 1)[-1]

        section_path = str(getattr(h, "section_path", "") or "")
        chunk_id = str(getattr(h, "chunk_id", "") or "")
        score = float(getattr(h, "score", 0) or 0)

        citation_map[counter] = Citation(
            doc_id=doc_id,
            chunk_id=chunk_id,
            section_path=section_path,
            score=score,
            snippet=snippet,
        )

        # Inline anchoring: 【n】content (来源: file > section)
        # The marker is part of the sentence, so LLM filtering keeps them together.
        src_label = source_name or f"doc_{counter}"
        if section_path:
            src_label += f" > {section_path}"

        # Temporal hint: if the hit has a publication year in metadata, annotate it.
        # LLM sees e.g. "(2023, market_report.pdf)" and can prefer newer sources.
        pub_year = _extract_year(h)
        if pub_year:
            src_label = f"{pub_year}, {src_label}"

        parts.append(f"【{counter}】{snippet} (来源: {src_label})")
        total += len(snippet)
        if total >= effective_max:
            break

    return "\n\n".join(parts), citation_map


def renumber_citations(
    content: str,
    citation_map: dict[int, Citation],
) -> tuple[str, dict[int, Citation]]:
    """Post-processing: globally renumber citations across parallel-generated sections.

    Scans the final content for all ``【n】`` markers, deduplicates sources,
    assigns unique global IDs, and replaces all local IDs with global ones.

    Returns (renumbered_content, global_citation_map).
    """
    import re

    if not content:
        return content, citation_map
    # citation_map may be empty — proceed with auto-ID mode (sequential per-occurrence)

    # Step 1: Collect all unique citation markers 【n】 and [n] in content
    seen_global: set[tuple[str, str]] = set()
    local_to_global: dict[int, int] = {}
    global_map: dict[int, Citation] = {}
    global_counter = 0

    # Match both 【n】 (fullwidth) and [n] (halfwidth, common LLM mistake)
    local_ids_found = set()
    for m in re.finditer(r"[【\[](\d+)[】\]]", content):
        local_id = int(m.group(1))
        local_ids_found.add(local_id)

    if not citation_map and local_ids_found:
        # Auto-ID mode: no citation_map available. Assign unique sequential global
        # IDs per occurrence so 10x 【1】 becomes [1][2]...[10] instead of all [1].
        pass  # handled in replacement step below
    else:
        for local_id in sorted(local_ids_found):
            cit = citation_map.get(local_id)
            if cit is None:
                continue
            key = (cit.doc_id or "", cit.section_path or "")
            if key in seen_global:
                for gid, gc in global_map.items():
                    if (gc.doc_id or "", gc.section_path or "") == key:
                        local_to_global[local_id] = gid
                        break
                continue
            global_counter += 1
            seen_global.add(key)
            local_to_global[local_id] = global_counter
            global_map[global_counter] = Citation(
                doc_id=cit.doc_id,
                chunk_id=cit.chunk_id,
                section_path=cit.section_path,
                score=cit.score,
                snippet=cit.snippet,
            )

    # Count how many total markers exist in the document
    _total_markers = len(list(re.finditer(r"[【\[](\d+)[】\]]", content)))
    # Auto-ID when:
    #   1. map is empty (no citation data at all)
    #   2. LLM only used one number (e.g., only 【1】) → all would collapse to one ID
    #   3. map has too few entries to cover the markers
    _single_local_id = len(local_ids_found) == 1 and _total_markers >= 3
    _ratio_ok = _total_markers == 0 or len(global_map) >= _total_markers / 3
    _use_auto_id = (not citation_map) or _single_local_id or (_total_markers >= 4 and not _ratio_ok)

    if not local_to_global and citation_map and not _use_auto_id:
        return content, citation_map

    # Step 2: Replace all local IDs with global IDs — handle both bracket styles
    if _use_auto_id:
        # Auto-ID mode: sequential unique IDs per occurrence position.
        # Build a deduplicated reference list from the original citation_map.
        parts = re.split(r"([【\[]\d+[】\]])", content)
        renumbered_parts: list[str] = []
        counter = 0
        auto_map: dict[int, Citation] = {}
        _seen_dedup: set[tuple[str, str]] = set()  # (doc_id, chunk_id)
        for part in parts:
            if re.match(r"^[【\[]\d+[】\]]$", part):
                counter += 1
                renumbered_parts.append(f"【{counter}】")
                local_id = int(re.search(r"\d+", part).group(0))
                if citation_map and local_id in citation_map:
                    cit = citation_map[local_id]
                    dedup_key = (cit.doc_id or "", cit.chunk_id or cit.section_path or "")
                    if dedup_key not in _seen_dedup:
                        _seen_dedup.add(dedup_key)
                        auto_map[counter] = cit
            else:
                renumbered_parts.append(part)
        return "".join(renumbered_parts), auto_map
    else:
        def _replace_id(m: re.Match) -> str:
            local_id = int(m.group(1))
            gid = local_to_global.get(local_id, local_id)
            return f"【{gid}】"
        renumbered = re.sub(r"[【\[](\d+)[】\]]", _replace_id, content)
        return renumbered, global_map


def verify_citations(
    content: str,
    citation_map: dict[int, Citation],
) -> list[dict]:
    """Post-processing: verify that each 【n】 citation references a real source.

    Uses entity-based matching (numbers, percentages, key nouns) rather than
    full word overlap to avoid false positives from paraphrasing.

    Only flags: fake citations (nonexistent IDs), and citations where the
    claimed context shares zero key entities with the source snippet.

    Returns a list of suspicious citations.
    """
    import re

    if not content or not citation_map:
        return []

    warnings: list[dict] = []
    max_valid_id = max(citation_map.keys()) if citation_map else 0

    for m in re.finditer(r"【(\d+)】", content):
        n = int(m.group(1))
        cit = citation_map.get(n)

        # Fake citation: ID not in map
        if cit is None:
            warnings.append({
                "id": n, "issue": "fake_citation",
                "detail": f"【{n}】 does not exist (max source ID: {max_valid_id})",
            })
            continue

        # Extract key entities from source snippet and claimed context
        start = max(0, m.start() - 60)
        end = min(len(content), m.end() + 100)
        context = content[start:end].replace("\n", " ")
        snippet = (cit.snippet or "").replace("\n", " ")

        src_entities = _extract_key_entities(snippet)
        ctx_entities = _extract_key_entities(context)

        if not src_entities:
            continue  # No entities to check against — skip

        # If zero entity overlap, flag as suspicious
        shared = src_entities & ctx_entities
        if not shared:
            warnings.append({
                "id": n,
                "issue": "no_entity_overlap",
                "claimed_context": context[:120],
                "source_snippet": snippet[:120],
                "source_entities": sorted(src_entities)[:8],
            })

    return warnings


def _extract_key_entities(text: str) -> set[str]:
    """Extract key entities from text for citation verification.

    Focuses on concrete items less likely to be paraphrased:
    numbers, percentages, technical terms, capitalized acronyms.
    """
    import re
    entities: set[str] = set()

    # Numbers with units: "1200万", "1200万辆", "35%", "400Wh/kg", "3.5亿"
    for m in re.finditer(r"\d+(?:\.\d+)?\s*(?:[万亿千百]?辆?|[%％]|[A-Za-z/]+)?", text):
        v = m.group().strip()
        if len(v) >= 2:
            entities.add(v)

    # Capitalized acronyms: "NEV", "BEV", "PHEV", "CR5"
    # Use lookaround instead of \b — \b is unreliable with mixed CJK/Latin text
    for m in re.finditer(r"(?<![A-Za-z])[A-Z]{2,}(?:\d+)?(?![A-Za-z])", text):
        entities.add(m.group())

    # Percentage-adjacent terms (catch "同比增长" after "35%")
    for m in re.finditer(r"(?:同比|环比|增长|下降|提升|减少)\s*(?:\d+(?:\.\d+)?\s*%?)?", text):
        v = m.group().strip()
        if len(v) >= 3:
            entities.add(v)

    return entities


def build_citation_map(
    all_section_plans: dict[str, dict],
) -> dict[int, Citation]:
    """Merge per-section citation maps into a single global citation map.

    Re-numbers citations so each source gets one unique ID across the
    entire document.  Deduplicates by doc_id + section_path.
    """
    global_map: dict[int, Citation] = {}
    seen: set[tuple[str, str]] = set()
    counter = 0

    for section_title, plan in (all_section_plans or {}).items():
        section_map: dict = plan.get("citation_map") or {}
        for local_id, cit in sorted(section_map.items(), key=lambda x: x[0]):
            if not isinstance(cit, Citation):
                continue
            # Use chunk_id for finer-grained dedup — different chunks of the
            # same document/section should get distinct citation IDs
            key = (cit.doc_id or "", cit.chunk_id or cit.section_path or "")
            if key in seen:
                continue
            counter += 1
            seen.add(key)
            global_map[counter] = Citation(
                doc_id=cit.doc_id,
                chunk_id=cit.chunk_id,
                section_path=cit.section_path,
                score=cit.score,
                snippet=cit.snippet,
            )

    return global_map


def format_citation_list(citation_map: dict[int, Citation]) -> str:
    """Format a citation map into a human-readable reference list.

    Uses ``【n】`` format matching the inline citation markers.
    Avoids ``[n]`` which conflicts with Markdown link reference syntax.
    """
    if not citation_map:
        return ""

    lines = []
    for n in sorted(citation_map):
        cit = citation_map[n]
        label = (cit.doc_name or cit.doc_id or f"来源_{n}").replace(".pdf", "").replace(".docx", "")
        if "/" in label:
            label = label.rsplit("/", 1)[-1]
        if "\\" in label:
            label = label.rsplit("\\", 1)[-1]
        lines.append(f"【{n}】{label}")
    return "\n".join(lines)


# ── Section-type-aware adaptive retrieval (Layer 2 + feedback) ─────────────

def _quality_ok(hits: list, min_score: float = 0.0, min_unique_docs: int = 1, min_hits: int = 5) -> bool:
    """Check if retrieval results are good enough. 0 LLM calls.

    Early return if >= 3 hits AND all top-3 have score >= 0.5 (consistent quality).
    Avoids a single 0.7-score hit masking four 0.1-score hits.
    """
    if not hits:
        return False
    scores = sorted(
        [float(getattr(h, "score", 0) or 0) for h in hits], reverse=True
    )
    # Consistent quality: at least 3 hits, and the 3rd-best is still decent
    if len(scores) >= 3 and scores[2] >= 0.5:
        return True
    if len(hits) < min_hits:
        return False
    avg_score = sum(scores) / len(scores)
    if avg_score < min_score:
        return False
    unique_docs = len(set(str(getattr(h, "doc_id", "") or "") for h in hits))
    if unique_docs < min_unique_docs:
        return False
    return True


def _retrieve_by_section_type(kb, kb_name: str, query: str, section_type: str) -> list:
    """Adaptive retrieval with quality feedback loop and query expansion.

    Why adaptive? Different section types need different search strategies:
    - "data" sections need precise keyword matching (adaptive w/ title boost)
    - "analysis" sections need conceptual depth (HyDE generates hypothetical answers first)
    - "review" sections need broad coverage (adaptive w/ fallback to HyDE)

    The quality feedback loop prevents bad retrievals from poisoning content
    generation: if the initial strategy returns low-quality results (low average
    score, too few unique docs), we escalate to a broader strategy and then to
    pure vector search. This costs 0 extra LLM calls and adds ~50ms latency.
    """
    all_results = []
    seen_chunk_ids = set()
    
    # First, try with the original query
    primary = {
        "data":              lambda q: kb.search_adaptive(kb=kb_name, query=q, top_k=8,
                                                           enable_diversity=False, enable_title_boost=True),
        "experiment_setup":  lambda q: kb.search_adaptive(kb=kb_name, query=q, top_k=10,
                                                           enable_title_boost=True),
        "analysis":          lambda q: kb.search_hyde(kb=kb_name, query=q, top_k=10),
        "review":            lambda q: kb.search_adaptive(kb=kb_name, query=q, top_k=8),
    }

    fallback1 = {
        "data":              lambda q: kb.search_adaptive(kb=kb_name, query=q, top_k=12),
        "experiment_setup":  lambda q: kb.search_adaptive(kb=kb_name, query=q, top_k=14),
        "analysis":          lambda q: kb.search_adaptive(kb=kb_name, query=q, top_k=15),
        "review":            lambda q: kb.search_hyde(kb=kb_name, query=q, top_k=12),
    }
    
    # Last resort: pure vector search with broader top_k
    def ultimate(q: str):
        return kb.search(kb=kb_name, query=q, top_k=12)
    
    # Generate expanded queries
    expanded_queries = _generate_expanded_queries(query)
    
    for q_idx, expanded_query in enumerate(expanded_queries):
        strategies = [
            primary.get(section_type, primary["review"]),
            fallback1.get(section_type, fallback1["review"]),
            ultimate,
        ]
        
        for step, strategy in enumerate(strategies):
            try:
                hits = strategy(expanded_query)
                for h in hits:
                    if h.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(h.chunk_id)
                        all_results.append(h)
                
                # ── Adaptive escalation: score-distribution-aware ──
                # 1. Early stop: all top-3 hits are decent quality → don't escalate
                if step == 0 and len(all_results) >= 3:
                    top3_scores = sorted(
                        [float(getattr(h, "score", 0) or 0) for h in all_results],
                        reverse=True,
                    )[:3]
                    if top3_scores and min(top3_scores) >= 0.5:
                        logger.debug("planner_early_stop query=%s type=%s hits=%d top3=%.2f/%.2f/%.2f",
                                     query[:30], section_type, len(all_results),
                                     top3_scores[0], top3_scores[1], top3_scores[2])
                        return all_results[:20]

                # 2. Escalate: low-quality results from current strategy
                if _quality_ok(all_results, min_hits=5):
                    if q_idx > 0 or step > 0:
                        logger.info("planner_retrieval_escalate query=%s q_idx=%d type=%s step=%d hits=%d",
                                    query[:30], q_idx, section_type, step, len(all_results))
                    return all_results[:20]
                elif step < len(strategies) - 1 and len(all_results) >= 1:
                    # Quality not met but we have some results — check score distribution
                    top3 = sorted(
                        [float(getattr(h, "score", 0) or 0) for h in all_results],
                        reverse=True,
                    )[:3]
                    avg_top3 = sum(top3) / len(top3) if top3 else 0
                    if avg_top3 < 0.5:
                        logger.debug("planner_escalate_low_quality query=%s step=%d avg_top3=%.2f",
                                     query[:30], step, avg_top3)
            except Exception:
                continue
    
    # Everything failed — last try with basic search (broader)
    try:
        # Final fallback: try with key concepts from query
        key_concepts = _extract_terms(query, min_len=2, max_len=5)
        for concept in key_concepts[:2]:
            try:
                hits = kb.search(kb=kb_name, query=concept, top_k=10)
                for h in hits:
                    if h.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(h.chunk_id)
                        all_results.append(h)
            except Exception:
                pass
        return all_results[:20] or []
    except Exception:
        return []


def _generate_expanded_queries(original_query: str) -> list[str]:
    """Generate expanded versions of the original query for better recall.
    
    When the original query is too specific or abstract, extract key concepts
    and try searching with broader terms.
    
    Examples:
        "RAG系统的发展历程概述" → ["RAG系统", "RAG 发展历程", "RAG技术", "RAG"]
        "多模态RAG的技术挑战" → ["多模态RAG", "RAG 多模态", "RAG技术", "RAG"]
    """
    queries = []
    original_query = str(original_query or "").strip()
    
    if not original_query:
        return []
    
    # 1. Original query first
    queries.append(original_query)
    
    # 2. Extract key terms using jieba
    import jieba
    words = jieba.lcut(original_query)
    key_terms = []
    for w in words:
        w = w.strip()
        if 2 <= len(w) <= 8:
            if w not in _STOPWORDS and not re.match(r'^[\d\W]+$', w):
                key_terms.append(w)
    
    # 3. Try combinations of key terms (2-3 terms)
    if len(key_terms) >= 2:
        queries.append(" ".join(key_terms[:2]))
    if len(key_terms) >= 3:
        queries.append(" ".join(key_terms[:3]))
    
    # 4. Try individual key terms
    for term in key_terms[:4]:
        queries.append(term)
    
    # 5. Extract English acronyms and try them
    acronyms = re.findall(r'[A-Z]{2,}(?:-[A-Z]+)?', original_query)
    for acronym in acronyms[:3]:
        queries.append(acronym)
        # Also try acronym + "技术" for Chinese context
        queries.append(acronym + "技术")
    
    # 6. Deduplicate and limit — cap at 4 to avoid redundant overlapping queries
    seen = set()
    result = []
    for q in queries:
        q = q.strip()
        if q and q not in seen and 2 <= len(q) <= 50:
            seen.add(q)
            result.append(q)
            if len(result) >= 4:
                break

    return result


def _multi_query_retrieve(kb, kb_name: str, queries: list[str], section_type: str, max_hits: int = 20) -> list:
    """Retrieve using multiple queries and merge results.
    
    Returns deduplicated hits, preserving order of first occurrence.
    """
    all_hits = []
    seen = set()
    
    for i, query in enumerate(queries):
        try:
            hits = _retrieve_by_section_type(kb, kb_name, query, section_type)
            logger.debug("planner_multi_query idx=%d query=%s hits=%d", i, query[:30], len(hits))
            
            for h in hits:
                if h.chunk_id not in seen:
                    seen.add(h.chunk_id)
                    all_hits.append(h)
                    if len(all_hits) >= max_hits:
                        break
        except Exception as e:
            logger.debug("planner_query_failed query=%s err=%s", query[:30], e)
            continue
        
        if len(all_hits) >= max_hits:
            break
    
    return all_hits


# ── Batch checklist generation (1 LLM call for all sections) ──────────────

_BATCH_CHECKLIST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个中文文档处理助手。"),
    ("human", """\
你是报告撰写规划助手。以下是一份报告的多个章节，请为每个章节列出3-5个撰写时需要的知识点。

每个知识点一句话描述，具体可检索（含关键实体、概念术语）。

{section_list}

格式：每章用"## 章节名"开头，下面每行一个知识点。"""),
])


def _batch_generate_checklists(
    section_titles: list[str],
    user_prompt: str,
) -> dict[str, list[str]]:
    """Generate knowledge checklists for ALL sections in 1 LLM call."""
    if not section_titles:
        return {}

    section_list = "\n".join(f"- {t}" for t in section_titles)
    try:
        llm = get_chat_model(
            style=CONTENT_API_STYLE, model=_PLANNER_MODEL,
            endpoint=CONTENT_API_ENDPOINT, api_key=CONTENT_API_KEY,
            temperature=0.1, max_tokens=600, timeout_s=60,
        )
        chain = _BATCH_CHECKLIST_PROMPT | llm | StrOutputParser()
        raw = (chain.invoke({
            "section_list": section_list,
        }) or "").strip()

        # Parse: "## Section Title\n- point1\n- point2\n\n## Section Title\n..."
        result: dict[str, list[str]] = {}
        current_title = ""
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("## "):
                current_title = line[3:].strip()
                result[current_title] = []
            elif line.startswith("- ") and current_title:
                pt = line[2:].strip()
                if len(pt) > 5:
                    result[current_title].append(pt)

        # Fuzzy match section titles to parsed titles
        final: dict[str, list[str]] = {}
        for st in section_titles:
            best = None
            for pt_key in result:
                if st[:4] in pt_key or pt_key[:4] in st:
                    best = pt_key
                    break
            if best:
                final[st] = result[best][:5]
        return final
    except Exception as e:
        logger.debug("batch_checklist_failed err=%s", e)
        return {}


# ── Section-level planner ───────────────────────────────────────────────────

def plan_section_knowledge(
    *,
    section_title: str,
    parent_title: str,
    user_prompt: str,
    kb=None,
    kb_name: str = "",
    max_points: int = 4,
    target_words: int = 0,
    knowledge_points: list[str] | None = None,
) -> dict:
    """Plan and retrieve knowledge for a single h2 section.

    If ``knowledge_points`` is provided, skips LLM checklist generation
    and uses the given list directly.  This lets ``plan_all_sections``
    reuse its batch checklist while still getting per-section LLM sentence
    filtering.

    ``target_words`` scales the adaptive context budget for longer documents.

    Returns: {
        "knowledge_points": [...],
        "materials": str (compressed retrieved context),
    }
    """
    result: dict[str, Any] = {
        "knowledge_points": [],
        "materials": "",
    }

    # Step 1: Generate knowledge checklist — or reuse externally provided KPs
    if knowledge_points:
        points = [kp for kp in knowledge_points if kp and len(str(kp).strip()) > 3][:max_points]
        result["knowledge_points"] = points
        logger.debug("planner_checklist_reused section=%s kps=%d", section_title, len(points))
    else:
        try:
            llm = get_chat_model(
                style=CONTENT_API_STYLE, model=_PLANNER_MODEL,
                endpoint=CONTENT_API_ENDPOINT, api_key=CONTENT_API_KEY,
                temperature=0.1, max_tokens=200, timeout_s=30,
            )
            chain = _CHECKLIST_PROMPT | llm | StrOutputParser()
            raw = (chain.invoke({
                "user_prompt": user_prompt,
                "section_title": section_title,
                "parent_title": parent_title or "（顶级章节）",
            }) or "").strip()
            points = [p.strip() for p in raw.splitlines()
                      if p.strip() and len(p.strip()) > 5][:max_points]
            result["knowledge_points"] = points
        except Exception as e:
            logger.debug("planner_checklist_failed section=%s err=%s", section_title, e)
            return result

    # Step 2: Retrieve — section-type-aware with multi-query strategy
    if not kb or not kb_name:
        return result

    from agent_file_create.document.content_generator import classify_section_type
    sec_type = classify_section_type(section_title)
    logger.debug("planner_section_type section=%s type=%s", section_title, sec_type)
    result["section_type"] = sec_type

    # Adaptive context budget based on section type + target document length
    ctx_budget = _get_context_budget(sec_type, target_words)
    logger.debug("planner_budget section=%s type=%s budget=%d", section_title, sec_type, ctx_budget)

    # Generate all queries: from knowledge points + section title
    all_queries: list[str] = []
    
    # Strategy 1: Generate queries from each knowledge point
    for kp in points:
        queries = _generate_queries_from_knowledge_point(kp, section_title)
        all_queries.extend(queries)
    
    # Strategy 2: Add concepts from section title (fallback for when LLM fails)
    if not all_queries:
        title_queries = _extract_concepts_from_title(section_title)
        all_queries.extend(title_queries[:3])
    
    # Strategy 3: Add parent topic and user prompt concepts
    if parent_title:
        parent_queries = _extract_concepts_from_title(parent_title)
        all_queries.extend(parent_queries[:2])
    if user_prompt:
        prompt_queries = _extract_terms(user_prompt, min_len=2, max_len=6)
        all_queries.extend(prompt_queries[:2])

    # Strategy 4 (P1): Extract key terms from section body text when available
    section_body = result.get("section_body", "")
    if not section_body:
        # Try to get from outline section data
        outline_data = result.get("_outline", {})
        if isinstance(outline_data, dict):
            section_body = str(outline_data.get("body", "") or "")
    if section_body and len(section_body) > 20:
        body_terms = _extract_key_terms_from_body(section_body, max_terms=5)
        for t in body_terms:
            if t not in seen:
                seen.add(t)
                all_queries.append(t)

    # Deduplicate and limit
    seen = set()
    queries: list[str] = []
    for q in all_queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            queries.append(q)
            if len(queries) >= 12:
                break
    
    if not queries:
        logger.debug("planner_no_queries section=%s", section_title)
        return result
    
    logger.debug("planner_queries section=%s count=%d queries=%s", 
                 section_title, len(queries), str(queries[:5]))

    # Multi-query retrieval
    all_hits = _multi_query_retrieve(kb, kb_name, queries, sec_type, max_hits=25)

    # ── Document diversity: ensure top hits include chunks from multiple sources ──
    if all_hits and len(all_hits) >= 6:
        _seen_docs: set[str] = set()
        _diverse: list = []
        _rest: list = []
        for h in all_hits:
            did = str(h.doc_id or "")
            if did not in _seen_docs and len(_diverse) < 15:
                _seen_docs.add(did)
                _diverse.append(h)
            else:
                _rest.append(h)
        # If fewer than 3 unique documents in top hits, inject top-scoring ones from rest
        if len(_seen_docs) < 3:
            _extra_docs: set[str] = set()
            for h in _rest:
                did = str(h.doc_id or "")
                if did not in _seen_docs and did not in _extra_docs:
                    _extra_docs.add(did)
                    _diverse.append(h)
                    if len(_seen_docs) + len(_extra_docs) >= 3:
                        break
        if len(_diverse) >= 3:
            all_hits = _diverse + _rest
            _total_docs = len(set(str(h.doc_id or '') for h in all_hits))
            logger.debug("planner_diversity section=%s docs=%d", section_title, _total_docs)

    # ── P3: Content hash dedup — drop near-duplicate chunks, keep highest-score ──
    if all_hits and len(all_hits) >= 4:
        all_hits = _dedup_hits_by_content(all_hits)

    # ── Parent document backtracking: expand top hits to full parent paragraphs ──
    if all_hits and hasattr(kb, "store") and hasattr(kb.store, "get_parent_context"):
        all_hits = _backtrack_parent_context(kb, kb_name, all_hits)

    if not all_hits:
        logger.debug("planner_no_hits section=%s queries=%s", section_title, str(queries[:3]))
        return result

    result["hits_count"] = len(all_hits)
    result["_raw_hits"] = all_hits  # preserved for cross-document conflict detection
    logger.debug("planner_hits section=%s count=%d", section_title, len(all_hits))

    # Step 3: Compress — keep only relevant sentences (1 LLM call)
    # Decompose all hits into sentences
    all_sentences: list[str] = []
    for h in all_hits:
        content = str(h.content or "").strip()
        if not content:
            continue
        for sent in re.split(r"[。！？.!?\n]+", content):
            sent = sent.strip()
            if len(sent) >= 8:
                all_sentences.append(sent)

    if not all_sentences:
        return result

    # Collect key terms from knowledge points for pre-filtering
    key_terms: set[str] = set()
    for kp in points:
        terms = _extract_terms(kp, min_len=2, max_len=6)
        key_terms.update(terms[:5])
    
    # Add section title terms
    title_terms = _extract_terms(section_title, min_len=2, max_len=6)
    key_terms.update(title_terms[:3])
    
    logger.debug("planner_key_terms section=%s terms=%s", section_title, str(list(key_terms)[:10]))

    # Pre-filter: prioritize sentences containing key terms + quantitative bonus
    if key_terms:
        # Score sentences by key term coverage + quantitative content bonus
        scored_sentences: list[tuple[int, int, str]] = []  # (score, index, sentence)
        for idx, sent in enumerate(all_sentences):
            score = sum(1 for term in key_terms if term in sent)
            # Quantitative bonus: sentences with numbers, percentages, or metrics get +2
            if re.search(r'\d+\.?\d*\s*[%％]|\d+\.?\d+\s*[倍秒时天]|GPU|epoch|batch|lr\b|learning.rate|accuracy|recall|NDCG|F1|BLEU|ROUGE', sent, re.IGNORECASE):
                score += 2
            scored_sentences.append((score, idx, sent))
        
        # Sort by score descending, keep top 25
        scored_sentences.sort(key=lambda x: (-x[0], x[1]))
        filtered_sentences: list[tuple[int, str]] = []  # (original_idx, sentence)
        for score, idx, sent in scored_sentences[:25]:
            filtered_sentences.append((idx, sent))
        
        # Sort by original index to maintain context order
        filtered_sentences.sort(key=lambda x: x[0])
        filtered_sentence_list = [s for _, s in filtered_sentences]
        original_indices = [i for i, _ in filtered_sentences]
        
        logger.debug("planner_pre_filter section=%s all=%d filtered=%d", 
                     section_title, len(all_sentences), len(filtered_sentence_list))
    else:
        # No key terms, use first 25
        filtered_sentence_list = all_sentences[:25]
        original_indices = list(range(25))

    if not filtered_sentence_list:
        # Fallback: use raw sentences
        result["materials"] = "。".join(all_sentences[:8])[:ctx_budget]
        return result

    # Use combined knowledge points as filter query
    filter_query = "。".join(points[:2]) if points else section_title

    # ── Sentence filtering: reranker (fast, 0-token) → LLM (fallback) ──
    indices: set[int] = set()
    from agent_file_create.config import RERANK_ENABLED as _RERANK_ENABLED

    if _RERANK_ENABLED:
        try:
            ranked = _rerank_sentences(filter_query, filtered_sentence_list, top_k=8)
            for ri in ranked:
                if 0 <= ri < len(filtered_sentence_list):
                    indices.add(original_indices[ri] if original_indices else ri)
            if indices:
                logger.debug("planner_reranker_ok section=%s selected=%d/%d",
                             section_title, len(indices), len(filtered_sentence_list))
        except Exception:
            pass  # reranker failed → fall through to LLM path

    # LLM fallback when reranker is disabled or produced no results
    if not indices:
        sentences_text = "\n".join(
            f"[S{i+1}] {s}" for i, s in enumerate(filtered_sentence_list))
        try:
            raw_filter = call_llm(
                f"任务：从以下句子中筛选与「{filter_query[:150]}」相关的句子。\n\n"
                f"筛选标准：\n"
                f"1. 句子包含相关概念、术语或关键词\n"
                f"2. 句子提供有用的背景信息、数据或分析\n"
                f"3. 句子与主题直接相关或有间接关联\n"
                f"4. 如果没有完全匹配的句子，请选择最相关的3-5个句子\n\n"
                f"句子列表：\n{sentences_text[:3000]}\n\n"
                f"输出格式：只输出相关句子的序号，用逗号分隔，例如：S1,S3,S5,S8",
                timeout_s=15, temperature=0.0, num_predict=100,
                model_name=_PLANNER_MODEL,
                system="你是一个专业的文档处理助手，擅长从大量文本中提取与主题相关的内容。请严格按照要求只输出句子序号。")

            for m in re.findall(r"S?(\d+)", raw_filter or ""):
                idx = int(m) - 1
                if 0 <= idx < len(filtered_sentence_list):
                    indices.add(original_indices[idx] if original_indices else idx)

            # If LLM selected nothing, fall back to keyword-based selection
            if not indices and key_terms:
                logger.debug("planner_llm_no_selection section=%s using keyword fallback", section_title)
                scored = []
                for idx, sent in enumerate(all_sentences):
                    score = sum(1 for term in key_terms if term in sent)
                    if score > 0:
                        scored.append((score, idx, sent))
                scored.sort(key=lambda x: (-x[0], x[1]))
                for _, idx, _ in scored[:8]:
                    indices.add(idx)
        except Exception as e:
            logger.debug("planner_compression_failed section=%s err=%s", section_title, e)

    # ── Assemble final materials from selected sentences ──
    if indices:
        materials = ""
        for i in sorted(indices):
            if 0 <= i < len(all_sentences):
                s = all_sentences[i]
                if len(materials) + len(s) + 2 > ctx_budget:
                    break
                materials += s + "。"
        if materials:
            result["materials"] = materials.strip()
        else:
            result["materials"] = "。".join(all_sentences[:8])[:ctx_budget]
    else:
        # No indices at all — use pre-filtered raw fallback
        logger.debug("planner_no_selection section=%s using fallback", section_title)
        result["materials"] = "。".join(filtered_sentence_list[:8])[:ctx_budget]

    return result


def plan_all_sections(
    *,
    outline: str,
    user_prompt: str,
    kb=None,
    kb_name: str = "",
    target_words: int = 0,
) -> dict[str, dict]:
    """Plan knowledge for all h2 sections in the outline.

    ``target_words`` is forwarded to ``_get_context_budget`` for adaptive scaling.

    Returns: {section_title: {knowledge_points: [...], materials: str}}
    """
    from agent_file_create.document.content_generator import parse_outline_sections

    sections = parse_outline_sections(outline or "")
    h2_sections = [s for s in sections if s["level"] == 2]

    # ── Batch: generate ALL checklists in 1 LLM call (instead of N) ──
    all_titles = [s["title"] for s in h2_sections[:8]]
    batch_checklists = _batch_generate_checklists(all_titles, user_prompt)

    # ── Parallel: per-section knowledge plan (KB retrieval + LLM sentence filter) ──
    # Each section's plan_section_knowledge is independent: read-only KB searches
    # plus lightweight LLM calls.  Running them in parallel cuts wall-clock time
    # from O(N) to O(1) for the LLM-dominated phase.
    plan: dict[str, dict] = {}

    def _plan_one(sec: dict) -> tuple[str, dict] | None:
        try:
            kps = batch_checklists.get(sec["title"], [])
            result = plan_section_knowledge(
                section_title=sec["title"],
                parent_title="",
                user_prompt=user_prompt,
                kb=kb,
                kb_name=kb_name,
                target_words=target_words,
                knowledge_points=kps if kps else None,
            )
            materials = result.get("materials", "")
            if materials:
                return (sec["title"], {
                    "knowledge_points": result.get("knowledge_points", []),
                    "materials": materials,
                    "section_type": result.get("section_type", "review"),
                    "hits_count": result.get("hits_count", 0),
                    "_raw_hits": result.get("_raw_hits", []),
                })
        except Exception as e:
            logger.debug("planner_section_failed section=%s err=%s", sec["title"], e)
        return None

    sections_to_plan = h2_sections[:8]
    if sections_to_plan:
        with ThreadPoolExecutor(max_workers=min(8, len(sections_to_plan))) as ex:
            futures = {ex.submit(_plan_one, sec): sec for sec in sections_to_plan}
            for fut in as_completed(futures):
                r = fut.result()
                if r is not None:
                    plan[r[0]] = r[1]

    # ── KB diversity audit: warn when retrieval misses most of the available docs ──
    if plan and kb and kb_name:
        try:
            all_doc_ids: set[str] = set()
            for _sec_plan in plan.values():
                for _h in (_sec_plan.get("_raw_hits") or []):
                    _did = str(getattr(_h, "doc_id", "") or "").strip()
                    if _did:
                        all_doc_ids.add(_did)

            _kb_stats = kb.kb_stats(kb=kb_name) if hasattr(kb, "kb_stats") else {}
            _total_docs = int(_kb_stats.get("doc_count") or 0) if isinstance(_kb_stats, dict) else 0

            if _total_docs > 0:
                _ratio = len(all_doc_ids) / _total_docs
                if _ratio < 0.3:
                    logger.warning(
                        "planner_diversity_low kb=%s retrieved=%d/%d docs ratio=%.0f%% — "
                        "KB has %d docs but retrieval only hit %d. Consider broadening "
                        "queries or checking embedding quality.",
                        kb_name, len(all_doc_ids), _total_docs,
                        _ratio * 100, _total_docs, len(all_doc_ids),
                    )
                else:
                    logger.info(
                        "planner_diversity_ok kb=%s retrieved=%d/%d docs ratio=%.0f%%",
                        kb_name, len(all_doc_ids), _total_docs, _ratio * 100,
                    )
        except Exception:
            pass

    if plan:
        logger.info("planner_done sections=%d total_points=%d",
                     len(plan), sum(len(v["knowledge_points"]) for v in plan.values()))
    return plan
