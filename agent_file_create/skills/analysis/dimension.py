"""Document analysis skill — statistical analysis of extracted materials.

Provides word frequency, coverage gap detection, and cross-document overlap analysis.
Runs during the *enrich* stage to give the LLM objective feedback on material quality
before outline generation.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from agent_file_create.skills.base import SkillResult, SkillMeta, skill

logger = logging.getLogger(__name__)

# Common stop words that shouldn't count as "key terms"
_STOP_WORDS = {
    "的", "是", "在", "了", "和", "也", "就", "都", "而", "及", "与",
    "着", "或", "一个", "没有", "我们", "你们", "他们", "它们", "这个",
    "那个", "这些", "那些", "可以", "需要", "已经", "因为", "所以",
    "但是", "然而", "因此", "如果", "虽然", "而且", "不过", "还是",
    "已经", "正在", "将要", "可能", "应该", "必须", "一定", "会",
    "能", "能够", "被", "把", "从", "对", "向", "以", "为", "到",
    "上", "中", "下", "内", "外", "前", "后", "左", "右",
    "个", "种", "次", "些", "点", "年", "月", "日", "时",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "not", "this", "that", "it", "its",
}


def _extract_terms(text: str, min_len: int = 2, max_terms: int = 50) -> list[tuple[str, int]]:
    """Extract frequent Chinese/English terms from text.

    Uses character bigrams for Chinese and word tokens for English.
    """
    if not text or not text.strip():
        return []

    # Chinese: character bigrams (fast, no jieba dependency)
    cn_chars = re.findall(r"[一-鿿]+", text)
    cn_bigrams: Counter[str] = Counter()
    for segment in cn_chars:
        for i in range(len(segment) - 1):
            bigram = segment[i:i + 2]
            if bigram not in _STOP_WORDS and len(bigram) >= min_len:
                cn_bigrams[bigram] += 1

    # English: word tokens
    en_words: Counter[str] = Counter()
    en_tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    for token in en_tokens:
        if token not in _STOP_WORDS:
            en_words[token] += 1

    # Merge and sort
    combined: Counter[str] = cn_bigrams + en_words
    return combined.most_common(max_terms)


def _detect_coverage_gaps(
    user_prompt: str,
    analysis_results: list[dict],
) -> list[str]:
    """Detect which user-requested concepts are missing from source materials.

    Compares key terms from the user prompt against all extracted summaries.
    Returns a list of concepts that appear in the prompt but have weak/absent
    coverage in the materials.
    """
    if not user_prompt or not analysis_results:
        return []

    # Extract key concepts from user prompt (nouns, proper nouns)
    prompt_terms = set()
    # Match Chinese compound terms (2-6 chars, contains at least one meaningful char)
    prompt_compounds = re.findall(r"[一-鿿]{2,6}", user_prompt)
    for term in prompt_compounds:
        if term not in _STOP_WORDS and len(term) >= 2:
            prompt_terms.add(term)

    # Match English words
    prompt_en = re.findall(r"[a-zA-Z]{3,}", user_prompt.lower())
    for term in prompt_en:
        prompt_terms.add(term)

    if not prompt_terms:
        return []

    # Build material text corpus
    material_text = ""
    for r in (analysis_results or [])[:10]:
        if not isinstance(r, dict):
            continue
        material_text += " " + str(r.get("summary", "") or "")
        material_text += " " + str(r.get("title", "") or "")

    material_text_lower = material_text.lower()

    # Check coverage
    gaps: list[str] = []
    for term in prompt_terms:
        if len(term) < 2:
            continue
        # Count occurrences
        count = material_text_lower.count(term.lower())
        if count == 0:
            gaps.append(term)
        elif count <= 2 and len(term) >= 3:
            gaps.append(f"{term}(覆盖薄弱)")

    return gaps[:15]


async def _execute_document_analysis(
    analysis_results: list | None = None,
    user_prompt: str = "",
    **kwargs,
) -> SkillResult:
    """Analyze extracted source materials for quality feedback.

    Produces:
        - Top frequent terms across all source documents
        - Coverage gaps: user-requested concepts missing from materials
        - Cross-document overlap: sections that appear in multiple sources

    This gives the LLM objective data to guide outline generation:
    which topics can be written confidently and which need supplementary research.
    """
    ar = analysis_results or []
    if not ar:
        return SkillResult(
            success=True,
            summary="（无提取结果可供分析）",
            data={"terms": [], "gaps": [], "overlaps": [], "file_count": 0},
        )

    valid = [r for r in ar if isinstance(r, dict)]
    if not valid:
        return SkillResult(success=True, summary="（无有效分析结果）", data={})

    # ── 1. Word frequency ─────────────────────────────────────────────────
    all_text = ""
    for r in valid[:10]:
        all_text += " " + str(r.get("summary", "") or "")
    terms = _extract_terms(all_text)
    top_terms = [(t, c) for t, c in terms[:20]]

    # ── 2. Coverage gaps ──────────────────────────────────────────────────
    gaps = _detect_coverage_gaps(user_prompt, valid) if user_prompt else []

    # ── 3. Cross-document overlap ─────────────────────────────────────────
    # Find terms that appear in multiple documents (potential redundancy)
    doc_term_sets: list[tuple[str, set]] = []
    for r in valid[:8]:
        fname = str(r.get("_file", "") or "")
        text = str(r.get("summary", "") or "")
        doc_terms = {t for t, _ in _extract_terms(text, max_terms=30)}
        if fname and doc_terms:
            doc_term_sets.append((fname, doc_terms))

    overlaps: list[str] = []
    if len(doc_term_sets) >= 2:
        for i in range(len(doc_term_sets)):
            for j in range(i + 1, len(doc_term_sets)):
                common = doc_term_sets[i][1] & doc_term_sets[j][1]
                if len(common) >= 3:
                    overlaps.append(
                        f"{doc_term_sets[i][0]} ⟷ {doc_term_sets[j][0]}: "
                        f"重叠 {len(common)} 个关键词 ({', '.join(sorted(common)[:5])})"
                    )

    # ── Build summary ─────────────────────────────────────────────────────
    lines: list[str] = ["📊 上传材料分析报告", ""]

    lines.append("**高频关键词**：")
    if top_terms:
        lines.append("  " + "、".join(f"{t}({c}次)" for t, c in top_terms[:15]))
    else:
        lines.append("  （无显著高频词）")

    lines.append("")
    lines.append("**与用户需求相比的材料覆盖缺口**：")
    if gaps:
        for g in gaps[:10]:
            lines.append(f"  ⚠️ {g}")
    else:
        lines.append("  ✅ 用户需求中的关键概念在材料中均有覆盖")

    lines.append("")
    lines.append("**文档间内容重叠**：")
    if overlaps:
        for o in overlaps[:5]:
            lines.append(f"  🔄 {o}")
    else:
        lines.append("  ✅ 各文档内容无明显重叠")

    return SkillResult(
        success=True,
        summary="\n".join(lines),
        data={
            "terms": [{"term": t, "count": c} for t, c in top_terms],
            "coverage_gaps": gaps,
            "overlaps": overlaps,
            "file_count": len(valid),
            "total_chars": len(all_text),
        },
    )


SKILL_META = skill(
    name="document_analysis",
    description="分析上传材料：提取高频关键词、检测与用户需求的覆盖缺口、识别文档间内容重叠",
    category="analysis",
    stage="enrich",
    parameters={
        "analysis_results": {
            "type": "array",
            "description": "文件提取结果列表（由 extract 阶段产出）",
        },
        "user_prompt": {
            "type": "string",
            "description": "用户原始需求描述",
            "default": "",
        },
    },
    timeout_s=30,
    max_retries=1,
)(_execute_document_analysis)
