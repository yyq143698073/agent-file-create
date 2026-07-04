"""Approach C: Decomposed evaluation — rule-based metrics that reuse existing
infrastructure.  No extra LLM calls needed.

Metrics
-------
* **Faithfulness** : ratio of claims in generated content that can be traced back
  to source materials (numbers, entities, years).
* **Completeness** : fraction of outline sections that have substantive content.
* **Coherence** : semantic overlap between adjacent sections (TF‑IDF cosine).
* **Relevance** : embedding cosine similarity between user_prompt and content.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agent_file_create.evaluation.models import DimensionScores


# ═══════════════════════════════════════════════════════════════════════════════
# Fact extraction — imported from _reviewer to avoid duplication
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise_number(num: str) -> str:
    """Strip Chinese modifiers and suffixes to get the numeric core."""
    n = (num or "").strip()
    # Drop leading modifiers
    n = re.sub(r"^(约|大约|近|超过|不足|至少|最多)\s*", "", n)
    # Drop trailing unit suffixes
    n = re.sub(r"\s*[万亿千百]?\s*(元|美元|亿|万|%|％|个|人|家|次|倍|吨|公斤|千米|公里|米|小时|天|年|月)?$", "", n)
    n = n.strip()
    return n


def _extract_facts(text: str) -> Dict[str, set]:
    """Extract verifiable data points — delegates to _reviewer module."""
    from agent_file_create.document._reviewer import extract_facts_from_materials
    return extract_facts_from_materials(text)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Faithfulness
# ═══════════════════════════════════════════════════════════════════════════════

def _score_faithfulness(
    content: str,
    analysis_results: List[dict],
) -> tuple[float, list[str]]:
    """Check generated content claims against source materials.

    Returns (score 0-1, list of warning strings).
    """
    if not content or not analysis_results:
        return 1.0, []

    # Collect facts from all source materials
    source_facts: Dict[str, set] = {"numbers": set(), "entities": set(), "years": set()}
    for ar in analysis_results:
        if not isinstance(ar, dict):
            continue
        digest = " ".join(str(v) for v in ar.values() if isinstance(v, str))
        file_facts = _extract_facts(digest)
        for key in source_facts:
            source_facts[key] |= file_facts[key]

    # Extract facts from generated content
    gen_facts = _extract_facts(content)

    warnings: list[str] = []

    # Normalise source numbers for fuzzy comparison
    source_nums_norm = {_normalise_number(n) for n in source_facts.get("numbers", set())}

    # Check numbers in generated content that are NOT in source
    for num in gen_facts.get("numbers", set()):
        if len(num) < 3:
            continue
        norm = _normalise_number(num)
        # Exact match or normalised match
        if num in source_facts.get("numbers", set()):
            continue
        if norm and norm in source_nums_norm:
            continue
        # Loose: core digits match (e.g. "近100个" vs "100")
        digits = re.sub(r"[^\d.]", "", norm)
        if digits and any(digits == re.sub(r"[^\d.]", "", sn) for sn in source_nums_norm if sn):
            continue
        warnings.append(f"数字不在原文中: {num}")

    # Check entities — only flag those that look like real org names (≥4 chars, contains 大学/公司/etc.)
    real_org_suffixes = re.compile(r"(公司|集团|大学|学院|研究所|研究院|中心|医院|银行|证券|保险|基金)")
    for ent in gen_facts.get("entities", set()):
        if len(ent) < 5:
            continue
        if not real_org_suffixes.search(ent):
            continue  # skip noise matches from generic suffixes
        source_ents = source_facts.get("entities", set())
        if ent in source_ents:
            continue
        # Fuzzy match: check if any source entity contains or is contained
        found = any(ent[:4] in s or s[:4] in ent for s in source_ents)
        if not found:
            warnings.append(f"机构不在原文中: {ent}")

    # Check years
    for yr in gen_facts.get("years", set()):
        if yr not in source_facts.get("years", set()):
            warnings.append(f"年份不在原文中: {yr}")

    # Score: penalty per unsupported claim
    total_claims = sum(len(gen_facts[k]) for k in gen_facts)
    unsupported = len(warnings)
    if total_claims == 0:
        return 1.0, []
    score = max(0.0, 1.0 - (unsupported / total_claims))
    return round(score, 3), warnings[:20]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Completeness
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_outline_sections(outline: str) -> list[dict]:
    """Parse markdown outline — delegates to content_generator."""
    from agent_file_create.document.content_generator import parse_outline_sections
    return parse_outline_sections(outline)


def _score_completeness(content: str, outline: str) -> tuple[float, dict]:
    """Check what fraction of outline sections have corresponding content."""
    sections = _parse_outline_sections(outline)
    if not sections:
        return 1.0, {"total_sections": 0, "covered": 0}

    covered = 0
    details: list[dict] = []
    for sec in sections:
        # Check if section title (or its key terms) appears in content
        title = sec["title"]
        # Extract key terms (first 4 chars of title, or full title for short ones)
        terms = [title[:4], title[:6]] if len(title) >= 6 else [title]
        found = any(t in content for t in terms)
        if found:
            covered += 1
        # Also check for substantial content around the keyword
        if found:
            idx = content.find(terms[0])
            nearby = content[max(0, idx - 20):idx + 100] if idx >= 0 else ""
            has_substance = len(nearby.strip()) >= 50
        else:
            has_substance = False
        details.append({
            "title": title,
            "level": sec["level"],
            "found": found,
            "has_substance": has_substance,
        })

    score = covered / len(sections) if sections else 1.0
    return round(score, 3), {
        "total_sections": len(sections),
        "covered": covered,
        "substance_count": sum(1 for d in details if d["has_substance"]),
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Coherence
# ═══════════════════════════════════════════════════════════════════════════════

def _split_sections(content: str) -> list[str]:
    """Split content at `##` headers.  Returns list of section texts."""
    # Find all ## header positions
    headers = list(re.finditer(r"^#{2,3}\s+.+$", content, re.MULTILINE))
    if len(headers) < 2:
        return [content]

    sections: list[str] = []
    for i, m in enumerate(headers):
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
        sections.append(content[start:end].strip())
    return sections


def _score_coherence(content: str) -> tuple[float, dict]:
    """Measure semantic overlap between adjacent sections via TF‑IDF."""
    sections = _split_sections(content)
    if len(sections) < 2:
        return 1.0, {"section_count": len(sections), "transitions": 0}

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        vec = TfidfVectorizer(max_features=300)
        tfidf = vec.fit_transform(sections)
        dense = tfidf.toarray()

        similarities: list[float] = []
        for i in range(len(dense) - 1):
            a, b = dense[i], dense[i + 1]
            dot = np.dot(a, b)
            norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
            similarities.append(float(dot / norm))

        avg_sim = float(np.mean(similarities)) if similarities else 0.0
        # Ideal coherence: moderate similarity (0.15–0.50), not identical nor disjoint
        # Map to 0-1: penalize both too-low (<0.05: disjoint) and too-high (>0.9: duplicate)
        if avg_sim < 0.05:
            coherence = max(0.0, avg_sim * 10)  # 0.05 → 0.5
        elif avg_sim > 0.80:
            coherence = max(0.0, (1.0 - avg_sim) * 5)  # 0.80 → 1.0, 0.95 → 0.25
        else:
            coherence = min(1.0, avg_sim * 2)  # 0.15 → 0.3, 0.40 → 0.8

        return round(coherence, 3), {
            "section_count": len(sections),
            "transitions": len(similarities),
            "avg_similarity": round(avg_sim, 4),
            "similarities": [round(s, 4) for s in similarities],
        }
    except ImportError:
        # scikit-learn not available — fallback: check if section headers exist
        return 0.5, {"section_count": len(sections), "error": "scikit-learn not installed"}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Relevance (to user prompt)
# ═══════════════════════════════════════════════════════════════════════════════

def _score_relevance_bigram_fallback(content: str, user_prompt: str) -> tuple[float, dict]:
    """Fallback: character bigram overlap — lightweight but surface-level proxy."""
    if not user_prompt or not content:
        return 0.5, {}

    prompt_bigrams = {user_prompt[i:i + 2] for i in range(len(user_prompt) - 1)}
    if not prompt_bigrams:
        return 0.5, {}

    hits = sum(1 for bg in prompt_bigrams if bg in content)
    coverage = hits / len(prompt_bigrams)

    first_chunk = content[:max(200, len(content) // 5)]
    early_hits = sum(1 for bg in prompt_bigrams if bg in first_chunk)
    early_coverage = early_hits / len(prompt_bigrams) if prompt_bigrams else 0.0

    score = 0.6 * coverage + 0.4 * early_coverage
    score = min(1.0, score * 1.5)

    return round(score, 3), {
        "method": "bigram_fallback",
        "prompt_bigram_count": len(prompt_bigrams),
        "coverage": round(coverage, 3),
        "early_coverage": round(early_coverage, 3),
    }


def _score_relevance(content: str, user_prompt: str) -> tuple[float, dict]:
    """Measure how well the generated content addresses the user prompt.

    Uses embedding-based cosine similarity for genuine semantic relevance,
    avoiding the surface-level false positives of character bigram overlap
    (e.g. "人工智能报告" vs "人工智障报告" would score near-perfect with
    bigrams but near-zero with embeddings).
    """
    if not user_prompt or not content:
        return 0.5, {}

    try:
        import math
        from agent_file_create.rag.embedder import embed_texts

        # Sample the first ~3000 chars — typically covers the abstract /
        # executive summary, which captures the document's overall topic.
        content_sample = content[:3000]

        vecs = embed_texts([user_prompt, content_sample], timeout_s=30, max_batch=2)
        if len(vecs) != 2 or not vecs[0] or not vecs[1]:
            raise RuntimeError("embed_texts returned insufficient vectors")

        prompt_vec, content_vec = vecs[0], vecs[1]

        dot = sum(a * b for a, b in zip(prompt_vec, content_vec))
        norm_a = math.sqrt(sum(a * a for a in prompt_vec))
        norm_b = math.sqrt(sum(b * b for b in content_vec))

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.5, {"method": "embedding_cosine", "error": "zero_norm_vector"}

        similarity = dot / (norm_a * norm_b)
        score = max(0.0, min(1.0, similarity))

        return round(score, 3), {
            "method": "embedding_cosine",
            "similarity": round(similarity, 4),
            "prompt_len": len(user_prompt),
            "content_sample_len": len(content_sample),
        }
    except Exception:
        return _score_relevance_bigram_fallback(content, user_prompt)


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_decomposed_eval(
    content: str,
    outline: str,
    analysis_results: List[dict],
    user_prompt: str = "",
) -> tuple[DimensionScores, dict]:
    """Run all four decomposed metrics. Returns (scores, details_dict)."""

    faith_score, warnings = _score_faithfulness(content, analysis_results)
    comp_score, comp_details = _score_completeness(content, outline)
    coh_score, coh_details = _score_coherence(content)
    rel_score, rel_details = _score_relevance(content, user_prompt)

    return DimensionScores(
        relevance=rel_score,
        faithfulness=faith_score,
        coherence=coh_score,
        completeness=comp_score,
    ), {
        "faithfulness_warnings": warnings,
        "completeness": comp_details,
        "coherence": coh_details,
        "relevance": rel_details,
    }
