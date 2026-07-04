"""Quality assurance functions extracted from document_service.py.

Includes: faithfulness check, hallucination re-retrieval, citation verification,
contrastive claim verification, aspect coverage, and factscore evaluation.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Atomic fact evaluation ────────────────────────────────────────────────────

def _decompose_to_atomic_facts(content: str) -> list[str]:
    """Break content into atomic facts — each one sentence, one claim."""
    from agent_file_create.llm_client import call_llm as _af_call

    text = (content or "").strip()
    if not text or len(text) < 100:
        return []

    prompt = (
        "请将以下文本分解为原子事实列表。每条原子事实应满足：\n"
        "1. 只包含一个可独立验证的信息点\n"
        "2. 是一句完整的陈述句\n"
        "3. 长度不超过 30 个汉字\n\n"
        f"文本：\n{text[:2000]}\n\n"
        "输出格式：每行一条原子事实，不要编号。"
    )
    try:
        raw = _af_call(prompt, timeout_s=20, temperature=0.0, num_predict=300,
                       system="你是一个中文文档处理助手。只输出原子事实，每行一条。")
        facts = [f.strip() for f in (raw or "").splitlines()
                 if f.strip() and len(f.strip()) > 5]
        return facts[:20]
    except Exception:
        return []


def _verify_atomic_facts(facts: list[str], source_text: str) -> dict:
    """Verify each atomic fact against source materials. Returns {fact: supported_bool}."""
    from agent_file_create.llm_client import call_llm as _vf_call

    if not facts or not source_text:
        return {}

    facts_text = "\n".join(f"[{i+1}] {f}" for i, f in enumerate(facts))
    prompt = (
        "请逐条判断以下原子事实是否能在参考资料中找到依据。\n\n"
        f"参考资料：\n{source_text[:5000]}\n\n"
        f"原子事实：\n{facts_text}\n\n"
        "对每条原子事实，如果能在参考资料中找到明确依据，输出 'OK'；否则输出 'MISS'。\n"
        "每行一条，格式：序号 OK|MISS"
    )
    try:
        raw = _vf_call(prompt, timeout_s=20, temperature=0.0, num_predict=200,
                       system="你是一个中文文档处理助手。只输出每条事实的判断结果。")
        result = {}
        for line in (raw or "").splitlines():
            line = line.strip()
            m = re.match(r"(\d+)\s*(OK|MISS)", line, re.IGNORECASE)
            if m:
                idx = int(m.group(1)) - 1
                result[facts[idx] if idx < len(facts) else line] = m.group(2).upper() == "OK"
        for f in facts:
            if f not in result:
                result[f] = False
        return result
    except Exception:
        return {f: False for f in facts}


# ── Aspect coverage evaluation ────────────────────────────────────────────────

def _extract_aspects(analysis_results: list) -> list[str]:
    """Extract key aspects/topics from source materials that the report should cover."""
    from agent_file_create.llm_client import call_llm as _ea_call

    source_summary = ""
    if analysis_results:
        for i, ar in enumerate(analysis_results[:10]):
            title = str(ar.get("title") or ar.get("filename") or "").strip()
            summary = str(ar.get("summary") or "").strip()
            if title or summary:
                source_summary += f"[{i+1}] {title}: {summary[:300]}\n"

    if len(source_summary.strip()) < 80:
        return []

    prompt = (
        "请从以下来源材料中提取 3-8 个关键主题/方面（aspects），"
        "这些是报告应该覆盖的重要内容维度。每个 aspect 用一句话描述。\n\n"
        f"来源材料：\n{source_summary[:1500]}\n\n"
        "输出：每行一个 aspect，不要编号。即使材料较少，也请尽力提取。"
    )
    try:
        raw = _ea_call(prompt, timeout_s=20, temperature=0.0, num_predict=200,
                       system="你是一个中文文档处理助手。只输出关键主题，每行一个。")
        aspects = [a.strip() for a in (raw or "").splitlines()
                   if a.strip() and len(a.strip()) > 5]
        return aspects[:8]
    except Exception:
        return []


# ── Refine retrieved context (CRAG-style) ─────────────────────────────────────

def _refine_retrieved_context(query: str, hits: list, max_chars: int = 2000) -> str:
    """Decompose retrieved chunks into sentences, filter irrelevant ones, recompose.

    CRAG-style: keeps only sentences relevant to the query, drops noise.
    """
    from agent_file_create.llm_client import call_llm as _rf_call

    if not hits:
        return ""

    all_sentences = []
    for h in hits:
        content = str(h.content or "").strip()
        if not content:
            continue
        for sent in re.split(r"[。！？.!?\n]+", content):
            sent = sent.strip()
            if len(sent) >= 8:
                all_sentences.append(sent)

    if not all_sentences:
        return ""

    sentences_text = "\n".join(f"[S{i+1}] {s}" for i, s in enumerate(all_sentences[:30]))
    prompt = (
        "请从以下检索到的句子中筛选出与问题相关的句子，过滤掉无关内容。\n\n"
        f"问题：{query[:200]}\n\n"
        f"候选句子：\n{sentences_text[:3000]}\n\n"
        "输出相关句子的序号，用逗号分隔（如：S1,S3,S5）。如果没有相关句子，回复 NONE。"
    )
    try:
        raw = _rf_call(prompt, timeout_s=15, temperature=0.0, num_predict=100,
                       system="你是一个中文文档处理助手。只输出相关句子序号。")
        indices = set()
        for m in re.findall(r"S?(\d+)", raw or ""):
            idx = int(m) - 1
            if 0 <= idx < len(all_sentences):
                indices.add(idx)
        if not indices:
            return ""
        refined = ""
        for i in sorted(indices):
            s = all_sentences[i]
            if len(refined) + len(s) + 2 > max_chars:
                break
            refined += s + "。"
        logger.info("refine_context query=%.50s hits=%d sents=%d kept=%d chars=%d",
                    query, len(hits), len(all_sentences), len(indices), len(refined))
        return refined.strip()
    except Exception:
        return "。".join(all_sentences[:15])[:max_chars]


# ── Cross-section consistency check ───────────────────────────────────────────

def _check_consistency(content: str, task_id: str = "") -> dict:
    """Check adjacent sections for contradictory or inconsistent claims."""
    from agent_file_create.llm_client import call_llm as _cc_call

    sections = []
    current_heading = None
    current_body = []
    for line in (content or "").splitlines():
        line = line.strip()
        m = re.match(r"^##\s+(.+)", line)
        if m:
            if current_heading and current_body:
                sections.append((current_heading, " ".join(current_body)[:300]))
            current_heading = m.group(1).strip()
            current_body = []
        elif current_heading and line and not line.startswith("> "):
            current_body.append(line)
    if current_heading and current_body:
        sections.append((current_heading, " ".join(current_body)[:300]))

    if len(sections) < 2:
        return {"consistency_score": 1.0, "flagged_pairs": []}

    flagged = []
    total_checks = 0
    for i in range(len(sections) - 1):
        h1, b1 = sections[i]
        h2, b2 = sections[i + 1]
        if not b1 or not b2:
            continue
        total_checks += 1
        prompt = (
            "请检查以下两个相邻章节的关键主张是否存在矛盾或显著不一致。\n"
            "只关注事实层面的冲突（如数据矛盾、结论相反），不关注措辞风格差异。\n\n"
            f"章节A「{h1}」：{b1[:200]}\n\n"
            f"章节B「{h2}」：{b2[:200]}\n\n"
            "回复 OK 表示一致，CONFLICT: <原因> 表示有矛盾。只回复一个词。"
        )
        try:
            raw = _cc_call(prompt, timeout_s=15, temperature=0.0, num_predict=80,
                           system="你是一个中文文档处理助手。只回复OK或CONFLICT。")
            if raw and "CONFLICT" in raw.upper():
                flagged.append({"section_a": h1, "section_b": h2, "reason": raw})
        except Exception:
            pass

    score = 1.0 - (len(flagged) / total_checks if total_checks > 0 else 0)
    return {"consistency_score": round(score, 4), "flagged_pairs": flagged}


# ── Coverage gap filling ──────────────────────────────────────────────────────

def _fill_coverage_gaps(content: str, uncovered_aspects: list[str],
                         analysis_results: list, task_id: str = "") -> str:
    """For uncovered aspects, retrieve from KB and generate supplemental content."""
    from agent_file_create.llm_client import call_llm as _fg_call

    if not uncovered_aspects or not analysis_results:
        return content

    for aspect in uncovered_aspects[:3]:
        kb_hits = []
        try:
            from agent_file_create.rag.kb import KnowledgeBase
            _kb = KnowledgeBase()
            kb_hits = _kb.search(kb=str(task_id), query=str(aspect), top_k=5)
        except Exception:
            pass

        ctx = ""
        if kb_hits:
            for j, h in enumerate(kb_hits[:3]):
                chunk = str(h.content or "")[:300]
                if chunk:
                    ctx += f"[KB{j+1}] {chunk}\n"
        if not ctx:
            ctx = "\n".join(
                str(ar.get("summary") or "")[:300]
                for ar in analysis_results[:2] if isinstance(ar, dict)
            )

        prompt = (
            f"你是一个中文文档处理助手。只输出补充段落。"
            f"请根据以下材料为报告补充关于「{aspect}」的段落（100-200字）：\n\n{ctx[:1200]}"
        )
        try:
            gap_section = _fg_call(prompt, timeout_s=20, temperature=0.1, num_predict=300,
                                   system="你是一个中文文档处理助手。只输出补充段落。")
            if gap_section and len(gap_section.strip()) > 20:
                content = content.rstrip() + "\n\n" + gap_section.strip()
        except Exception:
            pass

    return content


# ── Contrastive claim verification ────────────────────────────────────────────

def _verify_contrastive_claims(content: str, source_text: str,
                                task_id: str = "") -> dict:
    """Detect contrastive claims and verify both sides independently."""
    from agent_file_create.llm_client import call_llm as _vc_call

    claims = []
    for pat, label in [
        (r"(.{5,40})优于(.{5,40})", "优于"),
        (r"(.{5,40})高于(.{5,40})", "高于"),
        (r"(.{5,40})相比(.{5,40})更(.{5,20})", "相比更"),
    ]:
        for m in re.finditer(pat, content):
            claims.append({
                "pattern": label,
                "left": m.group(1).strip(),
                "right": m.group(2).strip(),
                "context": content[max(0,m.start()-30):m.end()+30],
            })

    if not claims:
        return {"flagged_count": 0, "total_count": 0, "details": []}

    unique = []
    seen = set()
    for c in claims:
        key = c["left"][:10] + c["right"][:10]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    claims = unique[:8]

    if not source_text:
        return {"flagged_count": 0, "total_count": len(claims), "details": []}

    flagged = []
    for c in claims:
        try:
            prompt = (
                f"判断以下对比论断是否能在来源材料中找到两侧的独立支撑。\n\n"
                f"来源：{source_text[:2000]}\n\n"
                f"论断：{c['context'][:200]}\n\n"
                f"如果对比双方都在来源中有依据，回复OK。如果有任一方不在来源中，回复MISS: <原因>。"
            )
            raw = _vc_call(prompt, timeout_s=15, temperature=0.0, num_predict=100,
                           system="你是一个中文文档处理助手。只回复OK或MISS。")
            if raw and "MISS" in raw.upper():
                flagged.append({"pattern": c["pattern"], "context": c["context"][:120],
                                "reason": (raw or "").strip()[:120]})
        except Exception:
            pass

    if flagged:
        logger.info("contrastive_verify task=%s flagged=%d/%d",
                    task_id, len(flagged), len(claims))
    return {"flagged_count": len(flagged), "total_count": len(claims),
            "details": flagged}


# ── FActScore + Coverage computation ──────────────────────────────────────────

def _compute_factscore_and_coverage(content: str, analysis_results: list,
                                     task_id: str = "") -> dict:
    """Compute FActScore-style faithfulness + aspect coverage metrics."""
    source_text = ""
    if analysis_results:
        for i, ar in enumerate(analysis_results[:8]):
            if isinstance(ar, dict):
                source_text += " ".join(
                    str(ar.get(k) or "") for k in ("title", "summary", "key_points")
                    if ar.get(k)
                ) + "\n"

    if len(source_text.strip()) < 100:
        logger.info("factscore_source_too_short task=%s chars=%d", task_id, len(source_text))
        return {"factscore": 0.5, "coverage": 0.5, "facts_count": 0,
                "verified_count": 0, "covered_count": 0, "aspects_count": 0,
                "uncovered_aspects": []}

    facts = _decompose_to_atomic_facts(content)
    verification = _verify_atomic_facts(facts, source_text) if facts else {}
    verified = sum(1 for v in verification.values() if v)
    factscore = verified / len(facts) if facts else 0.5

    aspects = _extract_aspects(analysis_results)
    if aspects and content:
        covered = 0
        for asp in aspects:
            terms = re.findall(r'[\w一-鿿]{2,}', asp)[:3]
            if terms and sum(1 for t in terms if t in content) >= max(1, len(terms) // 2):
                covered += 1
        coverage = covered / len(aspects)
        uncovered = [asp for asp in aspects
                     if not (re.findall(r'[\w一-鿿]{2,}', asp)[:3]
                             and sum(1 for t in re.findall(r'[\w一-鿿]{2,}', asp)[:3]
                                     if t in content) >= max(1, len(re.findall(r'[\w一-鿿]{2,}', asp)[:3]) // 2))]
    else:
        coverage = 0.5
        uncovered = []

    return {"factscore": round(factscore, 3), "coverage": round(coverage, 3),
            "verified_count": verified, "facts_count": len(facts),
            "covered_count": covered if aspects else 0,
            "aspects_count": len(aspects), "uncovered_aspects": uncovered}


# ── Hallucination-triggered re-retrieval (main faithfulness hook) ──────────────

def _run_faithfulness_checks(
    *,
    content: str,
    analysis_results: list,
    task_id: str,
    output_dir: str,
) -> str:
    """Run faithfulness check and hallucination-triggered re-retrieval on content.

    Delegates to QualityPipeline for structured, decomposable execution.
    """
    from agent_file_create.quality import QualityPipeline, QualityContext

    ctx = QualityContext(
        content=content,
        analysis_results=analysis_results or [],
        task_id=str(task_id),
        output_dir=str(output_dir),
    )
    result = QualityPipeline().run(ctx)
    return str(result.content or "")
