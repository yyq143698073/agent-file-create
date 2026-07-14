# -*- coding: utf-8 -*-
"""
Critic — 正文生成后的自动质检节点。

在 content 生成完毕后、satisfaction_content 人工确认前，自动对照大纲和原始材料
审查生成的正文，输出问题列表和修正建议。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent_file_create.config import (
    CONTENT_API_ENDPOINT,
    CONTENT_API_KEY,
    CONTENT_API_STYLE,
    CONTENT_MODEL_NAME,
)
from agent_file_create.llm_factory import get_chat_model

logger = logging.getLogger(__name__)

# ── Critic model: uses the content model by default ───────────────────
# For cloud APIs we substitute "pro" with "flash" for speed.  For local
# (Ollama) models the name passes through unchanged — there's no
# distinction between reasoning and generation variants.
_CRITIC_MODEL = (CONTENT_MODEL_NAME or "").strip()
if "pro" in _CRITIC_MODEL.lower() and "flash" not in _CRITIC_MODEL.lower():
    _CRITIC_MODEL = _CRITIC_MODEL.replace("-pro", "-flash")

# ── Critic prompt ─────────────────────────────────────────────────────────────

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个严格的文档质检员。对照原始材料和报告大纲，审查报告正文质量。"),
    ("human", """\
请审查以下报告正文，只标记明确的问题。

<报告大纲>
{outline}
</报告大纲>

<原始材料摘要>
{materials}
</原始材料摘要>

<报告正文>
{content}
</报告正文>

审查维度：
1. 事实准确性 — 正文中的数据、实体、结论是否能在材料中找到依据？没有依据的要标记。
2. 结构完整性 — 是否覆盖了大纲中的所有要点？遗漏的要指出。
3. 逻辑连贯性 — 章节之间衔接是否自然？是否有矛盾？
4. 数据重复 — 同一个具体数值（如百分比、绝对数量）是否在多处重复出现？如果同一数字在3个以上不同章节中被完整引用，标记为"数据过度重复"。
5. 缩写合规 — 首次出现的英文缩写是否在括号中给出了全称解释？未解释的要标记。
6. 证据缺口 — 如果发现高严重度问题是因为检索材料不足导致的，逐条给出 1-2 个可检索的关键词。

输出格式：
- 如果正文质量合格，只回复：OK
- 如果有问题，先逐条列出：## 问题 N | 类型 | 位置 | 描述 | 严重程度(高/中/低)
- 如果存在证据缺口，最后追加一行：@@SEARCH: 关键词1, 关键词2"""),
])

FALLBACK_CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个严格的事实审查员。只输出 OK 或固定格式的问题列表。"),
    ("human", """\
请根据材料审查正文，只关注最明确的问题。

<报告大纲>
{outline}
</报告大纲>

<原始材料摘要>
{materials}
</原始材料摘要>

<报告正文>
{content}
</报告正文>

检查规则：
1. 如果正文中的实体、数字、结论在材料中找不到依据，标记为事实准确性问题。
2. 如果正文明显没有覆盖大纲中的关键点，标记为结构完整性问题。
3. 不要解释过程，不要输出额外段落。

输出格式：
- 如果没有明确问题，只输出：OK
- 如果有问题，每行输出：## 问题 N | 类型 | 位置 | 描述 | 严重程度(高/中/低)
- 可选：最后一行输出 @@SEARCH: 关键词1, 关键词2"""),
])


def _invoke_critic_prompt(
    *,
    prompt: ChatPromptTemplate,
    outline: str,
    materials: str,
    content: str,
    max_tokens: int,
    timeout_s: int,
) -> str:
    llm = get_chat_model(
        style=CONTENT_API_STYLE,
        model=_CRITIC_MODEL,
        endpoint=CONTENT_API_ENDPOINT,
        api_key=CONTENT_API_KEY,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout_s=min(timeout_s, 45),  # flash models respond faster
    )
    chain = prompt | llm | StrOutputParser()
    return (chain.invoke({
        "outline": outline,
        "materials": materials,
        "content": content,
    }) or "").strip()


def _parse_critic_output(raw: str) -> tuple[list[dict[str, str]], list[str]]:
    issues: list[dict[str, str]] = []
    suggested_queries: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("##") and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                issues.append({
                    "raw": line,
                    "type": parts[1] if len(parts) > 1 else "",
                    "location": parts[2] if len(parts) > 2 else "",
                    "description": parts[3] if len(parts) > 3 else "",
                    "severity": parts[4] if len(parts) > 4 else "中",
                })
        elif line.upper().startswith("@@SEARCH:"):
            qs = line.split(":", 1)[1].strip()
            suggested_queries = [q.strip() for q in qs.split(",") if q.strip()]
    return issues, suggested_queries


def _heuristic_critic_review(*, outline: str, materials: str, content: str) -> dict[str, Any]:
    """Deterministic fallback when the local LLM returns empty output.

    This is intentionally conservative: it only flags obvious unsupported
    numbers/entities and clear outline omissions so local evaluation can finish
    with an interpretable result.
    """
    def _norm_compare(text: str) -> str:
        return re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower()).strip()

    materials_norm = _norm_compare(materials)
    content_norm = (content or "").lower()
    issues: list[dict[str, str]] = []
    generic_title_words = {
        "A", "An", "The", "Several", "English", "British", "American", "Original",
        "Radio", "Television", "Film", "Movie", "Actor", "Actress", "BBC", "He",
        "She", "They", "It", "This", "That", "These", "Those",
    }

    sentences = [s.strip() for s in re.split(r"[。！？.!?]\s*", content or "") if s.strip()]
    for sent in sentences[:10]:
        nums = re.findall(r"\b\d[\d,\.:%-]*\b", sent)
        missing_nums = [n for n in nums if n and n not in materials]
        if missing_nums:
            issues.append({
                "raw": "",
                "type": "事实准确性",
                "location": sent[:80],
                "description": f"正文包含材料中未出现的数字/年份：{missing_nums[0]}。",
                "severity": "高",
            })
            continue

        entity_phrases = []
        for ent in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", sent):
            parts = ent.split()
            if all(p in generic_title_words for p in parts):
                continue
            entity_phrases.append(ent)
        missing_entities = []
        for ent in entity_phrases:
            ent_norm = _norm_compare(ent)
            if len(ent_norm) < 4:
                continue
            if ent_norm not in materials_norm:
                missing_entities.append(ent)
        if missing_entities:
            issues.append({
                "raw": "",
                "type": "事实准确性",
                "location": sent[:80],
                "description": f"正文提到了材料中未找到依据的实体：{missing_entities[0]}。",
                "severity": "高",
            })
            continue

    point_lines = [ln.strip() for ln in (outline or "").splitlines() if ln.strip().startswith("## Point")]
    stop_words = {"what", "when", "where", "which", "who", "whom", "whose", "the", "and", "with"}
    for ln in point_lines[:3]:
        title = ln.split(":", 1)[-1].strip().lower()
        title_terms = [
            tok for tok in re.findall(r"[a-z]{4,}", title)
            if tok not in stop_words
        ][:4]
        if title_terms and not any(tok in content_norm for tok in title_terms[:2]):
            issues.append({
                "raw": "",
                "type": "结构完整性",
                "location": ln[:80],
                "description": "正文可能未覆盖该大纲要点的核心信息。",
                "severity": "中",
            })
            break

    if not issues:
        return {
            "issues": [],
            "raw": "OK",
            "passed": True,
            "suggested_queries": [],
            "error": "",
            "fallback_used": "heuristic",
        }

    rendered = []
    for idx, issue in enumerate(issues[:4], start=1):
        rendered.append(
            f"## 问题 {idx} | {issue['type']} | {issue['location']} | {issue['description']} | {issue['severity']}"
        )
    return {
        "issues": issues[:4],
        "raw": "\n".join(rendered),
        "passed": False,
        "suggested_queries": [],
        "error": "",
        "fallback_used": "heuristic",
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_critic(
    *,
    content: str,
    outline: str = "",
    materials: str = "",
) -> dict[str, Any]:
    """Run automated quality review on generated content.

    Args:
        content: 生成的报告正文
        outline: 报告大纲
        materials: 原始材料摘要（来自 analysis_results）

    Returns:
        {"issues": [...], "raw": str, "passed": bool, "error": str}
    """
    if not content or len(content) < 100:
        logger.info("critic skip (content too short: %d chars)", len(content or ""))
        return {"issues": [], "raw": "", "passed": True}

    outline_short = (outline or "")[:2000]
    materials_short = (materials or "")[:2500]
    # Use up to 12000 chars so the critic can see the majority of a typical
    # 5000-8000 word report.  Previously 5000 missed the last ~15 %.
    content_short = content[:12000] if len(content) > 12000 else content

    try:
        raw = _invoke_critic_prompt(
            prompt=CRITIC_PROMPT,
            outline=outline_short,
            materials=materials_short,
            content=content_short,
            max_tokens=600,
            timeout_s=60,
        )

        # Local models may occasionally return empty text or drift away from the
        # requested schema. Retry once with a shorter, more constrained prompt.
        if not raw:
            logger.info("critic fallback retry (empty primary response)")
            raw = _invoke_critic_prompt(
                prompt=FALLBACK_CRITIC_PROMPT,
                outline=outline_short[:1200],
                materials=materials_short[:1600],
                content=content_short[:2200],
                max_tokens=240,
                timeout_s=45,
            )

        if not raw:
            logger.info("critic heuristic fallback (empty response)")
            return _heuristic_critic_review(
                outline=outline_short[:1200],
                materials=materials_short[:1600],
                content=content_short[:2200],
            )

        if raw.upper() == "OK":
            logger.info("critic passed")
            return {"issues": [], "raw": raw, "passed": True, "error": ""}

        issues, suggested_queries = _parse_critic_output(raw)

        # Treat non-empty but unparsable output as a failed review instead of
        # silently passing; otherwise local-model format drift can mask issues.
        if raw and not issues:
            logger.info("critic fallback retry (schema drift)")
            retry_raw = _invoke_critic_prompt(
                prompt=FALLBACK_CRITIC_PROMPT,
                outline=outline_short[:1200],
                materials=materials_short[:1600],
                content=content_short[:2200],
                max_tokens=240,
                timeout_s=45,
            )
            if retry_raw and retry_raw.upper() == "OK":
                logger.info("critic fallback passed")
                return {"issues": [], "raw": retry_raw, "passed": True, "error": ""}
            if retry_raw:
                raw = retry_raw
                issues, suggested_queries = _parse_critic_output(raw)

        if raw and not issues:
            logger.info("critic heuristic fallback (schema drift)")
            heuristic = _heuristic_critic_review(
                outline=outline_short[:1200],
                materials=materials_short[:1600],
                content=content_short[:2200],
            )
            if heuristic.get("raw"):
                return heuristic
            issues.append({
                "raw": raw[:240],
                "type": "格式异常",
                "location": "",
                "description": "critic 返回了未按约定格式组织的结果，需要人工复核原始输出。",
                "severity": "中",
            })

        logger.info("critic done issues=%d suggested_queries=%d",
                    len(issues), len(suggested_queries))
        return {
            "issues": issues, "raw": raw, "passed": len(issues) == 0,
            "suggested_queries": suggested_queries, "error": "",
        }

    except Exception as e:
        logger.exception("critic failed")
        err = f"{type(e).__name__}: {e}"
        return {
            "issues": [{
                "raw": err,
                "type": "执行失败",
                "location": "",
                "description": f"critic 调用失败，需要人工复核。{err}",
                "severity": "高",
            }],
            "raw": "",
            "passed": False,
            "suggested_queries": [],
            "error": err,
        }


# ── Auto-fix prompt ──────────────────────────────────────────────────────────

FIX_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个精准的文档修正助手。只修正下面列出的具体问题，其余内容一字不改。"),
    ("human", """\
请修正以下报告正文中的指定问题。

<需要修正的问题>
{issues_text}
</需要修正的问题>

<原始材料（可作为修正依据）>
{materials}
</原始材料（可作为修正依据）>

<报告正文>
{content}
</报告正文>

修正规则：
1. 只修改问题中提到的具体数据/实体/描述，其余内容原封不动
2. 修正后的数据必须能在原始材料中找到依据
3. 如果材料中找不到依据，删除该处声明，替换为"数据待核实"
4. 输出完整的修正后正文"""),
])


def run_critic_fix(
    *,
    content: str,
    issues: list[dict],
    materials: str = "",
) -> str:
    """Auto-fix low/medium severity issues found by the critic.

    Only corrects issues with severity != "高". High-severity issues are
    left for human review.

    Args:
        content: 原始正文
        issues: Critic 发现的问题列表
        materials: 原始材料摘要

    Returns:
        修正后的正文（如果没有可修正的问题则返回原内容）
    """
    fixable = [i for i in issues if i.get("severity") != "高"]
    if not fixable:
        logger.info("critic_fix skip (no fixable issues)")
        return content

    # ── Skip fix when only a few low-severity issues ──────────────────
    # For ≤3 issues all marked "低", the LLM fix call (60-120s) isn't
    # worth the marginal quality improvement.
    low_only = [i for i in fixable if i.get("severity") == "低"]
    if len(fixable) <= 3 and len(low_only) == len(fixable):
        logger.info("critic_fix skip (%d low-severity issues, not worth the LLM call)", len(fixable))
        return content

    issues_text = "\n".join(
        f"- [{i.get('severity')}] {i.get('location')}: {i.get('description')}"
        for i in fixable
    )

    try:
        llm = get_chat_model(
            style=CONTENT_API_STYLE,
            model=_CRITIC_MODEL,
            endpoint=CONTENT_API_ENDPOINT,
            api_key=CONTENT_API_KEY,
            temperature=0.0,
            max_tokens=4000,
            timeout_s=60,  # flash models don't need 120s
        )
        chain = FIX_PROMPT | llm | StrOutputParser()
        fixed = (chain.invoke({
            "issues_text": issues_text,
            "materials": (materials or "")[:2500],
            "content": content[:6000],
        }) or "").strip()

        if fixed and len(fixed) > len(content) * 0.5:
            logger.info(
                "critic_fix done fixed=%d issues", len(fixable)
            )
            return fixed
        else:
            logger.warning("critic_fix returned too-short content, keeping original")
            return content

    except Exception as e:
        logger.warning("critic_fix failed: %s", e)
        return content
