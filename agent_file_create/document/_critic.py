# -*- coding: utf-8 -*-
"""
Critic — 正文生成后的自动质检节点。

在 content 生成完毕后、satisfaction_content 人工确认前，自动对照大纲和原始材料
审查生成的正文，输出问题列表和修正建议。
"""

from __future__ import annotations

import logging
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
4. 证据缺口 — 如果发现高严重度问题是因为检索材料不足导致的，逐条给出 1-2 个可检索的关键词。

输出格式：
- 如果正文质量合格，只回复：OK
- 如果有问题，先逐条列出：## 问题 N | 类型 | 位置 | 描述 | 严重程度(高/中/低)
- 如果存在证据缺口，最后追加一行：@@SEARCH: 关键词1, 关键词2"""),
])


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
        {"issues": [...], "raw": str, "passed": bool}
    """
    if not content or len(content) < 100:
        logger.info("critic skip (content too short: %d chars)", len(content or ""))
        return {"issues": [], "raw": "", "passed": True}

    outline_short = (outline or "")[:1500]
    materials_short = (materials or "")[:2000]
    content_short = content[:5000]

    try:
        llm = get_chat_model(
            style=CONTENT_API_STYLE,
            model=CONTENT_MODEL_NAME,
            endpoint=CONTENT_API_ENDPOINT,
            api_key=CONTENT_API_KEY,
            temperature=0.0,
            max_tokens=600,
            timeout_s=60,
        )
        chain = CRITIC_PROMPT | llm | StrOutputParser()
        raw = (chain.invoke({
            "outline": outline_short,
            "materials": materials_short,
            "content": content_short,
        }) or "").strip()

        if raw.upper() == "OK":
            logger.info("critic passed")
            return {"issues": [], "raw": raw, "passed": True}

        # Parse issues
        issues: list[dict] = []
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

        logger.info("critic done issues=%d suggested_queries=%d",
                    len(issues), len(suggested_queries))
        return {
            "issues": issues, "raw": raw, "passed": len(issues) == 0,
            "suggested_queries": suggested_queries,
        }

    except Exception as e:
        logger.warning("critic failed: %s", e)
        return {"issues": [], "raw": "", "passed": True}


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

    issues_text = "\n".join(
        f"- [{i.get('severity')}] {i.get('location')}: {i.get('description')}"
        for i in fixable
    )

    try:
        llm = get_chat_model(
            style=CONTENT_API_STYLE,
            model=CONTENT_MODEL_NAME,
            endpoint=CONTENT_API_ENDPOINT,
            api_key=CONTENT_API_KEY,
            temperature=0.0,
            max_tokens=4000,
            timeout_s=120,
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
