"""Approach A: LLM-as-Judge — automated quality scoring via an independent LLM.

Uses a separate LLM to evaluate the generated document on four dimensions
(Relevance, Faithfulness, Coherence, Completeness) on a 1–5 scale.

The judge LLM receives:
- The user's original prompt
- A summary of the source materials (analysis_results)
- The generated document content

and returns structured scores with reasoning.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from agent_file_create.llm_factory import get_chat_model
from agent_file_create.config import (
    CONTENT_API_STYLE,
    CONTENT_API_ENDPOINT,
    CONTENT_API_KEY,
    CONTENT_MODEL_NAME,
    MODEL_TIMEOUT,
)
from agent_file_create.evaluation.models import DimensionScores

logger = logging.getLogger(__name__)

# ── Judge prompt ──────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
你是一个严格、公正的文档质量评测员。你需要对一份 AI 生成的报告进行四维评分。

核心原则：
- 先列出每个维度的具体问题，再根据问题数量与严重程度打分。
- 5 分 = 近乎完美，极少出现；4 分 = 良好，有小瑕疵；3 分 = 合格，存在明显不足；
  2 分 = 较差，有较多或较重问题；1 分 = 极差，基本不可用。
- 只输出 JSON，不要任何额外文字。
"""

_JUDGE_USER_TEMPLATE = """\
【用户原始需求】
{user_prompt}

【源材料摘要】
{source_summary}

【生成的文档内容】
{content}

请按以下步骤评测，只输出 JSON：

1. 先列出每个维度发现的具体问题（没有则写"无"）
2. 再参照评分锚定给出 1–5 分

━━━ 评分锚定 ━━━

**relevance（相关性）—— 内容是否紧扣用户需求？**
- 5分：完全覆盖用户需求，无任何偏题或冗余段落
- 4分：主体切题，个别段落与核心需求关联较弱
- 3分：大约有 1/4 内容偏题或属于不相关的背景介绍
- 2分：近一半内容与需求无关，核心问题未得到充分回应
- 1分：基本文不对题，完全偏离用户需求

**faithfulness（忠实度）—— 断言是否有源材料依据？**
- 5分：所有关键断言、数据、结论均可在源材料中找到直接依据
- 4分：主体内容有依据，存在 ≤2 处推测性表述，但未歪曲事实
- 3分：存在 3–5 处无依据的断言，或 1 处明显事实错误
- 2分：多处关键数据/结论疑似编造，或 ≥2 处事实错误
- 1分：大量虚构内容，严重偏离源材料信息

**coherence（连贯性）—— 章节逻辑是否顺畅？**
- 5分：章节递进自然，无跳跃、无重复，读起来一气呵成
- 4分：整体连贯，个别段落衔接稍显生硬或有 1 处轻微重复
- 3分：存在明显的逻辑跳跃或 2–3 处内容重复，但不影响理解
- 2分：多处断裂或重复，读者需要自行推断逻辑关系
- 1分：章节堆砌，无逻辑顺序，基本无法流畅阅读

**completeness（完整性）—— 是否覆盖了应有要点？**
- 5分：覆盖了用户需求的所有关键维度，无重要遗漏
- 4分：覆盖了主要维度，遗漏 ≤1 个次要方面
- 3分：遗漏了 1 个重要方面，或多个次要方面
- 2分：遗漏了 ≥2 个重要方面，报告结构明显不完整
- 1分：仅覆盖了需求的很小一部分，大量关键内容缺失

━━━ 输出格式 ━━━

{{
  "issues": {{
    "relevance": ["<问题1>", "<问题2> 或 '无'"],
    "faithfulness": ["<问题1>", "<问题2> 或 '无'"],
    "coherence": ["<问题1>", "<问题2> 或 '无'"],
    "completeness": ["<问题1>", "<问题2> 或 '无'"]
  }},
  "scores": {{
    "relevance": {{"score": <1-5>, "reason": "<结合上述问题，说明为何打此分>"}},
    "faithfulness": {{"score": <1-5>, "reason": "<结合上述问题，说明为何打此分>"}},
    "coherence": {{"score": <1-5>, "reason": "<结合上述问题，说明为何打此分>"}},
    "completeness": {{"score": <1-5>, "reason": "<结合上述问题，说明为何打此分>"}}
  }},
  "overall_comment": "<总体评价，不超过100字>"
}}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_source_summary(analysis_results: List[dict], max_chars: int = 2000) -> str:
    """Condense analysis_results into a compact summary for the judge."""
    parts: list[str] = []
    chars = 0
    for i, ar in enumerate(analysis_results):
        if not isinstance(ar, dict):
            continue
        title = str(ar.get("title", "") or "").strip()
        summary = str(ar.get("summary", "") or "").strip()
        key_points = ar.get("key_points", [])
        if isinstance(key_points, list):
            kp_text = "; ".join(str(k) for k in key_points[:5])
        else:
            kp_text = ""

        chunk = f"[{title}] " if title else ""
        chunk += summary[:200] if summary else ""
        if kp_text:
            chunk += f" 要点: {kp_text}"
        if chars + len(chunk) > max_chars:
            parts.append(chunk[:max_chars - chars])
            break
        if chunk.strip():
            parts.append(chunk)
            chars += len(chunk)

    return "\n".join(parts) if parts else "（无源材料）"


def _parse_judge_response(raw: str) -> tuple[DimensionScores, str]:
    """Extract scores from LLM JSON response. Returns (scores, reasoning)."""
    # Clean markdown fences
    cleaned = re.sub(r"```json\s*|```", "", raw).strip()
    # Find JSON object
    s = cleaned.find("{")
    e = cleaned.rfind("}")
    if s != -1 and e != -1 and e > s:
        cleaned = cleaned[s:e + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return DimensionScores(), raw[:500]

    # Support both new format {"scores": {...}} and old flat format
    scores_block = data.get("scores", data)

    scores = DimensionScores()
    for dim in ("relevance", "faithfulness", "coherence", "completeness"):
        entry = scores_block.get(dim, {})
        if isinstance(entry, dict):
            val = float(entry.get("score", 0))
            scores.__setattr__(dim, val / 5.0)  # Normalize 1-5 → 0-1
        elif isinstance(entry, (int, float)):
            scores.__setattr__(dim, float(entry) / 5.0)

    reasoning = data.get("overall_comment", "") or ""
    # Collect per-dimension reasons
    for dim in ("relevance", "faithfulness", "coherence", "completeness"):
        entry = scores_block.get(dim, {})
        if isinstance(entry, dict) and entry.get("reason"):
            reasoning += f" [{dim}: {entry['reason']}]"
    # Append issues if present
    issues_block = data.get("issues", {})
    if isinstance(issues_block, dict):
        for dim in ("relevance", "faithfulness", "coherence", "completeness"):
            dim_issues = issues_block.get(dim, [])
            if isinstance(dim_issues, list) and dim_issues:
                reasoning += f" [{dim}_issues: {'; '.join(dim_issues)}]"

    return scores, reasoning


# ── Public API ────────────────────────────────────────────────────────────────

def run_llm_judge(
    content: str,
    analysis_results: List[dict],
    user_prompt: str = "",
    *,
    model_name: Optional[str] = None,
    api_style: Optional[str] = None,
    api_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_s: int = MODEL_TIMEOUT,
) -> tuple[DimensionScores, str]:
    """Evaluate generated *content* with an independent LLM judge.

    Returns (scores_normalized_0_to_1, raw_reasoning_string).
    Returns zero scores if the LLM call fails.
    """
    if not content.strip():
        return DimensionScores(), "（内容为空，跳过评估）"

    source_summary = _build_source_summary(analysis_results)
    user_msg = _JUDGE_USER_TEMPLATE.format(
        user_prompt=user_prompt or "（未指定）",
        source_summary=source_summary,
        content=content[:10000],  # Truncate for judge context window
    )

    style = (api_style or CONTENT_API_STYLE).strip().lower()
    model = model_name or CONTENT_MODEL_NAME
    endpoint = api_endpoint or CONTENT_API_ENDPOINT
    key = api_key or CONTENT_API_KEY

    if not endpoint and not key:
        logger.warning("LLM judge: no API endpoint configured, skipping")
        return DimensionScores(), "（未配置 API，跳过 LLM 评估）"

    try:
        llm = get_chat_model(
            style=style,
            model=model,
            endpoint=endpoint,
            api_key=key,
            temperature=0.0,
            max_tokens=800,
            timeout_s=int(timeout_s),
        )
        response = llm.invoke([
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ])
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.warning("LLM judge call failed: %s", exc)
        return DimensionScores(), f"（LLM 调用失败: {exc}）"

    scores, reasoning = _parse_judge_response(raw)
    return scores, reasoning
