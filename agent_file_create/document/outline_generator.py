import logging
import re
import time
from typing import Any, Dict, List

from agent_file_create.config import OUTLINE_API_ENDPOINT, OUTLINE_API_KEY, OUTLINE_API_STYLE, OUTLINE_MODEL_NAME, MODEL_TIMEOUT
from agent_file_create.llm_client import call_llm

logger = logging.getLogger(__name__)

def _multimodal_digest(multimodal_results: Dict[str, Any], max_chars: int = 3500) -> str:
    items_raw = [(k, v) for k, v in (multimodal_results or {}).items() if isinstance(v, dict)]
    if not items_raw:
        return ""

    # Fair quota: each source gets at least 300 chars
    per_item = max(300, max_chars // len(items_raw))
    total = 0
    parts: list[str] = []

    for k, v in items_raw:
        title = str(v.get("title") or "").strip()
        summary = str(v.get("summary") or "").strip()
        key_points = v.get("key_points") if isinstance(v.get("key_points"), list) else []
        kp = "；".join([str(x).strip() for x in key_points[:4] if str(x).strip()])
        combined = " | ".join([x for x in [title, summary, kp] if x])
        if not combined:
            continue
        line = f"- {k}: {combined}"
        if len(line) > per_item:
            line = line[:per_item] + "…"
        parts.append(line)
        total += len(line)
        if total >= max_chars:
            break

    return "\n".join(parts).strip()


def _validate_outline(outline: str) -> List[str]:
    """Validate outline structure. Returns list of issues; empty list means valid."""
    issues: list[str] = []

    lines = [l for l in (outline or "").splitlines() if l.strip()]
    if not lines:
        return ["大纲为空"]

    # Must start with # heading
    if not re.match(r"^#\s", lines[0].strip()):
        issues.append("缺少一级标题(# )开头")

    # Must have at least one ## heading
    if not any(re.match(r"^##\s", l.strip()) for l in lines):
        issues.append("缺少二级标题(## )")

    # Must have at least one ### (core content subsection)
    if not any(re.match(r"^###\s", l.strip()) for l in lines):
        issues.append("缺少三级标题(### )子节")

    # Check heading level jumps (no skipping levels)
    levels: list[tuple[int, str]] = []
    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.+)", line.strip())
        if m:
            levels.append((len(m.group(1)), m.group(2).strip()))
    for i in range(1, len(levels)):
        if levels[i][0] - levels[i-1][0] > 1:
            issues.append(f"标题层级跳跃: L{levels[i-1][0]}→L{levels[i][0]} ({levels[i-1][1]} → {levels[i][1]})")

    # Minimum section count
    h2_count = sum(1 for lv, _ in levels if lv == 2)
    if h2_count < 3:
        issues.append(f"二级标题过少({h2_count}个)，至少需要3个")

    return issues


def _clean_llm_output(text: str) -> str:
    out = (text or "").strip()
    out = re.sub(r"^```[a-zA-Z]*\s*", "", out).strip()
    out = re.sub(r"\s*```$", "", out).strip()
    return out


def _build_outline_prompt(user_req: str, digest: str, feedback: str) -> str:
    rules = [
        "你是一个专业报告的大纲生成助手。请基于参考材料与用户需求输出 Markdown 大纲。",
        "",
        "结构规则（必须遵守）：",
        "1) # 一级标题 = 报告总标题（1个），应概括报告的核心主题。",
        "2) ## 二级标题 = 主章节（至少3个，建议不超过8个）。",
        "   命名原则：从材料中提取最突出的主题作为章节名，而不是套用「背景/分析/建议」的固定模板。",
        "   好的命名示例：「华东区Q3销售异常分析」「竞品A产品策略拆解」「供应链成本优化路径」",
        "   差的命名示例：「第一章 背景」「第一部分 概述」（过于空洞）",
        "3) ### 三级标题 = 子节（每个 ## 下至少1个，建议不超过5个）。",
        "   子节应拆解主章节的具体维度，如数据拆解、原因分析、方案对比、案例举证等。",
        "4) 建议不超过 #### 四级标题（如需更深层级，优先考虑拆分 ##）。",
        "5) 层级必须连续，禁止 # 直接跳 ###（跳过 ##）。",
        "6) 如果材料中没有支撑某类内容，不要强行编造该章节。宁可大纲短一些，也不要无中生有。",
        "7) 只输出 Markdown 大纲本身，不要解释、不要前言后语、不要代码块包裹。",
        "",
        "用户需求：",
        user_req,
        "",
        "参考材料摘要：",
        digest or "（无）",
    ]
    if feedback:
        rules += [
            "",
            "⚠️ 上一版大纲结构需要调整，具体问题：",
            feedback,
            "请针对性修正后重新输出。结构规则是硬性要求必须满足，但章节命名可以灵活调整。",
        ]
    rules.append("\n输出：")
    return "\n\n".join(rules).strip()


def generate_outline(multimodal_results: Dict[str, Any], user_prompt: str) -> str:
    digest = _multimodal_digest(multimodal_results)
    user_req = (user_prompt or "").strip() or "生成一份报告"

    t0 = time.perf_counter()
    best_outline = ""
    last_issues: list[str] = []

    for attempt in range(3):
        prompt = _build_outline_prompt(user_req, digest, "\n".join(last_issues) if last_issues else "")
        text = call_llm(
            prompt,
            timeout_s=MODEL_TIMEOUT,
            temperature=0.2,
            num_predict=900,
            system="你是一个中文报告助手，只输出 Markdown 大纲。",
            api_style=OUTLINE_API_STYLE,
            api_endpoint=OUTLINE_API_ENDPOINT,
            api_key=OUTLINE_API_KEY,
            model_name=OUTLINE_MODEL_NAME,
        )
        out = _clean_llm_output(text)

        if not out.startswith("#"):
            out = "# 报告\n\n" + out

        best_outline = out
        issues = _validate_outline(out)
        if not issues:
            t1 = time.perf_counter()
            logger.info(f"outline_done seconds={t1 - t0:.2f} prompt_chars={len(prompt)} outline_chars={len(out)} attempts={attempt + 1}")
            return out

        last_issues = issues
        logger.warning(f"outline_validation_failed attempt={attempt + 1} issues={issues}")

    t1 = time.perf_counter()
    logger.warning(f"outline_done_with_issues seconds={t1 - t0:.2f} attempts=3 issues={last_issues}")
    return best_outline
