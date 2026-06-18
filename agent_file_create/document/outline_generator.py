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


def _add_outline_numbering(outline: str) -> str:
    """Add hierarchical numbering to markdown outline headings.

    Transforms::

        # 报告标题
        ## 背景分析
        ### 行业现状
        ## 数据解读

    into::

        # 1. 报告标题
        ## 1.1 背景分析
        ### 1.1.1 行业现状
        ## 1.2 数据解读
    """
    lines = (outline or "").splitlines()
    result: list[str] = []
    # counters[i] = current count at heading level i (1-indexed: counters[1] for #, counters[2] for ##, ...)
    counters: list[int] = [0] * 10

    for line in lines:
        stripped = line.strip()
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
        if not m:
            result.append(line)
            continue

        level = len(m.group(1))
        title = m.group(2).strip()

        # Increment counter at this level, reset all deeper levels
        counters[level] += 1
        for lv in range(level + 1, len(counters)):
            counters[lv] = 0

        # Build hierarchical number: e.g. "1.1.2" for level 3
        number_parts = [str(counters[lv]) for lv in range(1, level + 1) if counters[lv] > 0]
        number = ".".join(number_parts) + "."

        # Preserve original indentation
        indent = line[:len(line) - len(line.lstrip())]
        result.append(f"{indent}{'#' * level} {number} {title}")

    return "\n".join(result)


def _build_outline_prompt(user_req: str, digest: str, feedback: str, target_words: int = 0, template_sections: list = None) -> str:
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
    ]
    if template_sections:
        sections_str = "、".join(template_sections)
        rules += [
            f"模板期望章节：{sections_str}",
            "上述章节名称来自用户选用的输出模板，请以此为主要结构来组织 ## 二级标题：",
            "• 措辞可以微调（如『局限性』→『现有方法的局限性分析』），但保持语义对应，确保最终渲染时模板占位符能匹配到内容；",
            "• 相近章节可以合并（如『研究背景』与『研究目的』合为一个章节），但不能直接删除；",
            "• 如果材料中有特别突出但模板未覆盖的主题，可额外增加 1~2 个章节；",
            "• 只有在材料确实完全没有相关内容时，才允许省略某个章节。",
            "• 若模板包含『局限性』『不足』『问题与挑战』类章节，请在大纲后半部分安排一个反思性章节，讨论本方法/框架自身的问题和不足，而非仅讨论前人工作的局限。",
            "",
        ]
    if target_words and target_words > 0:
        rules += [
            f"目标总字数：约 {target_words} 字。请据此调整章节数量（建议 {max(3, target_words // 1500)}~{max(5, target_words // 800)} 个 ## 章节）和每个章节的子节深度。",
            "字数越多，章节应越多、子节越深；字数越少，章节应精简、聚焦核心。",
        ]
    rules += [
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


def _check_outline_coverage(outline: str, user_req: str, template_sections: list = None) -> list[str]:
    """Lightweight LLM check: does the outline cover user requirements and template sections?

    Returns a list of missing or inadequately covered section names (empty = all good).
    """
    sections_to_check = (template_sections or [])[:]
    if not sections_to_check and not user_req:
        return []

    check_prompt = (
        "你是一个大纲质量审查助手。请检查以下大纲是否覆盖了所有要求的主题。\n\n"
        f"用户需求：{user_req[:500]}\n\n"
    )
    if sections_to_check:
        check_prompt += f"必须覆盖的章节：{'、'.join(sections_to_check)}\n\n"
    check_prompt += (
        f"大纲：\n{outline[:1500]}\n\n"
        '如果所有关键主题都已覆盖，回复"OK"。'
        "如果有缺失或覆盖不足的主题，只回复缺失的主题名称（一行一个），不要任何解释。"
    )

    try:
        text = call_llm(
            check_prompt,
            timeout_s=15,
            temperature=0.0,
            num_predict=120,
            system="你是一个中文文档处理助手。",
            api_style=OUTLINE_API_STYLE,
            api_endpoint=OUTLINE_API_ENDPOINT,
            api_key=OUTLINE_API_KEY,
            model_name=OUTLINE_MODEL_NAME,
        )
        result = (text or "").strip()
        if result.upper() in {"OK", "OK。", "无", "无缺失", "NONE"}:
            return []
        # Parse each line as a missing section
        missing = [line.strip().lstrip("-•·1234567890. ") for line in result.splitlines()]
        missing = [m for m in missing if m and len(m) > 1]
        return missing[:8]  # cap
    except Exception:
        return []  # Don't block on check failure


def generate_outline(multimodal_results: Dict[str, Any], user_prompt: str,
                     feedback: str = "", enriched_context: str = "",
                     target_words: int = 0, template_sections: list = None) -> str:
    digest = _multimodal_digest(multimodal_results)
    user_req = (user_prompt or "").strip() or "生成一份报告"

    # Prepend skill-enriched context to the digest if available
    if enriched_context.strip():
        digest = f"[技能搜集到的补充信息]\n{enriched_context.strip()}\n\n[文件内容摘要]\n{digest}"

    t0 = time.perf_counter()
    best_outline = ""
    last_issues: list[str] = []

    for attempt in range(2):
        validation_feedback = "\n".join(last_issues) if last_issues else ""
        combined_feedback = ""
        if feedback and validation_feedback:
            combined_feedback = f"{feedback}\n此外，验证工具发现问题：{validation_feedback}"
        elif feedback:
            combined_feedback = feedback
        elif validation_feedback:
            combined_feedback = validation_feedback

        prompt = _build_outline_prompt(user_req, digest, combined_feedback, target_words, template_sections)
        text = call_llm(
            prompt,
            timeout_s=MODEL_TIMEOUT,
            temperature=0.2,
            num_predict=900,
            system="你是一个中文文档处理助手。",
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
            # Format OK — check content coverage
            coverage_issues = _check_outline_coverage(out, user_req, template_sections)
            if not coverage_issues:
                t1 = time.perf_counter()
                logger.info(f"outline_done seconds={t1 - t0:.2f} prompt_chars={len(prompt)} outline_chars={len(out)} attempts={attempt + 1}")
                return _add_outline_numbering(out)
            last_issues = [f"内容覆盖不足，缺失或需加强：{'、'.join(coverage_issues)}"]
            logger.warning(f"outline_coverage_failed attempt={attempt + 1} missing={coverage_issues}")
        else:
            last_issues = issues
            logger.warning(f"outline_validation_failed attempt={attempt + 1} issues={issues}")

    t1 = time.perf_counter()
    logger.warning(f"outline_done_with_issues seconds={t1 - t0:.2f} attempts=3 issues={last_issues}")
    return _add_outline_numbering(best_outline)
