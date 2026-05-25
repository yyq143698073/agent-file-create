import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent_file_create.config import CONTENT_API_ENDPOINT, CONTENT_API_KEY, CONTENT_API_STYLE, CONTENT_MODEL_NAME, MAX_WORKERS_DEFAULT, MODEL_TIMEOUT
from agent_file_create.llm_client import call_llm
from agent_file_create.llm_factory import get_chat_model

logger = logging.getLogger(__name__)

# ── Section progress notification & cancel support ──────────────────────────


class TaskCanceledException(Exception):
    pass


def _notify_section_progress(task_id: str, done: int, total: int, section_title: str) -> None:
    if not task_id or total <= 0:
        return
    try:
        from agent_file_create.task.manager import TaskManager

        TaskManager().write_status(
            str(task_id),
            "processing",
            stage="document",
            message=f"正在生成 {done}/{total} 章节：{section_title}",
        )
    except Exception:
        pass


def _check_cancel(task_id: str) -> None:
    if not task_id:
        return
    try:
        from agent_file_create.task.manager import TaskManager

        _, cancel_ev = TaskManager().get_control_events(str(task_id))
        if cancel_ev.is_set():
            raise TaskCanceledException(f"Task {task_id} was canceled")
    except TaskCanceledException:
        raise
    except Exception:
        pass

# ── LLM-based section summary for cross-section coherence ──────────────────

_SECTION_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """\
将以下报告章节内容压缩为一段不超过200字的摘要，重点提取：
1) 核心论点与结论
2) 涉及的关键实体、数据、概念
3) 本章节在报告逻辑链中的角色（是铺垫、论证、对比、还是总结）

章节标题：{title}
章节内容：
{content}

摘要："""),
])

# ── Final coherence / fact-consistency review ──────────────────────────────

_COHERENCE_REVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """\
你是资深编辑，请检查以下报告的章节间逻辑连贯性和事实一致性。

报告全文：
{full_text}

参考材料摘要：
{material_digest}

检查要点：
1) 相邻章节之间是否存在逻辑断裂或跳跃？
2) 不同章节是否存在相互矛盾的陈述（例如前面说增长、后面说下降）？
3) 是否存在材料中无依据的具体数字、人名、机构名、年份？
4) 章节之间的术语使用是否一致？

输出格式：
- 如无问题，只回复：PASS
- 如有问题，逐条列出（格式：## 章节名: 问题描述 → 建议修改）

只标记明确的问题，不要吹毛求疵。"""),
])

# ── Per-section fact grounding check (lightweight) ─────────────────────────

_SECTION_FACT_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """\
检查以下报告章节内容是否严格基于参考材料，标记出无依据的断言。

参考材料：
{material_digest}

章节内容：
{section_text}

检查要点：
- 具体数字（金额、百分比、数量等）是否在材料中有出处？
- 人名、机构名是否在材料中出现过？
- 结论是否有材料中的证据支撑？

输出格式：
- 如无问题，只回复：PASS
- 如有问题，逐条列出：问题类型 | 具体内容 | 严重程度（高/中/低）"""),
])


# ── Structured fact extraction for cross-checking ──────────────────────────

def _extract_facts_from_materials(multimodal_digest: str) -> dict:
    """Extract structured data points from source materials using regex.

    Returns a dict with keys: 'numbers', 'entities', 'years'.
    Each value is a set of strings found in the materials.
    """
    text = str(multimodal_digest or "")
    if not text.strip():
        return {"numbers": set(), "entities": set(), "years": set()}

    # Numbers with units/context: "120亿", "35.6%", "2000万元", "1.5万"
    num_patterns = [
        r"\d+(?:\.\d+)?\s*[万亿千百]?\s*(?:元|美元|亿|万|%|％|个|人|家|次|倍|吨|公斤|千米|公里|米|小时|天|年|月)",
        r"\d+(?:\.\d+)?\s*%",
        r"(?:约|大约|近|超过|不足|至少|最多)\s*\d+(?:\.\d+)?",
    ]
    numbers: set[str] = set()
    for pat in num_patterns:
        for m in re.finditer(pat, text):
            v = m.group().strip()
            if len(v) >= 2:
                numbers.add(v)

    # Organization/institution names: "XX公司", "XX集团", "XX大学", "XX部门"
    org_patterns = [
        r"[一-鿿]{2,16}(?:公司|集团|大学|学院|研究所|研究院|中心|部门|委员会|协会|基金会|医院|银行|证券|保险|基金|科技|技术|网络|数据|人工智能|机器人)",
        r"[一-鿿]{2,8}(?:有限|股份|责任|合资|独资)公司",
    ]
    entities: set[str] = set()
    for pat in org_patterns:
        for m in re.finditer(pat, text):
            v = m.group().strip()
            if len(v) >= 3 and len(v) <= 24:
                entities.add(v)

    # Years and date ranges
    years: set[str] = set()
    for m in re.finditer(r"(?:19|20|21)\d{2}(?:[-—–/](?:19|20|21)?\d{2})?", text):
        years.add(m.group().strip())

    return {"numbers": numbers, "entities": entities, "years": years}


def _cross_check_facts(section_text: str, material_facts: dict) -> list[str]:
    """Check generated content for claims not supported by source material facts.

    Returns a list of warning strings (empty if clean).
    """
    text = str(section_text or "")
    if not text.strip() or not material_facts:
        return []

    issues: list[str] = []

    # Check numbers
    for m in re.finditer(r"\d+(?:\.\d+)?\s*[万亿千百]?\s*(?:元|美元|亿|万|%|％|个|人|家|次|倍)", text):
        val = m.group().strip()
        if len(val) >= 2 and val not in material_facts.get("numbers", set()):
            # Only flag if it looks like a specific data point (not generic)
            if any(c.isdigit() for c in val) and len(val) >= 3:
                issues.append(f"数字:{val}")

    # Check entities
    for pat in [r"[一-鿿]{2,16}(?:公司|集团|大学|银行|医院|研究所|中心)", r"[一-鿿]{2,8}(?:有限|股份)公司"]:
        for m in re.finditer(pat, text):
            val = m.group().strip()
            if val not in material_facts.get("entities", set()):
                exists = any(val[:4] in e or e[:4] in val for e in material_facts.get("entities", set()))
                if not exists:
                    issues.append(f"机构:{val}")

    # Check years
    for m in re.finditer(r"(?:19|20|21)\d{2}", text):
        val = m.group().strip()
        if val not in material_facts.get("years", set()):
            issues.append(f"年份:{val}")

    return issues[:12]  # cap at 12 flagged items


def parse_outline_sections(outline: str) -> list[dict]:
    """Parse a markdown outline into a list of {level, title} dicts."""
    sections: list[dict] = []
    for line in (outline or "").splitlines():
        s = line.strip()
        if not s.startswith("#"):
            continue
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", s)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        sections.append({"level": level, "title": title})
    return sections


def _multimodal_summary(multimodal_results: Dict[str, Any], max_chars: int = 1600) -> str:
    parts: List[str] = []
    for _, v in (multimodal_results or {}).items():
        if not isinstance(v, dict):
            continue
        title = str(v.get("title") or "").strip()
        summary = str(v.get("summary") or "").strip()
        conclusion = str(v.get("conclusion") or "").strip()
        key_points = v.get("key_points") if isinstance(v.get("key_points"), list) else []
        kp = "；".join([str(x).strip() for x in key_points[:3] if str(x).strip()])
        s = " | ".join([x for x in [title, summary, kp, conclusion] if x]).strip()
        if s:
            parts.append("- " + s)
        if sum(len(x) for x in parts) >= max_chars:
            break
    out = "\n".join(parts).strip()
    if len(out) > max_chars:
        return out[:max_chars] + "…"
    return out


def _build_section_prompt(
    *,
    section_title: str,
    parent_title: str,
    previous_summary: str,
    multimodal_digest: str,
    user_prompt: str,
    level: int,
    target_range: tuple[int, int],
    next_title: str = "",
    feedback: str = "",
) -> str:
    lo, hi = target_range
    parts = [
        "你是一个资深行业分析师和顶级文案，正在撰写一份专业的深度报告。",
        "",
        "核心指令：",
        "1) 拒绝复读：严禁直接复制粘贴参考材料中的长句，用全新语言重构核心观点。",
        "2) 逻辑流与衔接：用因果、递进、转折等连接词建立段落间关系。开篇用1句话承接前文（如有），结尾用1句话自然过渡到下一节（如有），不要生硬地写「接下来我们将讨论……」。",
        "3) 场景化扩写：解释数据和结论的业务含义，但场景必须是材料中有线索支撑的，不要凭空想象。",
        "4) 降低幻觉：不要编造具体的数字、机构名、人名、年份。材料中明确出现的数值可以引用，但要标注「据材料显示」等限定语。不确定就说「相关数据暂缺」。",
        "5) 溯源要求：每个关键论断后，用「（据<材料简称>）」标注信息来源。如多个材料共同支撑，标注「（综合多份材料）」。无材料支撑的推论，标注「（分析推测）」。",
        "",
        "结构化内容处理：",
        "- 如果材料中包含表格数据，用小段落描述关键趋势，不要逐行罗列。",
        "- 如果涉及多方案对比，用「相比之下」「与之相反」等短语体现对比关系。",
        "- 如果本节的论点需要数据支撑但材料中数据不足，说明「材料中暂缺该维度数据」而不是编造。",
        "",
        f"章节层级：{'## 主章节' if level==2 else '### 子节' if level==3 else '#'}",
        f"父章节：{parent_title or '（无，此为顶级章节）'}",
        f"当前章节：{section_title}",
    ]
    if next_title:
        parts.append(f"下一章节：{next_title}（本节结尾请做好内容铺垫）")
    parts += [
        "",
        "前文摘要（用于保持连贯）：",
        previous_summary.strip() or "（本节为开篇，无需承接前文）",
        "",
        "参考材料摘要：",
        multimodal_digest.strip() or "（无相关材料，请基于章节标题做合理的框架性论述，不要编造细节）",
        "",
        "用户需求：",
        (user_prompt or "").strip() or "生成报告正文",
        "",
    ]
    if feedback:
        parts += [
            "⚠️ 上一版报告的调整要求（来自用户反馈）：",
            feedback,
            "请特别注意上述反馈，在写作时针对性修正问题。",
            "",
        ]
    parts += [
        f"写作要求：本节建议 {lo}~{hi} 字。",
        "只输出本节正文，不要输出标题行。不要输出「本节」「本章」等元描述文字。",
        "",
        "正文：",
    ]
    return "\n\n".join(parts).strip()


def _call_qwen_text(prompt: str, *, timeout_s: int, num_predict: int) -> str:
    text = call_llm(
        prompt,
        timeout_s=timeout_s,
        temperature=0.4,
        num_predict=num_predict,
        stop=["```"],
        system="你是一个中文助手，只输出中文正文段落。",
        api_style=CONTENT_API_STYLE,
        api_endpoint=CONTENT_API_ENDPOINT,
        api_key=CONTENT_API_KEY,
        model_name=CONTENT_MODEL_NAME,
    )
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    if not t or t.startswith("{"):
        return ""
    return t


def _fallback_section(section_title: str, parent_title: str, previous_summary: str, multimodal_digest: str) -> str:
    p = []
    if previous_summary:
        p.append(previous_summary.strip())
    if multimodal_digest:
        lines = [x.strip("- ").strip() for x in multimodal_digest.splitlines() if x.strip()]
        if lines:
            p.append("；".join(lines[:3]))
    s = "。".join([x.strip("。") for x in p if x.strip()]) + "。"
    s = s.replace("。。", "。").strip()
    if not s or s == "。":
        s = "本节围绕相关材料进行归纳整理，并给出可执行的观点与结论。"
    return s


def generate_section_content(
    section_title: str,
    parent_title: str,
    multimodal_results: Dict[str, Any],
    user_prompt: str,
    previous_summary: str,
    level: int,
    next_title: str = "",
    *,
    task_id: str = "",
    feedback: str = "",
) -> str:
    multimodal_digest = _multimodal_summary(multimodal_results)
    if level == 2:
        target_range = (120, 180)
        num_predict = 500
    else:
        target_range = (180, 280)
        num_predict = 600

    prompt = _build_section_prompt(
        section_title=section_title,
        parent_title=parent_title,
        previous_summary=previous_summary,
        multimodal_digest=multimodal_digest,
        user_prompt=user_prompt,
        level=level,
        target_range=target_range,
        next_title=next_title,
        feedback=feedback,
    )

    for attempt in range(2):
        _check_cancel(task_id)
        t0 = time.perf_counter()
        text = _call_qwen_text(prompt, timeout_s=MODEL_TIMEOUT, num_predict=num_predict)
        t1 = time.perf_counter()
        if text:
            if t1 - t0 >= 8:
                logger.info(f"section_slow title={section_title} seconds={t1 - t0:.2f} prompt_chars={len(prompt)}")
            return text
        time.sleep(0.6 + attempt * 0.6)

    logger.warning(f"section_generate_failed title={section_title} level={level}")
    return _fallback_section(section_title, parent_title, previous_summary, multimodal_digest)


def _llm_summarize_for_next(section_title: str, content: str) -> str:
    """Generate a rolling summary for cross-section coherence."""
    s = re.sub(r"\s+", " ", (content or "").strip())
    if len(s) < 120:
        return s
    # Use LLM only for long h2 sections (>600 chars); text truncation for others
    if len(s) <= 600:
        return s[:200] + ("…" if len(s) > 200 else "")
    try:
        llm = get_chat_model(
            style=CONTENT_API_STYLE,
            model=CONTENT_MODEL_NAME,
            endpoint=CONTENT_API_ENDPOINT,
            api_key=CONTENT_API_KEY,
            temperature=0.1,
            max_tokens=160,
            timeout_s=30,
        )
        chain = _SECTION_SUMMARY_PROMPT | llm | StrOutputParser()
        summary = (chain.invoke({"title": section_title, "content": s[:2000]}) or "").strip()
        if summary and len(summary) >= 20:
            return summary[:280]
    except Exception:
        pass
    return s[:200] + ("…" if len(s) > 200 else "")


def _verify_section_facts(section_title: str, section_text: str, multimodal_digest: str) -> list[str]:
    """Lightweight fact-check: returns list of flagged issues (empty if clean)."""
    if not multimodal_digest.strip() or len(section_text) < 50:
        return []
    try:
        llm = get_chat_model(
            style=CONTENT_API_STYLE,
            model=CONTENT_MODEL_NAME,
            endpoint=CONTENT_API_ENDPOINT,
            api_key=CONTENT_API_KEY,
            temperature=0.0,
            max_tokens=200,
            timeout_s=60,
        )
        chain = _SECTION_FACT_CHECK_PROMPT | llm | StrOutputParser()
        result = (chain.invoke({
            "material_digest": multimodal_digest[:2000],
            "section_text": section_text[:1500],
        }) or "").strip()
        if not result or result.upper().startswith("PASS"):
            return []
        return [line.strip("- ").strip() for line in result.splitlines() if line.strip() and "问题" not in line[:4]]
    except Exception:
        return []


def _final_coherence_review(full_text: str, multimodal_digest: str) -> str:
    """Post-generation coherence review with layered clipping.

    Layer 3 (LLM contradiction check) and Layer 4 (regex fact cross-check) are
    expensive — each costs an extra LLM call.  Skip them for short reports where
    the base prompt constraints (Layer 1) and data-handling rules (Layer 2) already
    provide reasonable guardrails.
    """
    text_len = len(full_text)

    # Short report (< 3000 chars): layers 1 & 2 suffice
    if text_len < 3000:
        return full_text

    # Medium report (3000–8000 chars): layer 3 only (LLM review), skip layer 4
    run_layer4 = text_len >= 8000

    all_issues: list[str] = []

    # ── Step 1: LLM coherence review (layer 3) ──────────────────────────────
    try:
        llm = get_chat_model(
            style=CONTENT_API_STYLE,
            model=CONTENT_MODEL_NAME,
            endpoint=CONTENT_API_ENDPOINT,
            api_key=CONTENT_API_KEY,
            temperature=0.1,
            max_tokens=800,
            timeout_s=120,
        )
        chain = _COHERENCE_REVIEW_PROMPT | llm | StrOutputParser()
        review = (chain.invoke({
            "full_text": full_text[:8000],
            "material_digest": (multimodal_digest or "无")[:2000],
        }) or "").strip()
        if review and not review.upper().startswith("PASS"):
            for line in review.splitlines():
                s = line.strip()
                if s.startswith("##"):
                    all_issues.append(s)
    except Exception:
        pass

    # ── Step 2: Structured fact cross-check (layer 4, long reports only) ────
    if run_layer4:
        try:
            material_facts = _extract_facts_from_materials(multimodal_digest)
            fact_issues = _cross_check_facts(full_text, material_facts)
            for fi in fact_issues:
                all_issues.append(f"## 事实核查: {fi}")
        except Exception:
            pass

    if not all_issues:
        return full_text

    # ── Apply markers for flagged issues ─────────────────────────────────
    corrected = full_text
    for issue in all_issues[:8]:
        if "→" in issue:
            problem_part, suggestion = issue.split("→", 1)
            section_name = problem_part.split(":")[0].replace("##", "").strip()
            if section_name and section_name in corrected:
                marker = f"\n\n<!-- 核查提示：{suggestion.strip()[:120]} -->"
                idx = corrected.find(section_name)
                if idx >= 0:
                    next_section = corrected.find("\n## ", idx + len(section_name))
                    if next_section < 0:
                        next_section = len(corrected)
                    insert_at = min(next_section, idx + 2000)
                    if "<!-- 核查提示" not in corrected[max(0, insert_at - 200):insert_at + 200]:
                        corrected = corrected[:insert_at] + marker + corrected[insert_at:]
        else:
            # Fact-check issues without →: append at end as review notes
            marker = f"\n\n<!-- 核查提示：{issue.replace('##', '').strip()[:160]} -->"
            if marker not in corrected[-800:]:
                corrected = corrected.rstrip() + marker

    return corrected


def generate_full_content(outline: str, multimodal_results: Dict[str, Any], user_prompt: str, *, task_id: str = "", feedback: str = "") -> str:
    flat = parse_outline_sections(outline)
    if not flat:
        return ""

    # Precompute next-title lookup for content-level sections (level ≥ 2)
    content_indices = [i for i, item in enumerate(flat) if int(item.get("level") or 0) >= 2]
    next_title_map: dict[int, str] = {}
    for idx, ci in enumerate(content_indices):
        if idx + 1 < len(content_indices):
            next_ci = content_indices[idx + 1]
            next_title_map[ci] = str(flat[next_ci].get("title") or "").strip()

    previous_summary = ""
    out_lines: List[str] = []
    parent_stack: List[str] = []
    multimodal_digest = _multimodal_summary(multimodal_results)

    for i, item in enumerate(flat):
        level = int(item.get("level") or 0)
        title = str(item.get("title") or "").strip()
        if not title or level < 1:
            continue

        while len(parent_stack) >= level:
            parent_stack.pop()
        parent_title = parent_stack[-1] if parent_stack else ""

        if level == 1:
            out_lines.append(f"# {title}")
            parent_stack.append(title)
            continue

        next_title = next_title_map.get(i, "")

        _check_cancel(task_id)

        if level == 2:
            out_lines.append("")
            out_lines.append(f"## {title}")
            body = generate_section_content(title, parent_title, multimodal_results, user_prompt, previous_summary, level, next_title, task_id=task_id, feedback=feedback)
            out_lines.append("")
            out_lines.append(body)
            previous_summary = _llm_summarize_for_next(title, body)
            parent_stack.append(title)
            continue

        if level == 3:
            out_lines.append("")
            out_lines.append(f"### {title}")
            body = generate_section_content(title, parent_title, multimodal_results, user_prompt, previous_summary, level, next_title, task_id=task_id, feedback=feedback)
            out_lines.append("")
            out_lines.append(body)
            previous_summary = _llm_summarize_for_next(title, body)
            parent_stack.append(title)
            continue

        out_lines.append("")
        out_lines.append("#" * level + " " + title)
        body = generate_section_content(title, parent_title, multimodal_results, user_prompt, previous_summary, level, next_title, task_id=task_id, feedback=feedback)
        out_lines.append("")
        out_lines.append(body)
        previous_summary = _llm_summarize_for_next(title, body)
        parent_stack.append(title)

    raw = "\n".join(out_lines).strip() + "\n"
    return _final_coherence_review(raw, multimodal_digest)


def generate_content(outline: str, multimodal_results: Dict[str, Any], user_prompt: str, *, task_id: str = "", feedback: str = "") -> str:
    return generate_full_content_parallel(outline, multimodal_results, user_prompt, task_id=task_id, feedback=feedback)


def _prepare_section_tasks(flat: List[dict]) -> List[dict]:
    """Pre-compute section metadata (parent title, level) for each section that needs content generation.
    Returns a list of task dicts with keys: index, title, parent_title, level.
    """
    tasks: List[dict] = []
    parent_stack: List[str] = []
    for i, item in enumerate(flat):
        level = int(item.get("level") or 0)
        title = str(item.get("title") or "").strip()
        if not title or level < 1:
            continue
        while len(parent_stack) >= level:
            parent_stack.pop()
        parent_title = parent_stack[-1] if parent_stack else ""
        if level >= 2:
            tasks.append({"index": i, "title": title, "parent_title": parent_title, "level": level})
        parent_stack.append(title)
    return tasks


def _build_section_args(task: dict, multimodal_results: Dict[str, Any], user_prompt: str, parent_summary: str = "", *, task_id: str = "", feedback: str = "") -> dict:
    return {
        "section_title": task["title"],
        "parent_title": task["parent_title"],
        "multimodal_results": multimodal_results,
        "user_prompt": user_prompt,
        "previous_summary": parent_summary,
        "level": task["level"],
        "task_id": task_id,
        "feedback": feedback,
    }


def generate_full_content_parallel(outline: str, multimodal_results: Dict[str, Any], user_prompt: str, *, task_id: str = "", feedback: str = "") -> str:
    flat = parse_outline_sections(outline)
    if not flat:
        return ""

    tasks = _prepare_section_tasks(flat)
    if not tasks:
        return ""

    total_sections = len(tasks)
    done_count = 0

    multimodal_digest = _multimodal_summary(multimodal_results)

    # Tag tasks with their parent h2 for grouping
    parent_stack: list[str] = []
    h2_index_map: dict[int, int] = {}
    current_h2_idx = -1
    for i, item in enumerate(flat):
        level = int(item.get("level") or 0)
        title = str(item.get("title") or "").strip()
        if not title or level < 1:
            continue
        while len(parent_stack) >= level:
            parent_stack.pop()
        if level == 2:
            current_h2_idx = i
        if level >= 3:
            h2_index_map[i] = current_h2_idx if current_h2_idx >= 0 else -1
        parent_stack.append(title)

    results: dict[int, str] = {}
    max_workers = max(1, min(int(MAX_WORKERS_DEFAULT), len(tasks)))

    # Helper to write partial content so the frontend can preview completed sections
    def _flush_partial():
        if not task_id:
            return
        try:
            from pathlib import Path as _Path
            base = _Path(__file__).resolve().parent.parent.parent
            p = base / "result" / str(task_id) / "content.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            buf: list[str] = []
            pstack: list[str] = []
            for i, item in enumerate(flat):
                lv = int(item.get("level") or 0)
                ti = str(item.get("title") or "").strip()
                if not ti or lv < 1:
                    continue
                while len(pstack) >= lv:
                    pstack.pop()
                pstack.append(ti)
                if lv == 1:
                    buf.append(f"# {ti}")
                else:
                    buf.append("")
                    buf.append(f"{'#' * lv} {ti}")
                    body = results.get(i, "")
                    buf.append("")
                    buf.append(body if body else "（生成中…）")
            raw = "\n".join(buf).strip() + "\n"
            p.write_text(raw, encoding="utf-8")
        except Exception:
            pass

    # Phase 1: parallelize h2 sections
    h2_summaries: dict[int, str] = {}  # flat_index -> summary
    h2_tasks = [t for t in tasks if t["level"] == 2]
    if h2_tasks:
        _check_cancel(task_id)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for t in h2_tasks:
                args = _build_section_args(t, multimodal_results, user_prompt, task_id=task_id, feedback=feedback)
                futures[pool.submit(generate_section_content, **args)] = t["index"]
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = str(fut.result() or "")
                except Exception as e:
                    results[idx] = f"（本节生成失败：{str(e)[:120]}）"
                _check_cancel(task_id)
                done_count += 1
                section_title = str(flat[idx].get("title") or "") if idx < len(flat) else ""
                _notify_section_progress(task_id, done_count, total_sections, section_title)
        # Compute summaries after all h2s complete
        for t in h2_tasks:
            idx = t["index"]
            body = results.get(idx, "")
            h2_summaries[idx] = _llm_summarize_for_next(t["title"], body)
        _flush_partial()

    # Phase 2: parallelize h3+ sections under their parent h2 context
    sub_tasks = [t for t in tasks if t["level"] >= 3]
    if sub_tasks:
        _check_cancel(task_id)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for t in sub_tasks:
                parent_h2_idx = h2_index_map.get(t["index"], -1)
                parent_summary = h2_summaries.get(parent_h2_idx, "") if parent_h2_idx >= 0 else ""
                args = _build_section_args(t, multimodal_results, user_prompt, parent_summary=parent_summary, task_id=task_id, feedback=feedback)
                futures[pool.submit(generate_section_content, **args)] = t["index"]
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = str(fut.result() or "")
                except Exception as e:
                    results[idx] = f"（本节生成失败：{str(e)[:120]}）"
                done_count += 1
                _check_cancel(task_id)
                section_title = str(flat[idx].get("title") or "") if idx < len(flat) else ""
                _notify_section_progress(task_id, done_count, total_sections, section_title)
        _flush_partial()

    parent_stack = []
    out_lines: list[str] = []
    for i, item in enumerate(flat):
        level = int(item.get("level") or 0)
        title = str(item.get("title") or "").strip()
        if not title or level < 1:
            continue

        while len(parent_stack) >= level:
            parent_stack.pop()
        parent_stack.append(title)

        if level == 1:
            out_lines.append(f"# {title}")
        else:
            out_lines.append("")
            prefix = "#" * level
            out_lines.append(f"{prefix} {title}")
            body = results.get(i, "")
            out_lines.append("")
            out_lines.append(body)

    raw = "\n".join(out_lines).strip() + "\n"
    return _final_coherence_review(raw, multimodal_digest)


def regenerate_section(
    outline: str,
    content: str,
    section_name: str,
    multimodal_results: Dict[str, Any],
    user_prompt: str,
    *,
    task_id: str = "",
) -> str:
    """Regenerate a single section (and its children) in the content.
    Returns the full updated content string, or empty string if the section
    could not be found.
    """
    search = (section_name or "").strip()
    if not search or not content:
        return ""

    lines = content.splitlines()
    # Parse content into heading blocks: {heading, level, start_idx, end_idx}
    blocks: list[dict] = []
    current = None
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            if current is not None:
                current["end_idx"] = i
                blocks.append(current)
            current = {
                "heading": m.group(2).strip(),
                "level": len(m.group(1)),
                "start_idx": i,
                "end_idx": len(lines),
            }
    if current is not None:
        blocks.append(current)

    if not blocks:
        return ""

    # Find best matching section
    best_idx = -1
    best_score = 0.0
    for bi, blk in enumerate(blocks):
        title = blk["heading"]
        if search == title:
            best_idx = bi
            break
        if search in title or title in search:
            score = len(set(search) & set(title)) / max(len(set(search)), 1)
            if score > best_score:
                best_score = score
                best_idx = bi

    if best_idx < 0:
        return ""

    target = blocks[best_idx]
    target_level = target["level"]

    # Determine line range to replace (target heading + body + children)
    replace_end = target["end_idx"]
    for bi in range(best_idx + 1, len(blocks)):
        if blocks[bi]["level"] <= target_level:
            replace_end = blocks[bi]["start_idx"]
            break

    section_title = target["heading"]

    # Collect child sections
    child_titles: list[tuple[str, int]] = []
    for bi in range(best_idx + 1, len(blocks)):
        if blocks[bi]["level"] <= target_level:
            break
        child_titles.append((blocks[bi]["heading"], blocks[bi]["level"]))

    # Build parent summary from preceding section at same or higher level
    parent_summary = ""
    if best_idx > 0:
        for bi in range(best_idx - 1, -1, -1):
            if blocks[bi]["level"] <= target_level:
                ps = blocks[bi]["start_idx"]
                pe = blocks[bi]["end_idx"]
                prev_body = "\n".join(lines[ps + 1:pe]).strip()[:500]
                parent_summary = _llm_summarize_for_next(blocks[bi]["heading"], prev_body)
                break

    # Regenerate the target section
    _check_cancel(task_id)
    new_body = generate_section_content(
        section_title, "", multimodal_results, user_prompt,
        parent_summary, target_level, task_id=task_id,
    )

    # Regenerate children
    child_bodies: dict[str, str] = {}
    child_rolling = _llm_summarize_for_next(section_title, new_body)
    for child_title, child_level in child_titles:
        _check_cancel(task_id)
        try:
            child_body = generate_section_content(
                child_title, section_title, multimodal_results, user_prompt,
                child_rolling, child_level, task_id=task_id,
            )
            child_bodies[child_title] = child_body
            child_rolling = _llm_summarize_for_next(child_title, child_body)
        except Exception:
            child_bodies[child_title] = ""

    # Rebuild content with regenerated section inserted
    result: list[str] = []
    # Lines before the target section
    if target["start_idx"] > 0:
        before = lines[:target["start_idx"]]
        result.extend(before)
        if before[-1].strip():
            result.append("")

    result.append(f"{'#' * target_level} {section_title}")
    result.append("")
    result.append(new_body)

    for child_title, child_level in child_titles:
        result.append("")
        result.append(f"{'#' * child_level} {child_title}")
        result.append("")
        body = child_bodies.get(child_title)
        if body:
            result.append(body)
        else:
            # Keep old child body if regeneration produced nothing
            for bi in range(best_idx + 1, len(blocks)):
                if blocks[bi]["heading"] == child_title:
                    cs = blocks[bi]["start_idx"] + 1
                    ce = blocks[bi]["end_idx"]
                    result.append("\n".join(lines[cs:ce]).strip())
                    break

    # Lines after the replaced range
    if replace_end < len(lines):
        tail = "\n".join(lines[replace_end:]).strip()
        if tail:
            result.append("")
            result.append(tail)

    return "\n".join(result).strip() + "\n"