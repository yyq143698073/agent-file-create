import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent_file_create.config import CONTENT_API_ENDPOINT, CONTENT_API_KEY, CONTENT_API_STYLE, CONTENT_MODEL_NAME, MAX_WORKERS_DEFAULT, MODEL_TIMEOUT
from agent_file_create.llm_client import call_llm
from agent_file_create.llm_factory import get_chat_model
from agent_file_create.document._reviewer import (
    SECTION_SUMMARY_PROMPT as _SECTION_SUMMARY_PROMPT,
    COHERENCE_REVIEW_PROMPT as _COHERENCE_REVIEW_PROMPT,
    SECTION_FACT_CHECK_PROMPT as _SECTION_FACT_CHECK_PROMPT,
    extract_facts_from_materials as _extract_facts_from_materials,
    cross_check_facts as _cross_check_facts,  # kept as utility, called from _node_critic
    final_coherence_review as _final_coherence_review,  # deprecated, replaced by Critic node
)

logger = logging.getLogger(__name__)

# ── Section progress notification & cancel support ──────────────────────────


class TaskCanceledException(Exception):
    pass


# ── Module-level singletons ────────────────────────────────────────────
# Avoids constructing TaskManager() and summary LLM chains on every call.
_tm_instance: Any = None
_tm_lock = threading.Lock()


def _get_task_manager():
    """Return a shared TaskManager singleton."""
    global _tm_instance
    if _tm_instance is None:
        with _tm_lock:
            if _tm_instance is None:
                from agent_file_create.task.manager import TaskManager
                _tm_instance = TaskManager()
    return _tm_instance


_summary_chain = None
_summary_chain_lock = threading.Lock()


def _get_summary_chain():
    """Return a cached LLM chain for section summarization."""
    global _summary_chain
    if _summary_chain is None:
        with _summary_chain_lock:
            if _summary_chain is None:
                llm = get_chat_model(
                    style=CONTENT_API_STYLE,
                    model=CONTENT_MODEL_NAME,
                    endpoint=CONTENT_API_ENDPOINT,
                    api_key=CONTENT_API_KEY,
                    temperature=0.1,
                    max_tokens=160,
                    timeout_s=30,
                )
                _summary_chain = _SECTION_SUMMARY_PROMPT | llm | StrOutputParser()
    return _summary_chain


def _notify_section_progress(task_id: str, done: int, total: int, section_title: str) -> None:
    if not task_id or total <= 0:
        return
    try:
        _get_task_manager().write_status(
            str(task_id),
            "processing",
            stage="document",
            message=f"正在生成 {done}/{total} 章节：{section_title}",
            extra={"sections_done": done, "sections_total": total, "section_title": section_title},
        )
    except Exception:
        pass


def _check_cancel(task_id: str) -> None:
    if not task_id:
        return
    try:
        _, cancel_ev = _get_task_manager().get_control_events(str(task_id))
        if cancel_ev.is_set():
            raise TaskCanceledException(f"Task {task_id} was canceled")
    except TaskCanceledException:
        raise
    except Exception:
        pass

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


# ── Section type classification ──────────────────────────────────────────

_SECTION_TYPE_KEYWORDS: dict[str, list[str]] = {
    "data": [
        "实验", "数据", "结果", "性能", "评估", "测试", "指标", "统计",
        "对比", "比较", "精度", "准确率", "召回率", "F1", "BLEU", "ROUGE",
        "消融", "参数", "配置", "超参数", "训练", "推理", "延迟", "吞吐",
        "baseline", "基线", "对比实验", "定量", "数值", "百分比",
    ],
    "experiment_setup": [
        "实验设定", "实验设计", "实验设置", "方法", "设置",
        "数据集", "实现细节", "模型架构", "训练配置", "评测方案",
        "预处理", "特征工程", "采样", "划分", "验证策略",
    ],
    "analysis": [
        "讨论", "分析", "展望", "启示", "建议", "未来", "趋势", "影响",
        "意义", "价值", "优劣", "权衡", "局限", "不足", "改进方向",
        "综合", "解读", "思考", "反思", "启示", "对策", "路径",
    ],
}

def classify_section_type(section_title: str) -> str:
    """Classify a section heading into data / experiment_setup / analysis / review.

    - ``data``: experiments, metrics, quantitative results — strict sourcing, low temperature
    - ``experiment_setup``: methods, datasets, model config — data-adjacent but method-focused
    - ``analysis``: discussion, implications, future work — more inference, higher temperature
    - ``review``: background, definitions, frameworks — balanced (default)
    """
    title = (section_title or "").strip()
    data_score = sum(1 for kw in _SECTION_TYPE_KEYWORDS["data"] if kw in title)
    experiment_score = sum(1 for kw in _SECTION_TYPE_KEYWORDS["experiment_setup"] if kw in title)
    analysis_score = sum(1 for kw in _SECTION_TYPE_KEYWORDS["analysis"] if kw in title)

    # Tie-break: experiment_setup > data > analysis > review
    # experiment_setup is a data-adjacent type that wins ties against pure data
    scores = [
        (experiment_score, "experiment_setup"),
        (data_score, "data"),
        (analysis_score, "analysis"),
    ]
    # Sort by score descending; on tie, first in list wins (experiment_setup priority)
    scores.sort(key=lambda x: (-x[0], ["experiment_setup", "data", "analysis"].index(x[1])))
    if scores[0][0] > 0:
        return scores[0][1]
    return "review"


def _extract_kps_from_context(enriched_context: str, section_title: str) -> list[str]:
    """Extract knowledge_points from enriched_context text for a given section.

    Looks for blocks like::

        [章节素材: Section Title]
        知识点: kp1; kp2; kp3

    Returns a list of knowledge point strings, or empty list.
    """
    if not enriched_context or not section_title:
        return []
    # Find the block for this section
    pattern = rf"\[章节素材:\s*{re.escape(section_title)}\][^\[]*?知识点:\s*(.+?)(?:\n|$)"
    m = re.search(pattern, enriched_context, re.DOTALL)
    if not m:
        # Fuzzy match: first 4 chars
        probe = section_title[:4]
        pattern = rf"\[章节素材:\s*{re.escape(probe)}[^\]]*\][^\[]*?知识点:\s*(.+?)(?:\n|$)"
        m = re.search(pattern, enriched_context, re.DOTALL)
    if m:
        raw = m.group(1).strip()
        return [kp.strip() for kp in re.split(r"[；;]", raw) if kp.strip() and len(kp.strip()) >= 3]
    return []


def _compute_coverage_map(
    knowledge_points: list[str],
    materials_text: str,
) -> list[tuple[str, str]]:
    """Check each knowledge point against retrieved materials using
    jieba token overlap (Jaccard similarity).

    Returns [(kp, status), ...] where status is `充足` / `有限` / `无`.
    """
    if not knowledge_points or not materials_text:
        return []

    try:
        import jieba
        material_words = set(w for w in jieba.lcut(materials_text) if len(w.strip()) >= 2)
    except Exception:
        material_words = set()
        for i in range(len(materials_text) - 1):
            chunk = materials_text[i:i+2]
            if chunk.strip() and len(chunk) >= 2:
                material_words.add(chunk)

    result = []
    for kp in knowledge_points:
        kp = str(kp).strip()
        if not kp or len(kp) < 4:
            continue

        try:
            import jieba
            kp_words = set(w for w in jieba.lcut(kp) if len(w.strip()) >= 2)
        except Exception:
            kp_words = set()
            for i in range(len(kp) - 1):
                chunk = kp[i:i+2]
                if chunk.strip() and len(chunk) >= 2:
                    kp_words.add(chunk)

        if not kp_words or not material_words:
            count = materials_text.count(kp[:6])
            status = "充足" if count >= 2 else ("有限" if count == 1 else "无")
            result.append((kp, status))
            continue

        intersection = kp_words & material_words
        union = kp_words | material_words
        jaccard = len(intersection) / max(len(union), 1)

        if jaccard >= 0.3:
            status = "充足"
        elif jaccard >= 0.1:
            status = "有限"
        else:
            status = "无"
        result.append((kp, status))

    return result


# ── Section-writing system prompt ────────────────────────────────────
# Static writing instructions reused across all sections via the system
# message, cutting per-section prompt size by ~60%.  For a 15-section
# report that saves ~30 KB of redundant input tokens.
SECTION_SYSTEM_PROMPT = (
    "你是一个资深行业分析师和顶级文案，正在撰写专业的深度报告。\n"
    "\n"
    "核心规则：\n"
    "1) 用全新语言重构核心观点，严禁直接复制粘贴材料中的长句。\n"
    "2) 用因果、递进、转折等连接词建立段落间关系。主章节(H2)开篇承接前文、结尾过渡到下一节。子节(H3)直接切入。去重：前文已详述的论点只做一句话回顾。\n"
    "3) 优先引用定量内容（具体数值、超参数、指标），避免空洞定性描述。\n"
    "4) 降低幻觉：不编造数字、机构名、人名、年份。材料中明确出现的数值可引用并标注具体来源。不确定就说「相关数据暂缺」。\n"
    "5) 溯源简称：每个【n】编号从对应文件名取简短关键词作简称。不同编号必须有不同的简称。\n"
    "6) 每个关键论断标注【n】（据XX），编号和口头引用一一对应。\n"
    "7) 时效优先：优先采信年份较新的来源。引用3年以上文献需注明年份。\n"
    "8) 标注格式：同一编号全文使用完全相同的口头引用文本。\n"
    "9) 同一段落每个【n】编号最多出现1次，段落结尾统一标注。\n"
    "10) 每章至少使用2种不同编号。通篇只用【1】判定为不合格。\n"
    "11) 完成后数一数【n】种类，不到2种则重写。\n"
    "12) 讨论方法局限时直接说明。首次英文缩写必须给全称。\n"
    "\n"
    "内容处理：表格数据用小段落描述趋势不逐行罗列。多方案对比用「相比之下」等短语。数据不足时说明「材料中暂缺该维度数据」不编造。"
)


def _build_system_prompt_for_section(section_type: str) -> str:
    """Return the system prompt for a section, including type-specific hints."""
    hints = {
        "data": "【数据型章节】必须逐条引用来源数据，不可笼统概括。每个数据点标注出处。数据不足只写已有数据，禁止推测。",
        "experiment_setup": "【实验设定型章节】准确描述实验配置和参数。数据集描述含规模、来源、划分。必须列出至少2个对比基线。评估指标必须具体。",
        "analysis": "【分析型章节】可在材料事实基础上做合理推理延伸但标注「（分析推测）」。鼓励多材料综合对比。可提出推理观点但不可与材料事实矛盾。",
    }
    base = SECTION_SYSTEM_PROMPT
    hint = hints.get(section_type, "")
    if hint:
        return base + "\n\n" + hint
    return base


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
    section_type: str = "review",
    knowledge_points: list[str] | None = None,
    enriched_context: str = "",
) -> str:
    """Build the per-section user prompt — variable content only.

    Static writing rules live in SECTION_SYSTEM_PROMPT (module-level constant)
    and are passed as the system message, avoiding ~2KB of redundant tokens
    per section.
    """
    lo, hi = target_range
    parts: list[str] = []
    parts.append(f"章节层级：{'## 主章节' if level==2 else '### 子节' if level==3 else '#'}")
    parts.append(f"父章节：{parent_title or '（无，此为顶级章节）'}")
    parts.append(f"当前章节：{section_title}")
    if next_title:
        parts.append(f"下一章节：{next_title}（本节结尾请做好内容铺垫）")
    else:
        parts.append("⚠️ 这是本文最后一章（总结/展望/结论）。请提出 2-3 个具体的、可验证的未来研究方向或改进建议，不要停留在「需要进一步研究」的笼统陈述。")
    parts += [
        "",
        "前文摘要（用于保持连贯）：",
        previous_summary.strip() or (
            "（本节为子节，父章节正文正在并行生成中暂不可见，请直接围绕子主题展开，"
            "无需写「承接前文」的过渡句）" if level == 3
            else "（本节为开篇，无需承接前文）"
        ),
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
    # ── Coverage map: tell the LLM what we have / don't have ──
    # Try to extract knowledge_points from enriched_context if not explicitly provided.
    # Enriched context contains lines like "[章节素材: Title]\n知识点: kp1; kp2; kp3"
    if not knowledge_points and enriched_context:
        kps = _extract_kps_from_context(enriched_context, section_title)
        if kps:
            knowledge_points = kps

    if knowledge_points:
        coverage = _compute_coverage_map(knowledge_points, multimodal_digest)
        if coverage:
            parts.append("")
            parts.append("## 知识覆盖度提示")
            parts.append("以下知识点在当前检索材料中的覆盖情况：")
            for kp, status in coverage:
                icon = {"充足": "[FULL]", "有限": "[LIMITED]", "无": "[MISSING]"}.get(status, "?")
                parts.append(f"  {icon} {status}: {kp}")
            parts.append("")
            parts.append("写作时请注意：")
            parts.append("- [充足] 的知识点可以深入展开，引用具体数据。")
            parts.append("- [有限] 的知识点只写材料中已有的内容，不要延伸推测。")
            parts.append("- [无] 的知识点：如果跳过不影响章节完整性则跳过；")
            parts.append("  如果必须提及，用一句话概括并标注 [需补充数据]。")
            parts.append("")

    parts += [
        f"写作要求：本节严格控制在 {lo}~{hi} 字以内。超出字数限制的内容会被截断。",
        "只输出本节正文，不要输出标题行。不要输出「本节」「本章」等元描述文字。",
        "",
        "正文：",
    ]
    return "\n\n".join(parts).strip()


def _call_qwen_text(prompt: str, *, timeout_s: int, num_predict: int, temperature: float = 0.4,
                    system: str | None = None) -> str:
    text = call_llm(
        prompt,
        timeout_s=timeout_s,
        temperature=temperature,
        num_predict=num_predict,
        stop=["```"],
        system=system or "你是一个中文文档处理助手。",
        api_style=CONTENT_API_STYLE,
        api_endpoint=CONTENT_API_ENDPOINT,
        api_key=CONTENT_API_KEY,
        model_name=CONTENT_MODEL_NAME,
    )
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    # Only filter genuine JSON errors, not content starting with {
    if not t:
        return ""
    if t.startswith('{"error"') or t.startswith('{"success": false'):
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


# ── Q4 P0: Citation compliance checker ────────────────────────────────────────


def _check_citation_compliance(text: str) -> list[str]:
    """Post-generation check: does the section body follow citation rules?

    Returns list of issue strings (empty = compliant).
    Checks:
    1. At least 2 different citation numbers (per Rule 10)
    2. No duplicate 【n】 in the same paragraph (per Rule 9)
    3. Each 【n】 has a corresponding verbal reference (据XX / 来源XX)
    4. Citation numbers are ≤ 20 (sanity check against hallucination)
    """
    if not text or "【" not in text:
        return ["缺少引用标注【n】：正文未使用任何引用标记"]

    issues: list[str] = []

    # 1. Count unique citation numbers
    markers = re.findall(r"【(\d+)】", text)
    unique_nums = set(int(m) for m in markers)
    if len(unique_nums) < 2:
        issues.append(f"引用来源过少：仅使用 {len(unique_nums)} 种引用编号（需≥2）")

    # 2. Check for duplicate 【n】 in same paragraph
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    for i, para in enumerate(paragraphs):
        para_markers = re.findall(r"【(\d+)】", para)
        if len(para_markers) > len(set(para_markers)):
            issues.append(f"同段重复引用：第{i+1}段中同一编号出现多次")

    # 3. Check that 【n】 has verbal reference nearby
    for m in re.finditer(r"【(\d+)】", text):
        n = m.group(1)
        # Check 30 chars after the marker for source text
        after = text[m.end():m.end() + 40]
        if not re.search(r"[据参来][^，。；\n]{2,20}", after):
            issues.append(f"引用【{n}】缺少口头引用说明（如'据XX年报'）")
            break  # one example is enough

    # 4. Sanity check: citation numbers shouldn't be absurdly high
    if unique_nums and max(unique_nums) > 20:
        issues.append(f"引用编号异常：最大编号 {max(unique_nums)} > 20，可能存在幻觉")

    return issues


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
    target_words: int = 0,
    section_type: str = "",
    enriched_context: str = "",
) -> str:
    # Auto-classify if caller didn't specify
    section_type = section_type or classify_section_type(section_title)
    multimodal_digest = _multimodal_summary(multimodal_results)
    if target_words and target_words > 0:
        lo = max(80, target_words // 2)
        hi = max(120, target_words)
        target_range = (lo, hi)
        # Chinese text ≈ 2 tokens / char; hi*2.2 gives ~10% headroom
        num_predict = max(200, int(hi * 2.2))
    elif level == 2:
        target_range = (160, 260)
        num_predict = 800
    else:  # H3 sub-sections
        target_range = (120, 190)
        num_predict = 500

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
        section_type=section_type,
        enriched_context=enriched_context,
    )

    # Temperature by section type: data=0.1, analysis=0.4, review=0.2
    temp_map = {"data": 0.1, "analysis": 0.4, "review": 0.2}
    gen_temp = temp_map.get(section_type, 0.2)
    logger.info("section_type title=%s type=%s temp=%.2f", section_title[:40], section_type, gen_temp)

    # Build system prompt with static writing rules (shared across all sections)
    section_system = _build_system_prompt_for_section(section_type)

    # ── Q4 P0: Citation compliance check + auto-retry ──────────────────────
    citation_feedback = ""
    for attempt in range(3):
        _check_cancel(task_id)
        if citation_feedback and attempt > 0:
            # Inject citation compliance feedback into the prompt
            prompt = _build_section_prompt(
                section_title=section_title,
                parent_title=parent_title,
                previous_summary=previous_summary,
                multimodal_digest=multimodal_digest,
                user_prompt=user_prompt,
                level=level,
                target_range=target_range,
                next_title=next_title,
                feedback=(feedback + "\n" + citation_feedback).strip(),
                section_type=section_type,
                enriched_context=enriched_context,
            )

        t0 = time.perf_counter()
        text = _call_qwen_text(prompt, timeout_s=MODEL_TIMEOUT, num_predict=num_predict,
                               temperature=gen_temp, system=section_system)
        t1 = time.perf_counter()
        if not text:
            time.sleep(0.6 + attempt * 0.6)
            continue

        if t1 - t0 >= 8:
            logger.info(f"section_slow title={section_title} seconds={t1 - t0:.2f} prompt_chars={len(prompt)}")

        # Q4 P0: Post-generation citation compliance check
        if enriched_context and "【" in enriched_context:
            cite_issues = _check_citation_compliance(text)
            if cite_issues:
                if attempt < 2:  # still have retries
                    citation_feedback = "引用合规问题（请修正后重新输出）：\n" + "\n".join(
                        f"- {issue}" for issue in cite_issues
                    )
                    logger.warning(f"section_citation_retry title={section_title} attempt={attempt+1} issues={len(cite_issues)}")
                    continue  # retry with citation feedback
                else:
                    logger.warning(f"section_citation_failed title={section_title} issues={len(cite_issues)} giving_up")

        return text

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
        chain = _get_summary_chain()
        summary = (chain.invoke({"title": section_title, "content": s[:2000]}) or "").strip()
        if summary and len(summary) >= 20:
            return summary[:280]
    except Exception:
        pass
    return s[:200] + ("…" if len(s) > 200 else "")


def generate_full_content(outline: str, multimodal_results: Dict[str, Any], user_prompt: str,
                          *, task_id: str = "", feedback: str = "", enriched_context: str = "",
                          target_words: int = 0) -> str:
    """Generate report body section by section, streaming progress to disk.

    Why sequential? Each section's content depends on the previous section's summary
    for coherence (cross-references, avoiding repetition). The previous_summary
    carries forward key entities and conclusions so later sections can refer back
    naturally. Sections are flushed to content.md after each write so the frontend
    can show live preview via SSE polling.
    """
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

    # Count total sections for progress tracking; weight H2 chapters 2× vs H3
    total_sections = sum(1 for item in flat if int(item.get("level") or 0) >= 2)
    _h2n = sum(1 for item in flat if int(item.get("level") or 0) == 2)
    _h3n = sum(1 for item in flat if int(item.get("level") or 0) == 3)
    _total_w = _h2n * 2 + _h3n
    if target_words and target_words > 0 and _total_w > 0:
        _h2_budget = max(100, (target_words * 2) // _total_w)
        _h3_budget = max(80, target_words // _total_w)
    else:
        _h2_budget = _h3_budget = 0
    done_count = 0

    def _flush_seq():
        """Write current out_lines to content.md for frontend preview."""
        if not task_id:
            return
        try:
            from pathlib import Path as _Path
            base = _Path(__file__).resolve().parent.parent.parent
            p = base / "result" / str(task_id) / "content.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            raw = "\n".join(out_lines).strip() + "\n"
            p.write_text(raw, encoding="utf-8")
        except Exception:
            pass

    previous_summary = ""
    out_lines: List[str] = []
    parent_stack: List[str] = []
    multimodal_digest = _multimodal_summary(multimodal_results)
    # Prepend skill-enriched context if available
    if enriched_context.strip():
        multimodal_digest = f"[技能搜集到的补充信息]\n{enriched_context.strip()}\n\n{multimodal_digest}"

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
            body = generate_section_content(title, parent_title, multimodal_results, user_prompt, previous_summary, level, next_title, task_id=task_id, feedback=feedback, target_words=_h2_budget, enriched_context=enriched_context)
            out_lines.append("")
            out_lines.append(body)
            previous_summary = _llm_summarize_for_next(title, body)
            parent_stack.append(title)
            done_count += 1
            _notify_section_progress(task_id, done_count, total_sections, title)
            _flush_seq()
            continue

        if level == 3:
            out_lines.append("")
            out_lines.append(f"### {title}")
            body = generate_section_content(title, parent_title, multimodal_results, user_prompt, previous_summary, level, next_title, task_id=task_id, feedback=feedback, target_words=_h3_budget, enriched_context=enriched_context)
            out_lines.append("")
            out_lines.append(body)
            previous_summary = _llm_summarize_for_next(title, body)
            parent_stack.append(title)
            done_count += 1
            _notify_section_progress(task_id, done_count, total_sections, title)
            _flush_seq()
            continue

        out_lines.append("")
        out_lines.append("#" * level + " " + title)
        body = generate_section_content(title, parent_title, multimodal_results, user_prompt, previous_summary, level, next_title, task_id=task_id, feedback=feedback, target_words=_h2_budget)
        out_lines.append("")
        out_lines.append(body)
        previous_summary = _llm_summarize_for_next(title, body)
        parent_stack.append(title)
        done_count += 1
        _notify_section_progress(task_id, done_count, total_sections, title)
        _flush_seq()

    raw = "\n".join(out_lines).strip() + "\n"
    return raw  # _final_coherence_review replaced by Critic graph node


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


def _build_section_args(task: dict, multimodal_results: Dict[str, Any], user_prompt: str, parent_summary: str = "", *, task_id: str = "", feedback: str = "", target_words: int = 0) -> dict:
    return {
        "section_title": task["title"],
        "parent_title": task["parent_title"],
        "multimodal_results": multimodal_results,
        "user_prompt": user_prompt,
        "previous_summary": parent_summary,
        "level": task["level"],
        "task_id": task_id,
        "feedback": feedback,
        "target_words": target_words,
    }


# ── Section-level caching: skip LLM calls for unchanged sections ────────────

def _section_cache_key(title: str, parent: str, multimodal_digest: str,
                        user_prompt: str, previous_summary: str, level: int,
                        target_words: int, feedback: str) -> str:
    """Hash all inputs affecting section content. Same hash = same output."""
    raw = "|".join([
        title, parent,
        (multimodal_digest or "")[:800],
        (user_prompt or "")[:300],
        (previous_summary or "")[:300],
        str(level), str(target_words),
        (feedback or "")[:200],
    ])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _load_section_cache(task_id: str) -> Dict[str, str]:
    """Load previous section key→content map from disk."""
    if not task_id:
        return {}
    try:
        base = Path(__file__).resolve().parent.parent.parent
        p = base / "result" / str(task_id) / "_section_cache.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_section_cache(task_id: str, cache: Dict[str, str]) -> None:
    """Persist section key→content map to disk."""
    if not task_id:
        return
    try:
        base = Path(__file__).resolve().parent.parent.parent
        p = base / "result" / str(task_id) / "_section_cache.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _find_parent_h2_index(flat: list[dict], h3_index: int) -> int:
    """Find the parent H2 index for a given H3 section."""
    target_level = flat[h3_index].get("level", 0) if h3_index < len(flat) else 0
    if target_level != 3:
        return -1
    for i in range(h3_index - 1, -1, -1):
        if flat[i].get("level") == 2:
            return i
    return -1


def generate_full_content_parallel(outline: str, multimodal_results: Dict[str, Any], user_prompt: str, *, task_id: str = "", feedback: str = "", enriched_context: str = "", target_words: int = 0) -> str:
    flat = parse_outline_sections(outline)
    if not flat:
        return ""

    tasks = _prepare_section_tasks(flat)
    if not tasks:
        return ""

    total_sections = len(tasks)
    # Weight H2 (chapter overview) sections 2× vs H3 (detail) so that
    # the word budget is proportional to each section's structural role.
    _h2n = sum(1 for _t in tasks if _t["level"] == 2)
    _h3n = sum(1 for _t in tasks if _t["level"] == 3)
    _total_w = _h2n * 2 + _h3n
    if target_words and target_words > 0 and _total_w > 0:
        _h2_budget = max(100, (target_words * 2) // _total_w)
        _h3_budget = max(80, target_words // _total_w)
    else:
        _h2_budget = _h3_budget = 0
    done_count = 0

    multimodal_digest = _multimodal_summary(multimodal_results)
    if enriched_context.strip():
        multimodal_digest = f"[技能搜集到的补充信息]\n{enriched_context.strip()}\n\n{multimodal_digest}"

    results: dict[int, str] = {}
    # Cap concurrent LLM calls to avoid overwhelming local models
    _llm_limit = int(os.environ.get("CONCURRENT_LLM_CALLS", str(MAX_WORKERS_DEFAULT)))
    max_workers = max(1, min(_llm_limit, len(tasks)))

    # ── Incremental flush state ───────────────────────────────────────
    # Instead of rebuilding content.md from flat+results on every section
    # completion (O(N²) I/O), we build the skeleton once and patch bodies
    # in-place.  A per-section lock ensures thread safety.
    base_dir = Path(__file__).resolve().parent.parent.parent
    _content_lines: list[str] = []
    _section_body_line: dict[int, int] = {}  # flat_index → line index in _content_lines
    _flush_lock = threading.Lock()
    _section_titles: dict[int, str] = {}  # flat_index → title (for stream events)

    # Build skeleton once
    _pstack: list[str] = []
    for _i, _item in enumerate(flat):
        _lv = int(_item.get("level") or 0)
        _ti = str(_item.get("title") or "").strip()
        if not _ti or _lv < 1:
            continue
        while len(_pstack) >= _lv:
            _pstack.pop()
        _pstack.append(_ti)
        if _lv == 1:
            _content_lines.append(f"# {_ti}")
        else:
            _content_lines.append("")
            _content_lines.append(f"{'#' * _lv} {_ti}")
            _content_lines.append("")
            _section_body_line[_i] = len(_content_lines)
            _content_lines.append("（生成中…）")
            _section_titles[_i] = _ti

    def _flush_partial():
        """Write current _content_lines to content.md — O(1) per section body update."""
        if not task_id:
            return
        with _flush_lock:
            try:
                p = base_dir / "result" / str(task_id) / "content.md"
                p.parent.mkdir(parents=True, exist_ok=True)
                # Patch bodies for completed sections
                newly_completed: list[dict] = []
                for _idx, _body in results.items():
                    if _body and _idx in _section_body_line:
                        _li = _section_body_line[_idx]
                        if _content_lines[_li] != _body:
                            _content_lines[_li] = _body
                            newly_completed.append({
                                "level": int(flat[_idx].get("level") or 0) if _idx < len(flat) else 0,
                                "title": _section_titles.get(_idx, ""),
                            })
                raw = "\n".join(_content_lines).strip() + "\n"
                p.write_text(raw, encoding="utf-8")
                if newly_completed:
                    stream_p = base_dir / "result" / str(task_id) / "stream.jsonl"
                    with open(stream_p, "a", encoding="utf-8") as sf:
                        for sec in newly_completed:
                            sf.write(json.dumps({
                                "type": "section_done", "title": sec["title"],
                                "level": sec["level"], "done": done_count,
                                "total": total_sections,
                            }, ensure_ascii=False) + "\n")
            except Exception:
                pass

    # ── Q4 P1: Two-stage parallel generation (H2 first → then H3) ──────────
    # Stage 1: H2 chapters generate in parallel; their summaries feed Stage 2.
    # Stage 2: H3 sub-sections generate with parent H2 summaries for coherence.
    # This provides H3 sections with actual context instead of empty placeholders.

    _flush_partial()  # skeleton for frontend

    section_cache = _load_section_cache(task_id)
    cache_hits = 0

    # Check cache for all tasks
    h2_tasks: list[dict] = []
    h3_tasks: list[dict] = []
    for t in tasks:
        _sec_budget = _h2_budget if t["level"] == 2 else _h3_budget
        key = _section_cache_key(
            t["title"], t["parent_title"], multimodal_digest,
            user_prompt, "", t["level"], _sec_budget, feedback)
        if key in section_cache:
            results[t["index"]] = section_cache[key]
            done_count += 1
            cache_hits += 1
            logger.info(f"section_cache_hit section={t['title'][:40]}")
        else:
            t["_cache_key"] = key
            if t["level"] == 2:
                h2_tasks.append(t)
            else:
                h3_tasks.append(t)

    def _generate_batch(task_list: list[dict], parent_summaries: dict[int, str] | None = None):
        """Generate a batch of sections in parallel, returning {index: content}."""
        if not task_list:
            return {}
        _check_cancel(task_id)
        names = [t["title"][:30] for t in task_list]
        logger.info(f"content_generating sections={len(task_list)} titles={names} task={task_id}")
        batch_results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {}
            for t in task_list:
                _sec_budget = _h2_budget if t["level"] == 2 else _h3_budget
                ps = ""
                if parent_summaries is not None and t["level"] == 3:
                    # Find parent H2 index
                    parent_idx = _find_parent_h2_index(flat, t["index"])
                    ps = parent_summaries.get(parent_idx, "")
                args = _build_section_args(t, multimodal_results, user_prompt,
                                           parent_summary=ps,
                                           task_id=task_id, feedback=feedback,
                                           target_words=_sec_budget)
                futs[pool.submit(generate_section_content, **args)] = t["index"]
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    batch_results[idx] = str(fut.result() or "")
                except Exception as e:
                    batch_results[idx] = f"（本节生成失败：{str(e)[:120]}）"
                _check_cancel(task_id)
                nonlocal done_count
                done_count += 1
                title = str(flat[idx].get("title") or "") if idx < len(flat) else ""
                logger.info(f"content_section_done {done_count}/{total_sections} title={title[:40]}")
                _notify_section_progress(task_id, done_count, total_sections, title)
                _flush_partial()
        return batch_results

    # Stage 1: Generate all H2 in parallel
    h2_results = _generate_batch(h2_tasks)
    results.update(h2_results)

    # Compute H2 summaries for Stage 2 (H3 sub-sections)
    h2_summaries: dict[int, str] = {}
    for t in h2_tasks:
        idx = t["index"]
        body = results.get(idx, "")
        if body:
            h2_summaries[idx] = _llm_summarize_for_next(t["title"], body)

    # Stage 2: Generate all H3 in parallel (with parent H2 summary)
    h3_results = _generate_batch(h3_tasks, parent_summaries=h2_summaries)
    results.update(h3_results)

    # Update cache for all generated sections
    for t in tasks:
        ck = t.get("_cache_key")
        body = results.get(t["index"], "")
        if ck and body:
            section_cache[ck] = body

    # Persist section cache
    _save_section_cache(task_id, section_cache)
    if cache_hits:
        logger.info(f"section_cache_summary task={task_id} hits={cache_hits}/{total_sections}")

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

    # ── Q4 P2: Transition sentence injection between adjacent H2 sections ──
    if _h2n >= 2:
        raw = _inject_h2_transitions(raw, task_id)

    return raw


def _inject_h2_transitions(content: str, task_id: str) -> str:
    """Check adjacent H2 sections for missing transition sentences, inject if absent.

    A transition is a short sentence at the end of section N that logically
    connects to section N+1.  Without transitions, the document reads as a
    collection of standalone essays rather than a cohesive report.
    """
    # Parse into H2 blocks
    blocks = re.split(r"\n(?=## )", content)
    if len(blocks) < 2:
        return content

    result_blocks = [blocks[0]]
    for i in range(1, len(blocks)):
        prev = result_blocks[-1]
        curr = blocks[i]

        # Extract H2 titles
        prev_title_match = re.search(r"^##\s+(.+)", prev, re.MULTILINE)
        curr_title_match = re.search(r"^##\s+(.+)", curr, re.MULTILINE)
        if not prev_title_match or not curr_title_match:
            result_blocks.append(curr)
            continue

        prev_title = prev_title_match.group(1).strip()
        curr_title = curr_title_match.group(1).strip()

        # Check if a transition already exists: look for connector words in
        # the last 2 sentences of the previous section
        last_sentences = re.split(r"[。！？\n](?=\s*[^\s])", prev.strip())[-3:]
        last_text = "".join(last_sentences[-2:])
        connectors = ["接下来", "下面", "下一", "进一步", "在此基础", "基于以上",
                      "承接", "前述", "综上所", "在此背景", "另一方"]
        has_transition = any(c in last_text for c in connectors)

        if not has_transition:
            # Generate a lightweight transition via LLM
            transition = _generate_transition(prev_title, curr_title, prev[-300:], task_id)
            if transition:
                # Insert before the next H2 heading
                prev += "\n\n" + transition
                result_blocks[-1] = prev

        result_blocks.append(curr)

    return "\n".join(result_blocks)


def _generate_transition(prev_title: str, next_title: str,
                         context: str, task_id: str) -> str:
    """Generate a 1-2 sentence transition connecting two H2 sections."""
    try:
        from agent_file_create.llm_client import call_llm
    except Exception:
        return ""

    prompt = (
        "你是一个报告连贯性编辑。下面有两个相邻章节，请生成1个过渡句（15-40字），"
        "自然地将上一章节引向下一章节。过渡句放在上一章节的末尾。\n\n"
        f"上一章：{prev_title}\n"
        f"下一章：{next_title}\n"
        f"上一章末尾内容（参考）：\n{context[-300:]}\n\n"
        "要求：只输出过渡句本身，不要加任何前缀、编号或解释。"
        "过渡句应该自然流畅，不要生硬地写'下面我们来讨论...'。"
    )
    try:
        text = call_llm(
            prompt,
            timeout_s=20,
            temperature=0.3,
            num_predict=60,
            system="你是一个中文报告编辑助手。",
            api_style=CONTENT_API_STYLE,
            api_endpoint=CONTENT_API_ENDPOINT,
            api_key=CONTENT_API_KEY,
            model_name=CONTENT_MODEL_NAME,
        )
        result = str(text or "").strip()
        # Validate: not too short, not too long, no markdown
        if 10 <= len(result) <= 80 and not result.startswith("#"):
            return result
    except Exception:
        pass
    return ""


def regenerate_section(
    outline: str,
    content: str,
    section_name: str,
    multimodal_results: Dict[str, Any],
    user_prompt: str,
    *,
    task_id: str = "",
    guidance: str = "",
) -> str:
    """Regenerate a single section (and its children) in the content.

    If *guidance* is provided (e.g. user-edited draft), it is injected as
    strong preference into the generation prompt so the LLM preserves the
    user's intent while polishing structure and coherence.

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
        feedback=guidance,
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