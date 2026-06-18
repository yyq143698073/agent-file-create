import hashlib
import json
import logging
import os
import re
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
            extra={"sections_done": done, "sections_total": total, "section_title": section_title},
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
    lo, hi = target_range
    parts = [
        "你是一个资深行业分析师和顶级文案，正在撰写一份专业的深度报告。",
        "",
        "核心指令：",
        "1) 拒绝复读：严禁直接复制粘贴参考材料中的长句，用全新语言重构核心观点。",
        "2) 逻辑流与衔接：用因果、递进、转折等连接词建立段落间关系。开篇用1句话承接前文（如有），结尾用1句话自然过渡到下一节（如有），不要生硬地写「接下来我们将讨论……」。",
        "3) 场景化扩写：解释数据和结论的业务含义，但场景必须是材料中有线索支撑的，不要凭空想象。",
        "4) 降低幻觉：不要编造具体的数字、机构名、人名、年份。材料中明确出现的数值可以引用，但要标注具体来源（如「据某论文实验数据」），禁止使用「据材料显示」「据资料记载」等笼统表述。不确定就说「相关数据暂缺」。",
        "5) 溯源要求：每个关键论断后，用「（据+来源文件具体关键词）」标注——如引用自「RAG技术综述_张三.pdf」则标注为「（据RAG技术综述）」。多个材料支撑时标注「（综合多份材料）」。无任何材料支撑的推论标注「（分析推测）」。",
        "6) 编号引用：参考材料中的 【1】【2】 等编号对应来源文件。当你引用某个编号材料的具体数据时，在句末标注引用编号，如「实验显示准确率达95.3%【1】」。可同时使用自然语言引用和编号引用。注意使用【】而非[]，避免 Markdown 链接语法冲突。",
        "7) 时效优先：参考材料标注了发表年份（如「2023, xxx.pdf」）。优先采信年份较新的来源。如果引用了较旧的文献（3年以上），请在文中注明其发表年份或标注'据20XX年研究'。",
        "8) 简化标注：你只需使用两套标注——①自然语言引用「（据XXX）」用于标明来源文件，②【n】编号引用用于对标检索片段。两者可以同时出现在同一论断中。不再需要第三套标注系统。",
        "",
    ]
    # ── Type-specific instructions ──
    if section_type == "data":
        parts += [
            "⚠️ 本节为「数据型」章节（含实验、性能、对比数据），特别要求：",
            "• 必须逐条引用来源材料中的数据，不可笼统概括（如材料说「35.1%」不要写成「约三分之一」）。",
            "• 每个数据点后必须标注出处：「（据<材料名>）」。",
            "• 如果材料中数据不足，只写已有数据，禁止推测数值或编造比较对象。",
            "• 温度极低：行文可以干练，但数据必须精确。",
            "",
        ]
    elif section_type == "experiment_setup":
        parts += [
            "🔧 本节为「实验设定/方法型」章节（含实验设计、数据集、实现细节），特别要求：",
            "• 准确描述实验配置和参数设定，不遗漏关键超参数（学习率、batch size、epoch 等）。",
            "• 数据集描述需包含规模、来源、划分方式，不可笼统说「使用公开数据集」。",
            "• 方法描述要能复现：模型架构、训练策略、评估方案逐条写清。",
            "• 可以引用材料中的配置表，但用自己的语言组织，标注出处。",
            "",
        ]
    elif section_type == "analysis":
        parts += [
            "💡 本节为「分析型」章节（含讨论、展望、建议、启示），特别要求：",
            "• 可以在材料事实基础上做合理的推理和延伸判断，但需标注「（分析推测）」。",
            "• 鼓励多材料综合对比——如果材料A和材料B的结论有冲突或互补，主动指出来。",
            "• 可以提出材料本身未明确表述但经你推理得出的观点，但不可与材料事实矛盾。",
            "• 温度较高：鼓励有洞察力的分析，不追求逐句引用。",
            "",
        ]
    # review type uses defaults (no extra instructions)
    parts += [
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
        f"写作要求：本节建议 {lo}~{hi} 字。",
        "只输出本节正文，不要输出标题行。不要输出「本节」「本章」等元描述文字。",
        "",
        "正文：",
    ]
    return "\n\n".join(parts).strip()


def _call_qwen_text(prompt: str, *, timeout_s: int, num_predict: int, temperature: float = 0.4) -> str:
    text = call_llm(
        prompt,
        timeout_s=timeout_s,
        temperature=temperature,
        num_predict=num_predict,
        stop=["```"],
        system="你是一个中文文档处理助手。",
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
        num_predict = max(300, hi * 3)
    elif level == 2:
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
        section_type=section_type,
        enriched_context=enriched_context,
    )

    # Temperature by section type: data=0.1, analysis=0.4, review=0.2
    temp_map = {"data": 0.1, "analysis": 0.4, "review": 0.2}
    gen_temp = temp_map.get(section_type, 0.2)
    logger.info("section_type title=%s type=%s temp=%.2f", section_title[:40], section_type, gen_temp)

    for attempt in range(2):
        _check_cancel(task_id)
        t0 = time.perf_counter()
        text = _call_qwen_text(prompt, timeout_s=MODEL_TIMEOUT, num_predict=num_predict, temperature=gen_temp)
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

    # Count total sections for progress tracking
    total_sections = sum(1 for item in flat if int(item.get("level") or 0) >= 2)
    per_section_words = max(80, target_words // max(total_sections, 1)) if target_words and target_words > 0 else 0
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
            body = generate_section_content(title, parent_title, multimodal_results, user_prompt, previous_summary, level, next_title, task_id=task_id, feedback=feedback, target_words=per_section_words, enriched_context=enriched_context)
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
            body = generate_section_content(title, parent_title, multimodal_results, user_prompt, previous_summary, level, next_title, task_id=task_id, feedback=feedback, target_words=per_section_words, enriched_context=enriched_context)
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
        body = generate_section_content(title, parent_title, multimodal_results, user_prompt, previous_summary, level, next_title, task_id=task_id, feedback=feedback, target_words=per_section_words)
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


def generate_full_content_parallel(outline: str, multimodal_results: Dict[str, Any], user_prompt: str, *, task_id: str = "", feedback: str = "", enriched_context: str = "", target_words: int = 0) -> str:
    flat = parse_outline_sections(outline)
    if not flat:
        return ""

    tasks = _prepare_section_tasks(flat)
    if not tasks:
        return ""

    total_sections = len(tasks)
    per_section_words = max(80, target_words // max(total_sections, 1)) if target_words and target_words > 0 else 0
    done_count = 0

    multimodal_digest = _multimodal_summary(multimodal_results)
    if enriched_context.strip():
        multimodal_digest = f"[技能搜集到的补充信息]\n{enriched_context.strip()}\n\n{multimodal_digest}"

    results: dict[int, str] = {}
    # Cap concurrent LLM calls to avoid overwhelming local models
    _llm_limit = int(os.environ.get("CONCURRENT_LLM_CALLS", str(MAX_WORKERS_DEFAULT)))
    max_workers = max(1, min(_llm_limit, len(tasks)))

    # Helper to write partial content and stream events
    base_dir = Path(__file__).resolve().parent.parent.parent  # pre-compute for both paths
    def _flush_partial():
        if not task_id:
            return
        try:
            p = base_dir / "result" / str(task_id) / "content.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            buf: list[str] = []
            pstack: list[str] = []
            completed_sections = []
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
                    if body and lv >= 2:
                        completed_sections.append({"level": lv, "title": ti})
            raw = "\n".join(buf).strip() + "\n"
            p.write_text(raw, encoding="utf-8")
            # Write stream event for SSE (append-only JSONL)
            if completed_sections:
                stream_p = base_dir / "result" / str(task_id) / "stream.jsonl"
                with open(stream_p, "a", encoding="utf-8") as sf:
                    for sec in completed_sections:
                        sf.write(json.dumps({"type": "section_done", "title": sec["title"],
                                             "level": sec["level"], "done": done_count,
                                             "total": total_sections}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ── Unified parallel phase: submit ALL sections (h2 + h3) at once ──
    # h3 sections use empty parent_summary initially (h2 content not ready yet),
    # but gain coherence from parent_title + multimodal_digest + user_prompt.
    # This eliminates the h2→h3 serial barrier, cutting wall-clock time by ~40%.

    _flush_partial()  # skeleton for frontend

    section_cache = _load_section_cache(task_id)
    cache_hits = 0

    # Check cache first for all tasks
    uncached: list[dict] = []
    for t in tasks:
        parent_summary = ""  # always empty — h2 content may not be ready
        key = _section_cache_key(
            t["title"], t["parent_title"], multimodal_digest,
            user_prompt, parent_summary, t["level"], per_section_words, feedback)
        if key in section_cache:
            results[t["index"]] = section_cache[key]
            done_count += 1
            cache_hits += 1
            logger.info(f"section_cache_hit section={t['title'][:40]}")
        else:
            t["_cache_key"] = key
            uncached.append(t)

    if uncached:
        _check_cancel(task_id)
        section_names = [t["title"][:30] for t in uncached]
        logger.info(f"content_generating sections={len(uncached)}/{total_sections} titles={section_names} task={task_id}")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for t in uncached:
                args = _build_section_args(t, multimodal_results, user_prompt,
                                           parent_summary="",
                                           task_id=task_id, feedback=feedback,
                                           target_words=per_section_words)
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
                logger.info(f"content_section_done {done_count}/{total_sections} title={section_title[:40]}")
                _notify_section_progress(task_id, done_count, total_sections, section_title)
                _flush_partial()

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
    return raw  # _final_coherence_review replaced by Critic graph node


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