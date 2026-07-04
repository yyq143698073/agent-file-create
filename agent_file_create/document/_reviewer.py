"""Fact-checking and coherence review extracted from content_generator.py.

All functions here are stateless and can be imported freely.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent_file_create.config import (
    CONTENT_API_ENDPOINT,
    CONTENT_API_KEY,
    CONTENT_API_STYLE,
    CONTENT_MODEL_NAME,
)
from agent_file_create.llm_factory import get_chat_model

# ── Prompt templates ──────────────────────────────────────────────────────────

SECTION_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
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

COHERENCE_REVIEW_PROMPT = ChatPromptTemplate.from_messages([
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

SECTION_FACT_CHECK_PROMPT = ChatPromptTemplate.from_messages([
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


# ── Structured fact extraction ────────────────────────────────────────────────

def extract_facts_from_materials(multimodal_digest: str) -> dict:
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

    # Organization/institution names
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


def cross_check_facts(section_text: str, material_facts: dict) -> list[str]:
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

    return issues[:12]


def patch_unverified_claims(content: str, material_facts: dict) -> tuple[str, list[str]]:
    """Auto-replace unverifiable numeric/entity claims with ``[数据待核实]``.

    Collects all matches first, then replaces in reverse order to avoid
    index drift from earlier replacements shifting later positions.

    Returns (patched_content, list of replacements made).
    """
    text = str(content or "")
    if not text.strip() or not material_facts:
        return text, []

    material_numbers = material_facts.get("numbers", set())
    material_entities = material_facts.get("entities", set())

    # Collect all to-replace spans: (start, end, label)
    to_replace: list[tuple[int, int, str]] = []

    # ── Numbers (with units) — only flag large/specific values ──
    for m in re.finditer(
        r"\d{2,}(?:\.\d+)?\s*[万亿千百]?\s*(?:元|美元|亿|万|倍|辆)",
        text,
    ):
        val = m.group().strip()
        if len(val) >= 4 and val not in material_numbers:
            to_replace.append((m.start(), m.end(), f"数字:{val}"))

    # ── Percentages — only flag if specific (>= 2 digits before decimal) ──
    for m in re.finditer(r"\d{2,}(?:\.\d+)?\s*[%％]", text):
        val = m.group().strip()
        if len(val) >= 3 and val not in material_numbers:
            to_replace.append((m.start(), m.end(), f"百分比:{val}"))

    # ── Bare large numbers (>= 4 chars): "2024", "1000", "15项" ──
    for m in re.finditer(
        r"(?<![a-zA-Z0-9_])\d{3,}(?:\.\d+)?(?:\s*[BbKkMm]|[项指标维度轮层])?(?![a-zA-Z0-9_%％])",
        text,
    ):
        val = m.group().strip()
        if len(val) >= 4 and val not in material_numbers:
            already = any(s == m.start() for s, _, _ in to_replace)
            if not already:
                to_replace.append((m.start(), m.end(), f"裸数字:{val}"))

    # ── Entities ──
    for pat in [
        r"[一-鿿]{2,16}(?:公司|集团|大学|银行|医院|研究所|中心)",
        r"[一-鿿]{2,8}(?:有限|股份)公司",
    ]:
        for m in re.finditer(pat, text):
            val = m.group().strip()
            if val not in material_entities:
                exists = any(
                    val[:4] in e or e[:4] in val for e in material_entities
                )
                if not exists:
                    to_replace.append((m.start(), m.end(), f"机构:{val}"))

    if not to_replace:
        return text, []

    # Sort reverse by start position — replace right-to-left, no index drift
    to_replace.sort(key=lambda x: -x[0])

    patches: list[str] = []
    result = text
    for start, end, label in to_replace:
        result = result[:start] + "[数据待核实]" + result[end:]
        patches.append(f"{label} -> [数据待核实]")

    return result, patches


def detect_cross_document_conflicts(hits: list, materials_text: str = "") -> list[dict]:
    """Detect conflicting numeric claims across documents on the same topic.

    Groups numeric claims by context (the 3-6 chars before each number).
    If two documents claim different values for the same metric, flags it.

    Returns list of conflicts: ``[{"metric": "...", "values": {"doc_a": "95.3%", "doc_b": "92.1%"}}]``
    """
    if len(hits) < 2:
        return []

    import re
    from collections import defaultdict

    # Extract (metric_name, numeric_value, doc_id) tuples from all hits
    claims: list[tuple[str, str, str]] = []
    # Known metric names to detect (common in Chinese academic/research text)
    _METRIC_NAMES = [
        r"准确率", r"精确率", r"召回率", r"F1", r"BLEU", r"ROUGE",
        r"精度", r"误差", r"正确率", r"覆盖率", r"成功率",
        r"延迟", r"吞吐", r"响应时间", r"推理速度",
        r"参数量", r"模型大小", r"训练时间",
        r"准确度", r"精确度",
    ]
    _METRIC_PATTERN = "|".join(_METRIC_NAMES)

    for h in hits:
        text = str(getattr(h, "content", "") or "")
        doc_id = str(getattr(h, "doc_id", "") or "")
        if "/" in doc_id:
            doc_id = doc_id.rsplit("/", 1)[-1]
        if "\\" in doc_id:
            doc_id = doc_id.rsplit("\\", 1)[-1]
        for m in re.finditer(
            rf"({_METRIC_PATTERN})(?:达|约|为|是|近|超|低至|高于|达到)?"
            r"\s*(\d+(?:\.\d+)?\s*(?:%|％|[万亿千百]?\s*(?:元|美元|亿|万))?)",
            text,
        ):
            metric = m.group(1).strip()
            value = m.group(2).strip()
            if metric and value and len(value) >= 2:
                claims.append((metric, value, doc_id or "unknown"))

    if not claims:
        return []

    # Group by metric name
    by_metric: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for metric, value, doc_id in claims:
        by_metric[metric][doc_id].add(value)

    # Find conflicts: same metric, different values from different docs
    conflicts: list[dict] = []
    for metric, doc_values in by_metric.items():
        if len(doc_values) < 2:
            continue
        # Collect unique (value, doc) pairs across all docs
        unique: dict[str, str] = {}
        for doc_id, values in doc_values.items():
            for v in values:
                if v not in unique:
                    unique[v] = doc_id
        if len(unique) < 2:
            continue
        # ── Threshold check: only flag if values differ numerically ──
        _numeric_values = []
        for v in unique:
            num_match = re.search(r"(\d+(?:\.\d+)?)", v)
            if num_match:
                _numeric_values.append(float(num_match.group(1)))
        if len(_numeric_values) >= 2:
            _diff = max(_numeric_values) - min(_numeric_values)
            _maxv = max(_numeric_values)
            # Only flag if difference > 1% of max (avoids rounding noise)
            if _maxv > 0 and _diff / _maxv < 0.01:
                continue

        conflicts.append({
            "metric": metric,
            "claims": {doc: val for val, doc in unique.items()},
        })
        if len(conflicts) >= 5:
            break

    return conflicts


def annotate_conflicts_in_materials(materials: str, conflicts: list[dict]) -> str:
    """Append conflict warnings to the materials text so LLM sees them.

    Format: ``⚠️ 数据冲突: [context] doc_a 声称 X, doc_b 声称 Y``
    """
    if not conflicts or not materials:
        return materials
    lines = [materials, "", "── 检测到以下跨文档数据冲突，请主动呈现而非掩盖 ──"]
    for c in conflicts[:5]:
        claims_str = ", ".join(
            f"{doc} 声称 {val}" for val, doc in c["claims"].items()
        )
        lines.append(f"⚠️ 「{c['context']}」: {claims_str}")
    return "\n".join(lines)


# ── LLM-based verification ────────────────────────────────────────────────────

def verify_section_facts(section_title: str, section_text: str, multimodal_digest: str) -> list[str]:
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
        chain = SECTION_FACT_CHECK_PROMPT | llm | StrOutputParser()
        result = (chain.invoke({
            "material_digest": multimodal_digest[:2000],
            "section_text": section_text[:1500],
        }) or "").strip()
        if not result or result.upper().startswith("PASS"):
            return []
        return [line.strip("- ").strip() for line in result.splitlines() if line.strip() and "问题" not in line[:4]]
    except Exception:
        return []


def final_coherence_review(full_text: str, multimodal_digest: str) -> str:
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

    # ── Step 1: LLM coherence review (layer 3) ──────────────────────────────────
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
        chain = COHERENCE_REVIEW_PROMPT | llm | StrOutputParser()
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

    # ── Step 2: Structured fact cross-check (layer 4, long reports only) ────────
    if run_layer4:
        try:
            material_facts = extract_facts_from_materials(multimodal_digest)
            fact_issues = cross_check_facts(full_text, material_facts)
            for fi in fact_issues:
                all_issues.append(f"## 事实核查: {fi}")
        except Exception:
            pass

    if not all_issues:
        return full_text

    # ── Apply markers for flagged issues ────────────────────────────────────────
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
            marker = f"\n\n<!-- 核查提示：{issue.replace('##', '').strip()[:160]} -->"
            if marker not in corrected[-800:]:
                corrected = corrected.rstrip() + marker

    return corrected
