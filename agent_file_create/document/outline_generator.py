import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Dict, List, Optional

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


# ── Q2 P0: Naming quality checks ─────────────────────────────────────────────

_TEMPLATE_TITLE_PATTERNS = [
    r"背景", r"概述", r"分析", r"建议", r"总结",
    r"引言", r"展望", r"介绍", r"现状", r"意义",
    r"讨论", r"结论", r"对策", r"方案", r"趋势",
]


def _check_naming_quality(outline: str) -> list[str]:
    """Check outline naming quality — returns non-blocking warnings.

    Detects:
    - Template-like headings (空洞词: 背景/概述/分析/建议 etc.)
    - Overly short headings (<4 characters)
    - Adjacent similar headings (edit distance > 70%)
    """
    warnings: list[str] = []
    headings: list[tuple[int, str]] = []
    for line in (outline or "").splitlines():
        m = re.match(r"^(#{2,3})\s+(.+)$", line.strip())
        if m:
            headings.append((len(m.group(1)), m.group(2).strip()))

    if not headings:
        return warnings

    # 1. Template-like + short heading detection
    for level, title in headings:
        clean = re.sub(r"^[\d.]+\s*", "", title).strip()
        if len(clean) < 4:
            warnings.append(f"标题过短(<4字): '{title}'")
        for pat in _TEMPLATE_TITLE_PATTERNS:
            if re.search(pat, clean) and len(clean) <= 8:
                warnings.append(f"模板化标题(含'{pat}'): '{title}'")
                break

    # 2. Adjacent heading similarity
    for i in range(len(headings) - 1):
        t1 = re.sub(r"^[\d.]+\s*", "", headings[i][1]).strip()
        t2 = re.sub(r"^[\d.]+\s*", "", headings[i + 1][1]).strip()
        dist = _levenshtein(t1, t2)
        sim = 1.0 - dist / max(len(t1), len(t2), 1)
        if sim > 0.70:
            warnings.append(f"相邻标题高度相似({sim:.0%}): '{t1}' ↔ '{t2}'")

    return warnings


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


# ── Q2 P1: Critical section content check ────────────────────────────────────

_CRITICAL_KEYWORDS = [
    "局限", "不足", "限制", "缺口", "改进", "风险",
    "挑战", "不确定性", "失败", "对比", "争议", "缺陷",
]


def _check_critical_section(outline: str) -> list[str]:
    """Verify the critical/limitation chapter has concrete content.

    Checks:
    - A critical chapter exists in the last 2 H2 positions
    - It contains at least one quantitative indicator or source reference
    """
    h2s: list[str] = []
    for line in (outline or "").splitlines():
        m = re.match(r"^##\s+(.+)$", line.strip())
        if m:
            h2s.append(m.group(1).strip())

    if len(h2s) < 3:
        return []  # structural validation handles this

    # Check last 2 H2 positions
    last_two = h2s[-2:]
    has_critical = any(
        any(kw in h for kw in _CRITICAL_KEYWORDS)
        for h in last_two
    )

    if not has_critical:
        return ["批判章节缺失：最后2个H2中未找到局限性/不足/挑战等关键词"]

    # Check for quantitative indicators in headings
    critical_h2 = [h for h in last_two if any(kw in h for kw in _CRITICAL_KEYWORDS)]
    if critical_h2:
        h3s_under_critical = []
        capture = False
        for line in (outline or "").splitlines():
            m = re.match(r"^(#{2,3})\s+(.+)$", line.strip())
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                if level == 2:
                    capture = any(kw in title for kw in _CRITICAL_KEYWORDS)
                elif level == 3 and capture:
                    h3s_under_critical.append(title)

        has_detail = any(
            re.search(r"\d+[%％]|\d+[个项种类条]|[0-9]+(?:\.\d+)?", h)
            for h in h3s_under_critical
        )
        if not has_detail and len(h3s_under_critical) <= 1:
            return ["批判章节缺少具体子节：建议增加包含量化数据或具体技术细节的子节"]

    return []


# ── Q2 P2: Topic coverage check (jieba-based, no LLM call) ──────────────────


def _check_topic_coverage(outline: str, user_prompt: str) -> dict:
    """Check whether outline headings cover key topics from the user prompt.

    Uses jieba token-frequency extraction on the user prompt and matches
    against outline heading text.  Returns coverage ratio + uncovered terms.
    Coverage < 0.5 triggers feedback injection for retry.
    """
    if not user_prompt or len(user_prompt.strip()) < 10:
        return {"coverage": 1.0, "covered": [], "uncovered": []}

    try:
        import jieba
    except Exception:
        return {"coverage": 1.0, "covered": [], "uncovered": []}

    # Extract key terms from user prompt (TF top-10, skip stopwords)
    stop = set("的了吗呢吧啊是都在和与或对从到用把被让给为以而因但就这那也都还没又再很更太只就可会要能说看想他她它我你".split())
    tokens = [t.strip() for t in jieba.lcut(user_prompt) if len(t.strip()) >= 2 and t.strip() not in stop]
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    user_terms = [t for t, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]]

    if not user_terms:
        return {"coverage": 1.0, "covered": [], "uncovered": []}

    # Extract headings text
    headings = re.findall(r"^#{1,3}\s+(.+)$", outline, re.MULTILINE)
    heading_text = " ".join(headings)

    covered = []
    uncovered = []
    for term in user_terms:
        if term in heading_text or any(term in h for h in headings):
            covered.append(term)
        else:
            uncovered.append(term)

    cov = len(covered) / max(len(user_terms), 1)
    return {"coverage": round(cov, 3), "covered": covered, "uncovered": uncovered}


def _clean_llm_output(text: str) -> str:
    out = (text or "").strip()
    out = re.sub(r"^```[a-zA-Z]*\s*", "", out).strip()
    out = re.sub(r"\s*```$", "", out).strip()
    # Demote headings deeper than ### to bold list items
    out = re.sub(r"^#{4,}\s+(.+)", r"- **\1**", out, flags=re.MULTILINE)
    return out


def _add_outline_numbering(outline: str) -> str:
    """Add hierarchical numbering to markdown outline headings.

    Transforms::

        # 报告标题
        ## 背景分析
        ### 行业现状
        ## 数据解读

    into::

        # 报告标题               ← single H1: no number (document title)
        ## 1.1 背景分析
        ### 1.1.1 行业现状
        ## 1.2 数据解读

    When there are 2+ ``#`` headings they become numbered chapters::

        # 1. 营收分析
        ## 1.1 产品线
        # 2. 成本分析
        ## 2.1 原材料
    """
    lines = (outline or "").splitlines()

    # Only number H1 when the outline has multiple top-level chapters
    h1_count = sum(1 for line in lines if re.match(r"^#\s", line.strip()))
    number_h1 = h1_count > 1

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

        # Build hierarchical number.  When H1 is not numbered (single-chapter
        # report), shift the numbering baseline to level 2 so H2 gets "1."
        # instead of the orphaned "1.1." that implies an invisible chapter.
        start_lv = 1 if number_h1 else 2
        number_parts = [str(counters[lv]) for lv in range(start_lv, level + 1) if counters[lv] > 0]
        number = ".".join(number_parts) + "." if number_parts else ""

        # Preserve original indentation
        indent = line[:len(line) - len(line.lstrip())]
        if level == 1 and not number_h1:
            result.append(f"{indent}# {title}")
        elif level == 1:
            result.append(f"{indent}# {number} {title}")
        else:
            result.append(f"{indent}{'#' * level} {number} {title}")

    return "\n".join(result)


def _build_outline_prompt(user_req: str, digest: str, feedback: str, target_words: int = 0, template_sections: list = None) -> str:
    rules = [
        "你是一个专业报告的大纲生成助手。请基于参考材料与用户需求输出 Markdown 大纲。",
        "",
        "结构规则（必须遵守）：",
        "1) # 一级标题 = 章标题（至少1个，内容丰富的报告可增加到不超过6个）。",
        "   每个 # 代表一个独立的「章」（chapter），编号由系统自动生成（1. 2. 3. …）。",
        "   单主题短报告使用 1 个 # 即可；多维度综合报告可将每个维度设为独立的 # 章。",
        "2) ## 二级标题 = 主章节（至少3个，建议不超过8个）。",
        "   命名原则：从材料中提取最突出的主题作为章节名，而不是套用「背景/分析/建议」的固定模板。",
        "   好的命名示例：「华东区Q3销售异常分析」「竞品A产品策略拆解」「供应链成本优化路径」",
        "   差的命名示例：「第一章 背景」「第一部分 概述」（过于空洞）",
        "3) ### 三级标题 = 子节（每个 ## 下至少1个，建议不超过5个）。",
        "   子节应拆解主章节的具体维度，如数据拆解、原因分析、方案对比、案例举证等。",
        "4) 最多到 ### 三级标题。禁止使用 #### 或更深的标题。如果内容需要更细粒度，用列表项或粗体文本组织，不要增加标题层级。",
        "5) 层级必须连续，禁止 # 直接跳 ###（跳过 ##）。",
        "6) 如果材料中没有支撑某类内容，不要强行编造该章节。宁可大纲短一些，也不要无中生有。",
        "7) 只输出 Markdown 大纲本身，不要解释、不要前言后语、不要代码块包裹。",
        "8) 避免同质化结构：连续多个 ## 章节如果采用完全相同的「概述—拆解—数据」模式，请做结构调整——例如将某一章改为问题驱动型（以具体问题开头）、案例分析型（以实例贯穿）、或对比分析型（多方案并列）。相邻章节的结构类型不应完全相同。",
        "9) 强制批判章节：大纲必须包含一个 ## 二级章节来讨论方法的局限性、失败场景、未解决的问题或与其他方法的对比差距。要求：a) 必须引用至少一个具体来源（标注材料编号如'根据材料1...'），b) 至少包含一项量化数据或具体技术细节，c) 不得使用'相关研究不足''数据有限'等无实质内容的表述。该章节放在大纲后半部分（倒数第1~2个 ## 的位置），不能是 ### 三级子节。",
        "",
        "【标题命名示例】",
        "✅ 好的标题（具体、可从材料中定位）：",
        "- 锂电池能量密度五年提升路径：从200Wh/kg到500Wh/kg",
        "- Q3华东区销售异常的三重归因：价格、渠道、竞品",
        "- 供应链成本优化：物流外包 vs 自建仓储的ROI对比",
        "❌ 差的标题（空洞、模板化，请避免）：",
        "- 第一章 背景分析",
        "- 第二节 数据概况",
        "- 相关技术概述",
        "- 未来展望与建议",
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
    # Fast skip: no template sections → coverage check is redundant with format validation
    sections_to_check = (template_sections or [])[:]
    if not sections_to_check:
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
        # ── Run format validation + coverage check in parallel ──────
        # Validation is pure regex (instant); coverage check is an LLM
        # call (1-2 s).  Running them concurrently eliminates the serial
        # wait for the LLM on the first (passing) attempt.
        issues: list[str]
        coverage_future: Optional[Future] = None
        if template_sections:
            with ThreadPoolExecutor(max_workers=2) as _pool:
                _val_future = _pool.submit(_validate_outline, out)
                coverage_future = _pool.submit(_check_outline_coverage, out, user_req, template_sections)
                issues = _val_future.result()
        else:
            issues = _validate_outline(out)

        # ── Q2: Run naming + critical + topic-coverage checks (non-blocking) ──
        naming_warnings = _check_naming_quality(out)
        critical_issues = _check_critical_section(out)
        topic_cov = _check_topic_coverage(out, user_prompt)

        if not issues:
            coverage_issues = coverage_future.result() if coverage_future else []
            soft_warnings = naming_warnings + critical_issues
            if not coverage_issues and not soft_warnings:
                t1 = time.perf_counter()
                logger.info(f"outline_done seconds={t1 - t0:.2f} prompt_chars={len(prompt)} outline_chars={len(out)} attempts={attempt + 1}")
                return _add_outline_numbering(out)

            # Feed soft warnings back for retry
            has_hard_issue = bool(coverage_issues)
            soft_feedback: list[str] = []

            if coverage_issues:
                soft_feedback.append(f"内容覆盖不足，缺失或需加强：{'、'.join(coverage_issues)}")
            if topic_cov["coverage"] < 0.50:
                soft_feedback.append(
                    f"用户需求关键词未覆盖：{'、'.join(topic_cov['uncovered'][:6])}"
                )
                has_hard_issue = True
            if soft_warnings:
                soft_feedback.append(f"命名质量/批判章节建议：{'；'.join(soft_warnings[:4])}")

            if not has_hard_issue:
                t1 = time.perf_counter()
                logger.info(f"outline_done_with_soft_warnings seconds={t1 - t0:.2f} warnings={len(soft_warnings)} topic_cov={topic_cov['coverage']}")
                return _add_outline_numbering(out)

            last_issues = ["；".join(soft_feedback)]
            logger.warning(f"outline_quality_retry attempt={attempt + 1} coverage={bool(coverage_issues)} topic_cov={topic_cov['coverage']:.2f}")
        else:
            # Inject naming/critical/topic warnings into structural feedback
            fb_parts = issues[:]
            if naming_warnings:
                fb_parts.append(f"命名质量建议：{'；'.join(naming_warnings[:3])}")
            if critical_issues:
                fb_parts.append(f"批判章节建议：{'；'.join(critical_issues[:2])}")
            if topic_cov["coverage"] < 0.40:
                fb_parts.append(f"用户需求关键词未覆盖：{'、'.join(topic_cov['uncovered'][:6])}")
            last_issues = fb_parts
            logger.warning(f"outline_validation_failed attempt={attempt + 1} issues={issues} naming={len(naming_warnings)} critical={len(critical_issues)} topic_cov={topic_cov['coverage']:.2f}")

    t1 = time.perf_counter()
    logger.warning(f"outline_done_with_issues seconds={t1 - t0:.2f} attempts=3 issues={last_issues}")
    return _add_outline_numbering(best_outline)
