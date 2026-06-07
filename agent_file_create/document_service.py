import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Atomic fact evaluation (FActScore-style) ──────────────────────────────

def _decompose_to_atomic_facts(content: str) -> list[str]:
    """Break content into atomic facts — each one sentence, one claim."""
    from agent_file_create.llm_client import call_llm as _af_call

    text = (content or "").strip()
    if not text or len(text) < 100:
        return []

    prompt = (
        "请将以下文本分解为原子事实列表。每条原子事实应满足：\n"
        "1. 只包含一个可独立验证的信息点\n"
        "2. 是一句完整的陈述句\n"
        "3. 长度不超过 30 个汉字\n\n"
        f"文本：\n{text[:2000]}\n\n"
        "输出格式：每行一条原子事实，不要编号。"
    )
    try:
        raw = _af_call(prompt, timeout_s=20, temperature=0.0, num_predict=300,
                       system="你是一个文本分解助手。只输出原子事实，每行一条。")
        facts = [f.strip() for f in (raw or "").splitlines()
                 if f.strip() and len(f.strip()) > 5]
        return facts[:20]
    except Exception:
        return []


def _verify_atomic_facts(facts: list[str], source_text: str) -> dict:
    """Verify each atomic fact against source materials. Returns {fact: supported_bool}."""
    from agent_file_create.llm_client import call_llm as _vf_call

    if not facts or not source_text:
        return {}

    # Batch verification: check all facts at once
    facts_text = "\n".join(f"[{i+1}] {f}" for i, f in enumerate(facts))
    prompt = (
        "请逐条判断以下原子事实是否能在参考资料中找到依据。\n\n"
        f"参考资料：\n{source_text[:5000]}\n\n"
        f"原子事实：\n{facts_text}\n\n"
        "对每条原子事实，如果能在参考资料中找到明确依据，输出 'OK'；否则输出 'MISS'。\n"
        "每行一条，格式：序号 OK|MISS"
    )
    try:
        raw = _vf_call(prompt, timeout_s=20, temperature=0.0, num_predict=200,
                       system="你是事实核查助手。只输出每条事实的判断结果。")
        result = {}
        for line in (raw or "").splitlines():
            line = line.strip()
            m = re.match(r"(\d+)\s*(OK|MISS)", line, re.IGNORECASE)
            if m:
                idx = int(m.group(1)) - 1
                result[facts[idx] if idx < len(facts) else line] = m.group(2).upper() == "OK"
        # Fill in missing
        for f in facts:
            if f not in result:
                result[f] = False
        return result
    except Exception:
        return {f: False for f in facts}


# ── Aspect coverage evaluation (ICAT-style) ────────────────────────────────

def _extract_aspects(analysis_results: list) -> list[str]:
    """Extract key aspects/topics from source materials that the report should cover."""
    from agent_file_create.llm_client import call_llm as _ea_call

    if not analysis_results:
        return []

    source_summary = ""
    for i, ar in enumerate(analysis_results[:5]):
        title = str(ar.get("title") or ar.get("filename") or "").strip()
        summary = str(ar.get("summary") or "").strip()
        if title or summary:
            source_summary += f"[{i+1}] {title}: {summary[:200]}\n"

    if not source_summary:
        return []

    prompt = (
        "请从以下来源材料中提取 5-10 个关键主题/方面（aspects），"
        "这些是报告应该覆盖的重要内容维度。每个 aspect 用一句话描述。\n\n"
        f"来源材料：\n{source_summary[:1500]}\n\n"
        "输出：每行一个 aspect，不要编号。"
    )
    try:
        raw = _ea_call(prompt, timeout_s=20, temperature=0.0, num_predict=200,
                       system="你只输出关键主题，每行一个。")
        aspects = [a.strip() for a in (raw or "").splitlines()
                   if a.strip() and len(a.strip()) > 5]
        return aspects[:10]
    except Exception:
        return []


# ── Cross-section consistency check ────────────────────────────────────────

def _check_consistency(content: str, task_id: str = "") -> dict:
    """Check adjacent sections for contradictory or inconsistent claims.

    Returns dict with: consistency_score, flagged_pairs
    """
    from agent_file_create.llm_client import call_llm as _cc_call

    # Extract section headings and their first 2-3 sentences
    sections = []
    current_heading = None
    current_body = []
    for line in (content or "").splitlines():
        line = line.strip()
        m = re.match(r"^##\s+(.+)", line)
        if m:
            if current_heading and current_body:
                sections.append((current_heading, " ".join(current_body)[:300]))
            current_heading = m.group(1).strip()
            current_body = []
        elif current_heading and line and not line.startswith("> "):
            current_body.append(line)
    if current_heading and current_body:
        sections.append((current_heading, " ".join(current_body)[:300]))

    if len(sections) < 2:
        return {"consistency_score": 1.0, "flagged_pairs": []}

    # Check adjacent pairs
    flagged = []
    total_checks = 0
    for i in range(len(sections) - 1):
        h1, b1 = sections[i]
        h2, b2 = sections[i + 1]
        if not b1 or not b2:
            continue
        total_checks += 1
        prompt = (
            "请检查以下两个相邻章节的关键主张是否存在矛盾或显著不一致。\n"
            "只关注事实层面的冲突（如数据矛盾、结论相反），不关注措辞风格差异。\n\n"
            f"章节A「{h1}」：{b1[:200]}\n\n"
            f"章节B「{h2}」：{b2[:200]}\n\n"
            "如果两节一致，回复OK。如果不一致，回复 CONFLICT: <简述矛盾点>"
        )
        try:
            raw = _cc_call(prompt, timeout_s=15, temperature=0.0, num_predict=100,
                           system="你是文档一致性审查助手。只回复OK或CONFLICT。")
            result = (raw or "").strip()
            if result.upper().startswith("CONFLICT"):
                reason = result[len("CONFLICT"):].strip().lstrip(":：").strip()
                flagged.append({"section_a": h1, "section_b": h2, "reason": reason})
                logger.info("consistency_conflict task=%s a=%s b=%s reason=%s",
                            task_id, h1[:30], h2[:30], reason)
        except Exception:
            pass

    score = 1.0 - (len(flagged) / max(1, total_checks))
    if flagged:
        logger.info("consistency_check task=%s score=%.2f flagged=%d/%d",
                    task_id, score, len(flagged), total_checks)
    return {"consistency_score": round(score, 4), "flagged_pairs": flagged}


# ── Coverage gap filling ──────────────────────────────────────────────────

def _fill_coverage_gaps(content: str, uncovered_aspects: list[str],
                         analysis_results: list, task_id: str = "") -> str:
    """For uncovered aspects, retrieve from KB and generate supplemental content.

    Returns the augmented content string, or original content if no gaps filled.
    """
    if not uncovered_aspects:
        return content

    # Build source text for search
    source_text = ""
    for ar in (analysis_results or [])[:8]:
        title = str(ar.get("title") or ar.get("filename") or "").strip()
        summary = str(ar.get("summary") or "").strip()
        if title or summary:
            source_text += f"[{title}]\n{summary[:400]}\n\n"

    if not source_text:
        return content

    # For each uncovered aspect, find relevant source material
    supplemental_parts = []
    for aspect in uncovered_aspects[:3]:  # Max 3 gaps to fill
        try:
            from agent_file_create.llm_client import call_llm as _gf_call
            # Search existing source text for aspect-related content
            prompt = (
                "从以下来源材料中，找出与指定主题最相关的内容。"
                "如果材料中有相关内容，摘录关键信息；如果没有，回复 NONE。\n\n"
                f"来源材料：\n{source_text[:2000]}\n\n"
                f"需要查找的主题：{aspect}\n\n"
                "相关材料摘录（如无相关内容回复NONE）："
            )
            raw = _gf_call(prompt, timeout_s=15, temperature=0.0, num_predict=200,
                           system="你只输出相关材料或NONE。")
            material = (raw or "").strip()
            if material and material.upper() != "NONE" and len(material) > 10:
                # Generate supplemental paragraph
                gen_prompt = (
                    f"请基于以下材料，撰写一段关于「{aspect}」的补充内容（100-200字），"
                    "以填补报告中缺失的维度。只输出正文段落。\n\n"
                    f"参考材料：{material[:800]}"
                )
                para = _gf_call(gen_prompt, timeout_s=20, temperature=0.3, num_predict=300,
                                system="你是报告撰写助手。只输出补充段落。")
                if para and len(str(para).strip()) > 20:
                    supplemental_parts.append((aspect, str(para).strip()))
                    logger.info("coverage_gap_filled aspect=%s", aspect)
        except Exception as e:
            logger.warning("coverage_gap_fill_failed aspect=%s err=%s", aspect, str(e)[:100])

    if not supplemental_parts:
        return content

    # Append supplemental section to content
    gap_section = "\n\n## 补充：材料中提及但正文未覆盖的主题\n\n"
    gap_section += "> ⚠️ 以下内容由系统根据主题覆盖度评估自动补充，建议人工核实。\n\n"
    for aspect, para in supplemental_parts:
        gap_section += f"### {aspect}\n{para}\n\n"

    logger.info("coverage_gap_filled_summary task=%s filled=%d/%d",
                task_id, len(supplemental_parts), len(uncovered_aspects))
    return content.rstrip() + "\n" + gap_section


# ── Contrastive claim verification ────────────────────────────────────────

_CONTRASTIVE_PATTERNS = [
    # Entity groups: must match 2+ Chinese/English chars (not whitespace/punctuation)
    r"([一-鿿\w]{2,})优于([一-鿿\w]{2,})",
    r"([一-鿿\w]{2,})快于([一-鿿\w]{2,})",
    r"([一-鿿\w]{2,})超过([一-鿿\w]{2,})",
    r"([一-鿿\w]{2,})比([一-鿿\w]{2,})更",
    r"([一-鿿\w]{2,})高于([一-鿿\w]{2,})",
    r"([一-鿿\w]{2,})胜过([一-鿿\w]{2,})",
    r"([一-鿿\w]{2,})领先于?([一-鿿\w]{2,})",
    r"([A-Za-z]+)\s+outperforms?\s+([A-Za-z]+)",
]


def _verify_contrastive_claims(content: str, source_text: str,
                                 task_id: str = "") -> dict:
    """Detect contrastive claims and verify both sides independently.

    Returns: {flagged_count, total_count, details: [...]}
    """
    from agent_file_create.llm_client import call_llm as _vc_call

    # Find contrastive claims
    claims = []
    for pat in _CONTRASTIVE_PATTERNS:
        for m in re.finditer(pat, content):
            start = max(0, m.start() - 40)
            end = min(len(content), m.end() + 60)
            ctx = content[start:end].replace("\n", " ").strip()
            claims.append({
                "entity_a": m.group(1).strip(),
                "entity_b": m.group(2).strip(),
                "context": ctx,
            })

    if not claims:
        return {"flagged_count": 0, "total_count": 0, "details": []}

    # Dedup by context
    seen = set()
    unique = []
    for c in claims:
        key = c["context"][:60]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    claims = unique[:8]

    if not source_text:
        return {"flagged_count": 0, "total_count": len(claims), "details": []}

    # Verify each claim
    flagged = []
    for c in claims:
        prompt = (
            "以下报告中出现了一条对比型论断。"
            "请判断对比的双方（A和B）的数据是否都能在材料中找到独立支撑。\n\n"
            f"来源材料：\n{source_text[:1500]}\n\n"
            f"论断上下文：「{c['context'][:200]}」\n\n"
            "如果对比双方的数据都可以在材料中找到，回复OK。"
            "如果缺少任意一方的数据支撑，回复 MISS: <缺少哪一方>"
        )
        try:
            raw = _vc_call(prompt, timeout_s=15, temperature=0.0, num_predict=80,
                           system="你是文档事实核查助手。只回复OK或MISS。")
            result = (raw or "").strip()
            if result.upper().startswith("MISS"):
                reason = result[4:].strip().lstrip(":：").strip()
                flagged.append({**c, "reason": reason})
                logger.info("contrastive_miss task=%s claim=%s", task_id, c["context"][:60])
        except Exception:
            pass

    if flagged:
        logger.info("contrastive_verify task=%s flagged=%d/%d",
                    task_id, len(flagged), len(claims))

    return {"flagged_count": len(flagged), "total_count": len(claims),
            "details": flagged}


def _align_facts_to_aspects(facts: list[str], aspects: list[str]) -> dict:
    """LLM-based alignment of atomic facts to aspects (ICAT-S approach).

    Returns: {aspect: [matched_fact1, matched_fact2, ...], ...}
    """
    from agent_file_create.llm_client import call_llm as _al_call

    if not facts or not aspects:
        return {}

    facts_text = "\n".join(f"[F{i+1}] {f}" for i, f in enumerate(facts))
    aspects_text = "\n".join(f"[A{i+1}] {a}" for i, a in enumerate(aspects))

    prompt = (
        "请将每条原子事实对齐到最相关的主题方面（aspect）。\n"
        "每条原子事实可以对应 0 个、1 个或多个方面。\n\n"
        f"主题方面：\n{aspects_text}\n\n"
        f"原子事实：\n{facts_text}\n\n"
        "输出格式：每行 F序号 -> A序号，如 'F1 -> A3, A5' 表示事实1覆盖了方面3和5。"
        "如果某条事实不匹配任何方面，输出 'F序号 -> NONE'。\n"
        "只输出对齐结果，不要解释。"
    )
    try:
        raw = _al_call(prompt, timeout_s=20, temperature=0.0, num_predict=200,
                       system="你只输出事实到方面的对齐结果。")
        alignment = {}
        for line in (raw or "").splitlines():
            line = line.strip()
            m = re.match(r"F(\d+)\s*->\s*(.+)", line, re.IGNORECASE)
            if m:
                fidx = int(m.group(1)) - 1
                fact = facts[fidx] if fidx < len(facts) else None
                if fact:
                    targets = m.group(2).strip()
                    if targets.upper() == "NONE":
                        continue
                    for aidx_str in re.findall(r"A?(\d+)", targets):
                        aidx = int(aidx_str) - 1
                        aspect = aspects[aidx] if aidx < len(aspects) else None
                        if aspect:
                            alignment.setdefault(aspect, []).append(fact)
        return alignment
    except Exception:
        return {}


def _compute_factscore_and_coverage(content: str, analysis_results: list,
                                     task_id: str = "") -> dict:
    """Run atomic fact evaluation + aspect coverage in one pass.

    Returns dict with: factscore, coverage, facts_count, verified_count, aspects_count, covered_count
    """
    results = {"factscore": None, "coverage": None,
               "facts_count": 0, "verified_count": 0,
               "aspects_count": 0, "covered_count": 0,
               "uncovered_aspects": []}

    # ── Source text for verification ──
    source_text = ""
    for ar in (analysis_results or [])[:12]:
        title = str(ar.get("title") or ar.get("filename") or "").strip()
        summary = str(ar.get("summary") or "").strip()
        full = str(ar.get("full_text") or ar.get("content") or "").strip()
        text = (full or summary or "")[:1200]
        # Clean PDF noise: skip lines that are pure metadata/formatting
        clean_lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line: continue
            # Skip pure metadata lines
            if re.match(r"^(ISSN|CN|ISBN|DOI|http|www\.|收稿日期|网络首发|Vol\.|No\.|第\d+卷)", line):
                continue
            if len(line) < 5: continue  # too short
            clean_lines.append(line)
        text = " ".join(clean_lines)[:1200]
        if title or text:
            source_text += f"[{title}]\n{text}\n\n"
    if len(source_text) < 500:
        # Too little source material — can't verify meaningfully
        logger.warning("factscore_source_too_short task=%s chars=%d", task_id, len(source_text))
        return results

    if not content:
        return results

    # ── FActScore ──
    facts = _decompose_to_atomic_facts(content)
    if facts:
        verification = _verify_atomic_facts(facts, source_text)
        results["facts_count"] = len(facts)
        results["verified_count"] = sum(1 for v in verification.values() if v)
        results["factscore"] = round(results["verified_count"] / max(1, len(facts)), 4)

    # ── Aspect Coverage (ICAT-S: LLM-based alignment) ──
    aspects = _extract_aspects(analysis_results)
    if aspects and facts:
        alignment = _align_facts_to_aspects(facts, aspects)
        results["aspects_count"] = len(aspects)
        results["covered_count"] = len(alignment)
        results["coverage"] = round(len(alignment) / max(1, len(aspects)), 4)
        results["uncovered_aspects"] = [a for a in aspects if a not in alignment]

    if results["factscore"] is not None or results["coverage"] is not None:
        uncovered_str = ""
        if results.get("uncovered_aspects"):
            uncovered_str = " uncovered=" + ", ".join(results["uncovered_aspects"][:3])
        logger.info("factscore_and_coverage task=%s factscore=%s coverage=%s facts=%d/%d aspects=%d/%d%s",
                    task_id,
                    results.get("factscore"), results.get("coverage"),
                    results.get("verified_count", 0), results.get("facts_count", 0),
                    results.get("covered_count", 0), results.get("aspects_count", 0),
                    uncovered_str)

    return results


def generate_document(
    *,
    user_prompt: str,
    analysis_results: List[Dict[str, Any]],
    document_type: str = "report",
    task_id: Optional[str] = None,
    template_dir_override: Optional[str] = None,
    outline: Optional[str] = None,
    content: Optional[str] = None,
) -> Dict[str, Any]:
    if not task_id:
        task_id = uuid.uuid4().hex[:8]

    base_dir = Path(__file__).resolve().parent.parent
    result_dir = base_dir / "result"
    output_dir = result_dir / str(task_id)
    default_template_dir = result_dir / "template"
    template_dir = Path(template_dir_override) if template_dir_override else default_template_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"result_dir_create_failed err={str(e)[:160]}")

    db_conn = None
    outline_id = ""
    db_ready = threading.Event()

    def _init_db():
        nonlocal db_conn
        try:
            from agent_file_create.db_service import create_task, get_db_connection, init_db
            db_conn = get_db_connection()
            init_db(db_conn)
            create_task(
                db_conn,
                task_id=str(task_id),
                title="",
                document_type=str(document_type or ""),
                user_prompt=str(user_prompt or ""),
                status="processing",
                output_dir=str(output_dir),
                meta={"template_dir": str(template_dir)},
            )
        except Exception as e:
            logger.warning(f"db_init_failed err={str(e)[:200]}")
        finally:
            db_ready.set()

    threading.Thread(target=_init_db, daemon=True).start()

    multimodal_results = {f"source_{i}": r for i, r in enumerate(analysis_results or [])}

    from agent_file_create.document.content_generator import TaskCanceledException

    # ── Outline ─────────────────────────────────────────────────────────
    if outline:
        logger.info("outline_reuse task=%s chars=%d", task_id, len(outline))
    else:
        logger.info("生成大纲...")
        from agent_file_create.document.outline_generator import generate_outline

        t0 = time.perf_counter()
        outline = generate_outline(multimodal_results, user_prompt)
        t1 = time.perf_counter()
        logger.info(f"outline_done seconds={t1 - t0:.2f} outline_chars={len(outline or '')}")

    try:
        (output_dir / "outline.md").write_text(str(outline or ""), encoding="utf-8")
    except Exception as e:
        logger.warning(f"write_outline_failed err={str(e)[:160]}")

    # Notify frontend that outline is complete
    try:
        from agent_file_create.task.manager import TaskManager

        task_mgr = TaskManager()
        task_mgr.write_status(
            str(task_id),
            "processing",
            stage="document",
            message="大纲生成完成，正在生成正文…",
        )
    except Exception:
        pass

    try:
        db_ready.wait(timeout=1.0)
        m = re.search(r"^\s*#\s+(.+?)\s*$", str(outline or ""), flags=re.M)
        doc_title = (m.group(1).strip() if m else "").strip()
        if doc_title and db_conn is not None:
            from agent_file_create.db_service import update_task_title

            update_task_title(db_conn, str(task_id), doc_title)
    except Exception as e:
        logger.warning(f"db_update_title_failed err={str(e)[:160]}")

    try:
        if db_conn is not None:
            from agent_file_create.document.content_generator import parse_outline_sections
            from agent_file_create.db_service import save_outline

            flat = parse_outline_sections(str(outline or ""))
            outline_id = save_outline(db_conn, task_id=str(task_id), outline_markdown=str(outline or ""), outline_sections=flat)
    except Exception as e:
        logger.warning(f"db_save_outline_failed err={str(e)[:200]}")

    # ── Content ─────────────────────────────────────────────────────────
    if content:
        logger.info("content_reuse task=%s chars=%d", task_id, len(content))
    else:
        logger.info("生成正文...")
        from agent_file_create.document.content_generator import generate_full_content

        t0 = time.perf_counter()
        try:
            content = generate_full_content(str(outline or ""), multimodal_results, str(user_prompt or ""), task_id=str(task_id))
        except TaskCanceledException:
            logger.info("content_canceled task_id=%s", str(task_id))
            try:
                TaskManager().write_status(str(task_id), "canceled", stage="document", message="已取消")
            except Exception:
                pass
            return {
                "task_id": str(task_id),
                "document_outline": outline,
                "document_content": "",
                "document_type": str(document_type or ""),
                "output_dir": str(output_dir),
                "template_dir": str(template_dir),
                "rendered_outputs": [],
                "status": "canceled",
            }
        t1 = time.perf_counter()
        logger.info(f"content_done seconds={t1 - t0:.2f} content_chars={len(content or '')}")

    try:
        _clean_content = re.sub(r"<!--.*?-->", "", str(content or ""), flags=re.DOTALL)
        (output_dir / "content.md").write_text(_clean_content, encoding="utf-8")
    except Exception as e:
        logger.warning(f"write_content_failed err={str(e)[:160]}")

    # ── Faithfulness check + hallucination-triggered re-retrieval ──
    try:
        from agent_file_create.llm_client import call_llm as _f_call
        source_text = ""
        for _i, _ar in enumerate((analysis_results or [])[:8]):
            _title = str(_ar.get("title") or _ar.get("filename") or "").strip()
            _summary = str(_ar.get("summary") or "").strip()
            if _title or _summary:
                source_text += f"[{_i+1}] {_title}: {_summary[:300]}\n"
        if not source_text:
            logger.warning("faithfulness_check skip task=%s reason=no_source_text", task_id)
            if content:
                # Source materials are empty — mark entire report as unverifiable
                _header = "> ⚠️ **事实核查无法执行**：未从上传材料中提取到足够内容，无法验证以下报告的事实准确性。所有数据和结论请以原始材料为准。\n\n"
                annotated = _header + str(content or "")
                try:
                    (output_dir / "content.md").write_text(annotated, encoding="utf-8")
                except Exception:
                    pass
                content = annotated
        elif not content:
            logger.warning("faithfulness_check skip task=%s reason=no_content", task_id)
        elif source_text and content:
            _check_prompt = (
                "你是文档事实核查助手。请检查以下报告正文的每个 ## 章节，判断其内容是否能在来源材料中找到依据。\n\n"
                f"来源材料摘要：\n{source_text[:3000]}\n\n"
                f"报告正文：\n{str(content)[:4000]}\n\n"
                "对于每个 ## 章节，判断其可信度：如果章节中的所有事实都能在来源材料中找到明确依据，标记OK；"
                "如果章节中有部分推断或数据无法在来源材料中验证，标记 WARN: <原因>。\n"
                "只输出有问题的章节（OK的不用输出），如果没有问题，回复ALL_OK。\n"
                "格式：## 章节名\nWARN: 原因（简短一句）"
            )
            _check_result = _f_call(
                _check_prompt, timeout_s=20, temperature=0.0, num_predict=300,
                system="你是文档事实核查助手，只输出有问题的章节。")
            _check_text = (_check_result or "").strip()
            if not _check_text:
                logger.warning("faithfulness_check empty_response task=%s", task_id)
            elif _check_text.upper() in {"ALL_OK", "OK", "无", "NONE"}:
                logger.info("faithfulness_check all_ok task=%s", task_id)
            else:
                # Parse warnings: collect (section_name, reason) pairs
                import re as _re
                _warnings: list[tuple[str, str]] = []
                _cur_section = ""
                for _line in _check_text.splitlines():
                    _line = _line.strip()
                    if _line.startswith("## "):
                        _cur_section = _line[3:].strip()
                    elif (_line.startswith("WARN:") or _line.startswith("WARN：")) and _cur_section:
                        _reason = _line[5:].strip().lstrip(":：").strip()
                        _warnings.append((_cur_section, _reason))
                        _cur_section = ""

                if _warnings:
                    # ── Phase 1: Attempt re-retrieval to fix each warning ──
                    _fixed_count = 0
                    _unfixed: list[tuple[str, str]] = []
                    try:
                        from agent_file_create.rag.kb import KnowledgeBase
                        _kb = KnowledgeBase()
                        _kb_name = str(task_id)
                    except Exception:
                        _kb = None
                        _kb_name = ""

                    annotated = str(content or "")
                    for _sec_name, _reason in _warnings:
                        _did_fix = False
                        if _kb:
                            try:
                                # Extract key claims from the section for re-retrieval
                                _sec_body = ""
                                _in_sec = False
                                for _cl in annotated.splitlines():
                                    if _cl.strip().startswith(f"## {_sec_name}"):
                                        _in_sec = True
                                        continue
                                    if _in_sec:
                                        if _cl.strip().startswith("## "):
                                            break
                                        if not _cl.strip().startswith("> ⚠️"):
                                            _sec_body += _cl + "\n"
                                _sec_body = _sec_body.strip()

                                if _sec_body and len(_sec_body) > 50:
                                    # Extract claims to verify
                                    _claim_prompt = (
                                        "从以下段落中提取1-3个需要验证的关键事实主张（如具体数据、结论性判断）。"
                                        "每行一个主张，只输出主张本身。\n\n"
                                        f"段落：{_sec_body[:800]}"
                                    )
                                    _claims_raw = _f_call(
                                        _claim_prompt, timeout_s=10, temperature=0.0, num_predict=150,
                                        system="你只输出关键主张，每行一个。")
                                    _claims = [c.strip() for c in (_claims_raw or "").splitlines()
                                               if c.strip() and len(c.strip()) > 5][:3]

                                    if _claims:
                                        # Re-retrieve for each claim
                                        _new_hits = []
                                        for _claim in _claims:
                                            try:
                                                _h = _kb.search(kb=_kb_name, query=_claim, top_k=3)
                                                _new_hits.extend(_h)
                                            except Exception:
                                                pass

                                        if _new_hits:
                                            _new_ctx = ""
                                            for _j, _h in enumerate(_new_hits[:5]):
                                                _chunk = str(_h.content or "")[:400]
                                                if _chunk:
                                                    _new_ctx += f"[新检索{j+1}] {_chunk}\n"

                                            if _new_ctx:
                                                # Attempt to fix the section
                                                _fix_prompt = (
                                                    "你是一个文档编辑助手。以下段落的事实核查未通过，"
                                                    "因为部分内容在原始材料中找不到依据。请用新增检索到的材料重写该段落，"
                                                    "只保留有材料支撑的内容。如果新材料能支撑原主张，保留它；"
                                                    "如果不能，用新材料替换或删除无支撑的内容。\n\n"
                                                    f"原始段落（有问题）：\n{_sec_body[:600]}\n\n"
                                                    f"新增检索材料：\n{_new_ctx[:1200]}\n\n"
                                                    "请输出修正后的段落（只输出正文，不使用HTML注释、不使用核查标记）："
                                                )
                                                _fixed_body = _f_call(
                                                    _fix_prompt, timeout_s=15, temperature=0.2, num_predict=500,
                                                    system="你是文档编辑助手，只输出修正后的正文段落，禁止添加HTML注释或核查标记。")
                                                # Strip HTML comments the LLM may have inserted
                                                _fixed_body = _re.sub(r"<!--.*?-->", "", str(_fixed_body or ""), flags=_re.DOTALL).strip()
                                                if _fixed_body and len(_fixed_body) > 20:
                                                    # Replace the section body in annotated content
                                                    _esc_sec = _re.escape(_sec_body[:120])
                                                    _pat = _re.compile(_re.escape(_sec_body[:120]) + r".*", _re.DOTALL)
                                                    # Simpler: find and replace by section boundaries
                                                    _lines = annotated.splitlines()
                                                    _new_lines = []
                                                    _skip = False
                                                    _found = False
                                                    for _cl in _lines:
                                                        if _cl.strip().startswith(f"## {_sec_name}"):
                                                            _new_lines.append(_cl)
                                                            _new_lines.append("")
                                                            _new_lines.append(str(_fixed_body).strip())
                                                            _skip = True
                                                            _found = True
                                                            continue
                                                        if _skip:
                                                            if _cl.strip().startswith("## "):
                                                                _new_lines.append("")
                                                                _new_lines.append(_cl)
                                                                _skip = False
                                                            continue
                                                        _new_lines.append(_cl)
                                                    if _found:
                                                        annotated = "\n".join(_new_lines)
                                                        _did_fix = True
                                                        _fixed_count += 1
                                                        logger.info(
                                                            "faithfulness_fixed section=%s claims=%d",
                                                            _sec_name, len(_claims))
                            except Exception as _e:
                                logger.warning("faithfulness_refix_failed section=%s err=%s",
                                               _sec_name, str(_e)[:100])

                        if not _did_fix:
                            _unfixed.append((_sec_name, _reason))
                            # Add ⚠️ marker for unfixed sections
                            _marker = f"> ⚠️ **事实核查提醒**：{_reason}\n> 已尝试增量检索修正，仍未找到充分依据。请人工核实。\n\n"
                            _esc_sec = _re.escape(f"## {_sec_name}")
                            _repl = f"## {_sec_name}\n{_marker}"
                            _new = _re.sub(_esc_sec, _repl, annotated, count=1)
                            if _new != annotated:
                                annotated = _new

                    if annotated != content:
                        # Final safety: strip any lingering HTML comments from the entire content
                        annotated = _re.sub(r"<!--.*?-->", "", annotated, flags=_re.DOTALL)
                        try:
                            (output_dir / "content.md").write_text(annotated, encoding="utf-8")
                        except Exception:
                            pass
                        content = annotated
                        if _fixed_count:
                            logger.info("faithfulness_summary fixed=%d unfixed=%d",
                                        _fixed_count, len(_unfixed))
    except Exception as _e:
        logger.warning("faithfulness_check_failed err=%s", str(_e)[:200])

    # ── Citation verification: check that cited claims match cited sources ──
    try:
        _raw_content = str(content or "")
        _citations = re.findall(r"[（(]据(.+?)[）)]", _raw_content)
        if _citations and analysis_results:
            _source_map = {}
            for _ar in (analysis_results or []):
                _fn = str(_ar.get("filename") or _ar.get("title") or "").strip()
                _summary = str(_ar.get("summary") or "").strip()
                if _fn:
                    _source_map[_fn] = _summary
            _bad_cites = []
            for _cite in _citations:
                _cite = _cite.strip()
                # Find surrounding sentence for context
                _idx = _raw_content.find(f"（据{_cite}）")
                if _idx < 0:
                    _idx = _raw_content.find(f"(据{_cite})")
                if _idx >= 0:
                    _start = max(0, _idx - 80)
                    _end = min(len(_raw_content), _idx + len(_cite) + 80)
                    _context = _raw_content[_start:_end].replace("\n", " ")
                else:
                    _context = _cite
                # Find matching source: check if citation label appears in any source filename
                _best_match = None
                for _fn in _source_map:
                    if _cite in _fn or _fn in _cite:
                        _best_match = _fn
                        break
                    # Try word-level matching
                    _cite_words = _cite.replace("、", " ").replace("，", " ").split()
                    if any(_w in _fn for _w in _cite_words if len(_w) >= 2):
                        _best_match = _fn
                        break
                if not _best_match:
                    _bad_cites.append((_cite, _context))
            if _bad_cites:
                # Group by citation label for dedup
                _by_label = {}
                for _bc in _bad_cites:
                    _label = _bc[0]
                    _by_label.setdefault(_label, []).append(_bc)

                _cite_lines = ["\n> ⚠️ **引用溯源提醒**："]
                for _label, _items in sorted(_by_label.items()):
                    _n = len(_items)
                    if _n >= 3:
                        _cite_lines.append(f"> - 「据{_label}」出现了 {_n} 次，但未在来源材料中找到匹配。建议改为具体文件名（如「据XX论文」），不要使用笼统占位符。")
                    else:
                        _ctx = _items[0][1][:60] if len(_items[0]) > 1 else ""
                        _cite_lines.append(f"> - 标注为「据{_label}」，未找到匹配来源 → {_ctx}...")
                _cite_lines.append("> 请核实引用标注与来源材料的一致性。\n")
                _cite_marker = "\n".join(_cite_lines)
                # Prepend citation warnings to the content
                _clean_content = re.sub(r"<!--.*?-->", "", str(content or ""), flags=re.DOTALL)
                content = _cite_marker + "\n" + _clean_content
                try:
                    (output_dir / "content.md").write_text(content, encoding="utf-8")
                except Exception:
                    pass
                logger.info("citation_verify bad=%d total=%d", len(_bad_cites), len(_citations))
            else:
                logger.info("citation_verify all_ok count=%d", len(_citations))
    except Exception as _e:
        logger.warning("citation_verify_failed err=%s", str(_e)[:200])

    # ── Contrastive claim verification ──
    try:
        _contrastive = _verify_contrastive_claims(str(content or ""),
                                                   source_text, task_id=str(task_id))
        if _contrastive.get("flagged_count"):
            _ct_lines = ["\n> ⚠️ **对比论断待核实**：以下对比型表述中的部分数据在材料中缺乏独立支撑："]
            for _c in _contrastive.get("details", [])[:3]:
                _ct_lines.append(f"> - 「{_c['context'][:80]}...」— {_c.get('reason', '需核实')}")
            _ct_lines.append("> 请核实对比双方的独立数据来源。\n")
            _ct_marker = "\n".join(_ct_lines)
            _clean = re.sub(r"<!--.*?-->", "", str(content or ""), flags=re.DOTALL)
            content = _ct_marker + "\n" + _clean
            try:
                (output_dir / "content.md").write_text(content, encoding="utf-8")
            except Exception:
                pass
            logger.info("contrastive_verify_summary flagged=%d/%d",
                        _contrastive.get("flagged_count", 0),
                        _contrastive.get("total_count", 0))
    except Exception as _e:
        logger.warning("contrastive_verify_failed err=%s", str(_e)[:200])

    # ── FActScore + Aspect Coverage evaluation ──
    _coverage_results = {}
    try:
        _coverage_results = _compute_factscore_and_coverage(
            str(content or ""), analysis_results or [], task_id=str(task_id))
    except Exception as _e:
        logger.warning("factscore_coverage_failed err=%s", str(_e)[:200])

    # ── Coverage gap filling (if coverage < 0.8) ──
    if (_coverage_results.get("coverage") or 1.0) < 0.8 and _coverage_results.get("uncovered_aspects"):
        try:
            _filled = _fill_coverage_gaps(str(content or ""),
                                           _coverage_results["uncovered_aspects"],
                                           analysis_results or [], task_id=str(task_id))
            if _filled != content:
                content = _filled
                try:
                    (output_dir / "content.md").write_text(content, encoding="utf-8")
                except Exception:
                    pass
        except Exception as _e:
            logger.warning("coverage_gap_fill_failed err=%s", str(_e)[:200])

    # ── Cross-section consistency check ──
    _consistency_results = {}
    try:
        _consistency_results = _check_consistency(str(content or ""), task_id=str(task_id))
    except Exception as _e:
        logger.warning("consistency_check_failed err=%s", str(_e)[:200])

    rendered_outputs: list[str] = []
    if template_dir.exists() and template_dir.is_dir():
        try:
            from agent_file_create.document.template_renderer import (
                _scan_md_placeholders,
                build_content_dict,
                render_markdown_template,
                render_pdf_template,
                render_word_template,
                should_skip_render,
                _changed_sections,
                _section_hashes,
                _save_render_cache,
                _render_markdown_to_docx,
            )

            # Collect expected section keys from .md templates for fuzzy matching
            _SYSTEM_TPL_KEYS = {"title", "task_id", "document_outline", "document_content"}
            _template_section_keys: set[str] = set()
            for _tp in sorted(template_dir.glob("*.md")):
                _template_section_keys.update(_scan_md_placeholders(str(_tp)))
            _template_section_keys -= _SYSTEM_TPL_KEYS

            content_dict = build_content_dict(
                str(content or ""),
                template_keys=sorted(_template_section_keys) if _template_section_keys else None,
            )
            # Extract and remove match info before rendering
            match_info = content_dict.pop("_template_match_info", None) or {}
            content_dict["task_id"] = str(task_id)
            content_dict["document_outline"] = str(outline or "")
            content_dict["document_content"] = str(content or "")

            # ── Lazy rendering: skip templates whose sections haven't changed ──
            changed = _changed_sections(content_dict, str(output_dir))
            if changed:
                logger.info(
                    "render_lazy_changed task=%s changed_sections=%s",
                    task_id, ", ".join(sorted(changed)[:10]),
                )
            else:
                logger.info("render_lazy_skip_all task=%s (no sections changed)", task_id)

            templates = sorted([p for p in template_dir.iterdir() if p.is_file()])
            for tp in templates:
                suf = tp.suffix.lower()
                if suf not in {".md", ".docx", ".pdf"}:
                    continue
                out_path = output_dir / f"{tp.stem}_rendered{suf}"

                # Check if we can skip this template
                if should_skip_render(str(tp), changed, is_pdf=(suf == ".pdf")):
                    if out_path.exists():
                        rendered_outputs.append(str(out_path))
                        logger.info("render_lazy_skip template=%s (no referenced sections changed)", tp.name)
                        continue
                    # If output doesn't exist yet, we must render regardless
                    logger.info("render_lazy_force template=%s (output missing)", tp.name)

                try:
                    if suf == ".md":
                        render_markdown_template(str(tp), content_dict, str(out_path))
                    elif suf == ".docx":
                        render_word_template(str(tp), content_dict, str(out_path))
                    else:
                        render_pdf_template(str(tp), content_dict, str(out_path))
                    rendered_outputs.append(str(out_path))
                    logger.info(f"template_rendered template={tp.name} output={out_path.name}")
                except Exception as e:
                    logger.warning(f"template_render_failed template={tp.name} err={str(e)[:200]}")

            # Persist section hashes for next incremental render
            _save_render_cache(str(output_dir), _section_hashes(content_dict))

            # ── Template match summary → chat ──
            if match_info:
                exact_n = len(match_info.get("exact") or [])
                fuzzy_n = len(match_info.get("fuzzy") or [])
                unmatched = match_info.get("unmatched") or []
                total = exact_n + fuzzy_n + len(unmatched)
                if total > 0:
                    parts = [f"模板渲染完成：{total} 个占位符中 {exact_n + fuzzy_n} 个成功匹配"]
                    if exact_n:
                        parts.append(f"{exact_n} 个精确匹配")
                    if fuzzy_n:
                        parts.append(f"{fuzzy_n} 个模糊匹配")
                    if unmatched:
                        parts.append(f"「{'」「'.join(unmatched)}」未找到对应章节，内容为空")
                    try:
                        TaskManager().append_chat_history(
                            task_id,
                            [{"role": "assistant", "content": "；".join(parts) + "。"}],
                        )
                    except Exception:
                        pass

            # ── Fallback: no DOCX template found → generate from markdown ──
            has_docx = any(
                str(p.suffix).lower() == ".docx"
                for p in template_dir.iterdir() if p.is_file()
            )
            if not has_docx and content:
                fallback_path = output_dir / "output.docx"
                try:
                    _render_markdown_to_docx(str(content), str(outline or ""), str(fallback_path))
                    rendered_outputs.append(str(fallback_path))
                    logger.info("render_fallback_docx output=%s", fallback_path.name)
                except Exception as e:
                    logger.warning("render_fallback_docx_failed err=%s", str(e)[:200])
        except Exception as e:
            logger.warning(f"template_render_setup_failed err={str(e)[:200]}")

    def _save_content_bg():
        db_ready.wait(timeout=1.0)
        try:
            if db_conn is not None:
                from agent_file_create.db_service import save_content, save_rendered_outputs, update_task_status

                save_content(
                    db_conn,
                    task_id=str(task_id),
                    markdown_content=str(content or ""),
                    meta={"outline_id": outline_id, "output_dir": str(output_dir), "template_dir": str(template_dir)},
                )
                save_rendered_outputs(db_conn, task_id=str(task_id), outputs=rendered_outputs)
                update_task_status(db_conn, str(task_id), "finished")
        except Exception as e:
            logger.warning(f"db_save_content_failed err={str(e)[:200]}")

    threading.Thread(target=_save_content_bg, daemon=True).start()

    return {
        "task_id": str(task_id),
        "document_outline": outline,
        "document_content": content,
        "document_type": str(document_type or ""),
        "output_dir": str(output_dir),
        "template_dir": str(template_dir),
        "rendered_outputs": rendered_outputs,
        "status": "finished",
        "factscore": _coverage_results.get("factscore"),
        "coverage": _coverage_results.get("coverage"),
        "facts_verified": _coverage_results.get("verified_count", 0),
        "facts_total": _coverage_results.get("facts_count", 0),
        "aspects_covered": _coverage_results.get("covered_count", 0),
        "aspects_total": _coverage_results.get("aspects_count", 0),
        "uncovered_aspects": _coverage_results.get("uncovered_aspects", []),
    }

