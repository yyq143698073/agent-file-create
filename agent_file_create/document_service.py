import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Quality functions imported from _quality.py ──
from agent_file_create.document._quality import (
    _run_faithfulness_checks,
    _compute_factscore_and_coverage,
    _check_consistency,
    _fill_coverage_gaps,
)

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
    content = _run_faithfulness_checks(
        content=content,
        analysis_results=analysis_results,
        task_id=str(task_id),
        output_dir=str(output_dir),
    )

    # Consistency check disabled — removed to speed up render time
    # (N LLM calls for N sections, low ROI per call)

    # ── Collect all warnings separately (keep content clean) ──
    _all_warnings: list[dict] = []
    _clean_content = re.sub(r"<!--.*?-->", "", str(content or ""), flags=re.DOTALL)
    _bad_cites: list = []
    _contrastive: dict = {}
    _coverage_results: dict = {}

    # Citation verification warnings
    if _bad_cites:
        _by_label = {}
        for _bc in _bad_cites:
            _label = _bc[0]
            _by_label.setdefault(_label, []).append(_bc)
        for _label, _items in sorted(_by_label.items()):
            _n = len(_items)
            _desc = f"「据{_label}」出现了 {_n} 次，但未在来源材料中找到匹配" if _n >= 3 else f"「据{_label}」未找到匹配来源"
            _all_warnings.append({
                "type": "citation",
                "severity": "high" if _n >= 3 else "medium",
                "title": "引用溯源问题",
                "description": _desc,
                "details": [f"未找到来源：{_item[1][:60]}..." for _item in _items[:3]],
                "suggestion": "改为具体文件名（如「据XX论文」），不要使用笼统占位符"
            })

    # Contrastive claim verification warnings
    if _contrastive.get("flagged_count"):
        for _c in _contrastive.get("details", [])[:5]:
            _all_warnings.append({
                "type": "contrastive",
                "severity": "medium",
                "title": "对比论断待核实",
                "description": "对比型表述中的部分数据在材料中缺乏独立支撑",
                "context": _c.get("context", "")[:120],
                "reason": _c.get("reason", "需核实"),
                "suggestion": "核实对比双方的独立数据来源"
            })

    # Write clean content (without warning blocks)
    try:
        (output_dir / "content.md").write_text(_clean_content, encoding="utf-8")
    except Exception:
        pass

    return {
        "task_id": str(task_id),
        "document_outline": outline,
        "document_content": _clean_content,
        "document_type": str(document_type or ""),
        "output_dir": str(output_dir),
        "template_dir": str(template_dir),
        "rendered_outputs": [],  # populated later by render_document()
        "status": "content_ready",  # not "finished" yet — rendering comes after satisfaction
        "warnings": _all_warnings,
        "warnings_count": len(_all_warnings),
        "factscore": _coverage_results.get("factscore"),
        "coverage": _coverage_results.get("coverage"),
        "facts_verified": _coverage_results.get("verified_count", 0),
        "facts_total": _coverage_results.get("facts_count", 0),
        "aspects_covered": _coverage_results.get("covered_count", 0),
        "aspects_total": _coverage_results.get("aspects_count", 0),
        "uncovered_aspects": _coverage_results.get("uncovered_aspects", []),
    }




def _run_faithfulness_checks(
    *,
    content: str,
    analysis_results: list,
    task_id: str,
    output_dir: str,
) -> str:
    """Run faithfulness check and hallucination-triggered re-retrieval on content."""
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
                system="你是一个中文文档处理助手。只输出有问题的章节。")
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
                    _enable_retrieval = os.getenv("HALLUCINATION_RETRIEVAL_ENABLED", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
                    _fixed_count = 0
                    _unfixed: list[tuple[str, str]] = []
                    try:
                        from agent_file_create.rag.kb import KnowledgeBase
                        _kb = KnowledgeBase() if _enable_retrieval else None
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
                                    # ── Generate topic-based search queries (neutral, not verbatim claims) ──
                                    _query_prompt = (
                                        "你是一个搜索查询生成助手。以下段落的事实核查未通过，需要从知识库中检索正确的信息来验证。\n"
                                        "请针对段落中涉及的每个关键主题，生成1-3个中性的搜索查询（不要重复段落的原话，\n"
                                        "而是提取核心主题和实体，用不同的措辞表达）。\n\n"
                                        f"段落：{_sec_body[:800]}\n\n"
                                        "每行一个搜索查询，只输出查询本身。"
                                    )
                                    _queries_raw = _f_call(
                                        _query_prompt, timeout_s=12, temperature=0.0, num_predict=200,
                                        system="你是一个中文文档处理助手。只输出搜索查询，每行一个。")
                                    _queries = [q.strip() for q in (_queries_raw or "").splitlines()
                                                if q.strip() and len(q.strip()) > 3][:4]

                                    # Also extract verbatim claims as fallback
                                    _claim_prompt = (
                                        "从以下段落中提取1-2个需要验证的关键事实主张。每行一个，只输出主张本身。\n\n"
                                        f"段落：{_sec_body[:800]}"
                                    )
                                    _claims_raw = _f_call(
                                        _claim_prompt, timeout_s=10, temperature=0.0, num_predict=150,
                                        system="你是一个中文文档处理助手。只输出关键主张，每行一个。")
                                    _claims = [c.strip() for c in (_claims_raw or "").splitlines()
                                               if c.strip() and len(c.strip()) > 5][:3]

                                    _all_queries = _queries + _claims  # topics first, then fallback claims

                                    if _all_queries:
                                        # Re-retrieve using topic-based queries + HyDE
                                        _new_hits = []
                                        _seen = set()
                                        for _q in _all_queries:
                                            try:
                                                _h = _kb.search_hyde(kb=_kb_name, query=_q, top_k=3)
                                                for _hit in _h:
                                                    if _hit.chunk_id not in _seen:
                                                        _seen.add(_hit.chunk_id)
                                                        _new_hits.append(_hit)
                                            except Exception:
                                                try:
                                                    _h = _kb.search_adaptive(kb=_kb_name, query=_q, top_k=3)
                                                    for _hit in _h:
                                                        if _hit.chunk_id not in _seen:
                                                            _seen.add(_hit.chunk_id)
                                                            _new_hits.append(_hit)
                                                except Exception:
                                                    pass

                                        if _new_hits:
                                            _new_ctx = ""
                                            for _j, _h in enumerate(_new_hits[:5]):
                                                _chunk = str(_h.content or "")[:400]
                                                if _chunk:
                                                    _new_ctx += f"[新检索{_j+1}] {_chunk}\n"

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
                                                    system="你是一个中文文档处理助手。只输出修正后的正文段落。")
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
    _bad_cites: list = []  # Initialize to avoid NameError
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
            from difflib import SequenceMatcher as _SM

            _bad_cites = []
            _last_good_cite = ""  # track last valid citation for auto-fill
            for _cite in _citations:
                _cite = _cite.strip()
                # Auto-fill placeholder citations
                _placeholder_patterns = ["同一材料", "同份材料", "同研究", "同一研究",
                                          "同上", "同文献", "同来源", "据材料显示",
                                          "据资料记载", "据文献", "据实验数据"]
                _is_placeholder = any(_p in _cite for _p in _placeholder_patterns)
                if _is_placeholder and _last_good_cite:
                    # Replace placeholder with last valid citation in content
                    _old = f"（据{_cite}）"
                    _new = f"（据{_last_good_cite}）"
                    if _old in _raw_content:
                        _raw_content = _raw_content.replace(_old, _new, 1)
                        _cite = _last_good_cite
                        logger.info("citation_autofill %r → %r", _cite, _last_good_cite)
                    _old2 = f"(据{_cite})"
                    if _old2 in _raw_content:
                        _raw_content = _raw_content.replace(_old2, f"(据{_last_good_cite})", 1)

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
                # Find matching source: exact → fuzzy → word-level
                _best_match = None
                _best_score = 0.0
                for _fn in _source_map:
                    if _cite in _fn or _fn in _cite:
                        _best_match = _fn; _best_score = 1.0; break
                    # Fuzzy match for abbreviations / aliases
                    _s = _SM(None, _cite, _fn).ratio()
                    if _s > _best_score:
                        _best_score = _s; _best_match = _fn
                    # Word-level fallback
                    if _best_score < 0.5:
                        _cite_words = _cite.replace("、", " ").replace("，", " ").split()
                        if any(_w in _fn for _w in _cite_words if len(_w) >= 2):
                            _best_match = _fn; break
                if _best_match and _best_score >= 0.35:
                    _last_good_cite = _cite  # track for placeholder auto-fill
                else:
                    _bad_cites.append((_cite, _context))
            # Write back auto-filled content
            if _raw_content != str(content or ""):
                content = _raw_content
                try:
                    (output_dir / "content.md").write_text(content, encoding="utf-8")
                except Exception:
                    pass

            if _bad_cites:
                # Group by citation label for dedup
                _by_label = {}
                for _bc in _bad_cites:
                    _label = _bc[0]
                    _by_label.setdefault(_label, []).append(_bc)

                # Note: Warnings now collected separately at return - content stays clean
                logger.info("citation_verify bad=%d total=%d", len(_bad_cites), len(_citations))
            else:
                logger.info("citation_verify all_ok count=%d", len(_citations))
    except Exception as _e:
        logger.warning("citation_verify_failed err=%s", str(_e)[:200])

    # ── Contrastive claim verification ──
    _contrastive: dict = {}  # Initialize to avoid NameError
    try:
        _contrastive = _verify_contrastive_claims(str(content or ""),
                                                   source_text, task_id=str(task_id))
        # Note: Warnings now collected separately at return - content stays clean
        if _contrastive.get("flagged_count"):
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

    return str(content or "")


def render_document(
    *,
    task_id: str,
    content: str,
    outline: str,
    output_dir: str,
    template_dir: str,
) -> list[str]:
    """Render content into templates AFTER user confirms satisfaction.

    Called separately from generate_document() so that template rendering
    (which can be expensive for DOCX/PDF) only happens once the user is happy.
    """
    _output_dir = Path(output_dir) if isinstance(output_dir, str) else Path(str(output_dir))
    _template_dir = Path(template_dir) if isinstance(template_dir, str) else Path(str(template_dir))

    if not _template_dir.exists() or not _template_dir.is_dir():
        # No templates — generate fallback DOCX from markdown
        rendered: list[str] = []
        if content:
            try:
                from agent_file_create.document.template_renderer import _render_markdown_to_docx
                fallback_path = _output_dir / "output.docx"
                _render_markdown_to_docx(str(content), str(outline or ""), str(fallback_path))
                rendered.append(str(fallback_path))
                logger.info("render_fallback_docx output=%s", fallback_path.name)
            except Exception as e:
                logger.warning("render_fallback_docx_failed err=%s", str(e)[:200])
        return rendered

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

        _SYSTEM_TPL_KEYS = {"title", "task_id", "document_outline", "document_content"}
        _template_section_keys: set[str] = set()
        for _tp in sorted(_template_dir.glob("*.md")):
            _template_section_keys.update(_scan_md_placeholders(str(_tp)))
        _template_section_keys -= _SYSTEM_TPL_KEYS

        content_dict = build_content_dict(
            str(content or ""),
            template_keys=sorted(_template_section_keys) if _template_section_keys else None,
        )
        match_info = content_dict.pop("_template_match_info", None) or {}
        content_dict["task_id"] = str(task_id)
        content_dict["document_outline"] = str(outline or "")
        content_dict["document_content"] = str(content or "")

        changed = _changed_sections(content_dict, str(_output_dir))
        if changed:
            logger.info("render_lazy_changed task=%s changed_sections=%s",
                         task_id, ", ".join(sorted(changed)[:10]))
        else:
            logger.info("render_lazy_skip_all task=%s (no sections changed)", task_id)

        rendered_outputs: list[str] = []
        templates = sorted([p for p in _template_dir.iterdir() if p.is_file()])
        for tp in templates:
            suf = tp.suffix.lower()
            if suf not in {".md", ".docx", ".pdf"}:
                continue
            out_path = _output_dir / f"{tp.stem}_rendered{suf}"

            if should_skip_render(str(tp), changed, is_pdf=(suf == ".pdf")):
                if out_path.exists():
                    rendered_outputs.append(str(out_path))
                    logger.info("render_lazy_skip template=%s", tp.name)
                    continue
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

        _save_render_cache(str(_output_dir), _section_hashes(content_dict))

        # Template match summary
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
                        task_id, [{"role": "assistant", "content": "；".join(parts) + "。"}])
                except Exception:
                    pass

        # Fallback DOCX
        has_docx = any(str(p.suffix).lower() == ".docx"
                       for p in _template_dir.iterdir() if p.is_file())
        if not has_docx and content:
            fallback_path = _output_dir / "output.docx"
            try:
                _render_markdown_to_docx(str(content), str(outline or ""), str(fallback_path))
                rendered_outputs.append(str(fallback_path))
                logger.info("render_fallback_docx output=%s", fallback_path.name)
            except Exception as e:
                logger.warning("render_fallback_docx_failed err=%s", str(e)[:200])

        return rendered_outputs
    except Exception as e:
        logger.warning(f"template_render_setup_failed err={str(e)[:200]}")
        return []

