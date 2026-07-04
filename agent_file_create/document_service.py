"""Legacy document generation service — **DEPRECATED**.

This module's `generate_document()` is the old monolithic pipeline.
New code should use `agent_file_create.agent.DocumentAgent` instead,
which provides the same functionality via a LangGraph StateGraph with
proper human-in-the-loop, checkpointing, and error recovery.

The helper functions imported from `_quality.py` are still used by the
new agent's `_node_quality_gate` and remain supported.
"""

import logging
import os
import re
import threading
import time
import uuid
import warnings
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
    warnings.warn(
        "generate_document() is deprecated. Use DocumentAgent instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    except Exception as e:
        logger.warning("write_clean_content_failed path=%s/content.md err=%s", output_dir, e)

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
    """Run faithfulness check and hallucination-triggered re-retrieval on content.

    Delegates to QualityPipeline for structured, decomposable execution.
    """
    from agent_file_create.quality import QualityPipeline, QualityContext

    ctx = QualityContext(
        content=content,
        analysis_results=analysis_results or [],
        task_id=str(task_id),
        output_dir=str(output_dir),
    )
    result = QualityPipeline().run(ctx)
    return str(result.content or "")


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

