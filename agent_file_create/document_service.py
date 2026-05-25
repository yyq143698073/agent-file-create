import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
        (output_dir / "content.md").write_text(str(content or ""), encoding="utf-8")
    except Exception as e:
        logger.warning(f"write_content_failed err={str(e)[:160]}")

    rendered_outputs: list[str] = []
    if template_dir.exists() and template_dir.is_dir():
        try:
            from agent_file_create.document.template_renderer import build_content_dict, render_markdown_template, render_pdf_template, render_word_template

            content_dict = build_content_dict(str(content or ""))
            content_dict["task_id"] = str(task_id)
            content_dict["document_outline"] = str(outline or "")
            content_dict["document_content"] = str(content or "")

            templates = sorted([p for p in template_dir.iterdir() if p.is_file()])
            for tp in templates:
                suf = tp.suffix.lower()
                if suf not in {".md", ".docx", ".pdf"}:
                    continue
                out_path = output_dir / f"{tp.stem}_rendered{suf}"
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
    }

