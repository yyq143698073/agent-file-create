"""Upload and append routes for file ingestion.

Extracted from web/server.py.
"""

import logging
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from agent_file_create.rag import get_kb
from agent_file_create.task.manager import TaskManager
from agent_file_create.web._utils import (
    ingest_files_to_kb,
    result_dir,
    sanitize_filename,
    validate_upload_file,
)
from agent_file_create.web.routes.task import _start_task_thread

logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])


@router.post("/api/upload")
async def api_upload(
    files: list[UploadFile] = File(default=[]),
    templates: list[UploadFile] = File(default=[]),
    user_prompt: str = Form("生成一份报告"),
    target_words: str = Form("0"),
    ab_eval: str = Form("false"),
    add_to_kb: str = Form("false"),
    kb_name: str = Form(""),
    kb_doc_ids: str = Form(""),
    retrieval_kb: str = Form(""),
):
    kb_docs: list[str] = [x.strip() for x in str(kb_doc_ids).split(",") if x.strip()]
    files_list = [f for f in (files or []) if f.filename]
    if not files_list and not kb_docs:
        raise HTTPException(400, "未收到文件，也未选择知识库文档")

    task_id = uuid.uuid4().hex[:8]
    base = result_dir() / task_id
    uploads = base / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for f in files_list:
        # Validate file before saving
        err = validate_upload_file(
            f.filename or "",
            content_type=f.content_type,
            file_size=f.size or 0,
        )
        if err:
            raise HTTPException(400, f"文件「{f.filename}」校验失败: {err}")
        fn = sanitize_filename(f.filename or "upload")
        fp = uploads / fn
        content = await f.read()
        # Double-check actual size against limit
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(400, f"文件「{f.filename}」超过 50MB 限制")
        fp.write_bytes(content)
        saved.append(str(fp))

    # When KB docs are selected, reconstruct text from chunks and save as temp files
    if kb_docs:
        kbn = (kb_name or "").strip() or "default"
        for did in kb_docs:
            try:
                text = get_kb().get_doc_text(kb=kbn, doc_id=did)
                if text.strip():
                    fn = sanitize_filename(did) + ".md"
                    fp = uploads / fn
                    fp.write_text(text, encoding="utf-8")
                    saved.append(str(fp))
                    logger.info("kb_doc_used doc=%s chars=%d", did, len(text))
                else:
                    logger.warning("kb_doc_empty doc=%s", did)
            except Exception as e:
                logger.warning("kb_doc_read_failed doc=%s err=%s", did, str(e)[:200])

    if not saved:
        raise HTTPException(400, "未能从上传文件或知识库文档中提取到有效内容")

    user_template_dir = base / "template"
    user_template_dir.mkdir(parents=True, exist_ok=True)

    saved_templates: list[str] = []
    for f in (templates or []):
        fn = sanitize_filename(f.filename or "")
        suf = fn.rsplit(".", 1)[-1].lower() if "." in fn else ""
        if suf not in {"md", "docx", "pdf"}:
            continue
        fp = user_template_dir / fn
        fp.write_bytes(await f.read())
        saved_templates.append(str(fp))

    ab_val = str(ab_eval).strip().lower() in {"1", "true", "yes", "y", "on"}
    template_override = str(user_template_dir) if saved_templates else None
    template_mode = "task" if saved_templates else "default"

    # If user opted to also add files to knowledge base, ingest in background
    if str(add_to_kb).strip().lower() in {"1", "true", "yes", "y", "on"}:
        kbn = (kb_name or "").strip() or "default"
        threading.Thread(target=ingest_files_to_kb, args=(kbn, list(saved)), daemon=True).start()

    retrieval_kb_val = str(retrieval_kb or "").strip()
    task_manager = TaskManager()
    task_manager.write_status(
        task_id, "queued", stage="uploaded", message="已上传，等待开始生成…",
        extra={"saved_files": [Path(x).name for x in saved], "saved_templates": [Path(x).name for x in saved_templates], "ab_eval": ab_val, "ab_results": [], "clarify_questions": [], "clarify_answers": "", "clarify_skip": False, "retrieval_kb": retrieval_kb_val},
    )
    try: tw = int(str(target_words).strip() or "0")
    except Exception: tw = 0
    task_manager.write_task_meta(task_id, {"uploads_dir": str(uploads), "template_dir": str(user_template_dir), "file_paths": list(saved), "saved_templates": list(saved_templates), "user_prompt": str(user_prompt), "target_words": tw, "ab_eval": ab_val, "template_mode": template_mode, "active_kb": retrieval_kb_val})
    _start_task_thread(task_id, user_prompt=user_prompt, file_paths=saved, target_words=tw, ab_eval=ab_val, template_dir_override=template_override, saved_templates=saved_templates, mode="all")
    return JSONResponse({"task_id": task_id, "status": task_manager.read_status(task_id), "downloads": task_manager.collect_downloads(task_id)}, status_code=202)


@router.post("/api/append")
async def api_append(
    files: list[UploadFile] = File(default=[]),
    templates: list[UploadFile] = File(default=[]),
    task_id_raw: str = Form("", alias="task_id"),
    user_prompt: str = Form(""),
    target_words: str = Form("0"),
    ab_eval: str = Form(""),
    add_to_kb: str = Form("false"),
    kb_name: str = Form(""),
    kb_doc_ids: str = Form(""),
):
    task_manager = TaskManager()
    task_id = task_manager.normalize_task_id(task_id_raw)
    if not task_id:
        raise HTTPException(400, "task_id 不能为空或非法")
    if task_manager.is_task_running(task_id):
        raise HTTPException(409, "任务正在运行，请先 /pause 或 /cancel 后再追加。")

    kb_docs: list[str] = [x.strip() for x in str(kb_doc_ids).split(",") if x.strip()]
    files_list = [f for f in (files or []) if f.filename]
    tmpl_list = [f for f in (templates or []) if f.filename]
    if not files_list and not tmpl_list and not kb_docs:
        raise HTTPException(400, "未收到文件、模板或知识库文档")

    base = result_dir() / task_id
    uploads = base / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    saved_new: list[str] = []
    for f in files_list:
        fp = uploads / sanitize_filename(f.filename or "upload")
        fp.write_bytes(await f.read())
        saved_new.append(str(fp))

    # When KB docs are selected, reconstruct text from chunks and save as temp files
    if kb_docs:
        kbn = (kb_name or "").strip() or "default"
        for did in kb_docs:
            try:
                text = get_kb().get_doc_text(kb=kbn, doc_id=did)
                if text.strip():
                    fn = sanitize_filename(did) + ".md"
                    fp = uploads / fn
                    fp.write_text(text, encoding="utf-8")
                    saved_new.append(str(fp))
            except Exception as e:
                logger.warning("kb_doc_read_failed doc=%s err=%s", did, str(e)[:200])

    # If user opted to also add files to knowledge base, ingest in background
    if str(add_to_kb).strip().lower() in {"1", "true", "yes", "y", "on"}:
        kbn = (kb_name or "").strip() or "default"
        threading.Thread(target=ingest_files_to_kb, args=(kbn, list(saved_new)), daemon=True).start()

    user_template_dir = base / "template"
    user_template_dir.mkdir(parents=True, exist_ok=True)

    saved_templates_new: list[str] = []
    for f in tmpl_list:
        fn = sanitize_filename(f.filename or "")
        suf = fn.rsplit(".", 1)[-1].lower() if "." in fn else ""
        if suf not in {"md", "docx", "pdf"}:
            continue
        fp = user_template_dir / fn
        fp.write_bytes(await f.read())
        saved_templates_new.append(str(fp))

    meta = task_manager.read_task_meta(task_id)
    old_files = meta.get("file_paths") if isinstance(meta.get("file_paths"), list) else []
    old_files2 = [str(x) for x in old_files if isinstance(x, str)]
    all_files = old_files2 + saved_new
    user_prompt_val = (user_prompt or "").strip() or str(meta.get("user_prompt") or "").strip() or "生成一份报告"
    ab_val = bool(meta.get("ab_eval"))
    if ab_eval.strip():
        ab_val = ab_eval.strip().lower() in {"1", "true", "yes", "y", "on"}

    template_override = str(user_template_dir) if (task_manager.list_task_templates(task_id) or saved_templates_new) else None
    template_mode = str(meta.get("template_mode") or "").strip().lower() or ("task" if saved_templates_new else "default")
    if saved_templates_new:
        template_mode = "task"

    task_manager.write_task_meta(task_id, {"uploads_dir": str(uploads), "template_dir": str(user_template_dir), "file_paths": list(all_files), "user_prompt": user_prompt_val, "ab_eval": bool(ab_val), "template_mode": template_mode, "active_kb": str(meta.get("active_kb") or "").strip()})
    task_manager.write_status(task_id, "queued", stage="uploaded", message="已追加文件，等待重新生成…", extra={"saved_files": [Path(x).name for x in all_files], "saved_templates": [Path(x).name for x in task_manager.list_task_templates(task_id)], "ab_eval": bool(ab_val)})
    _start_task_thread(task_id, user_prompt=user_prompt_val, file_paths=all_files, ab_eval=bool(ab_val), template_dir_override=template_override, saved_templates=task_manager.list_task_templates(task_id), mode="all")
    return JSONResponse({"task_id": task_id, "status": task_manager.read_status(task_id), "downloads": task_manager.collect_downloads(task_id)}, status_code=202)
