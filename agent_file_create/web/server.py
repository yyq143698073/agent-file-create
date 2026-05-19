import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent_file_create.chat.handler import ChatHandler
from agent_file_create.document.extractor import extract_from_file
from agent_file_create.logging_config import setup_logging
from agent_file_create.preprocessor import compute_quality_metrics
from agent_file_create.rag.kb import KnowledgeBase
from agent_file_create.task.manager import TaskManager

logger = logging.getLogger(__name__)

_KB = KnowledgeBase()


def _get_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _html_dir() -> Path:
    return _get_base_dir() / "html"


def _result_dir() -> Path:
    return _get_base_dir() / "result"


def _sanitize_filename(name: str) -> str:
    n = (name or "").strip()
    n = n.replace("\\", "/").split("/")[-1]
    n = re.sub(r"[^0-9A-Za-z一-鿿._-]+", "_", n)
    n = n.strip("._")
    return n or "upload"


def _split_questions(text: str) -> list[str]:
    out: list[str] = []
    cur = ""
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            if cur:
                out.append(cur)
                cur = ""
            continue
        is_option = bool(re.match(r"^[A-Z][.)、\s]", s))
        if is_option and cur:
            cur += "\n" + s
            continue
        if cur:
            out.append(cur)
        s = re.sub(r"^[0-9]+[.)、\s]+", "", s).strip()
        s = re.sub(r"^[-*]\s+", "", s).strip()
        if s:
            cur = s[:240]
        if len(out) >= 6:
            break
    if cur and len(out) < 6:
        out.append(cur)
    if out:
        return out
    s = str(text or "").strip()
    return [s[:240]] if s else []


def _better_quality(a: dict, b: dict) -> bool:
    try:
        af = int(a.get("filled_fields") or 0)
    except Exception:
        af = 0
    try:
        bf = int(b.get("filled_fields") or 0)
    except Exception:
        bf = 0
    if bf != af:
        return bf > af
    try:
        ar = float(a.get("field_ratio") or 0.0)
    except Exception:
        ar = 0.0
    try:
        br = float(b.get("field_ratio") or 0.0)
    except Exception:
        br = 0.0
    return br >= ar


def _run_task(task_id: str, file_paths: list[str], user_prompt: str, *, ab_eval: bool, template_dir_override: str | None, saved_templates: list[str] | None) -> None:
    task_manager = TaskManager()
    st_names = [Path(x).name for x in (saved_templates or [])]
    pause_ev, cancel_ev = task_manager.get_control_events(task_id)

    def _control(stage: str) -> bool:
        if cancel_ev.is_set():
            task_manager.write_status(task_id, "canceled", stage=stage, message="已取消", extra={"saved_templates": st_names})
            return False
        resumed = False
        notified = False
        while pause_ev.is_set():
            if cancel_ev.is_set():
                task_manager.write_status(task_id, "canceled", stage=stage, message="已取消", extra={"saved_templates": st_names})
                return False
            if not notified:
                task_manager.write_status(task_id, "paused", stage=stage, message="已暂停（发送 /resume 继续，/cancel 取消）", extra={"saved_templates": st_names})
                notified = True
            time.sleep(0.6)
            resumed = True
        if resumed:
            task_manager.write_status(task_id, "processing", stage=stage, message="已继续执行…", extra={"saved_templates": st_names})
        return True

    if not _control("extract"):
        return
    task_manager.write_status(task_id, "processing", stage="extract", message="开始并行解析文件…", extra={"total_files": len(file_paths), "done_files": 0, "ab_eval": bool(ab_eval), "ab_results": [], "saved_templates": st_names})

    from agent_file_create.config import MAX_WORKERS_DEFAULT

    def _extract_one(fp: str) -> tuple[int, dict, dict | None]:
        fn = Path(fp).name
        ext = fn.rsplit(".", 1)[-1].lower() if "." in fn else ""
        is_text_like = ext in {"txt", "md"}
        try:
            if ab_eval and not is_text_like:
                res_a = extract_from_file(fp, preprocess=False)
                qa = compute_quality_metrics(res_a)
                res_b = extract_from_file(fp, preprocess=True)
                qb = compute_quality_metrics(res_b)
                use_b = _better_quality(qa, qb)
                chosen = res_b if use_b else res_a
                chosen["_file"] = fn
                return (0, chosen, {"file": fn, "a": qa, "b": qb, "chosen": "b" if use_b else "a"})
            else:
                res = extract_from_file(fp, preprocess=True)
                res["_file"] = fn
                ab_item = None
                if ab_eval:
                    q = compute_quality_metrics(res)
                    ab_item = {"file": fn, "a": q, "b": q, "chosen": "b", "note": "该类型不做A/B对比"}
                return (0, res, ab_item)
        except Exception as e:
            return (0, {"error": str(e), "_file": fn}, {"file": fn, "error": str(e)[:240]} if ab_eval else None)

    indexed: list[tuple[int, dict, dict | None]] = []
    max_workers = max(1, min(int(MAX_WORKERS_DEFAULT), len(file_paths)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_extract_one, fp): i for i, fp in enumerate(file_paths)}
        done = 0
        for fut in as_completed(futures):
            if not _control("extract"):
                for f in futures:
                    f.cancel()
                return
            try:
                _, ar, ab = fut.result()
            except Exception as e:
                idx = futures[fut]
                fn = Path(file_paths[idx]).name
                ar = {"error": str(e), "_file": fn}
                ab = {"file": fn, "error": str(e)[:240]} if ab_eval else None
            indexed.append((futures[fut], ar, ab))
            done += 1
            ab_snapshot = [x for _, _, x in indexed if x is not None]
            task_manager.write_status(task_id, "processing", stage="extract", message=f"已解析 {done}/{len(file_paths)} 个文件", extra={"total_files": len(file_paths), "done_files": done, "ab_results": ab_snapshot[-20:], "saved_templates": st_names})

    indexed.sort(key=lambda x: x[0])
    analysis_results = [ar for _, ar, _ in indexed]
    ab_results = [ab for _, _, ab in indexed if ab is not None]

    task_manager.write_analysis_results(task_id, analysis_results)
    task_manager.write_task_meta(task_id, {"file_paths": list(file_paths), "user_prompt": str(user_prompt or ""), "ab_eval": bool(ab_eval), "saved_templates": st_names})

    if not _control("document"):
        return
    try:
        from agent_core import DocumentAgent

        def _human_input(question: str) -> str:
            if cancel_ev.is_set():
                task_manager.write_status(task_id, "canceled", stage="clarify", message="已取消", extra={"saved_templates": st_names})
                return ""
            qs = _split_questions(question)
            task_manager.write_status(
                task_id, "need_user", stage="clarify",
                message="需要补充信息以便更好生成报告，请回答下列问题后点击提交。",
                extra={"clarify_questions": qs, "clarify_answers": "", "clarify_skip": False, "ab_eval": bool(ab_eval), "ab_results": ab_results[-20:], "saved_templates": st_names},
            )
            answers, skipped = task_manager.wait_for_clarify(task_id, timeout_s=1800)
            task_manager.write_status(
                task_id, "processing", stage="clarify", message="已收到补充信息，继续生成…",
                extra={"clarify_answers": answers, "clarify_skip": bool(skipped), "ab_eval": bool(ab_eval), "ab_results": ab_results[-20:], "saved_templates": st_names},
            )
            return answers

        if not _control("document"):
            return
        task_manager.write_status(task_id, "processing", stage="document", message="开始生成文档…", extra={"saved_templates": st_names, "ab_results": ab_results[-20:], "ab_eval": bool(ab_eval)})
        agent = DocumentAgent(task_id=task_id, user_prompt=user_prompt, file_paths=file_paths, template_dir_override=template_dir_override)
        agent.state["analysis_results"] = analysis_results
        agent.state["force_regen"] = False
        state = agent.run(max_turns=8, human_input_fn=_human_input)

        if cancel_ev.is_set():
            task_manager.write_status(task_id, "canceled", stage="done", message="已取消", extra={"saved_templates": st_names})
        else:
            task_manager.write_status(task_id, "finished", stage="done", message="生成完成", extra={"result": {"output_dir": state.get("output_dir")}, "saved_templates": st_names})
    except Exception as e:
        task_manager.write_status(task_id, "failed", stage="done", message="生成失败", extra={"error": str(e)[:400]})


def _run_document_only(task_id: str, *, user_prompt: str, file_paths: list[str], analysis_results: list[dict], template_dir_override: str | None, saved_templates: list[str] | None, ab_eval: bool) -> None:
    task_manager = TaskManager()
    st_names = [Path(x).name for x in (saved_templates or [])]
    pause_ev, cancel_ev = task_manager.get_control_events(task_id)

    def _control(stage: str) -> bool:
        if cancel_ev.is_set():
            task_manager.write_status(task_id, "canceled", stage=stage, message="已取消", extra={"saved_templates": st_names})
            return False
        resumed = False
        notified = False
        while pause_ev.is_set():
            if cancel_ev.is_set():
                task_manager.write_status(task_id, "canceled", stage=stage, message="已取消", extra={"saved_templates": st_names})
                return False
            if not notified:
                task_manager.write_status(task_id, "paused", stage=stage, message="已暂停（发送 /resume 继续，/cancel 取消）", extra={"saved_templates": st_names})
                notified = True
            time.sleep(0.6)
            resumed = True
        if resumed:
            task_manager.write_status(task_id, "processing", stage=stage, message="已继续执行…", extra={"saved_templates": st_names})
        return True

    if not _control("document"):
        return

    try:
        from agent_core import DocumentAgent

        task_manager.write_status(task_id, "processing", stage="document", message="开始重新生成文档…", extra={"saved_templates": st_names, "ab_eval": bool(ab_eval)})
        agent = DocumentAgent(task_id=task_id, user_prompt=user_prompt, file_paths=file_paths, template_dir_override=template_dir_override)
        agent.state["analysis_results"] = list(analysis_results or [])
        agent.state["force_regen"] = True
        state = agent.run(max_turns=8, human_input_fn=None)
        if cancel_ev.is_set():
            task_manager.write_status(task_id, "canceled", stage="done", message="已取消", extra={"saved_templates": st_names})
        else:
            task_manager.write_status(task_id, "finished", stage="done", message="重新生成完成", extra={"result": {"output_dir": state.get("output_dir")}, "saved_templates": st_names})
    except Exception as e:
        task_manager.write_status(task_id, "failed", stage="done", message="重新生成失败", extra={"error": str(e)[:400]})


def _start_task_thread(task_id: str, *, user_prompt: str, file_paths: list[str], ab_eval: bool, template_dir_override: str | None, saved_templates: list[str] | None, mode: str) -> tuple[bool, str]:
    task_manager = TaskManager()
    if task_manager.is_task_running(task_id):
        return False, "任务正在运行，无法启动新的生成。可先 /pause 或 /cancel。"
    pause_ev, cancel_ev = task_manager.get_control_events(task_id)
    pause_ev.clear()
    cancel_ev.clear()
    if mode == "document_only":
        analysis_results = task_manager.read_analysis_results(task_id)
        if not analysis_results:
            mode = "all"
        else:
            import threading
            th = threading.Thread(target=_run_document_only, kwargs={"task_id": task_id, "user_prompt": user_prompt, "file_paths": file_paths, "analysis_results": analysis_results, "template_dir_override": template_dir_override, "saved_templates": saved_templates, "ab_eval": bool(ab_eval)}, daemon=True)
            task_manager.start_task(task_id, th.start)
            return True, "已启动重新生成（仅文档阶段）。"
    import threading
    th = threading.Thread(target=_run_task, args=(task_id, file_paths, user_prompt), kwargs={"ab_eval": bool(ab_eval), "template_dir_override": template_dir_override, "saved_templates": saved_templates}, daemon=True)
    task_manager.start_task(task_id, th.start)
    return True, "已启动生成任务。"


def _run_section_regen(task_id: str, section_name: str) -> tuple[bool, str]:
    """Regenerate a single section in the background, write updated content."""
    try:
        from agent_file_create.document.content_generator import regenerate_section

        task_manager = TaskManager()
        meta = task_manager.read_task_meta(task_id)
        user_prompt = str(meta.get("user_prompt") or "").strip() or "生成一份报告"
        analysis_results = task_manager.read_analysis_results(task_id)

        base = _result_dir() / task_id
        outline_path = base / "outline.md"
        content_path = base / "content.md"
        outline = outline_path.read_text(encoding="utf-8") if outline_path.exists() else ""
        content = content_path.read_text(encoding="utf-8") if content_path.exists() else ""

        if not outline or not content:
            return False, "缺少大纲或正文，无法定位章节。请先确保任务已生成完成。"

        multimodal = {f"source_{i}": r for i, r in enumerate(analysis_results)} if analysis_results else {}

        task_manager.write_status(task_id, "processing", stage="document",
                                  message=f"正在重新生成章节「{section_name}」…")

        new_content = regenerate_section(
            outline, content, section_name, multimodal, user_prompt, task_id=task_id,
        )

        if not new_content:
            task_manager.write_status(task_id, "finished", stage="document",
                                      message=f"未找到匹配章节「{section_name}」，请检查章节标题是否正确。")
            return False, f"未找到匹配章节「{section_name}」。可尝试 /templates 或 /files 查看章节标题，使用精确标题重新生成。"

        content_path.write_text(new_content, encoding="utf-8")
        task_manager.write_status(task_id, "finished", stage="document",
                                  message=f"已重新生成章节「{section_name}」")
        return True, f"已重新生成章节「{section_name}」。请查看预览确认结果。"
    except Exception as e:
        logger.exception("section_regen_failed")
        try:
            TaskManager().write_status(task_id, "finished", stage="document",
                                       message=f"章节重生成失败：{str(e)[:120]}")
        except Exception:
            pass
        return False, f"章节生成失败：{str(e)[:200]}"


def _make_regenerate_fn(task_manager: TaskManager):
    def _fn(task_id: str, mode: str = "doc", section_name: str = "") -> tuple[bool, str]:
        # Single-section regeneration
        if section_name and section_name.strip():
            import threading
            th = threading.Thread(target=_run_section_regen, args=(task_id, section_name.strip()), daemon=True)
            task_manager.start_task(task_id, th.start)
            return True, f"已启动重新生成章节「{section_name}」，请稍后查看预览。"
        meta = task_manager.read_task_meta(task_id)
        user_prompt = str(meta.get("user_prompt") or "").strip() or "生成一份报告"
        file_paths = [str(x) for x in meta.get("file_paths") or [] if isinstance(x, str)]
        ab_eval = bool(meta.get("ab_eval"))
        saved_templates_raw = meta.get("saved_templates")
        saved_templates = [str(x) for x in saved_templates_raw] if isinstance(saved_templates_raw, list) else []
        template_dir_str = str(meta.get("template_dir") or "").strip() or None
        template_mode = str(meta.get("template_mode") or "").strip().lower()
        if template_mode == "default":
            template_dir_str = None
        mode_clean = "document_only" if mode in {"doc", "document_only"} else "all"
        if mode_clean == "document_only":
            analysis_results = task_manager.read_analysis_results(task_id)
            if not analysis_results:
                mode_clean = "all"
        return _start_task_thread(
            task_id,
            user_prompt=user_prompt,
            file_paths=file_paths,
            ab_eval=ab_eval,
            template_dir_override=template_dir_str,
            saved_templates=saved_templates,
            mode=mode_clean,
        )
    return _fn


# ── FastAPI Application ──

app = FastAPI(title="agent-file-create", version="1.0.0")

# Mount static file directories
_html = _html_dir()
_result = _result_dir()

if _html.exists():
    app.mount("/static", StaticFiles(directory=str(_html)), name="static")


# ── API Routes ──

@app.get("/api/kb/list")
def kb_list():
    items = _KB.list_kb()
    return {"kbs": items}


@app.get("/api/kb/docs")
def kb_docs(kb: str = Query("default")):
    docs = _KB.list_docs(kb=kb)
    return {"kb": kb, "docs": docs}


@app.post("/api/kb/query")
async def kb_query(request: Request):
    body = await request.json()
    kb = str(body.get("kb") or "").strip() or "default"
    question = str(body.get("question") or body.get("message") or "").strip()
    if not question:
        raise HTTPException(400, "question 不能为空")
    try:
        top_k = int(body.get("top_k") or 6)
    except Exception:
        top_k = 6
    filters = body.get("filters") if isinstance(body.get("filters"), dict) else None
    try:
        ans = _KB.answer(kb=kb, question=question, top_k=top_k, filters=filters)
        cits = [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "section_path": c.section_path, "score": float(c.score), "snippet": c.snippet} for c in ans.citations]
        return {"kb": ans.kb, "question": ans.question, "answer": ans.answer, "citations": cits}
    except Exception as e:
        raise HTTPException(500, str(e)[:240])


@app.post("/api/kb/upload")
async def kb_upload(
    files: list[UploadFile] = File(...),
    kb: str = Form("default"),
    doc_type: str = Form(""),
):
    kb = kb.strip() or "default"
    doc_type = doc_type.strip()
    if not files:
        raise HTTPException(400, "未收到文件")

    base = _result_dir() / "kb" / kb / "uploads"
    base.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for f in files:
        name = _sanitize_filename(f.filename or "upload")
        fp = base / (uuid.uuid4().hex[:8] + "_" + name)
        data = await f.read()
        try:
            fp.write_bytes(data)
        except Exception:
            results.append({"file": name, "ok": False, "error": "write_failed"})
            continue
        try:
            r = _KB.ingest_file(kb=kb, file_path=str(fp), doc_id=name, title=name, source=str(fp), doc_type=doc_type)
            r["file"] = name
            results.append(r)
        except Exception as e:
            results.append({"file": name, "ok": False, "error": str(e)[:240]})
    return {"kb": kb, "results": results}


@app.post("/api/kb/delete")
async def kb_delete(request: Request):
    body = await request.json()
    kb = str(body.get("kb") or "").strip()
    doc_id = str(body.get("doc_id") or "").strip()
    if not kb:
        raise HTTPException(400, "kb 不能为空")
    if doc_id:
        r = _KB.delete_doc(kb=kb, doc_id=doc_id)
    else:
        r = _KB.delete_kb(kb=kb)
    if not r.get("ok"):
        raise HTTPException(500, str(r.get("error") or "delete_failed")[:240])
    return r


@app.get("/api/kb/stats")
def kb_stats(kb: str = "default"):
    if not kb.strip():
        raise HTTPException(400, "kb 不能为空")
    try:
        return _KB.kb_stats(kb=kb.strip())
    except Exception as e:
        raise HTTPException(500, str(e)[:240])


@app.post("/api/kb/health")
def kb_health():
    return _KB.check_embed_health()


def _ingest_files_to_kb(kb_name: str, file_paths: list[str]) -> None:
    """Ingest saved files into a knowledge base (runs in background thread)."""
    import logging
    _log = logging.getLogger(__name__)
    from agent_file_create.rag.kb import KnowledgeBase
    kb = KnowledgeBase()
    kb_name = (kb_name or "").strip() or "default"
    ok = 0
    for fp in file_paths:
        try:
            r = kb.ingest_file(kb=kb_name, file_path=fp)
            if r.get("ok"):
                ok += 1
            else:
                _log.warning("kb_ingest_failed file=%s err=%s", fp, r.get("error", ""))
        except Exception as e:
            _log.warning("kb_ingest_failed file=%s err=%s", fp, str(e)[:200])
    _log.info("kb_ingest_done kb=%s ok=%d/%d", kb_name, ok, len(file_paths))


@app.post("/api/upload")
async def api_upload(
    files: list[UploadFile] = File(default=[]),
    templates: list[UploadFile] = File(default=[]),
    user_prompt: str = Form("生成一份报告"),
    ab_eval: str = Form("false"),
    add_to_kb: str = Form("false"),
    kb_name: str = Form(""),
    kb_doc_ids: str = Form(""),
):
    kb_docs: list[str] = [x.strip() for x in str(kb_doc_ids).split(",") if x.strip()]
    files_list = [f for f in (files or []) if f.filename]
    if not files_list and not kb_docs:
        raise HTTPException(400, "未收到文件，也未选择知识库文档")

    task_id = uuid.uuid4().hex[:8]
    base = _result_dir() / task_id
    uploads = base / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for f in files_list:
        fn = _sanitize_filename(f.filename or "upload")
        fp = uploads / fn
        fp.write_bytes(await f.read())
        saved.append(str(fp))

    # When KB docs are selected, reconstruct text from chunks and save as temp files
    if kb_docs:
        kbn = (kb_name or "").strip() or "default"
        for did in kb_docs:
            try:
                text = _KB.get_doc_text(kb=kbn, doc_id=did)
                if text.strip():
                    fn = _sanitize_filename(did) + ".md"
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
        fn = _sanitize_filename(f.filename or "")
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
        import threading
        threading.Thread(target=_ingest_files_to_kb, args=(kbn, list(saved)), daemon=True).start()

    task_manager = TaskManager()
    task_manager.write_status(
        task_id, "queued", stage="uploaded", message="已上传，等待开始生成…",
        extra={"saved_files": [Path(x).name for x in saved], "saved_templates": [Path(x).name for x in saved_templates], "ab_eval": ab_val, "ab_results": [], "clarify_questions": [], "clarify_answers": "", "clarify_skip": False},
    )
    task_manager.write_task_meta(task_id, {"uploads_dir": str(uploads), "template_dir": str(user_template_dir), "file_paths": list(saved), "saved_templates": list(saved_templates), "user_prompt": str(user_prompt), "ab_eval": ab_val, "template_mode": template_mode, "active_kb": (kb_name or "").strip() or "default"})
    _start_task_thread(task_id, user_prompt=user_prompt, file_paths=saved, ab_eval=ab_val, template_dir_override=template_override, saved_templates=saved_templates, mode="all")
    return JSONResponse({"task_id": task_id, "status": task_manager.read_status(task_id), "downloads": task_manager.collect_downloads(task_id)}, status_code=202)


@app.post("/api/append")
async def api_append(
    files: list[UploadFile] = File(default=[]),
    templates: list[UploadFile] = File(default=[]),
    task_id_raw: str = Form("", alias="task_id"),
    user_prompt: str = Form(""),
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

    base = _result_dir() / task_id
    uploads = base / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    saved_new: list[str] = []
    for f in files_list:
        fp = uploads / _sanitize_filename(f.filename or "upload")
        fp.write_bytes(await f.read())
        saved_new.append(str(fp))

    # When KB docs are selected, reconstruct text from chunks and save as temp files
    if kb_docs:
        kbn = (kb_name or "").strip() or "default"
        for did in kb_docs:
            try:
                text = _KB.get_doc_text(kb=kbn, doc_id=did)
                if text.strip():
                    fn = _sanitize_filename(did) + ".md"
                    fp = uploads / fn
                    fp.write_text(text, encoding="utf-8")
                    saved_new.append(str(fp))
            except Exception as e:
                logger.warning("kb_doc_read_failed doc=%s err=%s", did, str(e)[:200])

    # If user opted to also add files to knowledge base, ingest in background
    if str(add_to_kb).strip().lower() in {"1", "true", "yes", "y", "on"}:
        kbn = (kb_name or "").strip() or "default"
        import threading
        threading.Thread(target=_ingest_files_to_kb, args=(kbn, list(saved_new)), daemon=True).start()

    user_template_dir = base / "template"
    user_template_dir.mkdir(parents=True, exist_ok=True)

    saved_templates_new: list[str] = []
    for f in tmpl_list:
        fn = _sanitize_filename(f.filename or "")
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

    task_manager.write_task_meta(task_id, {"uploads_dir": str(uploads), "template_dir": str(user_template_dir), "file_paths": list(all_files), "user_prompt": user_prompt_val, "ab_eval": bool(ab_val), "template_mode": template_mode})
    task_manager.write_status(task_id, "queued", stage="uploaded", message="已追加文件，等待重新生成…", extra={"saved_files": [Path(x).name for x in all_files], "saved_templates": [Path(x).name for x in task_manager.list_task_templates(task_id)], "ab_eval": bool(ab_val)})
    _start_task_thread(task_id, user_prompt=user_prompt_val, file_paths=all_files, ab_eval=bool(ab_val), template_dir_override=template_override, saved_templates=task_manager.list_task_templates(task_id), mode="all")
    return JSONResponse({"task_id": task_id, "status": task_manager.read_status(task_id), "downloads": task_manager.collect_downloads(task_id)}, status_code=202)


@app.post("/api/clarify")
async def api_clarify(request: Request):
    body = await request.json()
    task_manager = TaskManager()
    task_id = task_manager.normalize_task_id(body.get("task_id"))
    if not task_id:
        raise HTTPException(400, "task_id 不能为空")
    answers = str(body.get("answers") or "").strip()
    skip = bool(body.get("skip"))
    task_manager.write_status(task_id, "processing", stage="clarify", message="已收到补充信息，正在继续…", extra={"clarify_answers": answers, "clarify_skip": skip, "clarify_submitted_at": float(time.time())})
    return {"task_id": task_id, "ok": True}


@app.post("/api/chat")
async def api_chat(request: Request):
    task_manager = TaskManager()
    chat_handler = ChatHandler(task_manager, regenerate_fn=_make_regenerate_fn(task_manager))
    body = await request.json()
    message = str(body.get("message") or "").strip()
    raw_tid = str(body.get("task_id") or "").strip()
    task_id = task_manager.normalize_task_id(raw_tid) if raw_tid else "lobby"
    history = body.get("history") if isinstance(body.get("history"), list) else []
    action = body.get("action")

    act = chat_handler._parse_chat_action(message, action)
    if not message and not act:
        raise HTTPException(400, "message 不能为空")
    if raw_tid and not task_id:
        raise HTTPException(400, "task_id 非法")

    if not history:
        history = task_manager.read_chat_history(task_id)

    st = task_manager.read_status(task_id)

    if act:
        if not message:
            message = "/" + str(act.get("type") or "action")
        reply = chat_handler._handle_chat_action(task_id, act)
        task_manager.append_chat_history(task_id, [{"role": "user", "content": message}, {"role": "assistant", "content": reply}])
        return {"task_id": task_id, "reply": reply, "action": act}

    if str(st.get("status") or "") == "need_user" and str(st.get("stage") or "") == "clarify":
        m = message.strip()
        low = m.lower()
        if low in {"skip", "跳过", "略过", "不用了", "不需要"}:
            task_manager.write_status(task_id, "processing", stage="clarify", message="用户选择跳过澄清，继续生成…", extra={"clarify_answers": "", "clarify_skip": True, "clarify_submitted_at": float(time.time())})
            reply = "已收到：跳过澄清。我会继续生成；如结果不符合预期，可再补充信息让我调整。"
        else:
            qs = st.get("clarify_questions") if isinstance(st.get("clarify_questions"), list) else []
            is_valid, warning = chat_handler.validate_clarify_answer(m, qs)
            if not is_valid:
                return {"task_id": task_id, "reply": warning}
            task_manager.write_status(task_id, "processing", stage="clarify", message="已收到用户补充信息，继续生成…", extra={"clarify_answers": m, "clarify_skip": False, "clarify_submitted_at": float(time.time())})
            reply = "已收到补充信息。我会继续生成文档；你也可以继续提问或补充更多要求。"
        task_manager.append_chat_history(task_id, [{"role": "user", "content": message}, {"role": "assistant", "content": reply}])
        return {"task_id": task_id, "reply": reply}

    reply = chat_handler.chat_reply(message, task_id, history)
    return {"task_id": task_id, "reply": reply}


@app.post("/api/chat/stream")
async def api_chat_stream(request: Request):
    task_manager = TaskManager()
    chat_handler = ChatHandler(task_manager, regenerate_fn=_make_regenerate_fn(task_manager))
    body = await request.json()
    message = str(body.get("message") or "").strip()
    raw_tid = str(body.get("task_id") or "").strip()
    task_id = task_manager.normalize_task_id(raw_tid) if raw_tid else "lobby"
    history = body.get("history") if isinstance(body.get("history"), list) else []
    action = body.get("action")

    act = chat_handler._parse_chat_action(message, action)
    if not message and not act:
        raise HTTPException(400, "message 不能为空")
    if raw_tid and not task_id:
        raise HTTPException(400, "task_id 非法")

    if not history:
        history = task_manager.read_chat_history(task_id)

    st = task_manager.read_status(task_id)

    if act:
        if not message:
            message = "/" + str(act.get("type") or "action")
        reply = chat_handler._handle_chat_action(task_id, act)
        task_manager.append_chat_history(task_id, [{"role": "user", "content": message}, {"role": "assistant", "content": reply}])
        return {"task_id": task_id, "reply": reply, "action": act}

    if str(st.get("status") or "") == "need_user" and str(st.get("stage") or "") == "clarify":
        m = message.strip()
        low = m.lower()
        if low in {"skip", "跳过", "略过", "不用了", "不需要"}:
            task_manager.write_status(task_id, "processing", stage="clarify", message="用户选择跳过澄清，继续生成…", extra={"clarify_answers": "", "clarify_skip": True, "clarify_submitted_at": float(time.time())})
            reply = "已收到：跳过澄清。我会继续生成；如结果不符合预期，可再补充信息让我调整。"
        else:
            qs = st.get("clarify_questions") if isinstance(st.get("clarify_questions"), list) else []
            is_valid, warning = chat_handler.validate_clarify_answer(m, qs)
            if not is_valid:
                return {"task_id": task_id, "reply": warning}
            task_manager.write_status(task_id, "processing", stage="clarify", message="已收到用户补充信息，继续生成…", extra={"clarify_answers": m, "clarify_skip": False, "clarify_submitted_at": float(time.time())})
            reply = "已收到补充信息。我会继续生成文档；你也可以继续提问或补充更多要求。"
        task_manager.append_chat_history(task_id, [{"role": "user", "content": message}, {"role": "assistant", "content": reply}])
        return {"task_id": task_id, "reply": reply}

    def sse_generate():
        try:
            for token in chat_handler.chat_reply_stream(message, task_id, history):
                if isinstance(token, dict):
                    yield f"data: {json.dumps(token, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)[:240]}, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse_generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@app.get("/api/chat/history")
def chat_history(task_id: str = Query("")):
    task_manager = TaskManager()
    tid = task_manager.normalize_task_id(task_id) if task_id else "lobby"
    if task_id and not tid:
        raise HTTPException(400, "task_id 非法")
    return {"task_id": tid, "history": task_manager.read_chat_history(tid), "summary": task_manager.read_chat_summary(tid)}


@app.get("/api/status")
def api_status(task_id: str = Query("")):
    task_manager = TaskManager()
    tid = task_manager.normalize_task_id(task_id)
    if not tid:
        raise HTTPException(400, "task_id 不能为空")
    data = task_manager.read_status(tid)
    data["downloads"] = task_manager.collect_downloads(tid)
    return data


@app.get("/api/tasks")
def api_tasks(task_id: str = Query("")):
    task_manager = TaskManager()
    # Single task detail
    if task_id:
        tid = task_manager.normalize_task_id(task_id)
        if not tid:
            raise HTTPException(400, "task_id 非法")
        data = task_manager.collect_downloads(tid)
        data["status"] = task_manager.read_status(tid)
        data["chat_summary"] = task_manager.read_chat_summary(tid) or ""
        return data

    # List all tasks
    base = _result_dir()
    items = []
    if base.exists():
        for d in sorted(base.iterdir(), key=lambda x: x.stat().st_mtime if x.is_dir() else 0, reverse=True):
            if not d.is_dir() or d.name.startswith(".") or d.name in {"template", "kb"}:
                continue
            try:
                st_path = d / "status.json"
                if not st_path.exists():
                    continue
                st = json.loads(st_path.read_text(encoding="utf-8"))
                if not isinstance(st, dict):
                    continue
                tid = str(st.get("task_id") or d.name).strip()
                # Check whether there is chat history
                has_chat = False
                try:
                    ch_path = d / "chat_history.json"
                    if ch_path.exists():
                        ch_data = json.loads(ch_path.read_text(encoding="utf-8"))
                        has_chat = isinstance(ch_data, list) and len(ch_data) > 0
                except Exception:
                    pass
                items.append({
                    "task_id": tid,
                    "status": str(st.get("status") or "").strip(),
                    "stage": str(st.get("stage") or "").strip(),
                    "message": str(st.get("message") or "")[:120],
                    "updated_at": st.get("updated_at", 0),
                    "has_chat": has_chat,
                })
            except Exception:
                continue
    return {"tasks": items, "total": len(items)}


# Serve result files and index.html
if _result.exists():
    app.mount("/result", StaticFiles(directory=str(_result)), name="result")

if _html.exists():
    @app.get("/")
    @app.get("/index.html")
    async def serve_index():
        index = _html / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return PlainTextResponse("Not Found", status_code=404)


application = app


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    setup_logging()
    logger.info(f"web_listen http://{host}:{port}/")
    uvicorn.run(app, host=host, port=int(port), log_level="warning")
