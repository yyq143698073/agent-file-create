"""Interactive routes — chat, clarify, satisfaction, versions, section, status, stream.

Extracted from web/server.py.
"""

import asyncio
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agent_file_create.chat.handler import ChatHandler
from agent_file_create.task.manager import TaskManager
from agent_file_create.web._utils import result_dir
from agent_file_create.web.routes.task import (
    _run_section_edit,
    _run_section_regen,
    _run_task,
    _start_task_thread,
    make_regenerate_fn,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["interact"])


# ── Clarify ──────────────────────────────────────────────────────────────────

@router.post("/api/clarify")
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


# ── Satisfaction ─────────────────────────────────────────────────────────────

@router.post("/api/satisfaction")
async def api_satisfaction(request: Request):
    """Handle user satisfaction feedback for outline, content, final, or quality_gate stage."""
    task_manager = TaskManager()
    body = await request.json()
    task_id = task_manager.normalize_task_id(str(body.get("task_id") or ""))
    if not task_id:
        raise HTTPException(400, "task_id 不能为空")

    stage = str(body.get("stage") or "").strip()  # "outline" | "content" | "final" | "quality"
    satisfied = bool(body.get("satisfied", True))
    feedback = str(body.get("feedback") or "").strip()
    scope = str(body.get("scope") or "outline").strip()  # "outline" | "content_only"
    selected_version = body.get("selected_version")

    if stage not in {"outline", "content", "final", "quality"}:
        raise HTTPException(400, "stage 必须是 outline、content、final 或 quality")

    current_st = task_manager.read_status(task_id)

    if stage == "final":
        extra_update: dict[str, Any] = {
            "final_satisfied": satisfied,
            "selected_version": selected_version,
        }
        stage_label = "最终确认"
        status_stage = "final_confirm"
        message = f"用户已完成最终确认，开始渲染报告…"
    elif stage == "quality":
        extra_update = {
            "quality_satisfied": satisfied,
            "satisfaction_feedback": feedback,
        }
        stage_label = "质量评估"
        status_stage = "quality_gate"
        message = f"用户选择{'开启' if satisfied else '跳过'}质量评估"
    else:
        extra_update = {
            f"{stage}_satisfied": satisfied,
            "satisfaction_feedback": feedback,
            "regeneration_scope": scope,
        }
        stage_label = "大纲" if stage == "outline" else "报告正文"
        status_stage = f"satisfaction_{stage}"
        if satisfied:
            message = f"用户对{stage_label}表示满意，继续生成…"
        else:
            message = f"用户对{stage_label}不满意，将重新生成（范围：{scope}）"

    task_manager.write_status(
        task_id, "processing",
        stage=status_stage,
        message=message,
        extra={**current_st, **extra_update},
    )

    return {"task_id": task_id, "ok": True, "satisfied": satisfied}


# ── Versions ─────────────────────────────────────────────────────────────────

@router.get("/api/versions")
def api_versions(task_id: str = Query(""), type: str = Query("outline")):
    """List versions of outline or content for a task."""
    task_manager = TaskManager()
    tid = task_manager.normalize_task_id(task_id)
    if not tid:
        raise HTTPException(400, "task_id 不能为空")
    if type not in {"outline", "content"}:
        raise HTTPException(400, "type 必须是 outline 或 content")
    versions = task_manager.list_versions(tid, type)
    return {"task_id": tid, "type": type, "versions": versions}


@router.post("/api/versions/select")
async def api_versions_select(request: Request):
    """Select a specific version as the final version for outline or content."""
    task_manager = TaskManager()
    body = await request.json()
    task_id = task_manager.normalize_task_id(str(body.get("task_id") or ""))
    if not task_id:
        raise HTTPException(400, "task_id 不能为空")
    vtype = str(body.get("type") or "outline").strip()
    if vtype not in {"outline", "content"}:
        raise HTTPException(400, "type 必须是 outline 或 content")
    try:
        version_num = int(body.get("version") or 0)
    except (ValueError, TypeError):
        raise HTTPException(400, "version 必须是整数")
    if version_num <= 0:
        raise HTTPException(400, "version 必须大于0")

    ok = task_manager.select_version(task_id, vtype, version_num)
    if not ok:
        raise HTTPException(404, f"版本 V{version_num} 不存在")

    return {"task_id": task_id, "type": vtype, "version": version_num, "ok": True}


@router.post("/api/versions/delete")
async def api_versions_delete(request: Request):
    task_manager = TaskManager()
    body = await request.json()
    task_id = task_manager.normalize_task_id(str(body.get("task_id") or ""))
    if not task_id:
        raise HTTPException(400, "task_id 不能为空")
    vtype = str(body.get("type") or "outline").strip()
    if vtype not in {"outline", "content"}:
        raise HTTPException(400, "type 必须是 outline 或 content")
    try:
        version_num = int(body.get("version") or 0)
    except (ValueError, TypeError):
        raise HTTPException(400, "version 必须是整数")
    if version_num <= 0:
        raise HTTPException(400, "version 必须大于0")

    # Check it's not the last remaining version
    versions = task_manager.list_versions(task_id, vtype)
    if len(versions) <= 1:
        raise HTTPException(400, "至少保留一个版本，无法删除")

    ok = task_manager.delete_version(task_id, vtype, version_num)
    return {"task_id": task_id, "type": vtype, "version": version_num, "ok": ok}


@router.post("/api/versions/redo")
async def api_versions_redo(request: Request):
    task_manager = TaskManager()
    body = await request.json()
    task_id = task_manager.normalize_task_id(str(body.get("task_id") or ""))
    if not task_id:
        raise HTTPException(400, "task_id 不能为空")
    vtype = str(body.get("type") or "outline").strip()
    if vtype not in {"outline", "content"}:
        raise HTTPException(400, "type 必须是 outline 或 content")
    try:
        base_version = int(body.get("base_version") or 0)
    except (ValueError, TypeError):
        raise HTTPException(400, "base_version 必须是整数")
    if base_version <= 0:
        raise HTTPException(400, "base_version 必须大于0")

    feedback = str(body.get("feedback") or "").strip()

    # Read task metadata
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
    try: target_words = int(meta.get("target_words") or 0)
    except Exception: target_words = 0

    # Build redo prompt with feedback and base version reference
    redo_prompt = user_prompt
    if feedback:
        redo_prompt += f"\n\n[用户改进意见]\n{feedback}"
    redo_prompt += f"\n\n[基于版本] V{base_version}"

    # Determine mode: redo from outline or content-only
    if vtype == "content":
        analysis_results = task_manager.read_analysis_results(task_id)
        if task_manager.is_task_running(task_id):
            raise HTTPException(400, "任务正在运行，请等待完成后再重做")
        ok, msg = _start_task_thread(
            task_id,
            user_prompt=redo_prompt,
            file_paths=file_paths,
            ab_eval=ab_eval,
            template_dir_override=template_dir_str,
            saved_templates=saved_templates,
            mode="document_only",
            target_words=target_words,
        )
    else:
        if task_manager.is_task_running(task_id):
            raise HTTPException(400, "任务正在运行，请等待完成后再重做")
        # Save the redo prompt to task meta so the agent picks it up
        task_manager.write_task_meta(task_id, {"user_prompt": redo_prompt, "redo_base_version": base_version, "redo_type": vtype})
        pause_ev, cancel_ev = task_manager.get_control_events(task_id)
        pause_ev.clear()
        cancel_ev.clear()
        th = threading.Thread(
            target=_run_task,
            args=(task_id, file_paths, redo_prompt),
            kwargs={"ab_eval": bool(ab_eval), "template_dir_override": template_dir_str, "saved_templates": saved_templates, "target_words": target_words},
            daemon=True,
        )
        task_manager.start_task(task_id, th.start)
        ok, msg = True, "已启动重新生成（全部流程）。"

    return {"task_id": task_id, "type": vtype, "base_version": base_version, "ok": ok, "message": msg}


# ── Version Cleanup ─────────────────────────────────────────────────────────

@router.post("/api/versions/clean")
async def api_versions_clean(request: Request):
    """Manually clean old versions for a task, keeping the latest N."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    task_id = str(body.get("task_id") or "").strip()
    if not task_id:
        raise HTTPException(400, "缺少 task_id")
    keep_last = int(body.get("keep_last") or 20)
    if keep_last < 3:
        keep_last = 3

    task_manager = TaskManager()
    deleted_total = 0
    for vt in ("outline", "content"):
        versions = task_manager.list_versions(task_id, vt)
        sorted_versions = sorted(versions, key=lambda x: x.get("version", 0))
        to_delete = sorted_versions[:-keep_last] if len(sorted_versions) > keep_last else []
        # Never delete v1
        to_delete = [v for v in to_delete if v.get("version") != 1]
        for v in to_delete:
            try:
                task_manager.delete_version(task_id, vt, v.get("version"))
                deleted_total += 1
            except Exception as e:
                logger.warning("version_clean_failed task=%s type=%s ver=%s err=%s",
                               task_id, vt, v.get("version"), e)

    return {"ok": True, "task_id": task_id, "deleted": deleted_total, "kept": keep_last}


# ── Section ──────────────────────────────────────────────────────────────────

def _save_section_direct(task_id: str, section_name: str, edited_content: str) -> tuple[bool, str]:
    """Directly save edited section content to content.md without AI rewrite."""
    try:
        base = result_dir() / task_id
        content_path = base / "content.md"
        if not content_path.exists():
            return False, "缺少正文文件"

        content = content_path.read_text(encoding="utf-8")
        search = section_name.strip()
        lines = content.splitlines()

        # Parse heading blocks
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
            return False, "无法解析正文结构"

        # Find matching section
        best_idx = -1
        for bi, blk in enumerate(blocks):
            if search == blk["heading"]:
                best_idx = bi
                break
            if search in blk["heading"] or blk["heading"] in search:
                best_idx = bi
                break

        if best_idx < 0:
            return False, f"未找到匹配章节「{search}」"

        target = blocks[best_idx]
        target_level = target["level"]

        # Determine line range to replace (body after heading + children)
        replace_start = target["start_idx"] + 1
        replace_end = target["end_idx"]
        for bi in range(best_idx + 1, len(blocks)):
            if blocks[bi]["level"] <= target_level:
                replace_end = blocks[bi]["start_idx"]
                break

        new_lines = lines[:replace_start] + edited_content.splitlines() + lines[replace_end:]
        content_path.write_text("\n".join(new_lines), encoding="utf-8")
        return True, f"已直接保存章节「{section_name}」"
    except Exception as e:
        logger.exception("section_save_direct_failed")
        return False, f"直接保存失败：{str(e)[:200]}"


@router.post("/api/section/save")
async def api_section_save(request: Request):
    """Save edited section content directly without AI rewrite."""
    task_manager = TaskManager()
    body = await request.json()
    task_id = task_manager.normalize_task_id(str(body.get("task_id") or ""))
    if not task_id:
        raise HTTPException(400, "task_id 不能为空")
    section_name = str(body.get("section_name") or "").strip()
    if not section_name:
        raise HTTPException(400, "section_name 不能为空")
    edited_content = str(body.get("edited_content") or "").strip()
    if not edited_content:
        raise HTTPException(400, "edited_content 不能为空")

    if task_manager.is_task_running(task_id):
        raise HTTPException(400, "任务正在运行，请等待完成后再保存")

    ok, msg = _save_section_direct(task_id, section_name, edited_content)
    return {"task_id": task_id, "section_name": section_name, "ok": ok, "message": msg}


@router.post("/api/section/edit")
async def api_section_edit(request: Request):
    """Edit a section: user provides edited content, LLM rewrites incorporating edits."""
    task_manager = TaskManager()
    body = await request.json()
    task_id = task_manager.normalize_task_id(str(body.get("task_id") or ""))
    if not task_id:
        raise HTTPException(400, "task_id 不能为空")
    section_name = str(body.get("section_name") or "").strip()
    if not section_name:
        raise HTTPException(400, "section_name 不能为空")
    edited_content = str(body.get("edited_content") or "").strip()
    if not edited_content:
        raise HTTPException(400, "edited_content 不能为空")

    if task_manager.is_task_running(task_id):
        raise HTTPException(400, "任务正在运行，请等待完成后再编辑")

    th = threading.Thread(
        target=_run_section_edit,
        args=(task_id, section_name, edited_content),
        daemon=True,
    )
    task_manager.start_task(task_id, th.start)
    return {"task_id": task_id, "section_name": section_name, "ok": True,
            "message": f"已启动章节「{section_name}」的编辑重写，请稍后查看预览。"}


# ── Chat routes ──────────────────────────────────────────────────────────────

@router.post("/api/chat")
async def api_chat(request: Request):
    """Chat endpoint — thin passthrough. All intent routing + clarify logic
    is handled inside ChatHandler.chat_reply()."""
    task_manager = TaskManager()
    body = await request.json()
    message = str(body.get("message") or "").strip()
    raw_tid = str(body.get("task_id") or "").strip()
    task_id = task_manager.normalize_task_id(raw_tid) if raw_tid else "lobby"
    history = body.get("history") if isinstance(body.get("history"), list) else []
    action = body.get("action")

    if not message and not action:
        raise HTTPException(400, "message 不能为空")
    if raw_tid and not task_id:
        raise HTTPException(400, "task_id 非法")

    quick_reply = ChatHandler._is_trivial_message(message)
    if quick_reply is not None and not action:
        task_manager.append_chat_history(task_id, [{"role": "user", "content": message}, {"role": "assistant", "content": quick_reply}])
        return {"task_id": task_id, "reply": quick_reply}

    chat_handler = ChatHandler(task_manager, regenerate_fn=make_regenerate_fn(task_manager))
    act = chat_handler._parse_chat_action(message, action)
    if not message and not act:
        raise HTTPException(400, "message 不能为空")
    if raw_tid and not task_id:
        raise HTTPException(400, "task_id 非法")

    if not history:
        history = task_manager.read_chat_history(task_id)

    if act:
        if not message:
            message = "/" + str(act.get("type") or "action")
        reply = chat_handler._handle_chat_action(task_id, act)
        task_manager.append_chat_history(task_id, [{"role": "user", "content": message}, {"role": "assistant", "content": reply}])
        return {"task_id": task_id, "reply": reply, "action": act}

    # All clarify / modify / question logic is now inside chat_reply()
    reply = chat_handler.chat_reply(message, task_id, history)
    return {"task_id": task_id, "reply": reply}


@router.post("/api/chat/stream")
async def api_chat_stream(request: Request):
    """Streaming chat endpoint — thin passthrough.

    All intent routing + clarify logic is handled inside
    ChatHandler.chat_reply_stream().
    """
    task_manager = TaskManager()
    body = await request.json()
    message = str(body.get("message") or "").strip()
    raw_tid = str(body.get("task_id") or "").strip()
    task_id = task_manager.normalize_task_id(raw_tid) if raw_tid else "lobby"
    history = body.get("history") if isinstance(body.get("history"), list) else []
    action = body.get("action")

    if not message and not action:
        raise HTTPException(400, "message 不能为空")
    if raw_tid and not task_id:
        raise HTTPException(400, "task_id 非法")

    quick_reply = ChatHandler._is_trivial_message(message)
    if quick_reply is not None and not action:
        task_manager.append_chat_history(task_id, [{"role": "user", "content": message}, {"role": "assistant", "content": quick_reply}])

        def sse_quick_reply():
            yield f"data: {json.dumps({'token': quick_reply}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_quick_reply(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                                          "X-Accel-Buffering": "no"})

    chat_handler = ChatHandler(task_manager, regenerate_fn=make_regenerate_fn(task_manager))
    act = chat_handler._parse_chat_action(message, action)
    if not message and not act:
        raise HTTPException(400, "message 不能为空")
    if raw_tid and not task_id:
        raise HTTPException(400, "task_id 非法")

    if not history:
        history = task_manager.read_chat_history(task_id)

    if act:
        if not message:
            message = "/" + str(act.get("type") or "action")
        reply = chat_handler._handle_chat_action(task_id, act)
        task_manager.append_chat_history(task_id, [{"role": "user", "content": message}, {"role": "assistant", "content": reply}])
        return {"task_id": task_id, "reply": reply, "action": act}

    # All clarify / modify / question logic is now inside chat_reply_stream()
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

    return StreamingResponse(sse_generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                                      "X-Accel-Buffering": "no"})


@router.get("/api/chat/history")
def chat_history(task_id: str = Query("")):
    task_manager = TaskManager()
    tid = task_manager.normalize_task_id(task_id) if task_id else "lobby"
    if task_id and not tid:
        raise HTTPException(400, "task_id 非法")
    return {"task_id": tid, "history": task_manager.read_chat_history(tid), "summary": task_manager.read_chat_summary(tid)}


@router.post("/api/chat/history/save")
async def save_chat_message(request: Request):
    """Save a single assistant or user message to the task's chat history."""
    body = await request.json()
    task_id = TaskManager().normalize_task_id(str(body.get("task_id") or ""))
    if not task_id:
        raise HTTPException(400, "task_id 不能为空")
    role = str(body.get("role") or "assistant")
    content = str(body.get("content") or "").strip()
    if role not in {"user", "assistant"} or not content:
        return {"ok": False}
    TaskManager().append_chat_history(task_id, [{"role": role, "content": content[:2000]}])
    return {"ok": True}


# ── Status / Stream / Tasks ──────────────────────────────────────────────────

@router.get("/api/status")
def api_status(task_id: str = Query("")):
    task_manager = TaskManager()
    tid = task_manager.normalize_task_id(task_id)
    if not tid:
        raise HTTPException(400, "task_id 不能为空")
    data = task_manager.read_status(tid)
    data["downloads"] = task_manager.collect_downloads(tid)
    return data


@router.get("/api/stream")
async def api_stream(task_id: str = Query("")):
    """SSE endpoint: streams section completion events as they happen."""
    if not task_id:
        raise HTTPException(400, "task_id 不能为空")

    stream_path = result_dir() / task_id / "stream.jsonl"

    async def event_generator():
        seen = 0
        # Clean old stream file
        if stream_path.exists():
            stream_path.unlink()
        while True:
            # Poll for new lines in stream file
            if stream_path.exists():
                lines = stream_path.read_text(encoding="utf-8").strip().split("\n")
                new_lines = [l for l in lines[seen:] if l.strip()]
                for line in new_lines:
                    yield f"data: {line}\n\n"
                    seen += 1
            # Check if task is done
            try:
                st = TaskManager().read_status(task_id)
                if str(st.get("status") or "") in ("finished", "canceled", "content_ready"):
                    yield f"data: {json.dumps({'type': 'done', 'status': st.get('status')}, ensure_ascii=False)}\n\n"
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)
        # Cleanup
        try:
            stream_path.unlink()
        except Exception:
            pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/api/tasks")
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
    base = result_dir()
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


# ── Gen from chat ───────────────────────────────────────────────────────────

@router.post("/api/gen")
async def api_gen(request: Request):
    """Trigger generation with a new prompt from chat, using existing task files."""
    body = await request.json()
    task_manager = TaskManager()
    task_id = task_manager.normalize_task_id(str(body.get("task_id") or ""))
    if not task_id:
        raise HTTPException(400, "task_id 不能为空")
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "prompt 不能为空")

    meta = task_manager.read_task_meta(task_id)
    file_paths = [str(x) for x in meta.get("file_paths") or [] if isinstance(x, str)]
    if not file_paths:
        raise HTTPException(400, "当前任务没有上传文件，请先在左侧面板上传材料")

    ab_eval = bool(meta.get("ab_eval"))
    saved_templates_raw = meta.get("saved_templates")
    saved_templates = [str(x) for x in saved_templates_raw] if isinstance(saved_templates_raw, list) else []
    template_dir_str = str(meta.get("template_dir") or "").strip() or None

    ok, msg = _start_task_thread(
        task_id, user_prompt=prompt, file_paths=file_paths,
        ab_eval=ab_eval, template_dir_override=template_dir_str,
        saved_templates=saved_templates, mode="all",
    )
    if not ok:
        raise HTTPException(409, msg)
    return {"task_id": task_id, "ok": True, "message": msg}
