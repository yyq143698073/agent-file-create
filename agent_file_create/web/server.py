import asyncio
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

import threading

from agent_file_create.chat.handler import ChatHandler
from agent_file_create.document.extractor import extract_from_file

# ── Task concurrency limiter ──
_MAX_CONCURRENT_TASKS = int(__import__("os").getenv("MAX_CONCURRENT_TASKS", "3"))
_task_semaphore = threading.BoundedSemaphore(_MAX_CONCURRENT_TASKS)
from agent_file_create.logging_config import setup_logging
from agent_file_create.preprocessor import compute_quality_metrics
from agent_file_create.rag.kb import KnowledgeBase
from agent_file_create.task.manager import TaskManager
from agent_file_create.web._kb_routes import init_kb_routes, router as kb_router

logger = logging.getLogger(__name__)

_kb_instance = None

def _get_kb():
    """Lazy-initialize KnowledgeBase singleton (replaces module-level global)."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance


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


def _run_task(task_id: str, file_paths: list[str], user_prompt: str, *, ab_eval: bool, template_dir_override: str | None, saved_templates: list[str] | None, target_words: int = 0) -> None:
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

    # ── Streaming extraction: collect results in background, feed to agent ──
    import threading as _threading
    analysis_lock = _threading.Lock()
    indexed_shared: list[tuple[int, dict, dict | None]] = []
    extraction_done = _threading.Event()
    total_files = len(file_paths)

    def _extract_all() -> None:
        nonlocal indexed_shared
        max_w = max(1, min(int(MAX_WORKERS_DEFAULT), total_files))
        with ThreadPoolExecutor(max_workers=max_w) as pool:
            fut_map = {pool.submit(_extract_one, fp): i for i, fp in enumerate(file_paths)}
            done = 0
            for fut in as_completed(fut_map):
                if cancel_ev.is_set():
                    for f in fut_map:
                        f.cancel()
                    break
                try:
                    _, ar, ab = fut.result()
                except Exception as e:
                    idx = fut_map[fut]
                    fn = Path(file_paths[idx]).name
                    ar = {"error": str(e), "_file": fn}
                    ab = {"file": fn, "error": str(e)[:240]} if ab_eval else None
                with analysis_lock:
                    indexed_shared.append((fut_map[fut], ar, ab))
                    done = len(indexed_shared)
                ab_snapshot = [x for _, _, x in indexed_shared if x is not None]
                task_manager.write_status(task_id, "processing", stage="extract",
                    message=f"已解析 {done}/{total_files} 个文件",
                    extra={"total_files": total_files, "done_files": done,
                           "ab_results": ab_snapshot[-20:], "saved_templates": st_names})
        extraction_done.set()

    # Launch background extraction
    extract_thread = _threading.Thread(target=_extract_all, daemon=True)
    extract_thread.start()

    # Wait for at least the first result before starting the agent
    while True:
        with analysis_lock:
            if len(indexed_shared) >= 1:
                break
        if cancel_ev.is_set():
            extraction_done.set()
            return
        time.sleep(0.2)

    def _get_latest_results() -> tuple[list[dict], list[dict]]:
        """Return (analysis_results, ab_results) snapshot under lock."""
        with analysis_lock:
            idx_sorted = sorted(indexed_shared, key=lambda x: x[0])
            ar_list = [ar for _, ar, _ in idx_sorted]
            ab_list = [ab for _, _, ab in idx_sorted if ab is not None]
        return ar_list, ab_list

    partial_results, ab_results = _get_latest_results()

    task_manager.write_analysis_results(task_id, partial_results)
    task_manager.write_task_meta(task_id, {"file_paths": list(file_paths), "user_prompt": str(user_prompt or ""), "ab_eval": bool(ab_eval), "saved_templates": st_names})

    if not _control("document"):
        return
    try:
        from agent_core import DocumentAgent

        # Reference to agent (set after creation) so _human_input can update its state
        _agent_ref: list = [None]

        def _human_input(question: str) -> str:
            if cancel_ev.is_set():
                task_manager.write_status(task_id, "canceled", stage="clarify", message="已取消", extra={"saved_templates": st_names})
                return ""

            q = (question or "").strip()

            # ── Refresh agent state with latest extraction results ──
            agent_obj = _agent_ref[0]
            if agent_obj is not None:
                latest_ar, latest_ab = _get_latest_results()
                if latest_ar:
                    agent_obj.state["analysis_results"] = latest_ar
                nonlocal ab_results
                ab_results = latest_ab

            # Detect clarify interrupt by stage prefix
            if q.startswith("[STAGE:clarify]"):
                q = q[len("[STAGE:clarify]"):].strip()

            # Detect final_confirm interrupt
            if q.startswith("[STAGE:final_confirm]"):
                import json as _json
                task_manager.write_status(
                    task_id, "need_user", stage="final_confirm",
                    message="请进行最终确认后渲染报告",
                    extra={"final_confirmed": None},
                )
                t0 = time.time()
                result = task_manager.wait_for_satisfaction(task_id, "final", timeout_s=1800)
                if cancel_ev.is_set():
                    return _json.dumps({"final_confirmed": True})
                return _json.dumps({"final_confirmed": bool(result.get("satisfied", True)), "selected_version": result.get("selected_version") or 1})

            # Detect satisfaction / quality_gate interrupts by stage prefix
            if q.startswith("[STAGE:satisfaction_outline]") or q.startswith("[STAGE:satisfaction_content]") or q.startswith("[STAGE:quality_gate]"):
                import json as _json
                is_quality = q.startswith("[STAGE:quality_gate]")
                is_outline = q.startswith("[STAGE:satisfaction_outline]")
                stage_name = "quality" if is_quality else ("outline" if is_outline else "content")
                scope_default = "outline" if is_outline else "content_only"
                label = "质量评估" if is_quality else ("大纲" if is_outline else "报告正文")

                if is_quality:
                    preview_text = ""
                    preview_version = 1
                    task_manager.write_status(
                        task_id, "need_user", stage="quality_gate",
                        message="报告已完成，是否进行质量评估？",
                        extra={"quality_satisfied": None},
                    )
                    t0 = time.time()
                    result = task_manager.wait_for_satisfaction(task_id, "quality", timeout_s=300)
                    if cancel_ev.is_set():
                        return _json.dumps({"satisfied": False, "feedback": "", "scope": ""})
                    elapsed = time.time() - t0
                    if elapsed >= 300 - 2:
                        return _json.dumps({"satisfied": False, "feedback": "超时未选择，跳过评估", "scope": ""})
                    return _json.dumps(result)
                else:
                    preview_text = ""
                    try:
                        idx1 = q.index("---")
                        idx2 = q.rindex("---")
                        if idx1 < idx2:
                            preview_text = q[idx1 + 3:idx2].strip()
                    except ValueError:
                        pass
                    import re as _re
                    preview_version = 1
                    vm = _re.search(r"当前版本[：:]\s*V(\d+)", q)
                    if vm:
                        try:
                            preview_version = int(vm.group(1))
                        except ValueError:
                            pass
                    task_manager.write_status(
                        task_id, "need_user", stage=f"satisfaction_{stage_name}",
                        message=f"{label}生成完成，请审阅并选择是否满意。",
                        extra={f"{stage_name}_satisfied": None, "satisfaction_feedback": "",
                               "regeneration_scope": scope_default,
                               "preview_text": preview_text,
                               "preview_version": preview_version,
                               "ab_eval": bool(ab_eval), "ab_results": ab_results[-20:],
                               "saved_templates": st_names},
                    )
                    result = task_manager.wait_for_satisfaction(task_id, stage_name, timeout_s=1800)
                    if cancel_ev.is_set():
                        return _json.dumps({"satisfied": True, "feedback": "", "scope": scope_default})
                    return _json.dumps(result)

            # Regular clarify interrupt (use cleaned q instead of raw question)
            qs = _split_questions(q)
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
        agent.state["analysis_results"] = partial_results
        agent.state["force_regen"] = False
        agent.state["target_words"] = int(target_words or 0)
        _agent_ref[0] = agent
        state = agent.run(max_turns=8, human_input_fn=_human_input)
        _agent_ref[0] = None

        # Wait for background extraction to complete (should already be done)
        extraction_done.wait(timeout=10)
        # Persist final results to disk
        final_results, final_ab = _get_latest_results()
        if final_results:
            task_manager.write_analysis_results(task_id, final_results)

        if cancel_ev.is_set():
            task_manager.write_status(task_id, "canceled", stage="done", message="已取消", extra={"saved_templates": st_names})
        else:
            extra: dict[str, Any] = {"result": {"output_dir": state.get("output_dir")}, "saved_templates": st_names, "eval_metrics": state.get("eval_metrics", {}), "warnings": state.get("warnings", []), "warnings_count": state.get("warnings_count", 0)}
            if state.get("eval_report"):
                extra["eval"] = state["eval_report"]
            task_manager.write_status(task_id, "finished", stage="done", message="生成完成", extra=extra)
    except Exception as e:
        task_manager.write_status(task_id, "failed", stage="done", message="生成失败", extra={"error": str(e)[:400]})


def _run_document_only(task_id: str, *, user_prompt: str, file_paths: list[str], analysis_results: list[dict], template_dir_override: str | None, saved_templates: list[str] | None, ab_eval: bool, target_words: int = 0) -> None:
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

        def _auto_approve(question: str) -> str:
            """Auto-approve satisfaction interrupts during redo/regeneration.
            For quality_gate, prompt user to decide (same as initial generation).
            """
            import json as _json
            q = str(question or "")
            if q.startswith("[STAGE:quality_gate]"):
                task_manager.write_status(
                    task_id, "need_user", stage="quality_gate",
                    message="报告已完成，是否进行质量评估？",
                    extra={"quality_satisfied": None},
                )
                t0 = time.time()
                result = task_manager.wait_for_satisfaction(task_id, "quality", timeout_s=300)
                if cancel_ev.is_set():
                    return _json.dumps({"satisfied": False, "feedback": "", "scope": ""})
                elapsed = time.time() - t0
                if elapsed >= 300 - 2:
                    return _json.dumps({"satisfied": False, "feedback": "超时未选择，跳过评估", "scope": ""})
                return _json.dumps(result)
            if "[STAGE:satisfaction_" in q:
                scope = "outline" if "outline" in q.split("\n")[0].lower() else "content_only"
                return _json.dumps({"satisfied": True, "feedback": "", "scope": scope})
            return "已收到，请继续。"

        task_manager.write_status(task_id, "processing", stage="document", message="开始重新生成文档…", extra={"saved_templates": st_names, "ab_eval": bool(ab_eval)})
        agent = DocumentAgent(task_id=task_id, user_prompt=user_prompt, file_paths=file_paths, template_dir_override=template_dir_override)
        agent.state["analysis_results"] = list(analysis_results or [])
        agent.state["force_regen"] = True
        agent.state["target_words"] = int(target_words or 0)
        state = agent.run(max_turns=8, human_input_fn=_auto_approve)
        if cancel_ev.is_set():
            task_manager.write_status(task_id, "canceled", stage="done", message="已取消", extra={"saved_templates": st_names})
        else:
            extra: dict[str, Any] = {"result": {"output_dir": state.get("output_dir")}, "saved_templates": st_names, "eval_metrics": state.get("eval_metrics", {}), "warnings": state.get("warnings", []), "warnings_count": state.get("warnings_count", 0)}
            if state.get("eval_report"):
                extra["eval"] = state["eval_report"]
            task_manager.write_status(task_id, "finished", stage="done", message="重新生成完成", extra=extra)
    except Exception as e:
        task_manager.write_status(task_id, "failed", stage="done", message="重新生成失败", extra={"error": str(e)[:400]})


def _start_task_thread(task_id: str, *, user_prompt: str, file_paths: list[str], ab_eval: bool, template_dir_override: str | None, saved_templates: list[str] | None, mode: str, target_words: int = 0) -> tuple[bool, str]:
    task_manager = TaskManager()
    if task_manager.is_task_running(task_id):
        return False, "任务正在运行，无法启动新的生成。可先 /pause 或 /cancel。"

    # Concurrency gate
    acquired = _task_semaphore.acquire(blocking=False)
    if not acquired:
        return False, f"系统繁忙（当前最多 {_MAX_CONCURRENT_TASKS} 个任务并行），请稍后重试。"

    def _release_on_done(target, *a, **kw):
        try:
            target(*a, **kw)
        finally:
            _task_semaphore.release()

    pause_ev, cancel_ev = task_manager.get_control_events(task_id)
    pause_ev.clear()
    cancel_ev.clear()
    if mode == "document_only":
        analysis_results = task_manager.read_analysis_results(task_id)
        if not analysis_results:
            mode = "all"
        else:
            import threading
            th = threading.Thread(
                target=_release_on_done, args=(_run_document_only,),
                kwargs={"task_id": task_id, "user_prompt": user_prompt, "file_paths": file_paths, "analysis_results": analysis_results, "template_dir_override": template_dir_override, "saved_templates": saved_templates, "ab_eval": bool(ab_eval)},
                daemon=True)
            task_manager.start_task(task_id, th.start)
            return True, "已启动重新生成（仅文档阶段）。"
    import threading
    th = threading.Thread(
        target=_release_on_done, args=(_run_task, task_id, file_paths, user_prompt),
        kwargs={"ab_eval": bool(ab_eval), "template_dir_override": template_dir_override, "saved_templates": saved_templates, "target_words": int(target_words or 0)},
        daemon=True)
    task_manager.start_task(task_id, th.start)
    return True, "已启动生成任务。"


def _run_section_regen(task_id: str, section_name: str, feedback: str = "") -> tuple[bool, str]:
    """Regenerate a single section in the background."""
    try:
        from agent_file_create.document.content_generator import regenerate_section
        from agent_file_create.document_service import render_document

        task_manager = TaskManager()
        pause_ev, cancel_ev = task_manager.get_control_events(task_id)
        meta = task_manager.read_task_meta(task_id)
        user_prompt = str(meta.get("user_prompt") or "").strip() or "生成一份报告"
        analysis_results = task_manager.read_analysis_results(task_id)

        current_st = task_manager.read_status(task_id)
        current_stage = str(current_st.get("stage") or "").strip()
        is_final_confirm = current_stage == "final_confirm"

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
            guidance=feedback.strip(),
        )

        if not new_content:
            if is_final_confirm:
                clean_extra_final = {k: v for k, v in current_st.items() if k not in {
                    "clarify_questions", "clarify_answers", "clarify_skip", "clarify_submitted_at",
                    "content_satisfied", "outline_satisfied", "satisfaction_feedback",
                    "regeneration_scope", "preview_text", "preview_version", "is_section_regen",
                }}
                clean_extra_final["final_confirmed"] = None
                task_manager.write_status(
                    task_id, "need_user", stage="final_confirm",
                    message=f"未找到匹配章节「{section_name}」。请进行最终确认。",
                    extra=clean_extra_final,
                )
            else:
                task_manager.write_status(task_id, "finished", stage="document",
                                          message=f"未找到匹配章节「{section_name}」，请检查章节标题是否正确。")
            return False, f"未找到匹配章节「{section_name}」。可尝试 /templates 或 /files 查看章节标题，使用精确标题重新生成。"

        content_path.write_text(new_content, encoding="utf-8")

        if is_final_confirm:
            clean_extra_final = {k: v for k, v in current_st.items() if k not in {
                "clarify_questions", "clarify_answers", "clarify_skip", "clarify_submitted_at",
                "content_satisfied", "outline_satisfied", "satisfaction_feedback",
                "regeneration_scope", "preview_text", "preview_version", "is_section_regen",
            }}
            clean_extra_final["final_confirmed"] = None
            task_manager.write_status(
                task_id, "need_user", stage="final_confirm",
                message=f"章节「{section_name}」已重新生成，请进行最终确认。",
                extra=clean_extra_final,
            )
            return True, f"章节「{section_name}」已重新生成。"

        clean_extra = {k: v for k, v in current_st.items() if k not in {
            "clarify_questions", "clarify_answers", "clarify_skip", "clarify_submitted_at",
            "outline_satisfied", "quality_wanted",
        }}
        clean_extra.update({
            "content_satisfied": None,
            "satisfaction_feedback": "",
            "regeneration_scope": "content_only",
            "preview_text": new_content[:3000],
            "preview_version": 1,
            "is_section_regen": True,
        })
        task_manager.write_status(
            task_id, "need_user", stage="satisfaction_content",
            message=f"章节「{section_name}」已重新生成，请审阅并确认是否为最终版本。",
            extra=clean_extra,
        )
        result = task_manager.wait_for_satisfaction(task_id, "content", timeout_s=1800)
        if cancel_ev.is_set():
            return False, "已取消"
        satisfied = bool(result.get("satisfied", True))

        if not satisfied:
            task_manager.write_status(
                task_id, "finished", stage="document",
                message=f"用户对章节「{section_name}」不满意，未渲染最终报告。",
            )
            return True, f"章节「{section_name}」已重新生成，但用户选择不渲染。"

        task_manager.write_status(task_id, "processing", stage="render",
                                  message="正在渲染最终报告…")
        output_dir = str(base)
        template_dir = str(base / "template")
        rendered = render_document(
            task_id=task_id,
            content=new_content,
            outline=outline,
            output_dir=output_dir,
            template_dir=template_dir,
        )

        current_st_qg = task_manager.read_status(task_id)
        clean_extra_qg = {k: v for k, v in current_st_qg.items() if k not in {
            "clarify_questions", "clarify_answers", "clarify_skip", "clarify_submitted_at",
            "content_satisfied", "outline_satisfied", "satisfaction_feedback",
            "regeneration_scope", "preview_text", "preview_version", "is_section_regen",
        }}
        clean_extra_qg["quality_satisfied"] = None
        task_manager.write_status(
            task_id, "need_user", stage="quality_gate",
            message="报告已渲染完成，是否进行质量评估？",
            extra=clean_extra_qg,
        )
        t0 = time.time()
        q_result = task_manager.wait_for_satisfaction(task_id, "quality", timeout_s=300)
        if cancel_ev.is_set():
            task_manager.write_status(task_id, "canceled", stage="done", message="已取消")
            return False, "已取消"
        elapsed = time.time() - t0
        want_eval = bool(q_result.get("satisfied", False)) if elapsed < 298 else False

        extra: dict[str, Any] = {"result": {"output_dir": output_dir}, "eval_metrics": {}, "warnings": [], "warnings_count": 0}
        if want_eval:
            try:
                from agent_file_create.document_service import _run_faithfulness_checks
                from agent_file_create.evaluation.orchestrator import evaluate as run_eval
                task_manager.write_status(task_id, "processing", stage="quality_gate",
                                          message="正在进行质量评估…")
                final_content = _run_faithfulness_checks(
                    content=new_content, analysis_results=analysis_results,
                    task_id=task_id, output_dir=output_dir,
                )
                eval_report = run_eval(
                    content=final_content or new_content,
                    outline=outline,
                    analysis_results=analysis_results,
                    user_prompt=user_prompt,
                )
                extra["eval_metrics"] = eval_report.to_dict()
                if final_content and final_content != new_content:
                    content_path.write_text(final_content, encoding="utf-8")
            except Exception as e:
                logger.warning("section_regen_quality_gate_failed err=%s", e)

        task_manager.write_status(task_id, "finished", stage="done",
                                  message="生成完成", extra=extra)
        return True, f"章节「{section_name}」已重新生成并渲染完成。"
    except Exception as e:
        logger.exception("section_regen_failed")
        try:
            current_st = TaskManager().read_status(task_id)
            current_stage = str(current_st.get("stage") or "").strip()
            is_final_confirm = current_stage == "final_confirm"
            if is_final_confirm:
                clean_extra_final = {k: v for k, v in current_st.items() if k not in {
                    "clarify_questions", "clarify_answers", "clarify_skip", "clarify_submitted_at",
                    "content_satisfied", "outline_satisfied", "satisfaction_feedback",
                    "regeneration_scope", "preview_text", "preview_version", "is_section_regen",
                }}
                clean_extra_final["final_confirmed"] = None
                TaskManager().write_status(
                    task_id, "need_user", stage="final_confirm",
                    message=f"章节重生成失败，请进行最终确认。",
                    extra=clean_extra_final,
                )
            else:
                TaskManager().write_status(task_id, "finished", stage="document",
                                           message=f"章节重生成失败：{str(e)[:120]}")
        except Exception:
            pass
        return False, f"章节生成失败：{str(e)[:200]}"


def _run_section_edit(task_id: str, section_name: str, edited_content: str) -> tuple[bool, str]:
    """Rewrite a section guided by user-edited content."""
    try:
        from agent_file_create.document.content_generator import regenerate_section
        from agent_file_create.document_service import render_document

        task_manager = TaskManager()
        pause_ev, cancel_ev = task_manager.get_control_events(task_id)
        meta = task_manager.read_task_meta(task_id)
        user_prompt = str(meta.get("user_prompt") or "").strip() or "生成一份报告"
        analysis_results = task_manager.read_analysis_results(task_id)

        current_st = task_manager.read_status(task_id)
        current_stage = str(current_st.get("stage") or "").strip()
        is_final_confirm = current_stage == "final_confirm"

        base = _result_dir() / task_id
        outline_path = base / "outline.md"
        content_path = base / "content.md"
        outline = outline_path.read_text(encoding="utf-8") if outline_path.exists() else ""
        content = content_path.read_text(encoding="utf-8") if content_path.exists() else ""

        if not outline or not content:
            return False, "缺少大纲或正文，无法定位章节。请先确保任务已生成完成。"

        multimodal = {f"source_{i}": r for i, r in enumerate(analysis_results)} if analysis_results else {}

        guidance = (
            "用户提供了以下编辑后的版本作为基础，请在此基础上进行润色和完善，"
            "保留用户的核心修改意图，优化表达、补充细节、增强连贯性：\n\n"
            + edited_content
        )

        task_manager.write_status(task_id, "processing", stage="document",
                                  message=f"正在根据编辑内容重写章节「{section_name}」…")

        new_content = regenerate_section(
            outline, content, section_name, multimodal, user_prompt,
            task_id=task_id, guidance=guidance,
        )

        if not new_content:
            if is_final_confirm:
                clean_extra_final = {k: v for k, v in current_st.items() if k not in {
                    "clarify_questions", "clarify_answers", "clarify_skip", "clarify_submitted_at",
                    "content_satisfied", "outline_satisfied", "satisfaction_feedback",
                    "regeneration_scope", "preview_text", "preview_version", "is_section_regen",
                }}
                clean_extra_final["final_confirmed"] = None
                task_manager.write_status(
                    task_id, "need_user", stage="final_confirm",
                    message=f"未找到匹配章节「{section_name}」。请进行最终确认。",
                    extra=clean_extra_final,
                )
            else:
                task_manager.write_status(task_id, "finished", stage="document",
                                          message=f"未找到匹配章节「{section_name}」，请检查章节标题是否正确。")
            return False, f"未找到匹配章节「{section_name}」。"

        content_path.write_text(new_content, encoding="utf-8")

        if is_final_confirm:
            clean_extra_final = {k: v for k, v in current_st.items() if k not in {
                "clarify_questions", "clarify_answers", "clarify_skip", "clarify_submitted_at",
                "content_satisfied", "outline_satisfied", "satisfaction_feedback",
                "regeneration_scope", "preview_text", "preview_version", "is_section_regen",
            }}
            clean_extra_final["final_confirmed"] = None
            task_manager.write_status(
                task_id, "need_user", stage="final_confirm",
                message=f"章节「{section_name}」已根据编辑内容重写，请进行最终确认。",
                extra=clean_extra_final,
            )
            return True, f"章节「{section_name}」已重写。"

        clean_extra = {k: v for k, v in current_st.items() if k not in {
            "clarify_questions", "clarify_answers", "clarify_skip", "clarify_submitted_at",
            "outline_satisfied", "quality_wanted",
        }}
        clean_extra.update({
            "content_satisfied": None,
            "satisfaction_feedback": "",
            "regeneration_scope": "content_only",
            "preview_text": new_content[:3000],
            "preview_version": 1,
            "is_section_regen": True,
        })
        task_manager.write_status(
            task_id, "need_user", stage="satisfaction_content",
            message=f"章节「{section_name}」已根据编辑内容重写，请审阅并确认是否为最终版本。",
            extra=clean_extra,
        )
        result = task_manager.wait_for_satisfaction(task_id, "content", timeout_s=1800)
        if cancel_ev.is_set():
            return False, "已取消"
        satisfied = bool(result.get("satisfied", True))

        if not satisfied:
            task_manager.write_status(
                task_id, "finished", stage="document",
                message=f"用户对章节「{section_name}」不满意，未渲染最终报告。",
            )
            return True, f"章节「{section_name}」已重写，但用户选择不渲染。"

        task_manager.write_status(task_id, "processing", stage="render",
                                  message="正在渲染最终报告…")
        output_dir = str(base)
        template_dir = str(base / "template")
        rendered = render_document(
            task_id=task_id,
            content=new_content,
            outline=outline,
            output_dir=output_dir,
            template_dir=template_dir,
        )

        current_st_qg = task_manager.read_status(task_id)
        clean_extra_qg = {k: v for k, v in current_st_qg.items() if k not in {
            "clarify_questions", "clarify_answers", "clarify_skip", "clarify_submitted_at",
            "content_satisfied", "outline_satisfied", "satisfaction_feedback",
            "regeneration_scope", "preview_text", "preview_version", "is_section_regen",
        }}
        clean_extra_qg["quality_satisfied"] = None
        task_manager.write_status(
            task_id, "need_user", stage="quality_gate",
            message="报告已渲染完成，是否进行质量评估？",
            extra=clean_extra_qg,
        )
        t0 = time.time()
        q_result = task_manager.wait_for_satisfaction(task_id, "quality", timeout_s=300)
        if cancel_ev.is_set():
            task_manager.write_status(task_id, "canceled", stage="done", message="已取消")
            return False, "已取消"
        elapsed = time.time() - t0
        want_eval = bool(q_result.get("satisfied", False)) if elapsed < 298 else False

        extra: dict[str, Any] = {"result": {"output_dir": output_dir}, "eval_metrics": {}, "warnings": [], "warnings_count": 0}
        if want_eval:
            try:
                from agent_file_create.document_service import _run_faithfulness_checks
                from agent_file_create.evaluation.orchestrator import evaluate as run_eval
                task_manager.write_status(task_id, "processing", stage="quality_gate",
                                          message="正在进行质量评估…")
                final_content = _run_faithfulness_checks(
                    content=new_content, analysis_results=analysis_results,
                    task_id=task_id, output_dir=output_dir,
                )
                eval_report = run_eval(
                    content=final_content or new_content,
                    outline=outline,
                    analysis_results=analysis_results,
                    user_prompt=user_prompt,
                )
                extra["eval_metrics"] = eval_report.to_dict()
                if final_content and final_content != new_content:
                    content_path.write_text(final_content, encoding="utf-8")
            except Exception as e:
                logger.warning("section_edit_quality_gate_failed err=%s", e)

        task_manager.write_status(task_id, "finished", stage="done",
                                  message="生成完成", extra=extra)
        return True, f"章节「{section_name}」已重写并渲染完成。"
    except Exception as e:
        logger.exception("section_edit_failed")
        try:
            current_st = TaskManager().read_status(task_id)
            current_stage = str(current_st.get("stage") or "").strip()
            is_final_confirm = current_stage == "final_confirm"
            if is_final_confirm:
                clean_extra_final = {k: v for k, v in current_st.items() if k not in {
                    "clarify_questions", "clarify_answers", "clarify_skip", "clarify_submitted_at",
                    "content_satisfied", "outline_satisfied", "satisfaction_feedback",
                    "regeneration_scope", "preview_text", "preview_version", "is_section_regen",
                }}
                clean_extra_final["final_confirmed"] = None
                TaskManager().write_status(
                    task_id, "need_user", stage="final_confirm",
                    message=f"章节编辑重写失败，请进行最终确认。",
                    extra=clean_extra_final,
                )
            else:
                TaskManager().write_status(task_id, "finished", stage="document",
                                           message=f"章节编辑重写失败：{str(e)[:120]}")
        except Exception:
            pass
        return False, f"章节编辑重写失败：{str(e)[:200]}"


def _make_regenerate_fn(task_manager: TaskManager):
    def _fn(task_id: str, mode: str = "doc", section_name: str = "", feedback: str = "") -> tuple[bool, str]:
        # Single-section regeneration
        if section_name and section_name.strip():
            import threading
            th = threading.Thread(target=_run_section_regen, args=(task_id, section_name.strip(), feedback.strip()), daemon=True)
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

# ── Include modular route handlers ──
from agent_file_create.web._kb_routes import router as kb_router, init_kb_routes

# Initialize KB routes with the server's KB getter
init_kb_routes(_get_kb)
app.include_router(kb_router)

# Mount static file directories
_html = _html_dir()
_result = _result_dir()

if _html.exists():
    app.mount("/static", StaticFiles(directory=str(_html)), name="static")


# KB routes moved to web/_kb_routes.py

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
                text = _get_kb().get_doc_text(kb=kbn, doc_id=did)
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


@app.post("/api/append")
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
                text = _get_kb().get_doc_text(kb=kbn, doc_id=did)
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

    task_manager.write_task_meta(task_id, {"uploads_dir": str(uploads), "template_dir": str(user_template_dir), "file_paths": list(all_files), "user_prompt": user_prompt_val, "ab_eval": bool(ab_val), "template_mode": template_mode, "active_kb": str(meta.get("active_kb") or "").strip()})
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


@app.post("/api/satisfaction")
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


@app.get("/api/versions")
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


@app.post("/api/versions/select")
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


@app.post("/api/versions/delete")
async def api_versions_delete(request: Request):
    """Delete a specific version."""
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


def _save_section_direct(task_id: str, section_name: str, edited_content: str) -> tuple[bool, str]:
    """Directly save edited section content to content.md without AI rewrite."""
    import re
    try:
        base = _result_dir() / task_id
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


@app.post("/api/section/save")
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


@app.post("/api/section/edit")
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

    import threading
    th = threading.Thread(
        target=_run_section_edit,
        args=(task_id, section_name, edited_content),
        daemon=True,
    )
    task_manager.start_task(task_id, th.start)
    return {"task_id": task_id, "section_name": section_name, "ok": True,
            "message": f"已启动章节「{section_name}」的编辑重写，请稍后查看预览。"}


@app.post("/api/versions/redo")
async def api_versions_redo(request: Request):
    """Redo generation from a specific version with user feedback."""
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
        import threading
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


# ── Template CRUD ──────────────────────────────────────────────────────────

_CUSTOM_TEMPLATE_DIR = _result_dir() / "template" / "custom"


@app.get("/api/template/variables")
def api_template_variables():
    """Return metadata about all available template placeholder variables."""
    return {
        "system_variables": [
            {"key": "title", "label": "文档标题", "description": "从大纲自动提取的文档主标题（# 开头）"},
            {"key": "task_id", "label": "任务ID", "description": "当前任务的8位唯一标识符"},
            {"key": "document_outline", "label": "文档大纲", "description": "完整的 Markdown 大纲内容"},
            {"key": "document_content", "label": "文档正文", "description": "完整的 Markdown 正文内容"},
        ],
        "section_variables_note": "章节级变量（如 {{背景分析}}、{{核心内容}}）由大纲的 ## 二级标题动态生成。你可以在模板中预先写入预期的章节变量名，生成时系统会自动匹配替换。",
    }


@app.get("/api/template/custom/list")
def api_template_custom_list():
    """List all user-created custom templates with metadata."""
    try:
        _CUSTOM_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    items = []
    if _CUSTOM_TEMPLATE_DIR.exists():
        from agent_file_create.document.template_renderer import _scan_md_placeholders
        for p in sorted(_CUSTOM_TEMPLATE_DIR.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                vars_set = _scan_md_placeholders(str(p))
                items.append({
                    "name": p.name,
                    "stripped": p.stem,
                    "size": p.stat().st_size,
                    "modified_at": p.stat().st_mtime,
                    "variable_count": len(vars_set),
                })
            except Exception:
                pass
    return {"templates": items}


@app.get("/api/template/custom/{name:path}")
def api_template_custom_get(name: str):
    """Get content of a single custom template by filename."""
    safe = _sanitize_filename(name)
    fp = _CUSTOM_TEMPLATE_DIR / safe
    if not fp.exists():
        raise HTTPException(404, f"模板不存在: {safe}")
    from agent_file_create.document.template_renderer import _scan_md_placeholders
    content = fp.read_text(encoding="utf-8")
    return {
        "name": safe,
        "content": content,
        "variables": sorted(_scan_md_placeholders(str(fp))),
    }


@app.post("/api/template/custom/save")
async def api_template_custom_save(request: Request):
    """Create or update a custom template."""
    body = await request.json()
    name = str(body.get("name") or "").strip()
    content = str(body.get("content") or "")
    if not name:
        raise HTTPException(400, "模板名称不能为空")
    safe = _sanitize_filename(name)
    if not safe.lower().endswith(".md"):
        safe += ".md"
    try:
        _CUSTOM_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    fp = _CUSTOM_TEMPLATE_DIR / safe
    try:
        fp.write_text(content, encoding="utf-8")
    except Exception as e:
        raise HTTPException(500, f"保存失败: {str(e)[:200]}")
    return {"name": safe, "ok": True}


@app.delete("/api/template/custom/{name:path}")
def api_template_custom_delete(name: str):
    """Delete a custom template."""
    safe = _sanitize_filename(name)
    fp = _CUSTOM_TEMPLATE_DIR / safe
    if not fp.exists():
        raise HTTPException(404, f"模板不存在: {safe}")
    try:
        fp.unlink()
    except Exception as e:
        raise HTTPException(500, f"删除失败: {str(e)[:200]}")
    return {"name": safe, "ok": True}


@app.post("/api/template/custom/use")
async def api_template_custom_use(request: Request):
    """Copy a custom template to a task's template directory."""
    body = await request.json()
    name = str(body.get("name") or "").strip()
    task_id = str(body.get("task_id") or "").strip()
    if not name or not task_id:
        raise HTTPException(400, "name 和 task_id 不能为空")
    safe = _sanitize_filename(name)
    if not safe.lower().endswith(".md"):
        safe += ".md"
    src = _CUSTOM_TEMPLATE_DIR / safe
    if not src.exists():
        raise HTTPException(404, f"模板不存在: {safe}")
    dest_dir = _result_dir() / task_id / "template"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    dst = dest_dir / safe
    try:
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception as e:
        raise HTTPException(500, f"复制失败: {str(e)[:200]}")
    return {"name": safe, "task_id": task_id, "ok": True}


def _is_waiting_clarify(st: dict) -> bool:
    """Check if task is waiting for a clarify answer, even if stage was
    overwritten by a racing background thread."""
    if str(st.get("status") or "") == "need_user" and str(st.get("stage") or "") == "clarify":
        return True
    # Fallback: clarify questions exist and no answer has been submitted yet
    has_questions = isinstance(st.get("clarify_questions"), list) and st.get("clarify_questions")
    has_answer = bool(str(st.get("clarify_answers") or "").strip()) or bool(st.get("clarify_skip"))
    return has_questions and not has_answer


def _handle_clarify_answer(task_manager: TaskManager, chat_handler: ChatHandler,
                           task_id: str, message: str, st: dict) -> str | None:
    """Process clarify answer. Returns reply string, or None if not a clarify answer."""
    m = message.strip()
    low = m.lower()
    if low in {"skip", "跳过", "略过", "不用了", "不需要"}:
        task_manager.write_status(task_id, "processing", stage="clarify",
            message="用户选择跳过澄清，继续生成…",
            extra={"clarify_answers": "", "clarify_skip": True,
                   "clarify_submitted_at": float(time.time())})
        return "已收到：跳过澄清。我会继续生成；如结果不符合预期，可再补充信息让我调整。"
    else:
        qs = st.get("clarify_questions") if isinstance(st.get("clarify_questions"), list) else []
        is_valid, warning = chat_handler.validate_clarify_answer(m, qs)
        if not is_valid:
            return None  # caller should return warning to user
        task_manager.write_status(task_id, "processing", stage="clarify",
            message="已收到用户补充信息，继续生成…",
            extra={"clarify_answers": m, "clarify_skip": False,
                   "clarify_submitted_at": float(time.time())})
        return "已收到补充信息。我会继续生成文档；你也可以继续提问或补充更多要求。"


# ── Built-in template marketplace ─────────────────────────────────────────

_BUILTIN_TEMPLATES_DIR = _get_base_dir() / "resource" / "templates"


@app.get("/api/template/builtin/list")
def api_template_builtin_list():
    """List all built-in report templates with metadata."""
    items = []
    if _BUILTIN_TEMPLATES_DIR.exists():
        for md_path in sorted(_BUILTIN_TEMPLATES_DIR.glob("*.md")):
            meta_path = md_path.with_suffix("").with_name(md_path.stem + "_meta.json")
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            try:
                from agent_file_create.document.template_renderer import _scan_md_placeholders
                vars_set = _scan_md_placeholders(str(md_path))
            except Exception:
                vars_set = set()
            items.append({
                "name": md_path.name,
                "id": md_path.stem,
                "size": md_path.stat().st_size,
                "variables": sorted(vars_set),
                "description": meta.get("description", ""),
                "category": meta.get("category", "通用"),
                "suggested_prompt": meta.get("suggested_prompt", ""),
            })
    return {"templates": items}


@app.get("/api/template/builtin/{name}")
def api_template_builtin_get(name: str):
    """Get a single built-in template by name."""
    safe = _sanitize_filename(name)
    if not safe.endswith(".md"):
        safe += ".md"
    fp = _BUILTIN_TEMPLATES_DIR / safe
    if not fp.exists():
        raise HTTPException(404, f"内置模板不存在: {safe}")
    meta = {}
    meta_path = _BUILTIN_TEMPLATES_DIR / (fp.stem + "_meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    content = fp.read_text(encoding="utf-8")
    from agent_file_create.document.template_renderer import _scan_md_placeholders
    return {
        "name": safe,
        "content": content,
        "variables": sorted(_scan_md_placeholders(str(fp))),
        "description": meta.get("description", ""),
        "category": meta.get("category", "通用"),
        "suggested_prompt": meta.get("suggested_prompt", ""),
    }


@app.post("/api/template/builtin/use")
async def api_template_builtin_use(request: Request):
    """Copy a built-in template to a task for use."""
    body = await request.json()
    task_id = str(body.get("task_id") or "").strip()
    name = str(body.get("name") or "").strip()
    if not task_id or not name:
        raise HTTPException(400, "task_id 和 name 不能为空")
    safe = _sanitize_filename(name)
    if not safe.endswith(".md"):
        safe += ".md"
    src = _BUILTIN_TEMPLATES_DIR / safe
    if not src.exists():
        raise HTTPException(404, f"内置模板不存在: {safe}")
    dest_dir = _result_dir() / task_id / "template"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / safe
    dst.write_bytes(src.read_bytes())
    return {"name": safe, "task_id": task_id, "ok": True}


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

    # Check if waiting for clarify answer (robust against stage race condition)
    if _is_waiting_clarify(st):
        reply = _handle_clarify_answer(task_manager, chat_handler, task_id, message, st)
        if reply is None:
            return {"task_id": task_id, "reply": chat_handler.validate_clarify_answer(message, st.get("clarify_questions") or [])[1]}
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

    # Check if waiting for clarify answer (robust against stage race condition)
    if _is_waiting_clarify(st):
        reply = _handle_clarify_answer(task_manager, chat_handler, task_id, message, st)
        if reply is None:
            return {"task_id": task_id, "reply": chat_handler.validate_clarify_answer(message, st.get("clarify_questions") or [])[1]}
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

    return StreamingResponse(sse_generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/chat/history")
def chat_history(task_id: str = Query("")):
    task_manager = TaskManager()
    tid = task_manager.normalize_task_id(task_id) if task_id else "lobby"
    if task_id and not tid:
        raise HTTPException(400, "task_id 非法")
    return {"task_id": tid, "history": task_manager.read_chat_history(tid), "summary": task_manager.read_chat_summary(tid)}


@app.post("/api/chat/history/save")
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


@app.get("/api/status")
def api_status(task_id: str = Query("")):
    task_manager = TaskManager()
    tid = task_manager.normalize_task_id(task_id)
    if not tid:
        raise HTTPException(400, "task_id 不能为空")
    data = task_manager.read_status(tid)
    data["downloads"] = task_manager.collect_downloads(tid)
    return data


@app.get("/api/stream")
async def api_stream(task_id: str = Query("")):
    """SSE endpoint: streams section completion events as they happen."""
    if not task_id:
        raise HTTPException(400, "task_id 不能为空")

    stream_path = _result_dir() / task_id / "stream.jsonl"

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
