"""Task execution functions — document generation workflow.

Extracted from web/server.py. Contains the core document generation logic
(_run_task, _run_document_only, section regen/edit) and the thread launcher.
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from agent_file_create.task.manager import TaskManager
from agent_file_create.utils import split_questions
from agent_file_create.web._utils import (
    _MAX_CONCURRENT_TASKS,
    _task_semaphore,
    better_quality,
    result_dir,
)

logger = logging.getLogger(__name__)


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
    from agent_file_create.document.extractor import extract_from_file
    from agent_file_create.preprocessor import compute_quality_metrics

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
                use_b = better_quality(qa, qb)
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
    analysis_lock = threading.Lock()
    indexed_shared: list[tuple[int, dict, dict | None]] = []
    extraction_done = threading.Event()
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
    extract_thread = threading.Thread(target=_extract_all, daemon=True)
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
        from agent_file_create.agent import DocumentAgent

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
                task_manager.write_status(
                    task_id, "need_user", stage="final_confirm",
                    message="请进行最终确认后渲染报告",
                    extra={"final_confirmed": None},
                )
                t0 = time.time()
                result = task_manager.wait_for_satisfaction(task_id, "final", timeout_s=1800)
                if cancel_ev.is_set():
                    return json.dumps({"final_confirmed": True})
                return json.dumps({"final_confirmed": bool(result.get("satisfied", True)), "selected_version": result.get("selected_version") or 1})

            # Detect satisfaction / quality_gate interrupts by stage prefix
            if q.startswith("[STAGE:satisfaction_outline]") or q.startswith("[STAGE:satisfaction_content]") or q.startswith("[STAGE:quality_gate]"):
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
                        return json.dumps({"satisfied": False, "feedback": "", "scope": ""})
                    elapsed = time.time() - t0
                    if elapsed >= 300 - 2:
                        return json.dumps({"satisfied": False, "feedback": "超时未选择，跳过评估", "scope": ""})
                    return json.dumps(result)
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
                        return json.dumps({"satisfied": True, "feedback": "", "scope": scope_default})
                    return json.dumps(result)

            # Regular clarify interrupt (use cleaned q instead of raw question)
            qs = split_questions(q)
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
        from agent_file_create.agent import DocumentAgent

        def _auto_approve(question: str) -> str:
            """Auto-approve satisfaction interrupts during redo/regeneration.
            For quality_gate, prompt user to decide (same as initial generation).
            """
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
                    return json.dumps({"satisfied": False, "feedback": "", "scope": ""})
                elapsed = time.time() - t0
                if elapsed >= 300 - 2:
                    return json.dumps({"satisfied": False, "feedback": "超时未选择，跳过评估", "scope": ""})
                return json.dumps(result)
            if "[STAGE:satisfaction_" in q:
                scope = "outline" if "outline" in q.split("\n")[0].lower() else "content_only"
                return json.dumps({"satisfied": True, "feedback": "", "scope": scope})
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
            th = threading.Thread(
                target=_release_on_done, args=(_run_document_only,),
                kwargs={"task_id": task_id, "user_prompt": user_prompt, "file_paths": file_paths, "analysis_results": analysis_results, "template_dir_override": template_dir_override, "saved_templates": saved_templates, "ab_eval": bool(ab_eval)},
                daemon=True)
            task_manager.start_task(task_id, th.start)
            return True, "已启动重新生成（仅文档阶段）。"
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

        base = result_dir() / task_id
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

        base = result_dir() / task_id
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


def make_regenerate_fn(task_manager: TaskManager):
    def _fn(task_id: str, mode: str = "doc", section_name: str = "", feedback: str = "") -> tuple[bool, str]:
        # Single-section regeneration
        if section_name and section_name.strip():
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
