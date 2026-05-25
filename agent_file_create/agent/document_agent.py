"""Document generation agent powered by LangGraph StateGraph.

Uses explicit graph nodes (not create_react_agent), native interrupt() for
human-in-the-loop, SqliteSaver checkpointing, and modular prompts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent_file_create.agent.prompts import CLARIFY_QUESTIONS_PROMPT
from agent_file_create.agent.state import AgentState
from agent_file_create.config import (
    CONTENT_API_ENDPOINT,
    CONTENT_API_KEY,
    CONTENT_API_STYLE,
    CONTENT_MODEL_NAME,
    MODEL_NAME,
    MODEL_TIMEOUT,
    MODEL_TIMEOUT_SHORT,
)
from agent_file_create.llm_factory import get_chat_model
from agent_file_create.utils import retry_call, safe_json

logger = logging.getLogger(__name__)

CHECKPOINT_DB_PATH = "result/checkpoints.db"

# ── Module-level SqliteSaver singleton ────────────────────────────────────────

_checkpointer: Optional[SqliteSaver] = None
_checkpointer_cm: Optional[object] = None


def _get_checkpointer() -> SqliteSaver:
    global _checkpointer, _checkpointer_cm
    if _checkpointer is None:
        _checkpointer_cm = SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH)
        _checkpointer = _checkpointer_cm.__enter__()
    return _checkpointer


# ── DocumentAgent ─────────────────────────────────────────────────────────────


class DocumentAgent:
    """Deterministic 7‑step document‑generation workflow backed by LangGraph.

    Steps
    -----
    1. extract   – parse all uploaded files
    2. assess    – compute quality metrics
    3. clarify   – ask the user for preferences (interrupt / human‑in‑the‑loop)
    4. outline   – generate a markdown outline
    5. content   – expand outline into full‑length content
    6. render    – produce final .md / .docx / .pdf outputs
    7. END

    The graph uses *SqliteSaver* so every successful step is persisted.
    Transient failures are retried with exponential backoff.
    """

    def __init__(
        self,
        *,
        task_id: str,
        user_prompt: str,
        file_paths: List[str],
        template_dir_override: Optional[str] = None,
    ) -> None:
        self.task_id = str(task_id)
        self.user_prompt = str(user_prompt or "")
        self.file_paths = [str(x) for x in (file_paths or [])]
        self.template_dir_override = template_dir_override
        # Public state dict — callers may pre‑populate analysis_results,
        # force_regen, or user_clarifications before calling run().
        self.state: Dict[str, Any] = {}
        self._graph = self._build_graph()
        self._stored_human_input_fn: Optional[Callable[[str], str]] = None

    # ── LLM helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_llm(timeout_s: int = MODEL_TIMEOUT_SHORT):
        """Lightweight LLM for clarification / routing (not content generation)."""
        style = (
            CONTENT_API_STYLE or "openai"
        ).strip().lower()
        endpoint = (CONTENT_API_ENDPOINT or "").strip()
        model = (CONTENT_MODEL_NAME or MODEL_NAME).strip()
        key = (CONTENT_API_KEY or "").strip()
        if not endpoint and not key:
            style = "ollama"
        return get_chat_model(
            style=style,
            model=model,
            endpoint=endpoint,
            api_key=key,
            temperature=0.2,
            max_tokens=1024,
            timeout_s=int(timeout_s),
        )

    # ── Graph construction ───────────────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(AgentState)

        # Nodes
        builder.add_node("extract", self._node_extract)
        builder.add_node("assess", self._node_assess)
        builder.add_node("clarify", self._node_clarify)
        builder.add_node("outline", self._node_outline)
        builder.add_node("satisfaction_outline", self._node_satisfaction_outline)
        builder.add_node("content", self._node_content)
        builder.add_node("satisfaction_content", self._node_satisfaction_content)
        builder.add_node("render", self._node_render)
        builder.add_node("eval", self._node_eval)
        builder.add_node("handle_error", self._node_handle_error)

        # Edges
        builder.add_edge(START, "extract")
        builder.add_edge("extract", "assess")
        builder.add_conditional_edges(
            "assess",
            self._route_after_assess,
            {"clarify": "clarify", "outline": "outline"},
        )
        builder.add_edge("clarify", "outline")

        # outline -> satisfaction_outline (then conditional)
        builder.add_edge("outline", "satisfaction_outline")
        builder.add_conditional_edges(
            "satisfaction_outline",
            self._route_after_satisfaction_outline,
            {"outline": "outline", "content": "content", "error": "handle_error"},
        )

        # content -> satisfaction_content (then conditional)
        builder.add_edge("content", "satisfaction_content")
        builder.add_conditional_edges(
            "satisfaction_content",
            self._route_after_satisfaction_content,
            {"outline": "outline", "content": "content", "render": "render", "error": "handle_error"},
        )

        # Error‑aware routing for render
        builder.add_conditional_edges(
            "render",
            self._route_after_render,
            {"eval": "eval", "end": END, "error": "handle_error"},
        )
        builder.add_edge("eval", END)
        builder.add_edge("handle_error", END)

        return builder.compile(checkpointer=_get_checkpointer())

    # ── Routing ──────────────────────────────────────────────────────────────

    @staticmethod
    def _route_after_assess(state: AgentState) -> str:
        """Skip clarification if the user has already provided preferences."""
        if state.get("force_regen"):
            return "outline"
        if (state.get("user_clarifications") or "").strip():
            return "outline"
        return "clarify"

    @staticmethod
    def _route_on_error(state: AgentState) -> str:
        """Route to error handler if a non‑recoverable error was set."""
        if state.get("error"):
            return "error"
        return "next"

    @staticmethod
    def _route_after_render(state: AgentState) -> str:
        """Route to eval if enabled, otherwise END."""
        if state.get("error"):
            return "error"
        if state.get("eval_enabled"):
            return "eval"
        return "end"

    @staticmethod
    def _route_after_satisfaction_outline(state: AgentState) -> str:
        """After satisfaction check on outline, decide next step."""
        if state.get("error"):
            return "error"
        if state.get("outline_satisfied"):
            return "content"
        # Not satisfied — check regeneration scope
        scope = state.get("regeneration_scope", "outline")
        if scope == "content_only":
            return "content"
        return "outline"

    @staticmethod
    def _route_after_satisfaction_content(state: AgentState) -> str:
        """After satisfaction check on content, decide next step."""
        if state.get("error"):
            return "error"
        if state.get("content_satisfied"):
            return "render"
        scope = state.get("regeneration_scope", "outline")
        if scope == "content_only":
            return "content"
        return "outline"

    # ── Node: extract ────────────────────────────────────────────────────────

    def _node_extract(self, state: AgentState) -> dict:
        logger.info("extract start  task=%s", self.task_id)
        if state.get("analysis_results") and not state.get("force_regen"):
            logger.info("extract skip   task=%s (already done)", self.task_id)
            return {}

        from agent_file_create.document.extractor import extract_from_file

        results: List[dict] = []
        for fp in state.get("file_paths", self.file_paths):
            res = retry_call(extract_from_file, fp, preprocess=True)
            res["_file"] = Path(fp).name
            results.append(res)

        logger.info("extract done   task=%s files=%d", self.task_id, len(results))
        return {"analysis_results": results, "force_regen": False}

    # ── Node: assess ─────────────────────────────────────────────────────────

    def _node_assess(self, state: AgentState) -> dict:
        logger.info("assess  start  task=%s", self.task_id)

        from agent_file_create.preprocessor import compute_quality_metrics

        ar = state.get("analysis_results") or []
        if not ar:
            return {"last_output": "（无文件可评估）"}

        lines: list[str] = []
        for it in ar[:8]:
            if not isinstance(it, dict):
                continue
            fn = str(it.get("_file") or "").strip()
            title = str(it.get("title") or "").strip()
            summary = str(it.get("summary") or "").strip()
            err = str(it.get("error") or "").strip()
            q = compute_quality_metrics(it)
            qtxt = (
                f"filled={q.get('filled_fields')}/7"
                f" r={float(q.get('field_ratio') or 0):.2f}"
            )
            s = " | ".join(
                x
                for x in [
                    title,
                    summary[:160] + ("…" if len(summary) > 160 else ""),
                ]
                if x
            ).strip()
            if err:
                s = (s + " | " if s else "") + ("ERROR=" + err[:120])
            head = (fn + ": ") if fn else ""
            lines.append(head + (s or "（无摘要）") + " | " + qtxt)

        assessment = "\n".join(lines).strip() or "（暂无抽取结果）"
        logger.info("assess  done   task=%s", self.task_id)
        return {"last_output": assessment}

    # ── Node: clarify ────────────────────────────────────────────────────────

    def _node_clarify(self, state: AgentState) -> dict:
        logger.info("clarify start  task=%s", self.task_id)

        assessment = state.get("last_output", "")
        user_prompt = state.get("user_prompt", self.user_prompt)

        # Generate clarification questions via LLM
        llm = self._build_llm()
        prompt = CLARIFY_QUESTIONS_PROMPT.invoke({
            "user_prompt": user_prompt,
            "assessment": assessment,
        })
        try:
            response = retry_call(llm.invoke, prompt)
            question = (
                response.content
                if hasattr(response, "content")
                else str(response)
            ).strip()
        except Exception as exc:
            logger.warning("clarify llm failed: %s, using fallback", exc)
            question = (
                "请确认以下偏好以便生成更精准的报告：\n"
                "1. 报告使用场景？A.内部决策/B.对外汇报/C.学术发表\n"
                "2. 侧重点偏向？A.技术深度/B.商业价值/C.风险分析\n"
                "3. 篇幅偏好？A.3000字精简/B.5000字标准/C.8000字详尽"
            )

        # Native LangGraph interrupt — pauses the graph here
        answer = interrupt(question)

        # ── Resumed with user answer ─────────────────────────────────
        prev = (state.get("user_clarifications") or "").strip()
        new_clarifications = (
            f"{prev}\n{str(answer or '').strip()}" if prev
            else str(answer or "").strip()
        )
        logger.info("clarify done   task=%s", self.task_id)
        return {
            "user_clarifications": new_clarifications,
            "clarify_question": "",
        }

    # ── Node: satisfaction_outline ───────────────────────────────────────────

    def _node_satisfaction_outline(self, state: AgentState) -> dict:
        logger.info("satisfaction_outline start  task=%s", self.task_id)

        current_ver = state.get("current_outline_version", 1)
        outline = state.get("outline", "")
        versions = state.get("outline_versions") or []

        # Build prompt showing the outline and asking for satisfaction
        prompt_parts = [
            "[STAGE:satisfaction_outline]",
            "📋 大纲已生成完成，请审阅：",
            "",
            "---",
            outline[:3000] if outline else "（大纲为空）",
            "---",
            "",
            f"当前版本：V{current_ver}",
        ]
        if len(versions) > 1:
            prompt_parts.append(f"共 {len(versions)} 个版本")
        prompt_parts.append("请选择：[满意] 继续生成报告  /  [不满意] 重新生成")
        question = "\n".join(prompt_parts)

        answer = interrupt(question)

        # Parse answer: user sends JSON-like structure via Command
        # The answer is a dict from the frontend: {"satisfied": bool, "feedback": "...", "scope": "outline|content_only"}
        import json as _json
        try:
            if isinstance(answer, str):
                ans = _json.loads(answer)
            else:
                ans = answer if isinstance(answer, dict) else {}
        except Exception:
            ans = {"satisfied": True, "feedback": "", "scope": "outline"}

        satisfied = bool(ans.get("satisfied", True))
        feedback = str(ans.get("feedback") or "").strip()
        scope = str(ans.get("scope") or "outline").strip()

        logger.info("satisfaction_outline done  task=%s satisfied=%s", self.task_id, satisfied)
        return {
            "outline_satisfied": satisfied,
            "satisfaction_feedback": feedback,
            "regeneration_scope": scope,
            "waiting_satisfaction": "",
        }

    # ── Node: satisfaction_content ───────────────────────────────────────────

    def _node_satisfaction_content(self, state: AgentState) -> dict:
        logger.info("satisfaction_content start  task=%s", self.task_id)

        current_ver = state.get("current_content_version", 1)
        content = state.get("content", "")
        versions = state.get("content_versions") or []

        prompt_parts = [
            "[STAGE:satisfaction_content]",
            "📄 报告正文已生成完成，请审阅：",
            "",
            "---",
            content[:3000] if content else "（报告为空）",
            "---",
            "",
            f"当前版本：V{current_ver}",
        ]
        if len(versions) > 1:
            prompt_parts.append(f"共 {len(versions)} 个版本")
        prompt_parts.append("请选择：[满意] 完成并渲染  /  [不满意] 重新生成")
        question = "\n".join(prompt_parts)

        answer = interrupt(question)

        import json as _json
        try:
            if isinstance(answer, str):
                ans = _json.loads(answer)
            else:
                ans = answer if isinstance(answer, dict) else {}
        except Exception:
            ans = {"satisfied": True, "feedback": "", "scope": "content_only"}

        satisfied = bool(ans.get("satisfied", True))
        feedback = str(ans.get("feedback") or "").strip()
        scope = str(ans.get("scope") or "content_only").strip()

        logger.info("satisfaction_content done  task=%s satisfied=%s scope=%s", self.task_id, satisfied, scope)
        result: dict = {
            "content_satisfied": satisfied,
            "satisfaction_feedback": feedback,
            "regeneration_scope": scope,
            "waiting_satisfaction": "",
        }
        # When user wants to restart from outline, reset outline_satisfied
        # so the outline node regenerates instead of skipping.
        if not satisfied and scope == "outline":
            result["outline_satisfied"] = False
        return result

    # ── Node: outline ────────────────────────────────────────────────────────

    def _node_outline(self, state: AgentState) -> dict:
        logger.info("outline start  task=%s", self.task_id)
        if (
            state.get("outline")
            and state.get("outline", "").strip()
            and state.get("outline_satisfied")
        ):
            logger.info("outline skip   task=%s (already satisfied)", self.task_id)
            return {}

        try:
            from agent_file_create.document.outline_generator import (
                generate_outline as _gen,
            )

            ar = state.get("analysis_results") or []
            multimodal = {f"source_{i}": r for i, r in enumerate(ar)}
            user_prompt = state.get("user_prompt", self.user_prompt)
            feedback = state.get("satisfaction_feedback", "")
            outline = _gen(multimodal, user_prompt, feedback=feedback)

            # Version management
            versions = list(state.get("outline_versions") or [])
            current_ver = int(state.get("current_outline_version") or 0) + 1
            import time
            versions.append({
                "version": current_ver,
                "content": outline,
                "feedback": feedback,
                "selected": False,
                "ts": time.time(),
            })

            logger.info("outline done   task=%s chars=%d ver=%d", self.task_id, len(outline or ""), current_ver)
            return {
                "outline": outline,
                "outline_versions": versions,
                "current_outline_version": current_ver,
                "outline_satisfied": False,  # reset for new version
                "satisfaction_feedback": "",
            }
        except Exception as exc:
            logger.warning("outline failed  task=%s err=%s", self.task_id, exc)
            return {"error": f"大纲生成失败: {exc}"}

    # ── Node: content ────────────────────────────────────────────────────────

    def _node_content(self, state: AgentState) -> dict:
        logger.info("content start  task=%s", self.task_id)
        if (
            state.get("content")
            and state.get("content", "").strip()
            and state.get("content_satisfied")
        ):
            logger.info("content skip   task=%s (already satisfied)", self.task_id)
            return {}

        try:
            from agent_file_create.document.content_generator import (
                generate_full_content as _gen,
            )

            ar = state.get("analysis_results") or []
            multimodal = {f"source_{i}": r for i, r in enumerate(ar)}
            outline = str(state.get("outline") or "")
            user_prompt = state.get("user_prompt", self.user_prompt)
            feedback = state.get("satisfaction_feedback", "")
            content = _gen(outline, multimodal, user_prompt, task_id=self.task_id, feedback=feedback)

            # Version management
            versions = list(state.get("content_versions") or [])
            current_ver = int(state.get("current_content_version") or 0) + 1
            import time
            versions.append({
                "version": current_ver,
                "content": content,
                "feedback": feedback,
                "selected": False,
                "ts": time.time(),
            })

            logger.info("content done   task=%s chars=%d ver=%d", self.task_id, len(content or ""), current_ver)
            return {
                "content": content,
                "content_versions": versions,
                "current_content_version": current_ver,
                "content_satisfied": False,
                "satisfaction_feedback": "",
            }
        except Exception as exc:
            logger.warning("content failed  task=%s err=%s", self.task_id, exc)
            return {"error": f"正文生成失败: {exc}"}

    # ── Node: render ─────────────────────────────────────────────────────────

    def _node_render(self, state: AgentState) -> dict:
        logger.info("render  start  task=%s", self.task_id)

        try:
            from agent_file_create.document_service import generate_document

            ar = state.get("analysis_results") or []
            user_prompt = state.get("user_prompt", self.user_prompt)
            result = retry_call(
                generate_document,
                user_prompt=user_prompt,
                analysis_results=ar,
                document_type="report",
                task_id=self.task_id,
                template_dir_override=state.get("template_dir_override") or "",
                outline=str(state.get("outline") or "") or None,
                content=str(state.get("content") or "") or None,
            )

            logger.info("render  done   task=%s", self.task_id)
            return {
                "outputs": result.get("rendered_outputs") or [],
                "output_dir": result.get("output_dir") or "",
                "outline": result.get("document_outline") or state.get("outline") or "",
                "content": result.get("document_content") or state.get("content") or "",
                "finished": True,
            }
        except Exception as exc:
            logger.warning("render failed   task=%s err=%s", self.task_id, exc)
            return {"error": f"文档渲染失败: {exc}"}

    # ── Node: eval ───────────────────────────────────────────────────────────

    def _node_eval(self, state: AgentState) -> dict:
        """Run post‑generation evaluation synchronously so the result is
        available when the status is written."""
        logger.info("eval    start  task=%s", self.task_id)

        content = str(state.get("content") or "")
        if not content.strip():
            logger.info("eval    skip   task=%s (no content)", self.task_id)
            return {}

        outline = str(state.get("outline") or "")
        analysis_results = list(state.get("analysis_results") or [])
        user_prompt = str(state.get("user_prompt") or self.user_prompt)

        try:
            from agent_file_create.evaluation import evaluate

            report = evaluate(
                content=content,
                outline=outline,
                analysis_results=analysis_results,
                user_prompt=user_prompt,
                enable_llm=True,
            )
            eval_dict = report.to_dict()
            logger.info(
                "eval    done   task=%s combined=%.2f/%.2f/%.2f/%.2f",
                self.task_id,
                report.combined.relevance,
                report.combined.faithfulness,
                report.combined.coherence,
                report.combined.completeness,
            )
            return {"eval_report": eval_dict}
        except Exception as exc:
            logger.warning("eval    failed task=%s err=%s", self.task_id, exc)
            return {}

    # ── Node: handle_error ───────────────────────────────────────────────────

    def _node_handle_error(self, state: AgentState) -> dict:
        err = state.get("error", "unknown error")
        logger.error(
            "workflow_error task=%s phase=%s err=%s",
            self.task_id,
            state.get("last_output", "")[:80],
            err,
        )
        return {
            "finished": True,
            "last_output": f"工作流异常终止：{err}",
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def run(
        self,
        *,
        max_turns: int = 0,  # no‑op — kept for backward compatibility
        human_input_fn: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        """Execute the document-generation workflow.

        Parameters
        ----------
        max_turns:
            Ignored (StateGraph has no turn limit).  Kept for backward
            compatibility with the old create_react_agent API.
        human_input_fn:
            If provided, called when the graph pauses for user clarification.
            When ``None`` the method returns early with ``need_user=True`` and
            the question so the caller can present it to the user and call
            :meth:`resume` later.

        Returns
        -------
        dict
            Always contains ``task_id``.  May contain ``need_user``,
            ``question`` when waiting for human input.
        """
        # Store human_input_fn for use in subsequent interrupts
        if human_input_fn is not None:
            self._stored_human_input_fn = human_input_fn

        config = {"configurable": {"thread_id": self.task_id}}

        # Build initial graph state, folding in any caller‑set values
        external = self.state
        initial: AgentState = {
            "task_id": self.task_id,
            "user_prompt": self.user_prompt,
            "file_paths": list(self.file_paths),
            "template_dir_override": str(self.template_dir_override or ""),
            "force_regen": bool(external.get("force_regen", False)),
            "user_clarifications": str(external.get("user_clarifications", "")),
            "eval_enabled": bool(external.get("eval_enabled", True)),
            "messages": [],
            "outline_versions": list(external.get("outline_versions") or []),
            "content_versions": list(external.get("content_versions") or []),
            "current_outline_version": int(external.get("current_outline_version") or 0),
            "current_content_version": int(external.get("current_content_version") or 0),
            "outline_satisfied": bool(external.get("outline_satisfied", False)),
            "content_satisfied": bool(external.get("content_satisfied", False)),
            "satisfaction_feedback": str(external.get("satisfaction_feedback") or ""),
            "regeneration_scope": str(external.get("regeneration_scope") or ""),
            "waiting_satisfaction": str(external.get("waiting_satisfaction") or ""),
        }
        # If caller pre‑populated analysis_results (e.g. from a previous run
        # or external extractor), carry them forward to skip extraction.
        if isinstance(external.get("analysis_results"), list) and external["analysis_results"]:
            initial["analysis_results"] = list(external["analysis_results"])
        if isinstance(external.get("outline"), str) and external["outline"].strip():
            initial["outline"] = external["outline"]
        if isinstance(external.get("content"), str) and external["content"].strip():
            initial["content"] = external["content"]

        try:
            result = self._graph.invoke(initial, config)
        except Exception as exc:
            # GraphInterrupt is expected — any other exception is a real error
            if "GraphInterrupt" in type(exc).__name__:
                return self._handle_interrupt(config, human_input_fn)
            logger.warning(
                "agent_graph_error task=%s err=%s", self.task_id, exc
            )
            self.state.update({
                "finished": True,
                "last_output": f"Agent error: {exc}",
                "error": str(exc),
            })
            return dict(self.state)

        # Check if graph was interrupted (some versions don't raise)
        snapshot = self._graph.get_state(config)
        if snapshot and snapshot.interrupts:
            return self._handle_interrupt(config, human_input_fn)

        # Write final state back so callers can read agent.state["output_dir"] etc.
        self.state.update(result)
        return dict(self.state)

    def resume(
        self,
        answer: str,
        *,
        human_input_fn: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        """Resume a paused workflow with the user's answer.

        Call this after :meth:`run` returned ``need_user=True``.
        """
        from langgraph.types import Command

        config = {"configurable": {"thread_id": self.task_id}}

        try:
            result = self._graph.invoke(
                Command(resume=answer),
                config,
            )
        except Exception as exc:
            if "GraphInterrupt" in type(exc).__name__:
                return self._handle_interrupt(config, human_input_fn)
            logger.warning(
                "agent_resume_error task=%s err=%s", self.task_id, exc
            )
            self.state.update({
                "finished": True,
                "last_output": f"Agent error: {exc}",
                "error": str(exc),
            })
            return dict(self.state)

        snapshot = self._graph.get_state(config)
        if snapshot and snapshot.interrupts:
            return self._handle_interrupt(config, human_input_fn)

        self.state.update(result)
        return dict(self.state)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _handle_interrupt(
        self,
        config: dict,
        human_input_fn: Optional[Callable[[str], str]],
    ) -> Dict[str, Any]:
        """Extract the interrupt question and either auto‑answer or signal UI."""
        snapshot = self._graph.get_state(config)
        question = ""
        interrupt_node = ""
        if snapshot and snapshot.interrupts:
            for interrupt_data in snapshot.interrupts:
                question = (
                    interrupt_data.value
                    if hasattr(interrupt_data, "value")
                    else str(interrupt_data)
                )
                # Detect which node triggered the interrupt
                if hasattr(interrupt_data, "ns"):
                    ns = interrupt_data.ns
                    if isinstance(ns, (list, tuple)):
                        interrupt_node = str(ns[-1]) if ns else ""
                    else:
                        interrupt_node = str(ns)
                break

        # Determine the interrupt type
        is_satisfaction = "satisfaction" in interrupt_node.lower() if interrupt_node else False
        is_clarify = "clarify" in interrupt_node.lower() if interrupt_node else False

        # Store human_input_fn for later resume calls
        fn = human_input_fn or self._stored_human_input_fn

        if fn:
            answer = str(fn(question) or "").strip()
            if answer:
                return self.resume(answer)
            return self.resume('{"satisfied": true, "feedback": "", "scope": "outline"}')

        return {
            "task_id": self.task_id,
            "need_user": True,
            "question": question or "请补充你的需求偏好。",
            "finished": False,
            "interrupt_node": interrupt_node,
            "is_satisfaction": is_satisfaction,
            "is_clarify": is_clarify,
        }
