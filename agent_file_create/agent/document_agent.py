"""Document generation agent powered by LangGraph StateGraph.

Uses explicit graph nodes (not create_react_agent), native interrupt() for
human-in-the-loop, SqliteSaver checkpointing, and modular prompts.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json as _json
import logging
import re
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent_file_create.errors import (
    DocAgentError,
    ExtractionError,
    LLMCallError,
    RAGRetrievalError,
    StepFatalError,
    StepRecoverableError,
)
from agent_file_create.prompts import CLARIFY_QUESTIONS_PROMPT
from agent_file_create.agent.state import AgentState
from agent_file_create.config import (
    CONTENT_API_ENDPOINT,
    CONTENT_API_KEY,
    CONTENT_API_STYLE,
    CONTENT_MODEL_NAME,
    GRAPH_RECURSION_LIMIT,
    MODEL_NAME,
    MODEL_TIMEOUT,
    MODEL_TIMEOUT_SHORT,
)
from agent_file_create.llm_factory import get_chat_model
from agent_file_create.utils import retry_call, safe_json

logger = logging.getLogger(__name__)

CHECKPOINT_DB_PATH = "result/checkpoints.db"

# ── Module-level SqliteSaver singleton (thread‑safe) ─────────────────────────

_checkpointer: Optional[SqliteSaver] = None
_checkpointer_cm: Optional[object] = None
_checkpointer_lock = threading.Lock()

# ── Module-level thread pool for _run_async (reused) ─────────────────────────

_async_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
_async_pool_lock = threading.Lock()


def _get_async_pool() -> concurrent.futures.ThreadPoolExecutor:
    global _async_pool
    if _async_pool is None:
        with _async_pool_lock:
            if _async_pool is None:
                _async_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    return _async_pool


def _get_checkpointer() -> SqliteSaver:
    global _checkpointer, _checkpointer_cm
    if _checkpointer is None:
        with _checkpointer_lock:
            if _checkpointer is None:
                _checkpointer_cm = SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH)
                _checkpointer = _checkpointer_cm.__enter__()
    return _checkpointer


def _prune_checkpoint(task_id: str) -> None:
    """Delete a completed task's checkpoint data to prevent unbounded DB growth."""
    try:
        cp = _get_checkpointer()
        # LangGraph SqliteSaver stores per-thread checkpoints
        cp.delete_thread(task_id)
        logger.debug("checkpoint_pruned task=%s", task_id)
    except Exception as e:
        logger.debug("checkpoint_prune_failed task=%s err=%s", task_id, e)


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
    def _run_async(coro):
        """Run an async coroutine safely in both sync and async contexts.

        Reuses a module-level ThreadPoolExecutor instead of creating one per call.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # Already in async context — use thread to avoid nested loop
        return _get_async_pool().submit(asyncio.run, coro).result()

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

        # Nodes — flow matches user expectation:
        # extract→plan→assess→(clarify)→enrich→outline→satisfaction_outline
        # →research→content→critic→satisfaction_content(版本比对+段落重生成)
        # →render→quality_gate(评估可选)→END
        builder.add_node("extract", self._node_extract)
        builder.add_node("plan", self._node_plan)           # ★ Planner
        builder.add_node("assess", self._node_assess)
        builder.add_node("clarify", self._node_clarify)
        builder.add_node("enrich", self._node_enrich)
        builder.add_node("outline", self._node_outline)
        builder.add_node("satisfaction_outline", self._node_satisfaction_outline)
        builder.add_node("research", self._node_research)
        builder.add_node("content", self._node_content)
        builder.add_node("critic_analyze", self._node_critic_analyze)      # ★ Critic: analyze
        builder.add_node("critic_auto_fix", self._node_critic_auto_fix)      # ★ Critic: auto-fix
        builder.add_node("satisfaction_content", self._node_satisfaction_content)
        builder.add_node("final_confirm", self._node_final_confirm)
        builder.add_node("render", self._node_render)
        builder.add_node("quality_gate", self._node_quality_gate)
        builder.add_node("handle_error", self._node_handle_error)

        # Edges
        builder.add_edge(START, "extract")
        builder.add_edge("extract", "plan")                  # ★ extract → plan
        builder.add_edge("plan", "assess")                   # ★ plan → assess
        builder.add_conditional_edges(
            "assess", self._route_after_assess,
            {"clarify": "clarify", "enrich": "enrich"},
        )
        builder.add_edge("clarify", "enrich")
        builder.add_edge("enrich", "outline")

        # outline → satisfaction_outline
        builder.add_edge("outline", "satisfaction_outline")
        builder.add_conditional_edges(
            "satisfaction_outline", self._route_after_satisfaction_outline,
            {"outline": "outline", "research": "research", "content": "content", "error": "handle_error"},
        )

        # research → content → critic_analyze → [conditional] → critic_auto_fix → satisfaction_content → final_confirm
        builder.add_edge("research", "content")
        builder.add_edge("content", "critic_analyze")
        builder.add_conditional_edges(
            "critic_analyze", self._route_after_critic_analyze,
            {"critic_auto_fix": "critic_auto_fix", "satisfaction_content": "satisfaction_content"},
        )
        builder.add_edge("critic_auto_fix", "satisfaction_content")
        builder.add_conditional_edges(
            "satisfaction_content", self._route_after_satisfaction_content,
            {"outline": "outline", "content": "content", "final_confirm": "final_confirm", "error": "handle_error"},
        )

        # final_confirm → render
        builder.add_edge("final_confirm", "render")

        # render → quality_gate(评估可选) → END
        builder.add_edge("render", "quality_gate")
        builder.add_edge("quality_gate", END)
        builder.add_edge("handle_error", END)

        return builder.compile(checkpointer=_get_checkpointer())

    # ── Routing ──────────────────────────────────────────────────────────────

    @staticmethod
    def _route_after_assess(state: AgentState) -> str:
        """Skip clarification if the user has already provided preferences."""
        if state.get("force_regen"):
            return "enrich"
        if (state.get("user_clarifications") or "").strip():
            return "enrich"
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
            # Run research skills before generating full content
            return "research"
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
            return "final_confirm"
        scope = state.get("regeneration_scope", "outline")
        if scope == "content_only":
            return "content"
        return "outline"

    @staticmethod
    def _route_after_final_confirm(state: AgentState) -> str:
        """After final confirm, decide next step."""
        if state.get("error"):
            return "error"
        if state.get("final_confirmed"):
            return "render"
        return "final_confirm"

    # ── Node: extract ────────────────────────────────────────────────────────

    def _node_extract(self, state: AgentState) -> dict:
        logger.info("extract start  task=%s", self.task_id)
        if state.get("analysis_results") and not state.get("force_regen"):
            logger.info("extract skip   task=%s (already done)", self.task_id)
            return {}

        from agent_file_create.document.extractor import extract_from_file
        from agent_file_create.config import MAX_WORKERS_DEFAULT

        fps = state.get("file_paths", self.file_paths)
        results: List[dict] = [{}] * len(fps)
        max_workers = max(1, min(int(MAX_WORKERS_DEFAULT), len(fps)))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(retry_call, extract_from_file, fp, preprocess=True): i
                for i, fp in enumerate(fps)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    res = fut.result()
                except ExtractionError as e:
                    logger.warning("extract file_failed task=%s file=%s err=%s",
                                   self.task_id, Path(fps[idx]).name, e)
                    res = {"error": str(e), "_file": Path(fps[idx]).name}
                except Exception as e:
                    logger.warning("extract file_failed task=%s file=%s err=%s",
                                   self.task_id, Path(fps[idx]).name, e)
                    res = {"error": str(e), "_file": Path(fps[idx]).name}
                if isinstance(res, dict):
                    res["_file"] = Path(fps[idx]).name
                results[idx] = res

        logger.info("extract done   task=%s files=%d", self.task_id, len(results))
        return {"analysis_results": results, "force_regen": False}

    # ── Node: plan (Planner) ─────────────────────────────────────────────────

    def _node_plan(self, state: AgentState) -> dict:
        """Task-level planner: decompose user request into sub-tasks.

        Runs once after file extraction. The plan is stored in state and guides
        subsequent steps (outline, research, content).
        """
        logger.info("plan    start  task=%s", self.task_id)
        if state.get("task_plan"):
            logger.info("plan    skip   task=%s (already done)", self.task_id)
            return {}

        user_prompt = state.get("user_prompt", self.user_prompt)
        ar = state.get("analysis_results") or []
        file_list = "\n".join(
            f"  - {r.get('_file', '?')}: {str(r.get('summary', ''))[:120]}"
            for r in ar[:8] if isinstance(r, dict)
        )

        prompt = (
            "你是一个报告撰写规划助手。请根据用户需求和已有材料，"
            "将任务分解为 3-6 个子任务。\n\n"
            f"用户需求：{user_prompt[:500]}\n\n"
            f"已有材料：\n{file_list or '（无）'}\n\n"
            "输出格式（每行一个子任务）：\n"
            "- 子任务描述 | 需要什么信息 | 优先级(高/中/低)"
        )

        try:
            from agent_file_create.utils import retry_call
            llm = self._build_llm(timeout_s=30)
            response = retry_call(llm.invoke, prompt, max_retries=2, delay=1.0)
            raw = (
                response.content if hasattr(response, "content")
                else str(response)
            ).strip()
        except LLMCallError as e:
            logger.warning("plan    llm_failed task=%s err=%s", self.task_id, e)
            raw = (
                "- 分析材料提取关键信息 | 材料内容 | 高\n"
                "- 生成报告大纲 | 结构规划 | 高\n"
                "- 撰写报告正文 | 大纲+材料 | 高"
            )

        # Parse plan items
        plan_items: list[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                parts = [p.strip() for p in line[2:].split("|")]
                if len(parts) >= 1 and parts[0]:
                    plan_items.append({
                        "task": parts[0],
                        "needs": parts[1] if len(parts) > 1 else "",
                        "priority": parts[2] if len(parts) > 2 else "中",
                    })

        logger.info("plan    done   task=%s items=%d", self.task_id, len(plan_items))
        return {"task_plan": plan_items, "plan_raw": raw}

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

    # ── Node: enrich (skill invocation before outline) ───────────────────────

    def _node_enrich(self, state: AgentState) -> dict:
        """Invoke skills to enrich material before generating the outline.

        1. Ask LLM which skills to call (based on user_prompt + files)
        2. Execute selected skills in parallel
        3. Collect results into enriched_context
        """
        logger.info("enrich  start  task=%s", self.task_id)
        if state.get("enriched_context") and state.get("skills_used"):
            logger.info("enrich  skip   task=%s (already done)", self.task_id)
            return {}

        # ── Query Rewrite: rewrite casual/spoken prompt into precise search query ──
        raw_prompt = state.get("user_prompt", self.user_prompt)
        if not state.get("rewritten_prompt") and raw_prompt and len(raw_prompt) > 8:
            try:
                from agent_file_create.rag.kb import KnowledgeBase
                kb = KnowledgeBase()
                rewritten = kb.rewrite_query(raw_prompt)
                if rewritten and rewritten != raw_prompt and len(rewritten) >= 4:
                    logger.info("enrich  query_rewritten  task=%s old=%.50s new=%.50s",
                                self.task_id, raw_prompt, rewritten)
                else:
                    rewritten = raw_prompt
            except Exception:
                rewritten = raw_prompt
        else:
            rewritten = state.get("rewritten_prompt") or raw_prompt

        try:
            from agent_file_create.skills import get_registry

            registry = get_registry()
            registry.discover()
        except RAGRetrievalError as exc:
            logger.warning("enrich  registry_failed task=%s err=%s", self.task_id, exc)
            return {"skills_used": [], "enriched_context": "", "skill_results": []}
        except Exception as exc:
            logger.warning("enrich  registry_failed task=%s err=%s", self.task_id, exc)
            return {"skills_used": [], "enriched_context": "", "skill_results": []}

        # Build prompt for LLM to select skills
        tools_prompt = registry.build_tools_prompt("enrich")
        user_prompt = state.get("user_prompt", self.user_prompt)
        analysis_results = state.get("analysis_results") or []
        file_list = "\n".join(
            f"  - {r.get('_file', '?')}" for r in analysis_results[:10]
            if isinstance(r, dict)
        )

        selection_prompt = (
            "你是一个报告撰写助手。在生成大纲之前，你可以选择性调用以下技能来"
            "获取更多信息，提升报告质量。\n\n"
            f"用户需求：{user_prompt[:500]}\n\n"
            f"已上传文件列表：\n{file_list or '（无）'}\n\n"
            + tools_prompt
        )

        # Ask LLM to select skills
        llm = self._build_llm(timeout_s=40)
        try:
            from agent_file_create.utils import retry_call
            import asyncio
            response = retry_call(llm.invoke, selection_prompt, max_retries=2, delay=1.0)
            raw = (
                response.content
                if hasattr(response, "content")
                else str(response)
            ).strip()
        except Exception as exc:
            logger.warning("enrich  llm_failed task=%s err=%s", self.task_id, exc)
            return {"skills_used": [], "enriched_context": "", "skill_results": [],
                    "skill_prompt": selection_prompt, "skill_calls_raw": ""}

        calls = registry.parse_skill_calls(raw)
        logger.info("enrich  llm_selected=%s task=%s", [c.skill_name for c in calls], self.task_id)

        if not calls:
            return {"skills_used": [], "enriched_context": "", "skill_results": [],
                    "skill_prompt": selection_prompt, "skill_calls_raw": raw}

        # Execute skills
        try:
            results = self._run_async(registry.execute_calls(calls))
        except Exception as exc:
            logger.warning("enrich  exec_failed task=%s err=%s", self.task_id, exc)
            return {"skills_used": [], "enriched_context": "", "skill_results": [],
                    "skill_prompt": selection_prompt, "skill_calls_raw": raw}

        # Collect results
        skill_results: list[dict] = []
        context_parts: list[str] = []
        skills_used: list[str] = []

        for call, result in results:
            skills_used.append(call.skill_name)
            skill_results.append({
                "skill": call.skill_name,
                "params": call.params,
                "success": result.success,
                "summary": result.summary,
                "data": result.data,
                "error": result.error,
            })
            ctx = result.to_context()
            if ctx:
                context_parts.append(f"### {call.skill_name}\n{ctx}")

        enriched = "\n\n".join(context_parts) if context_parts else ""
        logger.info("enrich  done   task=%s skills=%s chunks=%d chars=%d",
                     self.task_id, skills_used, len(context_parts), len(enriched))

        return {
            "skills_used": skills_used,
            "enriched_context": enriched,
            "skill_results": skill_results,
            "skill_prompt": selection_prompt,
            "skill_calls_raw": raw,
            "rewritten_prompt": rewritten,
        }

    # ── Node: research (skill invocation before content) ──────────────────────

    def _node_research(self, state: AgentState) -> dict:
        """Invoke research skills before generating full content.

        Called after the user approves the outline. Uses research-stage skills
        (e.g. web_search) to fetch up-to-date data for the content.
        """
        logger.info("research start  task=%s", self.task_id)

        try:
            from agent_file_create.skills import get_registry

            registry = get_registry()
            registry.discover()
        except Exception as exc:
            logger.warning("research registry_failed task=%s err=%s", self.task_id, exc)
            return {}

        outline = state.get("outline", "")
        user_prompt = state.get("user_prompt", self.user_prompt)

        # Combine previous enrich context with new research stage
        existing_context = state.get("enriched_context", "")

        tools_prompt = registry.build_tools_prompt("research")
        selection_prompt = (
            "你是一个报告撰写助手。即将开始撰写正文。大纲已经确定，你可以"
            "选择性调用以下技能来获取最新数据或补充信息。\n\n"
            f"用户需求：{user_prompt[:500]}\n\n"
            f"大纲要点：\n{outline[:600]}\n\n"
            + tools_prompt
        )

        llm = self._build_llm(timeout_s=40)
        try:
            from agent_file_create.utils import retry_call
            import asyncio
            response = retry_call(llm.invoke, selection_prompt, max_retries=2, delay=1.0)
            raw = (
                response.content
                if hasattr(response, "content")
                else str(response)
            ).strip()
        except Exception as exc:
            logger.warning("research llm_failed task=%s err=%s", self.task_id, exc)
            return {}

        calls = registry.parse_skill_calls(raw)
        logger.info("research llm_selected=%s task=%s", [c.skill_name for c in calls], self.task_id)

        if not calls:
            return {}

        # Execute skills
        try:
            results = self._run_async(registry.execute_calls(calls))
        except Exception as exc:
            logger.warning("research exec_failed task=%s err=%s", self.task_id, exc)
            return {}

        # Collect results, merge with existing context
        skill_results = list(state.get("skill_results") or [])
        context_parts = [existing_context] if existing_context else []
        skills_used = list(state.get("skills_used") or [])

        for call, result in results:
            if call.skill_name not in skills_used:
                skills_used.append(call.skill_name)
            skill_results.append({
                "skill": call.skill_name,
                "params": call.params,
                "success": result.success,
                "summary": result.summary,
                "data": result.data,
                "error": result.error,
            })
            ctx = result.to_context()
            if ctx:
                context_parts.append(f"### {call.skill_name}\n{ctx}")

        enriched = "\n\n".join(context_parts) if context_parts else existing_context
        logger.info("research done   task=%s skills=%s chars=%d",
                     self.task_id, skills_used, len(enriched))

        return {
            "skills_used": skills_used,
            "enriched_context": enriched,
            "skill_results": skill_results,
        }

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

        question = question or "请补充你希望生成文档的侧重点/受众/篇幅/风格等信息。"
        logger.info("clarify_question  task=%s question_chars=%d", self.task_id, len(question))

        question = "[STAGE:clarify]\n在开始写报告之前，想先确认几个事情：\n\n" + question

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
            prompt_parts.append(f"共 {len(versions)} 个历史版本，可切换对比")
            # Build a quick version comparison table
            prompt_parts.append("")
            prompt_parts.append("📊 版本对比摘要：")
            prompt_parts.append("| 版本 | 字数 | 段落数 | 章节数 | 评估分 |")
            prompt_parts.append("|------|------|--------|--------|--------|")
            _eval_report = state.get("eval_report") or {}
            _current_score = ""
            if _eval_report:
                _combined = _eval_report.get("combined", {})
                if _combined:
                    _avg = (_combined.get("faithfulness", 0) + _combined.get("completeness", 0)
                            + _combined.get("coherence", 0) + _combined.get("relevance", 0)) / 4.0
                    _current_score = f"{_avg:.0%}"
            for v in sorted(versions, key=lambda x: x.get("version", 0)):
                _vc = str(v.get("content", ""))
                _word_count = len(_vc.replace("\n", "").replace(" ", ""))
                _para_count = len([p for p in _vc.split("\n\n") if p.strip()])
                _section_count = len(re.findall(r"^##\s", _vc, re.MULTILINE))
                _is_current = v.get("version") == current_ver
                _marker = " ← 当前" if _is_current else ""
                _score = _current_score if _is_current else "-"
                prompt_parts.append(
                    f"| V{v.get('version', '?')} | {_word_count} | {_para_count} | {_section_count} | {_score}{_marker} |"
                )

        # ── ★ Critic 质检结果展示 ──
        critic_report = state.get("critic_report") or {}
        critic_issues = critic_report.get("issues", [])
        high_issues = [i for i in critic_issues if i.get("severity") == "高"]
        low_med_issues = [i for i in critic_issues if i.get("severity") != "高"]

        if critic_issues:
            prompt_parts.append("")
            prompt_parts.append("── 自动质检报告 ──")
            if not critic_report.get("passed", False):
                if low_med_issues:
                    prompt_parts.append(
                        f"✅ 已自动修正 {len(low_med_issues)} 处低/中严重度问题"
                    )
                if high_issues:
                    prompt_parts.append(f"⚠️ 发现 {len(high_issues)} 处高严重度问题，需人工确认：")
                    for i, issue in enumerate(high_issues[:5]):
                        prompt_parts.append(
                            f"  {i+1}. [{issue.get('type','')}] {issue.get('location','')}: "
                            f"{issue.get('description','')}"
                        )
                # Suggested search queries for missing evidence
                suggested = state.get("suggested_queries") or []
                if suggested:
                    prompt_parts.append("")
                    prompt_parts.append("🔍 证据不足？建议补充检索以下关键词后点 [不满意] 重新生成：")
                    prompt_parts.append(f"  {', '.join(suggested[:5])}")
            else:
                prompt_parts.append("✅ 质检通过，未发现问题")
            prompt_parts.append("──")

        prompt_parts.append(
            "操作选项：\n"
            "  [满意] → 渲染最终报告\n"
            "  [满意+备注] → 满意但留改进建议，不重跑流程\n"
            "  [不满意] → 重新生成正文\n"
            "  [编辑段落] → 点击预览区任意段落直接编辑\n"
            "  [版本对比] → 切换查看历史版本差异\n"
            "  [段落重生成] → 选中段落后AI重新生成该段"
        )
        question = "\n".join(prompt_parts)

        answer = interrupt(question)

        try:
            if isinstance(answer, str):
                ans = _json.loads(answer)
            else:
                ans = answer if isinstance(answer, dict) else {}
        except Exception:
            ans = {"satisfied": True, "feedback": "", "scope": "content_only"}

        # Detect "approve with note" mode
        approval_mode = str(ans.get("mode") or "").strip()
        approve_with_note = approval_mode == "approve_with_note"
        if approve_with_note:
            # User is satisfied but wants to leave a note — proceed without re-gen
            ans["satisfied"] = True
            logger.info("satisfaction_content approve_with_note task=%s note=%.100s",
                        self.task_id, str(ans.get("feedback", "")))

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

    # ── Node: final_confirm ───────────────────────────────────────────────────

    def _node_final_confirm(self, state: AgentState) -> dict:
        """Final confirmation before render. User can do version compare and section regen."""
        logger.info("final_confirm start  task=%s", self.task_id)

        current_ver = state.get("current_content_version", 1)
        content = state.get("content", "")
        versions = state.get("content_versions") or []

        prompt_parts = [
            "[STAGE:final_confirm]",
            "📄 报告正文已生成，请进行最终确认：",
            "",
            f"当前版本：V{current_ver}",
        ]
        if len(versions) > 1:
            prompt_parts.append(f"共 {len(versions)} 个历史版本")
        prompt_parts.append(
            "操作选项：\n"
            "  [版本对比] → 切换查看历史版本差异\n"
            "  [段落重生成] → 选中段落后AI重新生成该段\n"
            "  [编辑段落] → 直接编辑段落内容\n"
            "  [最终确认] → 确认后将渲染最终报告"
        )
        question = "\n".join(prompt_parts)

        answer = interrupt(question)

        try:
            if isinstance(answer, str):
                ans = _json.loads(answer)
            else:
                ans = answer if isinstance(answer, dict) else {}
        except Exception:
            ans = {"final_confirmed": True, "selected_version": current_ver}

        final_confirmed = bool(ans.get("final_confirmed", False))
        selected_version = int(ans.get("selected_version") or current_ver)

        logger.info("final_confirm done  task=%s confirmed=%s version=%s", self.task_id, final_confirmed, selected_version)
        result: dict = {
            "final_confirmed": final_confirmed,
            "selected_content_version": selected_version,
        }
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

            # Extract section-level placeholders from user templates
            template_sections: list[str] = []
            tpl_dir = self.template_dir_override or state.get("template_dir_override") or ""
            if tpl_dir:
                td = Path(tpl_dir)
                if td.exists() and td.is_dir():
                    from agent_file_create.document.template_renderer import _scan_md_placeholders
                    SYSTEM_VARS = {"title", "task_id", "document_outline", "document_content"}
                    all_sections: set[str] = set()
                    for tp in sorted(td.glob("*.md")):
                        all_sections.update(_scan_md_placeholders(str(tp)))
                    template_sections = sorted(all_sections - SYSTEM_VARS)
                    if template_sections:
                        logger.info("outline_template_sections task=%s sections=%s",
                                    self.task_id, template_sections)

            ar = state.get("analysis_results") or []
            multimodal = {f"source_{i}": r for i, r in enumerate(ar)}
            raw_prompt = state.get("user_prompt", self.user_prompt)
            user_prompt = state.get("rewritten_prompt") or raw_prompt
            feedback = state.get("satisfaction_feedback", "")
            enriched = state.get("enriched_context", "")
            target_words = int(state.get("target_words") or 0)
            outline = _gen(multimodal, user_prompt, feedback=feedback,
                          enriched_context=enriched, target_words=target_words,
                          template_sections=template_sections or None)

            # Version management
            versions = list(state.get("outline_versions") or [])
            current_ver = int(state.get("current_outline_version") or 0) + 1
            versions.append({
                "version": current_ver,
                "content": outline,
                "feedback": feedback,
                "selected": False,
                "ts": _time.time(),
            })

            # Persist version to disk
            try:
                from agent_file_create.task.manager import TaskManager
                TaskManager().save_version(self.task_id, "outline", current_ver, outline, feedback)
            except Exception as e:
                logger.debug("save_outline_version_failed ver=%d err=%s", current_ver, e)

            logger.info("outline done   task=%s chars=%d ver=%d", self.task_id, len(outline or ""), current_ver)
            # Write outline.md early so frontend can preview during satisfaction
            try:
                _out_path = Path(__file__).resolve().parent.parent.parent / "result" / str(self.task_id) / "outline.md"
                _out_path.parent.mkdir(parents=True, exist_ok=True)
                _out_path.write_text(str(outline or ""), encoding="utf-8")
            except Exception as e:
                logger.debug("write_outline_preview_failed path=%s err=%s", _out_path, e)
            return {
                "outline": outline,
                "outline_versions": versions,
                "current_outline_version": current_ver,
                "outline_satisfied": False,  # reset for new version
                "satisfaction_feedback": "",
            }
        except StepRecoverableError as exc:
            logger.warning("outline failed  task=%s err=%s", self.task_id, exc)
            return {"error": f"大纲生成失败: {exc}"}
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
            raw_prompt = state.get("user_prompt", self.user_prompt)
            user_prompt = state.get("rewritten_prompt") or raw_prompt
            feedback = state.get("satisfaction_feedback", "")
            enriched = state.get("enriched_context", "")
            target_words = int(state.get("target_words") or 0)

            # ── Planner: pre-plan knowledge (skip if KB is empty) ──
            try:
                from agent_file_create.rag.planner import plan_all_sections, build_citation_map, format_citation_list
                from agent_file_create.rag.kb import KnowledgeBase
                from agent_file_create.task.manager import TaskManager
                _kb = KnowledgeBase()
                _tm = TaskManager()
                _task_meta = _tm.read_task_meta(self.task_id)
                _active_kb = str(_task_meta.get("active_kb") or "").strip()
                _kb_name = str(_active_kb or "")
                # Fallback: auto-pick first non-empty KB
                if not _kb_name:
                    _kb_list = _kb.list_kb()
                    for _k in _kb_list:
                        _s = _kb.kb_stats(kb=_k)
                        if _s.get("doc_count", 0) > 0:
                            _kb_name = _k
                            logger.info("content planner_auto_kb task=%s kb=%s", self.task_id, _kb_name)
                            break
                _kb_stats = _kb.kb_stats(kb=_kb_name) if _kb_name else {}
                if _kb_stats.get("doc_count", 0) > 0:
                    logger.info("content planner_start  task=%s kb=%s docs=%d", self.task_id, _kb_name, _kb_stats["doc_count"])
                    t_plan = _time.perf_counter()
                    _plan = plan_all_sections(
                        outline=outline, user_prompt=user_prompt,
                        kb=_kb, kb_name=_kb_name,
                        target_words=target_words,
                    )
                    if _plan:
                        _plan_parts = [enriched] if enriched else []
                        for _sec_title, _sec_plan in _plan.items():
                            _mat = _sec_plan.get("materials", "")
                            _kps = "；".join(_sec_plan.get("knowledge_points", [])[:3])
                            if _mat or _kps:
                                _plan_parts.append(
                                    f"[章节素材: {_sec_title}]\n"
                                    f"知识点: {_kps}\n材料: {_mat}"
                                )
                        enriched = "\n\n".join(_plan_parts) if len(_plan_parts) > 1 else enriched

                        # Cross-document conflict detection
                        try:
                            from agent_file_create.document._reviewer import (
                                detect_cross_document_conflicts, annotate_conflicts_in_materials,
                            )
                            _all_plan_hits = []
                            for _sp in _plan.values():
                                _all_plan_hits.extend(_sp.get("_raw_hits", []))
                            if _all_plan_hits:
                                _conflicts = detect_cross_document_conflicts(_all_plan_hits)
                                if _conflicts:
                                    enriched = annotate_conflicts_in_materials(enriched, _conflicts)
                                    logger.info("content cross_doc_conflicts=%d", len(_conflicts))
                        except Exception:
                            pass

                        # Build global citation map from raw hits (plan_all_sections
                        # delegates to plan_section_knowledge which doesn't produce
                        # citation_map natively — rebuild from _raw_hits here)
                        _all_cit_maps: dict[str, dict] = {}
                        _annotated_parts: list[str] = []
                        try:
                            from agent_file_create.rag.planner import _compress_hits_annotated
                            for _sec_title, _sec_plan in _plan.items():
                                _raw = _sec_plan.get("_raw_hits") or []
                                if _raw:
                                    _annotated, _sec_cit_map = _compress_hits_annotated(
                                        _raw,
                                        _sec_plan.get("knowledge_points", [_sec_title])[0],
                                        section_type=_sec_plan.get("section_type", "review"),
                                    )
                                    _all_cit_maps[_sec_title] = {"citation_map": _sec_cit_map}
                                    if _annotated:
                                        _annotated_parts.append(f"## {_sec_title}\n{_annotated}")
                            _citation_map = build_citation_map(_all_cit_maps)  # once after all sections
                            # Append annotated materials to enriched context so LLM sees 【n】 markers
                            if _annotated_parts:
                                enriched = (enriched or "") + "\n\n---\n\n# 带编号引用的检索材料\n\n" + "\n\n".join(_annotated_parts)
                        except Exception:
                            _citation_map = {}
                        _citation_refs = format_citation_list(_citation_map) if _citation_map else ""
                        logger.info("content planner_done sections=%d citations=%d annotated_chars=%d elapsed=%.1fs",
                                    len(_plan), len(_citation_map), sum(len(p) for p in _annotated_parts), _time.perf_counter() - t_plan)
                    else:
                        logger.warning("content planner_empty_plan  task=%s — no sections planned", self.task_id)
                        _citation_map = {}
                        _citation_refs = ""
            except RAGRetrievalError as _pe:
                logger.debug("content planner_skip err=%s", _pe)
                _citation_map = {}
                _citation_refs = ""
            except Exception as _pe:
                logger.debug("content planner_skip err=%s", _pe)
                _citation_map = {}
                _citation_refs = ""

            content = _gen(outline, multimodal, user_prompt, task_id=self.task_id,
                          feedback=feedback, enriched_context=enriched,
                          target_words=target_words)

            # Faithfulness check moved to quality_gate node (optional, user-decided)

            # Version management
            versions = list(state.get("content_versions") or [])
            current_ver = int(state.get("current_content_version") or 0) + 1
            versions.append({
                "version": current_ver,
                "content": content,
                "feedback": feedback,
                "selected": False,
                "ts": _time.time(),
            })

            # Persist version to disk
            try:
                from agent_file_create.task.manager import TaskManager
                TaskManager().save_version(self.task_id, "content", current_ver, content, feedback)
            except Exception as e:
                logger.debug("save_content_version_failed ver=%d err=%s", current_ver, e)

            # Citation post-processing: renumber + optionally append reference list
            if content:
                try:
                    from agent_file_create.rag.planner import renumber_citations
                    content, _citation_map = renumber_citations(content, _citation_map or {})
                    _citation_refs = format_citation_list(_citation_map) if _citation_map else ""
                except Exception:
                    pass

            # Template-aware reference placement:
            # if the template has {{references}} placeholder, don't append —
            # the renderer will inject it there. Otherwise append at end.
            _template_has_refs = False
            try:
                tpl_dir = self.template_dir_override or state.get("template_dir_override") or ""
                if tpl_dir:
                    from agent_file_create.document.template_renderer import _get_template_placeholders
                    td = Path(tpl_dir)
                    if td.exists() and td.is_dir():
                        for tp in sorted(td.glob("*.md")):
                            if "references" in _get_template_placeholders(str(tp)):
                                _template_has_refs = True
                                break
            except Exception as e:
                logger.debug("template_placeholder_check_failed err=%s", e)

            # Check if outline already has a references section
            _outline_has_refs = bool(re.search(r'^#+\s*(?:参考|引用|文献|来源)', outline or "", re.MULTILINE | re.IGNORECASE))
            if _citation_refs and not _template_has_refs and not _outline_has_refs:
                content = (content or "") + "\n\n---\n\n" + _citation_refs

            logger.info("content done   task=%s chars=%d ver=%d citations=%d template_has_refs=%s",
                        self.task_id, len(content or ""), current_ver,
                        len(_citation_map), _template_has_refs)
            return {
                "content": content,
                "content_versions": versions,
                "current_content_version": current_ver,
                "content_satisfied": False,
                "satisfaction_feedback": "",
                "citation_map": _citation_map,
                "citation_refs": _citation_refs,
            }
        except StepRecoverableError as exc:
            logger.warning("content failed  task=%s err=%s", self.task_id, exc)
            return {"error": f"正文生成失败: {exc}", "citation_map": {}, "citation_refs": ""}
        except Exception as exc:
            logger.warning("content failed  task=%s err=%s", self.task_id, exc)
            return {"error": f"正文生成失败: {exc}", "citation_map": {}, "citation_refs": ""}

    # ── Node: critic_analyze (auto-review — analysis only) ────────────────────

    def _node_critic_analyze(self, state: AgentState) -> dict:
        """Automated quality review — analysis phase.

        Runs after content generation, before human satisfaction check.
        1. Regex pre-filter: fast number/entity/year cross-check (0 LLM calls).
        2. LLM review: check content against outline & materials.
        3. Citation verification: lightweight cross-reference.

        Does NOT modify content — only produces a critic_report.
        Auto-fix happens in the separate _node_critic_auto_fix node.
        """
        logger.info("critic_analyze start  task=%s", self.task_id)

        content = str(state.get("content") or "")
        outline = str(state.get("outline") or "")
        ar = state.get("analysis_results") or []

        # Build materials digest from analysis results
        materials_parts: list[str] = []
        materials_full = ""
        for r in ar[:8]:
            if isinstance(r, dict):
                title = str(r.get("title") or "").strip()
                summary = str(r.get("summary") or "").strip()
                if summary:
                    materials_parts.append(f"[{title}] {summary}" if title else summary)
        materials_full = "\n".join(materials_parts)
        materials = materials_full[:3000]

        if len(content) < 200:
            logger.info("critic_analyze skip   task=%s (content too short)", self.task_id)
            return {
                "critic_report": {"issues": [], "passed": True},
                "content_hardened": content,
            }

        # ── Layer 1: regex pre-filter + numerical hallucination hardening ──
        regex_issues: list[dict] = []
        content_patched = content
        try:
            from agent_file_create.document._reviewer import (
                extract_facts_from_materials, cross_check_facts, patch_unverified_claims,
            )
            material_facts = extract_facts_from_materials(materials_full)
            # 1a: detect issues
            raw_issues = cross_check_facts(content_patched, material_facts)
            for ri in raw_issues[:8]:
                regex_issues.append({
                    "type": "regex",
                    "location": ri.split(":")[0] if ":" in ri else "",
                    "description": ri,
                    "severity": "中",
                })
            # 1b: auto-patch unverifiable numbers/entities → [数据待核实]
            content_patched, patches = patch_unverified_claims(content_patched, material_facts)
            if patches:
                for p in patches:
                    regex_issues.append({
                        "type": "auto_patch",
                        "location": "numeral_guard",
                        "description": p,
                        "severity": "中",
                    })
                logger.info(
                    "critic_analyze regex_hardening patched=%d issues=%d",
                    len(patches), len(raw_issues),
                )
            elif regex_issues:
                logger.info("critic_analyze regex_filter found=%d issues", len(regex_issues))
        except Exception as e:
            logger.debug("critic_analyze regex_filter skipped: %s", e)

        try:
            from agent_file_create.document._critic import run_critic

            # LLM review (on already-hardened content)
            report = run_critic(content=content_patched, outline=outline, materials=materials)
        except Exception as e:
            logger.warning("critic_analyze failed task=%s err=%s", self.task_id, e)
            report = {"issues": [], "raw": "", "passed": True}

        issues = regex_issues + report.get("issues", [])
        n_issues = len(issues)
        high_issues = [i for i in issues if i.get("severity") == "高"]

        # ── Citation verification (optional, lightweight) ──
        citation_warnings: list[dict] = []
        try:
            _cit_map = state.get("citation_map") or {}
            if _cit_map and content:
                from agent_file_create.rag.planner import verify_citations
                citation_warnings = verify_citations(content_patched, _cit_map)
                if citation_warnings:
                    logger.info("critic_analyze citation_warnings=%d", len(citation_warnings))
        except Exception as e:
            logger.debug("critic_analyze citation_verify skipped: %s", e)

        # ── Quality Pipeline (optional, env-controlled) ─────────────────
        quality_issues: list[dict] = []
        try:
            import os as _os
            if _os.getenv("QUALITY_ENABLED", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
                from agent_file_create.quality import QualityPipeline, QualityContext
                _qp_ctx = QualityContext(
                    content=content_patched,
                    analysis_results=ar,
                    task_id=self.task_id,
                    output_dir=str(Path(__file__).resolve().parent.parent.parent / "result" / self.task_id),
                )
                _qp_result = QualityPipeline().run_parallel(_qp_ctx)
                for _sr in _qp_result.step_results:
                    if _sr.warnings:
                        for _w in _sr.warnings:
                            quality_issues.append({
                                "type": "quality_check",
                                "location": "",
                                "description": _w,
                                "severity": "中",
                            })
                if _qp_result.content and _qp_result.content != content_patched:
                    content_patched = _qp_result.content
                logger.info(
                    "critic_analyze quality_pipeline task=%s steps=%d warnings=%d changed=%s",
                    self.task_id, len(_qp_result.step_results),
                    sum(len(sr.warnings) for sr in _qp_result.step_results),
                    _qp_result.content != content,
                )
        except Exception as _qe:
            logger.debug("critic_analyze quality_pipeline skipped: %s", _qe)

        # Merge reports (no content modification)
        merged = dict(report)
        merged["issues"] = issues + quality_issues
        merged["regex_issues"] = len(regex_issues)
        merged["citation_warnings"] = citation_warnings
        merged["quality_issues"] = len(quality_issues)

        logger.info(
            "critic_analyze done   task=%s total=%d regex=%d llm=%d quality=%d high=%d",
            self.task_id, len(merged["issues"]), len(regex_issues),
            len(report.get("issues", [])), len(quality_issues), len(high_issues),
        )

        return {
            "content_hardened": content_patched,
            "critic_report": merged,
            "critic_issues_count": len(merged["issues"]),
            "critic_high_issues": len(high_issues),
            "suggested_queries": report.get("suggested_queries", []),
        }

    def _route_after_critic_analyze(self, state: AgentState) -> str:
        """Route: if there are fixable issues, run auto-fix; otherwise skip."""
        report = state.get("critic_report") or {}
        issues = report.get("issues", [])
        fixable = [i for i in issues if i.get("severity") != "高"]
        if fixable:
            return "critic_auto_fix"
        return "satisfaction_content"

    # ── Node: critic_auto_fix (auto-fix only) ─────────────────────────────

    def _node_critic_auto_fix(self, state: AgentState) -> dict:
        """Apply auto-fixes for low/medium severity issues.

        Only runs when _route_after_critic_analyze detects fixable issues.
        Uses the content_hardened from critic_analyze as the base for fixes.
        """
        logger.info("critic_auto_fix start task=%s", self.task_id)

        content = str(state.get("content") or "")
        content_patched = str(state.get("content_hardened") or content)
        report = state.get("critic_report") or {}
        issues = report.get("issues", [])

        fixable = [i for i in issues if i.get("severity") != "高"]
        if not fixable:
            logger.info("critic_auto_fix skip   task=%s (nothing fixable)", self.task_id)
            return {}

        # Build materials from analysis_results for context
        ar = state.get("analysis_results") or []
        materials_parts: list[str] = []
        for r in ar[:8]:
            if isinstance(r, dict):
                title = str(r.get("title") or "").strip()
                summary = str(r.get("summary") or "").strip()
                if summary:
                    materials_parts.append(f"[{title}] {summary}" if title else summary)
        materials = "\n".join(materials_parts)[:3000]

        fixed_content = content_patched
        try:
            from agent_file_create.document._critic import run_critic_fix

            fixed_content = run_critic_fix(
                content=content_patched, issues=fixable, materials=materials,
            )
            if fixed_content != content_patched:
                logger.info(
                    "critic_auto_fix applied task=%s low_med=%d chars=%d->%d",
                    self.task_id, len(fixable),
                    len(content_patched), len(fixed_content),
                )
            else:
                logger.info("critic_auto_fix no_change task=%s (LLM returned same content)", self.task_id)
        except Exception as e:
            logger.warning("critic_auto_fix failed task=%s err=%s", self.task_id, e)

        _had_changes = fixed_content != content
        logger.info(
            "critic_auto_fix done   task=%s fixed=%d changed=%s",
            self.task_id, len(fixable), _had_changes,
        )

        return {"content": fixed_content if _had_changes else content}

    # ── Node: render ─────────────────────────────────────────────────────────

    # ── Node: quality_gate ───────────────────────────────────────────────────

    def _node_quality_gate(self, state: AgentState) -> dict:
        """After render — ask user: '报告已完成，是否进行质量评估？'"""
        logger.info("quality_gate start  task=%s", self.task_id)

        question = (
            "[STAGE:quality_gate]\n"
            "📋 当前报告已完成，是否进行质量评估？\n\n"
            "开启后将核查每一章节的事实准确性，对可疑内容进行增量检索修正。\n\n"
            "请选择：[要] 开启质量评估  /  [不要] 跳过"
        )
        answer = interrupt(question)

        try:
            ans = _json.loads(answer) if isinstance(answer, str) else {}
        except Exception:
            ans = {}
        want_eval = bool(ans.get("satisfied", False))  # reuse satisfied=true for "要"

        if not want_eval:
            logger.info("quality_gate skip  task=%s (user declined)", self.task_id)
            return {"eval_skipped": True}

        logger.info("quality_gate run   task=%s", self.task_id)
        content = str(state.get("content") or "")
        ar = state.get("analysis_results") or []

        try:
            from agent_file_create.document_service import _run_faithfulness_checks
            from agent_file_create.evaluation.orchestrator import evaluate as run_eval
            output_dir = str(
                __import__('pathlib').Path(__file__).resolve().parent.parent.parent
                / "result" / self.task_id
            )
            new_content = _run_faithfulness_checks(
                content=content, analysis_results=ar,
                task_id=self.task_id, output_dir=output_dir,
            )

            # Run evaluation
            eval_report = run_eval(
                content=new_content or content,
                outline=str(state.get("outline") or ""),
                analysis_results=ar,
                user_prompt=str(state.get("user_prompt") or ""),
            )
            scores = eval_report.combined
            logger.info("quality_gate eval_done task=%s faith=%.2f comp=%.2f coh=%.2f rel=%.2f",
                        self.task_id, scores.faithfulness, scores.completeness,
                        scores.coherence, scores.relevance)

            # ── Per-section evaluation breakdown ───────────────────────
            section_evals: list[dict] = []
            try:
                from agent_file_create.evaluation.orchestrator import evaluate_by_section
                section_evals = evaluate_by_section(
                    content=new_content or content,
                    outline=str(state.get("outline") or ""),
                    analysis_results=ar,
                    user_prompt=str(state.get("user_prompt") or ""),
                    enable_llm=False,  # Decomposed only for speed
                )
                if section_evals:
                    logger.info(
                        "quality_gate sections task=%s n=%d weakest=%s(%.2f)",
                        self.task_id, len(section_evals),
                        section_evals[0]["title"], section_evals[0]["scores"].faithfulness,
                    )
            except Exception as _se:
                logger.debug("quality_gate section_eval_failed err=%s", _se)

            # ── Quality gate thresholds ───────────────────────────────
            from agent_file_create.config import (
                EVAL_MIN_FAITHFULNESS,
                EVAL_MIN_COMPLETENESS,
                EVAL_AUTO_RETRY,
            )
            below_threshold = []
            if scores.faithfulness < EVAL_MIN_FAITHFULNESS:
                below_threshold.append(f"忠实度({scores.faithfulness:.2f}<{EVAL_MIN_FAITHFULNESS})")
            if scores.completeness < EVAL_MIN_COMPLETENESS:
                below_threshold.append(f"完整性({scores.completeness:.2f}<{EVAL_MIN_COMPLETENESS})")

            auto_remediate = False
            remediation_target = ""
            if below_threshold and EVAL_AUTO_RETRY:
                auto_remediate = True
                if scores.faithfulness < EVAL_MIN_FAITHFULNESS:
                    remediation_target = "faithfulness"
                if scores.completeness < EVAL_MIN_COMPLETENESS:
                    remediation_target = (
                        remediation_target + "+completeness"
                        if remediation_target else "completeness"
                    )
                logger.info(
                    "quality_gate below_threshold task=%s thresholds=%s target=%s",
                    self.task_id, below_threshold, remediation_target,
                )

            _eval_dict = eval_report.to_dict()
            _eval_dict["_thresholds"] = {
                "min_faithfulness": EVAL_MIN_FAITHFULNESS,
                "min_completeness": EVAL_MIN_COMPLETENESS,
            }

            result = {
                "content": new_content if new_content != content else content,
                "eval_applied": True,
                "eval_metrics": _eval_dict,
                "eval_report": _eval_dict,
                "auto_remediate": auto_remediate,
                "remediation_target": remediation_target,
                "section_evals": [
                    {"title": s["title"], "chars": s["chars"],
                     "faithfulness": s["scores"].faithfulness,
                     "completeness": s["scores"].completeness,
                     "warnings": len(s.get("warnings", []))}
                    for s in section_evals
                ],
            }

            # Persist eval report to result dir
            try:
                import json as _j
                _eval_path = Path(__file__).resolve().parent.parent.parent / "result" / str(self.task_id) / "eval_report.json"
                _eval_path.parent.mkdir(parents=True, exist_ok=True)
                _eval_path.write_text(_j.dumps(_eval_dict, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("quality_gate eval_saved path=%s", _eval_path)
            except Exception:
                pass

            logger.info("quality_gate return task=%s auto_remediate=%s target=%s",
                        self.task_id, auto_remediate, remediation_target)
            return result
        except Exception as e:
            logger.warning("quality_gate failed  task=%s err=%s", self.task_id, e)
            return {"eval_skipped": True}

    def _node_render(self, state: AgentState) -> dict:
        """Render already-generated content into templates.

        Only called AFTER satisfaction_content confirms the user is happy.
        Does NOT re-generate — content comes from state.
        Uses user-selected version if specified.
        """
        logger.info("render  start  task=%s", self.task_id)

        try:
            from agent_file_create.document_service import render_document
            from pathlib import Path as _Path

            selected_version = state.get("selected_content_version") or state.get("current_content_version") or 1
            content_versions = state.get("content_versions") or []
            content = str(state.get("content") or "")

            if content_versions and selected_version:
                for v in content_versions:
                    if v.get("version") == selected_version:
                        content = str(v.get("content") or content)
                        logger.info("render using_selected_version  task=%s version=%s", self.task_id, selected_version)
                        break

            outline = str(state.get("outline") or "")
            output_dir = str(
                state.get("output_dir") or
                (_Path(__file__).resolve().parent.parent.parent / "result" / self.task_id)
            )
            template_dir = str(
                state.get("template_dir_override") or
                (_Path(__file__).resolve().parent.parent.parent / "result" / self.task_id / "template")
            )

            rendered = render_document(
                task_id=self.task_id,
                content=content,
                outline=outline,
                output_dir=output_dir,
                template_dir=template_dir,
            )

            logger.info("render  done   task=%s outputs=%d", self.task_id, len(rendered))
            return {
                "outputs": rendered,
                "output_dir": output_dir,
                "finished": True,
            }
        except StepFatalError as exc:
            logger.warning("render failed   task=%s err=%s", self.task_id, exc)
            return {"error": f"文档渲染失败: {exc}"}
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
        # Determine which node/phase failed from last_output or state context
        failed_node = ""
        last_out = str(state.get("last_output", ""))[:120]
        if state.get("content") and not state.get("outline"):
            failed_node = "outline"
        elif state.get("outline") and not state.get("content"):
            failed_node = "content"
        elif not state.get("analysis_results"):
            failed_node = "extract"
        elif state.get("outputs"):
            failed_node = "render"
        else:
            failed_node = "workflow"

        context_snapshot = {
            "stage": failed_node,
            "has_outline": bool(state.get("outline")),
            "has_content": bool(state.get("content")),
            "has_analysis": bool(state.get("analysis_results")),
            "files": len(state.get("file_paths", [])),
            "last_output_preview": last_out,
        }

        logger.error(
            "workflow_error task=%s node=%s err=%s ctx=%s",
            self.task_id,
            failed_node,
            err,
            context_snapshot,
        )

        # ── Save recovery snapshot ───────────────────────────────────────
        # Persist enough state so the user can retry from the failure point
        # without losing completed work (outline, extracted data, etc.).
        recovery_data = {
            "failed_node": failed_node,
            "error": str(err)[:500],
            "has_outline": bool(state.get("outline")),
            "has_content": bool(state.get("content")),
            "has_analysis": bool(state.get("analysis_results")),
            "user_prompt": str(state.get("user_prompt", self.user_prompt))[:2000],
            "retry_from": {
                "extract": "extract",
                "outline": "outline" if state.get("analysis_results") else "extract",
                "content": "content" if state.get("outline") else "outline",
                "render": "render" if state.get("content") else "content",
            }.get(failed_node, "outline"),
            "timestamp": _time.time(),
        }
        try:
            import json as _j
            _recovery_path = (
                Path(__file__).resolve().parent.parent.parent
                / "result" / str(self.task_id) / "recovery.json"
            )
            _recovery_path.parent.mkdir(parents=True, exist_ok=True)
            _recovery_path.write_text(_j.dumps(recovery_data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("recovery_saved task=%s node=%s path=%s", self.task_id, failed_node, _recovery_path)
        except Exception as _re:
            logger.debug("recovery_save_failed err=%s", _re)

        _prune_checkpoint(self.task_id)

        # User-friendly message with actionable hint
        stage_label = {
            "extract": "文件抽取",
            "outline": "大纲生成",
            "content": "正文生成",
            "render": "文档渲染",
        }.get(failed_node, "文档生成")

        # Build retry suggestion
        retry_hint = ""
        if failed_node in ("outline", "content"):
            retry_hint = (
                f"\n\n🔄 重试建议: 你可以修改需求描述后，在聊天框发送 "
                f"/regen 来从「{stage_label}」阶段重新生成，已完成的步骤将被复用。"
            )

        return {
            "finished": True,
            "last_output": (
                f"工作流在「{stage_label}」阶段异常终止：{err}\n\n"
                f"📎 已完成步骤: "
                + ("文件抽取 → " if state.get("analysis_results") else "")
                + ("大纲生成 → " if state.get("outline") else "")
                + ("正文生成 → " if state.get("content") else "")
                + ("文档渲染 → " if state.get("outputs") else "")
                + (f"❌ 失败于: {stage_label}"
                   f"{retry_hint}\n\n"
                   f"📋 恢复信息已保存到 result/{self.task_id}/recovery.json\n"
                   f"💡 如问题持续，请联系管理员查看日志 (task={self.task_id})")
            ),
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

        config = {"configurable": {"thread_id": self.task_id}, "recursion_limit": GRAPH_RECURSION_LIMIT}

        # Build initial graph state, folding in any caller‑set values
        external = self.state
        initial: AgentState = {
            "task_id": self.task_id,
            "user_prompt": self.user_prompt,
            "file_paths": list(self.file_paths),
            "template_dir_override": str(self.template_dir_override or ""),
            "force_regen": bool(external.get("force_regen", False)),
            "target_words": int(external.get("target_words") or 0),
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
            # Skill system
            "skill_results": list(external.get("skill_results") or []),
            "enriched_context": str(external.get("enriched_context") or ""),
            "skills_used": list(external.get("skills_used") or []),
            # Planner + Critic
            "task_plan": list(external.get("task_plan") or []),
            "plan_raw": str(external.get("plan_raw") or ""),
            "critic_report": dict(external.get("critic_report") or {}),
            "critic_issues_count": int(external.get("critic_issues_count") or 0),
            "critic_high_issues": int(external.get("critic_high_issues") or 0),
            "suggested_queries": list(external.get("suggested_queries") or []),
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
        if self.state.get("finished"):
            _prune_checkpoint(self.task_id)
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

        config = {"configurable": {"thread_id": self.task_id}, "recursion_limit": GRAPH_RECURSION_LIMIT}

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
            _prune_checkpoint(self.task_id)
            return dict(self.state)

        snapshot = self._graph.get_state(config)
        if snapshot and snapshot.interrupts:
            return self._handle_interrupt(config, human_input_fn)

        self.state.update(result)
        if self.state.get("finished"):
            _prune_checkpoint(self.task_id)
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
