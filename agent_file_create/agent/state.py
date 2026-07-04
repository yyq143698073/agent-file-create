"""Typed state for the document-generation agent StateGraph.

The ``messages`` key uses LangGraph's ``add_messages`` reducer so that
successive node returns are concatenated (not overwritten).
"""

from __future__ import annotations

from typing import Annotated, List, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Mutable state that flows through the agent graph.

    All keys are optional (``total=False``) — nodes return only the subset
    they changed.
    """

    # ── LangGraph managed ────────────────────────────────────────────
    messages: Annotated[list, add_messages]

    # ── Task identity ─────────────────────────────────────────────────
    task_id: str
    user_prompt: str
    file_paths: List[str]
    template_dir_override: str
    target_words: int

    # ── Workflow outputs ──────────────────────────────────────────────
    analysis_results: List[dict]
    outline: str
    content: str
    outputs: List[str]
    output_dir: str

    # ── Clarification / interrupt ─────────────────────────────────────
    user_clarifications: str
    clarify_question: str

    # ── Control flags ─────────────────────────────────────────────────
    force_regen: bool
    finished: bool

    # ── Evaluation ────────────────────────────────────────────────────
    eval_report: dict          # EvalReport.to_dict() result
    eval_enabled: bool         # Whether to run eval after render
    eval_metrics: dict         # FActScore, Coverage, etc. from generate_document

    # ── Version management ───────────────────────────────────────────
    outline_versions: list          # [{"version": N, "content": "...", "feedback": "", "selected": bool, "ts": float}, ...]
    content_versions: list
    current_outline_version: int
    current_content_version: int

    # ── Satisfaction control ─────────────────────────────────────────
    outline_satisfied: bool
    content_satisfied: bool
    satisfaction_feedback: str       # user's reason for dissatisfaction + new requirements
    regeneration_scope: str          # "outline" | "content_only"
    waiting_satisfaction: str        # "" | "outline" | "content"

    # ── Skill system ──────────────────────────────────────────────────
    skill_results: list              # [{"skill": "web_search", "success": True, "summary": "...", "data": {...}}, ...]
    enriched_context: str            # concatenated skill outputs for downstream prompts
    skills_used: list                # names of skills that were invoked
    skill_prompt: str                # the LLM prompt for skill selection (for UI display)
    skill_calls_raw: str             # raw LLM response for skill selection

    # ── Planner + Critic ──────────────────────────────────────────────
    task_plan: list                  # ★ Planner: [{"task": "...", "needs": "...", "priority": "高/中/低"}, ...]
    plan_raw: str                    # ★ Planner: raw LLM response
    content_hardened: str            # ★ Critic: regex-hardened content (between analyze and auto-fix)
    critic_report: dict              # ★ Critic: {"issues": [...], "raw": "...", "passed": bool, "suggested_queries": [...]}
    critic_issues_count: int         # ★ Critic: total issues found
    critic_high_issues: int          # ★ Critic: high-severity issues
    suggested_queries: list          # ★ Critic: suggested search keywords for missing evidence
    citation_map: dict               # ★ Citations: {n: Citation}
    citation_refs: str               # ★ Citations: formatted reference list (markdown)

    # ── Error / status ────────────────────────────────────────────────
    error: str
    last_output: str
