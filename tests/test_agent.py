"""Unit tests for agent module — state, prompts, routing, graph structure.

These tests exercise pure logic paths (no LLM / DB / file I/O required).
"""

from __future__ import annotations

import os
import sys

# ── Block modules that trigger KnowledgeBase(→Postgres) at import time ──────
_BASE = os.path.dirname(__file__)
_AGENT = os.path.join(_BASE, "..", "agent_file_create")


def _mk_mod(name: str, path: str) -> None:
    m = type(sys)(name)
    m.__path__ = [path]
    m.__file__ = os.path.join(path, "__init__.py")
    sys.modules[name] = m


_mk_mod("agent_file_create", _AGENT)
_mk_mod("agent_file_create.chat", os.path.join(_AGENT, "chat"))
_mk_mod("agent_file_create.rag", os.path.join(_AGENT, "rag"))
_mk_mod("agent_file_create.web", os.path.join(_AGENT, "web"))
_mk_mod("agent_file_create.task", os.path.join(_AGENT, "task"))
_mk_mod("agent_file_create.document", os.path.join(_AGENT, "document"))

import pytest

from agent_file_create.agent.prompts import (
    CLARIFY_QUESTIONS_PROMPT,
    CONTEXT_TEMPLATE,
    SYSTEM_PROMPT,
)
from agent_file_create.agent.state import AgentState
from agent_file_create.agent.tools import EmptyInput, create_tools
from agent_file_create.chat.prompts import (
    CHECK_RELEVANCE_PROMPT,
    FOLLOWUPS_PROMPT,
    REWRITE_QUERY_PROMPT,
    SUMMARIZE_HISTORY_PROMPT,
    lobby_prompt,
    task_chat_prompt,
)


# ── AgentState ────────────────────────────────────────────────────────────────

class TestAgentState:
    def test_create_empty(self):
        s: AgentState = {}
        assert s == {}

    def test_populate_required_fields(self):
        s: AgentState = {
            "task_id": "t1",
            "user_prompt": "生成报告",
            "file_paths": ["a.pdf"],
            "force_regen": False,
        }
        assert s["task_id"] == "t1"
        assert s["force_regen"] is False

    def test_optional_fields_default_to_none(self):
        s: AgentState = {"task_id": "t2"}
        assert s.get("outline") is None
        assert s.get("analysis_results") is None
        assert s.get("finished") is None

    def test_partial_update(self):
        """Nodes return partial dicts — only changed keys."""
        s: AgentState = {"task_id": "t3", "force_regen": False}
        s["analysis_results"] = [{"title": "测试"}]
        s["force_regen"] = True
        assert len(s["analysis_results"]) == 1
        assert s["force_regen"] is True


# ── EmptyInput schema ─────────────────────────────────────────────────────────

class TestEmptyInput:
    def test_empty_schema_has_no_required_fields(self):
        assert EmptyInput.model_fields == {}

    def test_can_instantiate(self):
        obj = EmptyInput()
        assert isinstance(obj, EmptyInput)


# ── Routing logic ─────────────────────────────────────────────────────────────

class TestRouting:
    @staticmethod
    def _route(state: AgentState) -> str:
        from agent_file_create.agent.document_agent import DocumentAgent
        return DocumentAgent._route_after_assess(state)

    def test_route_to_clarify_when_no_clarifications(self):
        state: AgentState = {
            "user_clarifications": "",
            "force_regen": False,
        }
        assert self._route(state) == "clarify"

    def test_route_to_outline_when_clarifications_exist(self):
        state: AgentState = {
            "user_clarifications": "用户希望生成财务分析报告",
            "force_regen": False,
        }
        assert self._route(state) == "outline"

    def test_route_to_outline_on_force_regen(self):
        state: AgentState = {
            "user_clarifications": "",
            "force_regen": True,
        }
        assert self._route(state) == "outline"


# ── Agent prompts ─────────────────────────────────────────────────────────────

class TestAgentPrompts:
    def test_system_prompt_is_message(self):
        from langchain_core.messages import SystemMessage
        assert isinstance(SYSTEM_PROMPT, SystemMessage)
        assert len(str(SYSTEM_PROMPT.content)) > 50

    def test_clarify_prompt_invocable(self):
        result = CLARIFY_QUESTIONS_PROMPT.invoke({
            "user_prompt": "生成一份市场分析报告",
            "assessment": "filled=3/7 r=0.43",
        })
        assert len(result.to_string()) > 0

    def test_context_template(self):
        rendered = CONTEXT_TEMPLATE.format(
            state_json='{"phase": "extract"}',
            input_text="请继续推进",
        )
        assert "extract" in rendered
        assert "请继续推进" in rendered


# ── Chat prompts ──────────────────────────────────────────────────────────────

class TestChatPrompts:
    def test_lobby_prompt_structure(self):
        msgs = lobby_prompt.format_messages(
            user_input="你好",
            history=[],
        )
        assert len(msgs) >= 2
        assert msgs[0].type == "system"
        assert msgs[-1].type == "human"

    def test_task_chat_prompt_structure(self):
        msgs = task_chat_prompt.format_messages(
            user_input="详细说明第三章",
            history=[],
            progress_hint="",
            context_text="",
        )
        assert len(msgs) >= 2
        assert msgs[0].type == "system"
        assert msgs[-1].type == "human"

    def test_summarize_history_prompt(self):
        result = SUMMARIZE_HISTORY_PROMPT.invoke({
            "transcript": "[user]: 你好\n[assistant]: 你好，有什么可以帮助你的？",
        })
        assert "你好" in result.to_string()

    def test_rewrite_query_prompt(self):
        result = REWRITE_QUERY_PROMPT.invoke({"question": "什么是RAG？"})
        assert "RAG" in result.to_string()

    def test_check_relevance_prompt(self):
        result = CHECK_RELEVANCE_PROMPT.invoke({
            "clarify_question": "请选择报告风格",
            "user_reply": "我要技术型报告",
        })
        assert "技术型报告" in result.to_string()

    def test_followups_prompt(self):
        result = FOLLOWUPS_PROMPT.invoke({
            "question": "什么是向量数据库？",
            "reply_summary": "介绍了FAISS和ChromaDB",
            "report_topics": "RAG, 向量检索",
        })
        assert "向量数据库" in result.to_string()


# ── Tool factory ──────────────────────────────────────────────────────────────

class TestToolFactory:
    @staticmethod
    def _make_tools(**overrides):
        base = {
            "state": {
                "task_id": "test",
                "force_regen": False,
                "user_clarifications": "",
            },
            "file_paths": [],
            "user_prompt": "测试",
            "task_id": "test",
            "template_dir_override": "",
        }
        base.update(overrides)
        return create_tools(
            state=base["state"],
            file_paths=base["file_paths"],
            user_prompt=base["user_prompt"],
            task_id=base["task_id"],
            template_dir_override=base["template_dir_override"],
        )

    def test_returns_seven_tools(self):
        tools = self._make_tools()
        assert len(tools) == 7

    def test_all_have_names(self):
        tools = self._make_tools()
        names = [t.name for t in tools]
        assert "extract_files" in names
        assert "assess_material" in names
        assert "generate_outline" in names
        assert "generate_content" in names
        assert "render_templates" in names
        assert "ask_user" in names
        assert "finish" in names

    def test_tools_are_independently_usable(self):
        """Each tool has .invoke() and metadata."""
        tools = self._make_tools()
        for t in tools:
            assert hasattr(t, "invoke")
            assert hasattr(t, "name")
            assert hasattr(t, "description")

    def test_empty_input_tools_have_empty_schema(self):
        tools = self._make_tools()
        empty_input_names = {
            "assess_material",
            "generate_outline",
            "generate_content",
            "render_templates",
            "finish",
        }
        for t in tools:
            if t.name in empty_input_names:
                assert t.args_schema is not None
                assert t.args_schema.model_fields == {}

    def test_extract_files_skips_when_already_done(self):
        state: AgentState = {
            "analysis_results": [{"title": "已有数据"}],
            "force_regen": False,
        }
        tools = self._make_tools(state=state)
        extract = [t for t in tools if t.name == "extract_files"][0]
        result = extract.invoke({"preprocess": "true"})
        assert "已存在" in result

    def test_assess_material_empty(self):
        state: AgentState = {"analysis_results": []}
        tools = self._make_tools(state=state)
        assess = [t for t in tools if t.name == "assess_material"][0]
        result = assess.invoke({})
        assert "暂无抽取结果" in result or "无文件" in result

    def test_ask_user_sets_state_flags(self):
        state: AgentState = {}
        tools = self._make_tools(state=state)
        ask = [t for t in tools if t.name == "ask_user"][0]
        result = ask.invoke({"question": "请选择报告类型？A.技术/B.商业"})
        assert state["awaiting_user"] is True
        assert "技术" in state["clarify_question"]

    def test_finish_sets_finished_flag(self):
        state: AgentState = {}
        tools = self._make_tools(state=state)
        finish = [t for t in tools if t.name == "finish"][0]
        result = finish.invoke({})
        assert state["finished"] is True
        assert result == "ok"
