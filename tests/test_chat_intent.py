from __future__ import annotations

import importlib.util
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_INTENT_PATH = _ROOT / "agent_file_create" / "chat" / "intent.py"
_SPEC = importlib.util.spec_from_file_location("chat_intent_for_test", _INTENT_PATH)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

ChatIntent = _MOD.ChatIntent
classify_intent = _MOD.classify_intent


def test_natural_language_status_is_control_intent():
    assert classify_intent("帮我看看当前进度") == ChatIntent.CONTROL_TASK


def test_natural_language_pause_is_control_intent():
    assert classify_intent("先暂停一下") == ChatIntent.CONTROL_TASK


def test_natural_language_regenerate_beats_modify_rule():
    assert (
        classify_intent("按刚才的意见重新生成一版", has_report_content=True)
        == ChatIntent.CONTROL_TASK
    )


def test_kb_question_still_routes_to_kb():
    assert classify_intent("什么是RAG") == ChatIntent.KB_QUERY


def test_natural_language_kb_use_is_control_intent():
    assert classify_intent("切换到产品知识库") == ChatIntent.CONTROL_TASK


def test_natural_language_kb_ask_with_explicit_kb_is_control_intent():
    assert classify_intent("在产品知识库里查一下RAG是什么") == ChatIntent.CONTROL_TASK


def test_natural_language_kb_clear_is_control_intent():
    assert classify_intent("先别用知识库了") == ChatIntent.CONTROL_TASK


def test_modify_feedback_still_routes_to_modify():
    assert (
        classify_intent("请把第三章展开一点，增加案例", has_report_content=True)
        == ChatIntent.MODIFY_REPORT
    )
