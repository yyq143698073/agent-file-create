from __future__ import annotations

import importlib.util
from pathlib import Path


class _FakeKB:
    def list_kb(self) -> list[str]:
        return ["产品", "研发资料"]


_ROOT = Path(__file__).resolve().parents[1]
_HANDLER_PATH = _ROOT / "agent_file_create" / "chat" / "handler.py"
_SPEC = importlib.util.spec_from_file_location("chat_handler_for_test", _HANDLER_PATH)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
_MOD.get_kb = lambda: _FakeKB()

ChatHandler = _MOD.ChatHandler


def _make_handler():
    return object.__new__(ChatHandler)


def test_parse_natural_kb_use_action():
    handler = _make_handler()
    act = handler._parse_chat_action("切换到产品知识库", None)
    assert act == {"type": "kb_use", "kb": "产品"}


def test_parse_natural_kb_query_with_explicit_kb():
    handler = _make_handler()
    act = handler._parse_chat_action("在产品知识库里查一下 RAG 是什么", None)
    assert act == {"type": "kb_ask", "kb": "产品", "question": "RAG 是什么"}


def test_parse_natural_kb_create_action():
    handler = _make_handler()
    act = handler._parse_chat_action("新建一个项目复盘知识库", None)
    assert act == {"type": "kb_create", "kb": "项目复盘"}


def test_parse_natural_kb_clear_action():
    handler = _make_handler()
    act = handler._parse_chat_action("先别用知识库了", None)
    assert act == {"type": "kb_clear"}
