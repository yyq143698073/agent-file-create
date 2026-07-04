"""Prompts for the document-generation agent.

All prompt templates are defined here so they can be versioned, reviewed,
and reused independently of the agent runtime.
"""

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# ── Agent system prompt (used when the LLM needs to decide *what* to ask) ─────

SYSTEM_PROMPT = SystemMessage(
    content=(
        "你是一个文档生成智能体。你的目标：基于用户提供的文件材料生成报告，"
        "并输出文档（md/docx/pdf 视模板而定）。"
    )
)

# ── Clarification-question generation ─────────────────────────────────────────
# Called inside the "clarify" node to turn a quality assessment into 2‑5
# actionable questions for the user.

CLARIFY_QUESTIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个帮助用户明确报告需求的助手。"
            "根据材料评估结果，生成 2-5 个具体、可操作的问题，"
            "方便用户快速选择或简短回答。"
            "如需选项，用 A./B./C. 格式写在同一行。",
        ),
        (
            "human",
            "用户需求：{user_prompt}\n\n"
            "材料评估结果：\n{assessment}\n\n"
            "请生成澄清问题（每行一个问题）：",
        ),
    ]
)

# ── Context injection wrapper ─────────────────────────────────────────────────
# Wraps the current state JSON and user input together for graph invocation.

CONTEXT_TEMPLATE = (
    "当前状态：\n{state_json}\n\n---\n\n{input_text}"
)
