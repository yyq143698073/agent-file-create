"""Intent classification for chat messages — routes user messages before LLM invocation.

Every user message is classified into one of five intents. This determines:
- Whether the LLM chain is invoked at all (MODIFY, CONTROL, CLARIFY bypass it)
- Which context layers are assembled (QUESTION loads report; KB_QUERY loads KB only)
- How the reply is generated (greeting → rule, general → LLM)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ChatIntent(str, Enum):
    """User intent — determined before any LLM invocation."""

    QUESTION_REPORT = "question_report"   # Asking about report content
    MODIFY_REPORT = "modify_report"       # Wanting to change the report
    CONTROL_TASK = "control_task"         # Task control (pause / resume / regen)
    CLARIFY_ANSWER = "clarify_answer"     # Answering a clarification question
    KB_QUERY = "kb_query"                 # Querying the knowledge base
    GENERAL_CHAT = "general_chat"         # Everything else


# ── Rule-based fast paths ──────────────────────────────────────────────

_SLASH_COMMANDS = {
    "help", "h", "status", "st", "pause", "resume", "cancel", "stop",
    "regen", "regenerate", "prompt", "prompt!", "kb", "files",
    "templates", "template", "append",
}

_MODIFICATION_INDICATORS = [
    "太长", "太短", "精简", "缩短", "删", "删除", "去掉", "移除",
    "增加", "添加", "补充", "加入", "扩展", "展开", "详细",
    "修改", "调整", "更改", "换成", "替换", "改成",
    "优化", "改进", "改善",
    "重点", "侧重", "强调", "突出",
    "多写", "少写", "减少", "扩充", "缩减",
    "重写", "重新", "重做", "再生成",
    "不对", "不好", "不行", "不对的", "有问题", "错误",
    "换一个", "另一种", "换个", "换个风格",
    "太啰嗦", "太繁琐", "太简略", "太笼统", "太抽象", "太具体",
    "数据", "图表", "案例", "分析", "对比",
]


def classify_intent(
    message: str,
    task_status: dict | None = None,
    is_clarify_phase: bool = False,
    has_report_content: bool = False,
    *,
    llm=None,
) -> ChatIntent:
    """Classify user intent before building context or invoking LLM.

    Priority order:
        1. Slash commands → CONTROL_TASK
        2. Clarify phase → CLARIFY_ANSWER
        3. Modification keywords → MODIFY_REPORT (if report exists)
        4. KB query indicators → KB_QUERY
        5. Short question-like messages → QUESTION_REPORT (if report exists)
        6. Ambiguous longer messages → LLM classification (if llm provided)
        7. Everything else → GENERAL_CHAT

    Args:
        message: The user's raw message text.
        task_status: Current task status dict (for phase detection).
        is_clarify_phase: Whether the task is waiting for clarify answers.
        has_report_content: Whether outline.md or content.md exists for this task.
        llm: Optional pre-built LLM. When provided, ambiguous messages (>20 chars)
             that don't match any rule are classified via LLM instead of defaulting
             to GENERAL_CHAT.

    Returns:
        The classified ChatIntent.
    """
    m = (message or "").strip()
    if not m:
        return ChatIntent.GENERAL_CHAT

    # 1. Slash commands → CONTROL_TASK (fastest path)
    if m.startswith("/"):
        cmd = m.split()[0][1:].lower()
        if cmd in _SLASH_COMMANDS:
            return ChatIntent.CONTROL_TASK
        return ChatIntent.GENERAL_CHAT

    # 2. Clarify phase → CLARIFY_ANSWER (unless it's a trivial greeting)
    if is_clarify_phase:
        if len(m) <= 5 and m.lower() in {"hi", "hello", "你好", "在吗"}:
            return ChatIntent.GENERAL_CHAT
        return ChatIntent.CLARIFY_ANSWER

    # 3. Trivial greetings — never report questions, even when report exists
    if _is_trivial_greeting(m):
        return ChatIntent.GENERAL_CHAT

    # 4. Modification intent → MODIFY_REPORT (keyword match, fast)
    if has_report_content and _detect_modification_intent_rule(m):
        return ChatIntent.MODIFY_REPORT

    # 5. KB query intent → KB_QUERY (explicit KB mention or conceptual question)
    #    Only triggers when NOT a modification intent (checked above)
    if _detect_kb_query(m):
        return ChatIntent.KB_QUERY
    if has_report_content and _detect_modification_intent_rule(m):
        return ChatIntent.MODIFY_REPORT

    # 5. Short message with report context → QUESTION_REPORT
    if has_report_content and _looks_like_question(m):
        return ChatIntent.QUESTION_REPORT

    # 6. Ambiguous longer messages → LLM classification (if available)
    #    Messages >20 chars that didn't match any rule are likely
    #    meaningful — use LLM to avoid misclassifying them as GENERAL_CHAT.
    if llm is not None and has_report_content and len(m) > 20:
        llm_intent = llm_classify_intent(message, task_status, llm=llm)
        if llm_intent != ChatIntent.GENERAL_CHAT:
            logger.debug("intent_classify llm_override rule=GENERAL_CHAT llm=%s msg=%.60s",
                         llm_intent.value, m)
            return llm_intent

    # 7. Fallback
    return ChatIntent.GENERAL_CHAT


_KB_QUERY_KEYWORDS = [
    "知识库", "资料库", "文档库",
    "有没有文档", "有没有资料", "有没有文件",
    "查一下", "搜一下", "检索", "搜索",
    "帮我查", "帮我找", "帮我搜",
]


def _detect_kb_query(message: str) -> bool:
    """Detect if a message is asking to search the knowledge base."""
    m = (message or "").strip()
    if not m:
        return False
    # Strong indicators: explicit KB mention
    if any(kw in m for kw in _KB_QUERY_KEYWORDS):
        return True
    # Pattern match: "关于XX的文档/资料"
    if ("文档" in m or "资料" in m) and len(m) <= 60:
        return True
    # Conceptual / framework questions — likely answerable from KB
    if any(kw in m for kw in [
        "是什么", "什么是", "怎样", "如何", "怎么", "为什么",
        "定义", "概念", "用途", "作用", "价值", "特点", "特征",
        "介绍一下", "解释一下", "说明一下",
        "有什么用", "能做什么", "可以用来", "怎么用",
    ]):
        return True
    return False


_GREETINGS = {
    "在吗", "在不在", "在了吗", "在不在呀",
    "你好", "您好", "你好啊", "嗨", "hi", "hello", "hey",
    "早上好", "下午好", "晚上好", "晚安", "中午好",
    "谢谢", "谢谢你", "感谢", "thx", "thanks", "thank you",
}


def _is_trivial_greeting(message: str) -> bool:
    """Check if message is a pure greeting/social chat (not a question)."""
    m = (message or "").strip()
    return m.lower() in _GREETINGS or len(m) <= 3 and m.lower() in {"hi", "嗨", "hey", "在吗", "你好"}


def _detect_modification_intent_rule(message: str) -> bool:
    """Fast rule-based modification intent detection.

    Stricter than the old keyword-only approach:
    - Message must be <= 120 chars
    - Must contain a modification keyword
    - Must NOT be a pure question (question marks, 吗-suffix without strong action verb)
    """
    m = (message or "").strip()
    if not m or len(m) > 120:
        return False
    if m.startswith("/"):
        return False
    if not any(kw in m for kw in _MODIFICATION_INDICATORS):
        return False
    # Question marks usually indicate a question, not a modification command
    if "?" in m or "？" in m:
        return False
    # 吗-suffix: "能帮我优化一下吗" → modify; "这个数据对吗" → question
    if m.endswith("吗") and len(m) <= 20:
        # Strong action verbs override the question form
        strong_verbs = {"优化", "修改", "重写", "删", "删除", "重做", "替换", "改成"}
        if not any(v in m for v in strong_verbs):
            return False
    return True


def _looks_like_question(message: str) -> bool:
    """Heuristic: does this message look like a question about the report?"""
    m = (message or "").strip()
    if not m:
        return False
    # Question words
    question_words = (
        "什么", "怎么", "如何", "为什么", "是否",
        "能不能", "可不可以", "能不能", "请解释", "请说明",
    )
    if any(m.startswith(q) for q in question_words):
        return True
    # Question marks
    if "?" in m or "？" in m:
        return True
    # 吗-suffix questions (对吗, 是吗, 行吗, etc.)
    if m.endswith("吗") and len(m) <= 30:
        return True
    # Short non-command messages: likely questions
    # But exclude messages with strong modification intent keywords
    _strong_mod_keywords = {"优化", "修改", "重写", "删", "重做", "替换", "改成", "去掉", "移除"}
    if len(m) <= 60 and not any(kw in m for kw in _strong_mod_keywords):
        return True
    return False


# ── LLM-based intent classification (for ambiguous cases) ──────────────

def llm_classify_intent(
    message: str,
    task_status: dict | None = None,
    *,
    llm=None,
) -> ChatIntent:
    """Use a lightweight LLM call to classify ambiguous intents.

    Only called when rule-based detection can't determine the intent.
    Uses a tiny prompt — single token output, temperature=0.

    Args:
        message: User message.
        task_status: Current task status (provides context about stage).
        llm: Optional pre-built LLM. If None, classification falls back to GENERAL_CHAT.
    """
    if llm is None:
        logger.debug("intent_classify no_llm — falling back to GENERAL_CHAT")
        return ChatIntent.GENERAL_CHAT

    stage = ""
    if isinstance(task_status, dict):
        stage = str(task_status.get("stage") or task_status.get("status") or "")

    prompt = (
        "分类用户意图。只输出一个选项名。\n\n"
        f"任务阶段：{stage or '无任务'}\n"
        f"用户消息：{message[:200]}\n\n"
        "选项：\n"
        "- question_report：用户对报告内容提问、询问、核实\n"
        "- modify_report：用户要求修改、重写、完善报告\n"
        "- general_chat：以上都不符合的普通对话\n\n"
        "意图："
    )

    try:
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = str(resp.content if hasattr(resp, "content") else resp).strip().lower()

        if "question" in raw:
            return ChatIntent.QUESTION_REPORT
        if "modify" in raw:
            return ChatIntent.MODIFY_REPORT
        return ChatIntent.GENERAL_CHAT
    except Exception as e:
        logger.debug("intent_classify llm_failed err=%s", e)
        return ChatIntent.GENERAL_CHAT
