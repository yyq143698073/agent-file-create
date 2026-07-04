"""Unified prompts module — single import source for all prompt templates.

Import everything from here regardless of which sub-module defined it:
    from agent_file_create.prompts import (
        SYSTEM_ASSISTANT, CLARIFY_QUESTIONS_PROMPT,
        lobby_prompt, ANSWER_PROMPT, HYDE_PROMPT, ...
    )

Prompt versions are tracked in PROMPT_VERSIONS. When you modify a prompt,
bump its minor version. Call get_prompt_info() to see all versions at runtime.
"""

from agent_file_create.prompts.system import (
    SYSTEM_ASSISTANT,
    SYSTEM_CLASSIFIER,
    SYSTEM_REASONING,
)
from agent_file_create.prompts.agent import (
    SYSTEM_PROMPT,
    CLARIFY_QUESTIONS_PROMPT,
    CONTEXT_TEMPLATE,
)
from agent_file_create.prompts.chat import (
    LOBBY_SYSTEM_TEMPLATE,
    lobby_prompt,
    TASK_CHAT_SYSTEM_TEMPLATE,
    task_chat_prompt,
    SUMMARIZE_HISTORY_PROMPT,
    REWRITE_QUERY_PROMPT,
    CHECK_RELEVANCE_PROMPT,
    FOLLOWUPS_PROMPT,
)
from agent_file_create.prompts.rag import (
    Citation,
    Answer,
    ANSWER_PROMPT,
    ANSWER_COT_PROMPT,
    HYDE_PROMPT,
    DECOMPOSE_PROMPT,
    QUERY_REWRITE_PROMPT,
    MULTI_QUERY_PROMPT,
    STEPBACK_PROMPT,
    QUERY_ROUTE_PROMPT,
    METADATA_FILTER_PROMPT,
)
from agent_file_create.prompts.document import (
    build_extract_prompt,
)

# ── Prompt version registry ──────────────────────────────────────────────────
# Bump minor version when you edit a prompt template.
# The version is logged at startup and can be queried via get_prompt_info().
# Format: "major.minor" — major bumps for rewrites, minor for tweaks.

PROMPT_VERSIONS: dict[str, str] = {
    # system
    "SYSTEM_ASSISTANT": "1.0",
    "SYSTEM_CLASSIFIER": "1.0",
    "SYSTEM_REASONING": "1.0",
    # agent
    "SYSTEM_PROMPT": "1.1",
    "CLARIFY_QUESTIONS_PROMPT": "1.0",
    "CONTEXT_TEMPLATE": "1.0",
    # chat
    "LOBBY_SYSTEM_TEMPLATE": "1.2",
    "TASK_CHAT_SYSTEM_TEMPLATE": "1.1",
    "SUMMARIZE_HISTORY_PROMPT": "1.0",
    "REWRITE_QUERY_PROMPT": "1.0",
    "FOLLOWUPS_PROMPT": "1.0",
    # rag
    "ANSWER_PROMPT": "1.1",
    "ANSWER_COT_PROMPT": "1.0",
    "HYDE_PROMPT": "1.0",
    "DECOMPOSE_PROMPT": "1.0",
    "QUERY_REWRITE_PROMPT": "1.0",
    "MULTI_QUERY_PROMPT": "1.0",
    "STEPBACK_PROMPT": "1.0",
    "QUERY_ROUTE_PROMPT": "1.0",
    "METADATA_FILTER_PROMPT": "1.0",
}


def get_prompt_info() -> dict:
    """Return prompt version information for diagnostics / A/B testing.

    Useful to log at startup and to include in evaluation reports,
    so you can correlate score changes with prompt changes.
    """
    return {
        "versions": dict(PROMPT_VERSIONS),
        "total": len(PROMPT_VERSIONS),
    }


__all__ = [
    "PROMPT_VERSIONS",
    "get_prompt_info",
    # system
    "SYSTEM_ASSISTANT",
    "SYSTEM_CLASSIFIER",
    "SYSTEM_REASONING",
    # agent
    "SYSTEM_PROMPT",
    "CLARIFY_QUESTIONS_PROMPT",
    "CONTEXT_TEMPLATE",
    # chat
    "LOBBY_SYSTEM_TEMPLATE",
    "lobby_prompt",
    "TASK_CHAT_SYSTEM_TEMPLATE",
    "task_chat_prompt",
    "SUMMARIZE_HISTORY_PROMPT",
    "REWRITE_QUERY_PROMPT",
    "CHECK_RELEVANCE_PROMPT",
    "FOLLOWUPS_PROMPT",
    # rag
    "Citation",
    "Answer",
    "ANSWER_PROMPT",
    "ANSWER_COT_PROMPT",
    "HYDE_PROMPT",
    "DECOMPOSE_PROMPT",
    "QUERY_REWRITE_PROMPT",
    "MULTI_QUERY_PROMPT",
    "STEPBACK_PROMPT",
    "QUERY_ROUTE_PROMPT",
    "METADATA_FILTER_PROMPT",
    # document
    "build_extract_prompt",
]
