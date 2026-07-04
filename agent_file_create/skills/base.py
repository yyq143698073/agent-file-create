"""Skill base types: SkillResult, SkillMeta, and the @skill decorator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass
class SkillResult:
    """Unified return type for all skills."""

    success: bool
    summary: str = ""          # injected into downstream LLM prompts
    data: dict = field(default_factory=dict)   # structured data for frontend / downstream
    error: str = ""
    tokens_used: int = 0

    def to_context(self) -> str:
        """Render this result as a context block for LLM prompts."""
        if not self.success:
            return f"[技能执行失败: {self.error}]"
        return self.summary


@dataclass
class SkillMeta:
    """Metadata for a registered skill."""

    name: str
    description: str
    category: str              # research | analysis | generation | file_processing
    stage: str                 # enrich(大纲前) | research(正文前) | both
    parameters: dict           # JSON Schema for the execute() kwargs
    execute: Callable          # async callable → SkillResult
    timeout_s: int = 60
    max_retries: int = 1


def skill(
    *,
    name: str,
    description: str,
    category: str = "research",
    stage: str = "both",
    parameters: dict | None = None,
    timeout_s: int = 60,
    max_retries: int = 1,
) -> Callable:
    """Decorator that wraps an async function into a SkillMeta.

    Usage::

        @skill(name="web_search", description="搜索互联网", category="research")
        async def web_search(query: str, max_results: int = 5, **kwargs) -> SkillResult:
            ...
    """
    def _wrap(fn: Callable) -> SkillMeta:
        params = dict(parameters or {})
        return SkillMeta(
            name=name,
            description=description,
            category=category,
            stage=stage,
            parameters=params,
            execute=fn,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )

    return _wrap
