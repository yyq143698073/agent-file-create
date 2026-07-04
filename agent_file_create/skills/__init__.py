"""Skill registry: auto-discovery, stage filtering, LLM tool-prompt generation."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from agent_file_create.skills.base import SkillMeta, SkillResult

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).resolve().parent


@dataclass
class SkillCall:
    """A parsed skill invocation from the LLM."""
    skill_name: str
    params: dict


class SkillRegistry:
    """Auto-discovers skills from the skills/ package and provides lookup."""

    def __init__(self) -> None:
        self._skills: Dict[str, SkillMeta] = {}
        self._discovered = False

    # ── Discovery ─────────────────────────────────────────────────────────

    def discover(self) -> int:
        """Scan the skills/ package tree and register all modules with a SKILL_META.

        Returns the number of skills registered.
        """
        if self._discovered:
            return len(self._skills)

        # Walk all sub-packages under skills/
        for _, name, is_pkg in pkgutil.iter_modules(
            [str(_SKILLS_DIR)], prefix="agent_file_create.skills."
        ):
            if is_pkg:
                self._discover_package(name)
            else:
                self._discover_module(name)

        self._discovered = True
        logger.info("SkillRegistry discovered %d skills: %s",
                     len(self._skills), list(self._skills.keys()))
        return len(self._skills)

    def _discover_package(self, pkg_name: str) -> None:
        """Recursively scan a sub-package for skill modules."""
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as exc:
            logger.debug("skill_package_import_failed pkg=%s err=%s", pkg_name, exc)
            return

        pkg_path = getattr(pkg, "__path__", None)
        if not pkg_path:
            return
        for _, name, is_pkg in pkgutil.iter_modules(pkg_path, prefix=pkg_name + "."):
            if is_pkg:
                self._discover_package(name)
            else:
                self._discover_module(name)

    def _discover_module(self, mod_name: str) -> None:
        """Import one module and register its SKILL_META if present."""
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            logger.debug("skill_import_failed mod=%s err=%s", mod_name, exc)
            return

        meta = getattr(mod, "SKILL_META", None)
        if not isinstance(meta, SkillMeta):
            return
        self.register(meta)

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, meta: SkillMeta) -> None:
        if meta.name in self._skills:
            logger.warning("skill_duplicate name=%s — overwriting", meta.name)
        self._skills[meta.name] = meta

    def unregister(self, name: str) -> None:
        self._skills.pop(name, None)

    # ── Query ─────────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[SkillMeta]:
        return self._skills.get(name)

    def list_all(self) -> List[SkillMeta]:
        return list(self._skills.values())

    def get_for_stage(self, stage: str) -> List[SkillMeta]:
        """Return skills applicable to *stage* (enrich / research)."""
        return [
            s for s in self._skills.values()
            if s.stage in (stage, "both")
        ]

    @property
    def skill_names(self) -> List[str]:
        return list(self._skills.keys())

    # ── LLM integration ───────────────────────────────────────────────────

    def build_tools_prompt(self, stage: str) -> str:
        """Build a prompt section listing available skills for the given stage.

        The LLM uses this to decide which skills to invoke.
        """
        candidates = self.get_for_stage(stage)
        if not candidates:
            return "（无可用技能）"

        lines = ["可用技能（你可以选择调用以获取更多信息）："]
        for i, s in enumerate(candidates, 1):
            params_desc = ""
            if s.parameters:
                params_desc = " 参数: " + ", ".join(
                    f"{k}({v.get('description', v.get('type', 'string'))})"
                    for k, v in s.parameters.items()
                )
            lines.append(f"{i}. {s.name} — {s.description}{params_desc}")

        lines.append("")
        lines.append("请判断是否需要调用技能。如果需要，输出JSON格式的调用列表：")
        lines.append('{"skills": [{"skill": "技能名", "params": {...}}, ...]}')
        lines.append("如果不需要，输出：")
        lines.append('{"skills": []}')
        return "\n".join(lines)

    def parse_skill_calls(self, llm_output: str) -> List[SkillCall]:
        """Parse the LLM's skill selection output into SkillCall objects."""
        import json as _json

        text = (llm_output or "").strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)
            text = "\n".join(text[1:]) if len(text) > 1 else text[0]
        if text.endswith("```"):
            text = text[: text.rfind("```")].strip()

        try:
            obj = _json.loads(text)
        except _json.JSONDecodeError:
            # Try to extract a JSON object from the text
            import re
            m = re.search(r'\{[^{}]*"skills"\s*:\s*\[.*?\][^{}]*\}', text, re.DOTALL)
            if m:
                try:
                    obj = _json.loads(m.group(0))
                except _json.JSONDecodeError:
                    return []
            else:
                return []

        skills_list = obj.get("skills") if isinstance(obj, dict) else []
        if not isinstance(skills_list, list):
            return []

        calls: List[SkillCall] = []
        for item in skills_list:
            if not isinstance(item, dict):
                continue
            name = str(item.get("skill") or "").strip()
            if not name or name not in self._skills:
                continue
            params = item.get("params") if isinstance(item.get("params"), dict) else {}
            calls.append(SkillCall(skill_name=name, params=params))
        return calls

    # ── Execution ─────────────────────────────────────────────────────────

    async def execute(self, name: str, **params) -> SkillResult:
        """Execute a skill by name with the given parameters."""
        meta = self._skills.get(name)
        if meta is None:
            return SkillResult(success=False, error=f"未知技能: {name}")

        import asyncio

        for attempt in range(meta.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    meta.execute(**params),
                    timeout=meta.timeout_s,
                )
                if isinstance(result, SkillResult):
                    return result
                # Allow plain dict returns
                if isinstance(result, dict):
                    return SkillResult(success=True, data=result)
                return SkillResult(success=True, summary=str(result))
            except asyncio.TimeoutError:
                if attempt < meta.max_retries:
                    continue
                return SkillResult(
                    success=False,
                    error=f"技能 {name} 执行超时 ({meta.timeout_s}s)",
                )
            except Exception as exc:
                if attempt < meta.max_retries:
                    continue
                return SkillResult(
                    success=False,
                    error=f"技能 {name} 执行失败: {str(exc)[:200]}",
                )

    async def execute_calls(
        self,
        calls: List[SkillCall],
        *,
        max_concurrent: int = 0,
        budget_ms: int = 0,
    ) -> List[tuple[SkillCall, SkillResult]]:
        """Execute multiple skill calls in parallel with optional budget control.

        Args:
            calls: List of skill calls to execute.
            max_concurrent: Max parallel executions (0 = unlimited).
            budget_ms: Total time budget in milliseconds (0 = unlimited).
                       Calls exceeding the budget are skipped gracefully.

        Returns:
            List of (call, result) pairs. Results for skipped calls have
            success=False with a "budget exceeded" error.
        """
        import asyncio
        import time as _t

        if not calls:
            return []

        # Apply concurrency limit
        _limit = max_concurrent if max_concurrent > 0 else len(calls)
        _sem = asyncio.Semaphore(_limit)

        async def _one(call: SkillCall) -> tuple[SkillCall, SkillResult]:
            async with _sem:
                result = await self.execute(call.skill_name, **call.params)
                return call, result

        if budget_ms <= 0:
            return await asyncio.gather(*[_one(c) for c in calls])

        # Budget-controlled execution
        budget_s = budget_ms / 1000.0
        t0 = _t.perf_counter()
        tasks = [asyncio.create_task(_one(c)) for c in calls]
        results: List[tuple[SkillCall, SkillResult]] = []

        for i, task in enumerate(tasks):
            elapsed = _t.perf_counter() - t0
            remaining = budget_s - elapsed
            if remaining <= 0:
                # Budget exhausted — skip remaining calls
                for j in range(i, len(tasks)):
                    tasks[j].cancel()
                    results.append((
                        calls[j],
                        SkillResult(
                            success=False,
                            error=f"技能执行预算超限 ({budget_ms}ms)，已跳过",
                        ),
                    ))
                break
            try:
                call_result = await asyncio.wait_for(task, timeout=remaining)
                results.append(call_result)
            except asyncio.TimeoutError:
                task.cancel()
                results.append((
                    calls[i],
                    SkillResult(
                        success=False,
                        error=f"技能执行超时 (预算 {budget_ms}ms)",
                    ),
                ))

        return results


# ── Module-level singleton ────────────────────────────────────────────────────

_registry: Optional[SkillRegistry] = None


def get_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
        _registry.discover()
    return _registry


def reset_registry() -> None:
    """Reset the global skill registry (useful for testing)."""
    global _registry
    _registry = None
