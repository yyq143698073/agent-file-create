"""Agent tools for document generation — standalone, testable, no closure capture.

Every tool receives *state*, *file_paths*, *user_prompt*, *task_id* and
*template_dir_override* explicitly via the factory function so there is no
hidden coupling to a ``DocumentAgent`` instance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

from langchain_core.tools import tool
from pydantic import BaseModel

from agent_file_create.agent.state import AgentState
from agent_file_create.utils import retry_call

logger = logging.getLogger(__name__)


class EmptyInput(BaseModel):
    """Schema for tools that take no arguments."""


def create_tools(
    *,
    state: AgentState,
    file_paths: List[str],
    user_prompt: str,
    task_id: str,
    template_dir_override: str,
):
    """Return the 7 tools bound to the supplied state and context."""

    # ── 1. extract_files ────────────────────────────────────────────

    @tool
    def extract_files(preprocess: str = "true") -> str:
        """从已上传文件中抽取结构化信息 (title/keywords/summary/key_points/data/conclusion/prediction)。

参数 preprocess: "true"(默认, 进行图片预处理) 或 "false"(跳过图片预处理)。
如果所有文件都返回 error, 说明格式不支持或文件损坏, 应调用 ask_user 告知用户。
        """
        if (
            not state.get("force_regen")
            and isinstance(state.get("analysis_results"), list)
            and state.get("analysis_results")
        ):
            return "analysis_results 已存在, 跳过抽取。"

        from agent_file_create.document.extractor import (
            deduplicate_extracted_results,
            extract_from_file,
        )
        from agent_file_create.config import MAX_WORKERS_DEFAULT
        from concurrent.futures import ThreadPoolExecutor, as_completed

        pp = str(preprocess or "").strip().lower() != "false"
        fps = list(file_paths)
        results: List[dict] = [{}] * len(fps)
        max_workers = max(1, min(int(MAX_WORKERS_DEFAULT), len(fps)))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fut_map = {
                pool.submit(retry_call, extract_from_file, fp, preprocess=pp): i
                for i, fp in enumerate(fps)
            }
            for fut in as_completed(fut_map):
                idx = fut_map[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"error": str(e), "_file": Path(fps[idx]).name}
                if isinstance(res, dict):
                    res["_file"] = Path(fps[idx]).name
                results[idx] = res
        state["analysis_results"] = deduplicate_extracted_results(results)
        state["force_regen"] = False
        return f"已抽取 {len(results)} 个文件，去重后保留 {len(state['analysis_results'])} 份材料。"

    # ── 2. assess_material ──────────────────────────────────────────

    @tool(args_schema=EmptyInput)
    def assess_material() -> str:
        """评估当前抽取结果的质量。

每行输出包含文件名、标题、摘要预览、filled 字段数(满分7)。
判断标准: 多数文件 filled<3 说明质量差; filled<5 说明信息可能不足, 应考虑 ask_user。
        """
        from agent_file_create.preprocessor import compute_quality_metrics

        ar = (
            state.get("analysis_results")
            if isinstance(state.get("analysis_results"), list)
            else []
        )
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
        return "\n".join(lines).strip() or "（暂无抽取结果）"

    # ── 3. generate_outline ─────────────────────────────────────────

    @tool(args_schema=EmptyInput)
    def generate_outline() -> str:
        """根据抽取结果和用户需求生成报告大纲(Markdown)。

输出包含 # 标题、## 主章节(>=3个)、### 子节的完整大纲。
调用前确保 analysis_results 存在且 assess_material 评估通过。
        """
        if not state.get("force_regen") and str(
            state.get("outline") or ""
        ).strip():
            return "outline 已存在, 跳过生成。"

        from agent_file_create.document.outline_generator import (
            generate_outline as _gen,
        )

        ar = state.get("analysis_results") or []
        multimodal = {f"source_{i}": r for i, r in enumerate(ar)}
        outline = retry_call(_gen, multimodal, user_prompt)
        state["outline"] = outline
        state["force_regen"] = False
        return f"outline 已生成 (chars={len(outline or '')})。"

    # ── 4. generate_content ─────────────────────────────────────────

    @tool(args_schema=EmptyInput)
    def generate_content() -> str:
        """根据大纲和抽取结果生成报告正文(Markdown)。

调用前确保 outline 已生成。
        """
        if not state.get("force_regen") and str(
            state.get("content") or ""
        ).strip():
            return "content 已存在, 跳过生成。"

        from agent_file_create.document.content_generator import (
            generate_full_content as _gen,
        )

        ar = state.get("analysis_results") or []
        multimodal = {f"source_{i}": r for i, r in enumerate(ar)}
        outline = str(state.get("outline") or "")
        content = retry_call(
            _gen, outline, multimodal, user_prompt, task_id=task_id,
        )
        state["content"] = content
        state["force_regen"] = False
        return f"content 已生成 (chars={len(content or '')})。"

    # ── 5. render_templates ─────────────────────────────────────────

    @tool(args_schema=EmptyInput)
    def render_templates() -> str:
        """将大纲和正文渲染为最终文档文件(md/docx/pdf), 输出到 result/<task_id>/ 目录。

调用前确保 content 已生成。
        """
        from agent_file_create.document_service import generate_document

        ar = state.get("analysis_results") or []
        result = retry_call(
            generate_document,
            user_prompt=user_prompt,
            analysis_results=ar,
            document_type="report",
            task_id=task_id,
            template_dir_override=template_dir_override,
        )
        state["outputs"] = result.get("rendered_outputs") or []
        state["output_dir"] = result.get("output_dir") or ""
        state["outline"] = (
            result.get("document_outline") or state.get("outline") or ""
        )
        state["content"] = (
            result.get("document_content") or state.get("content") or ""
        )
        return (
            f"已生成文档, output_dir={state.get('output_dir') or ''}"
            f" outputs={len(state.get('outputs') or [])}。"
        )

    # ── 6. ask_user ─────────────────────────────────────────────────

    @tool
    def ask_user(question: str) -> str:
        """向用户确认报告偏向性需求。assess_material 之后必须调用此工具。

question: 向用户提出的问题。2-5个换行分隔的问题。
质量差时聚焦: 是否补充材料、对内容缺失的接受度、是否缩小报告范围。
质量好时聚焦: 使用场景/目标受众/侧重点/篇幅/语气风格等偏好。
如需选项, 用 A./B./C. 格式写在同一行。
示例: "请确认以下偏好以便生成更精准的报告：
1. 报告使用场景？A.内部决策/B.对外汇报/C.学术发表
2. 侧重点偏向？A.技术深度/B.商业价值/C.风险分析
3. 篇幅偏好？A.3000字精简/B.5000字标准/C.8000字详尽"

重要: 调用后系统会暂停等待用户回复。不要在 ask_user 之后继续调用其他工具。
        """
        q = str(question or "").strip() or "请补充你希望生成文档的侧重点/受众/篇幅/风格等信息。"
        state["awaiting_user"] = True
        state["clarify_question"] = q
        return q

    # ── 7. finish ───────────────────────────────────────────────────

    @tool(args_schema=EmptyInput)
    def finish() -> str:
        """结束任务。调用条件: render_templates 已执行且 output_dir 存在。"""
        state["finished"] = True
        return "ok"

    return [
        extract_files,
        assess_material,
        generate_outline,
        generate_content,
        render_templates,
        ask_user,
        finish,
    ]
