"""Document generation agent powered by LangGraph.

Replaces the deprecated langchain_classic (create_react_agent / AgentExecutor /
ConversationSummaryBufferMemory) with langgraph.prebuilt.create_react_agent
and MemorySaver checkpointing.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from agent_file_create.config import (
    CONTENT_API_ENDPOINT,
    CONTENT_API_KEY,
    CONTENT_API_STYLE,
    CONTENT_MODEL_NAME,
    MODEL_NAME,
    MODEL_TIMEOUT_SHORT,
    PLANNER_API_ENDPOINT,
    PLANNER_API_KEY,
    PLANNER_API_STYLE,
    PLANNER_MODEL_NAME,
)
from agent_file_create.llm_factory import get_chat_model
from agent_file_create.utils import safe_json

logger = logging.getLogger(__name__)


# ── Tools ─────────────────────────────────────────────────────────────────
# Bound to a DocumentAgent instance so they can read/write agent.state.


def _build_tools(agent: "DocumentAgent"):
    from langchain_core.tools import tool

    @tool
    def extract_files(preprocess: str = "true") -> str:
        """从已上传文件中抽取结构化信息 (title/keywords/summary/key_points/data/conclusion/prediction)。

参数 preprocess: "true"(默认, 进行图片预处理) 或 "false"(跳过图片预处理)。
如果所有文件都返回 error, 说明格式不支持或文件损坏, 应调用 ask_user 告知用户。
        """
        if (
            not agent.state.get("force_regen")
            and isinstance(agent.state.get("analysis_results"), list)
            and agent.state.get("analysis_results")
        ):
            return "analysis_results 已存在, 跳过抽取。"

        from agent_file_create.document.extractor import extract_from_file

        pp = str(preprocess or "").strip().lower() != "false"
        results: List[dict] = []
        for fp in agent.file_paths:
            res = extract_from_file(fp, preprocess=pp)
            res["_file"] = Path(fp).name
            results.append(res)
        agent.state["analysis_results"] = results
        agent.state["force_regen"] = False
        return f"已抽取 {len(results)} 个文件。"

    @tool
    def assess_material(dummy: str = "") -> str:
        """评估当前抽取结果的质量。输入留空即可。

每行输出包含文件名、标题、摘要预览、filled 字段数(满分7)。
判断标准: 多数文件 filled<3 说明质量差; filled<5 说明信息可能不足, 应考虑 ask_user。
        """
        from agent_file_create.preprocessor import compute_quality_metrics

        ar = (
            agent.state.get("analysis_results")
            if isinstance(agent.state.get("analysis_results"), list)
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
                [
                    x
                    for x in [
                        title,
                        summary[:160] + ("…" if len(summary) > 160 else ""),
                    ]
                    if x
                ]
            ).strip()
            if err:
                s = (s + " | " if s else "") + ("ERROR=" + err[:120])
            head = (fn + ": ") if fn else ""
            lines.append(head + (s or "（无摘要）") + " | " + qtxt)
        return "\n".join(lines).strip() or "（暂无抽取结果）"

    @tool
    def generate_outline(dummy: str = "") -> str:
        """根据抽取结果和用户需求生成报告大纲(Markdown)。输入留空。

输出包含 # 标题、## 主章节(>=3个)、### 子节的完整大纲。
调用前确保 analysis_results 存在且 assess_material 评估通过。
        """
        if not agent.state.get("force_regen") and str(
            agent.state.get("outline") or ""
        ).strip():
            return "outline 已存在, 跳过生成。"

        from agent_file_create.document.outline_generator import (
            generate_outline as _gen,
        )

        ar = agent.state.get("analysis_results") or []
        multimodal = {f"source_{i}": r for i, r in enumerate(ar)}
        outline = _gen(multimodal, agent.user_prompt)
        agent.state["outline"] = outline
        agent.state["force_regen"] = False
        return f"outline 已生成 (chars={len(outline or '')})。"

    @tool
    def generate_content(dummy: str = "") -> str:
        """根据大纲和抽取结果生成报告正文(Markdown)。输入留空。

调用前确保 outline 已生成。
        """
        if not agent.state.get("force_regen") and str(
            agent.state.get("content") or ""
        ).strip():
            return "content 已存在, 跳过生成。"

        from agent_file_create.document.content_generator import (
            generate_full_content as _gen,
        )

        ar = agent.state.get("analysis_results") or []
        multimodal = {f"source_{i}": r for i, r in enumerate(ar)}
        outline = str(agent.state.get("outline") or "")
        content = _gen(outline, multimodal, agent.user_prompt, task_id=agent.task_id)
        agent.state["content"] = content
        agent.state["force_regen"] = False
        return f"content 已生成 (chars={len(content or '')})。"

    @tool
    def render_templates(dummy: str = "") -> str:
        """将大纲和正文渲染为最终文档文件(md/docx/pdf), 输出到 result/<task_id>/ 目录。

输入留空。调用前确保 content 已生成。
        """
        from agent_file_create.document_service import generate_document

        ar = agent.state.get("analysis_results") or []
        result = generate_document(
            user_prompt=agent.user_prompt,
            analysis_results=ar,
            document_type="report",
            task_id=agent.task_id,
            template_dir_override=agent.template_dir_override,
        )
        agent.state["outputs"] = result.get("rendered_outputs") or []
        agent.state["output_dir"] = result.get("output_dir") or ""
        agent.state["outline"] = (
            result.get("document_outline") or agent.state.get("outline") or ""
        )
        agent.state["content"] = (
            result.get("document_content") or agent.state.get("content") or ""
        )
        return (
            f"已生成文档, output_dir={agent.state.get('output_dir') or ''}"
            f" outputs={len(agent.state.get('outputs') or [])}。"
        )

    @tool
    def ask_user(question: str) -> str:
        """当材料信息不足、与需求不匹配、或侧重点不清晰时, 向用户提问。

question: 对用户提出的问题。可以是单个问题, 或换行分隔的多个问题(2-6个)。
如需选项, 用 A./B./C. 格式写在同一行。
示例: "报告偏向技术深度还是行业广度？A.技术深度/B.行业广度/C.两者兼顾"

重要: 调用后系统会暂停等待用户回复。在确实需要时才调用, 不要滥用。
工具执行后会立即返回, 请不要在调用 ask_user 后继续调用其他工具。
        """
        q = str(question or "").strip() or "请补充你希望生成文档的侧重点/受众/篇幅/风格等信息。"
        agent.state["awaiting_user"] = True
        agent.state["clarify_question"] = q
        return q

    @tool
    def finish(dummy: str = "") -> str:
        """结束任务。调用条件: render_templates 已执行且 output_dir 存在。输入留空。"""
        agent.state["finished"] = True
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


# ── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
你是一个文档生成智能体。你的目标：基于用户提供的文件材料生成报告，并输出文档（md/docx/pdf 视模板而定）。

工作流程（严格按顺序推进）：
1) 首先调用 extract_files 抽取所有文件的结构化信息。
2) 抽取后调用 assess_material 判断材料完整度。
   - 如果大部分文件解析失败（超过一半返回 error），必须先 ask_user 告知用户并建议重新上传或检查文件格式。
   - 如果材料明显与用户需求不相关（如用户要财务报告但材料全是技术文档），必须先 ask_user 确认。
   - 如果材料信息不足以支撑报告（摘要质量差、关键字段大量缺失），必须先 ask_user 请用户补充侧重点或材料。
3) 只有材料评估通过后，才依次调用 generate_outline → generate_content → render_templates。
4) 收到用户澄清回复后，根据当前进度继续推进（已有大纲则直接 generate_content，已有正文则 render_templates）。
5) render_templates 执行成功后调用 finish 结束。

重要约束：
- 调用 ask_user 后，你必须立即停止，等待用户回复。不要在 ask_user 之后调用任何其他工具。
- 如果用户已提供过澄清信息 (user_clarifications)，应直接推进而非重复提问。"""


# ── DocumentAgent ─────────────────────────────────────────────────────────


class DocumentAgent:
    def __init__(
        self,
        *,
        task_id: str,
        user_prompt: str,
        file_paths: List[str],
        template_dir_override: Optional[str] = None,
    ) -> None:
        self.task_id = str(task_id)
        self.user_prompt = str(user_prompt or "")
        self.file_paths = [str(x) for x in (file_paths or [])]
        self.template_dir_override = template_dir_override
        self.state: Dict[str, Any] = {
            "task_id": self.task_id,
            "force_regen": False,
            "file_paths": list(self.file_paths),
            "template_dir_override": str(self.template_dir_override or ""),
        }
        self._agent_llm = self._build_agent_llm()
        self._tools = _build_tools(self)
        self._graph = self._build_graph()

    # ── LLM ────────────────────────────────────────────────────────────

    def _build_agent_llm(self):
        style = (PLANNER_API_STYLE or CONTENT_API_STYLE or "").strip().lower()
        endpoint = (PLANNER_API_ENDPOINT or CONTENT_API_ENDPOINT or "").strip()
        model = (PLANNER_MODEL_NAME or CONTENT_MODEL_NAME or MODEL_NAME).strip()
        key = (PLANNER_API_KEY or CONTENT_API_KEY or "").strip()
        if not style:
            if endpoint and "/v1/" in endpoint:
                style = "openai"
            else:
                style = "ollama"

        return get_chat_model(
            style=style,
            model=model,
            endpoint=endpoint,
            api_key=key,
            temperature=0.2,
            max_tokens=420,
            timeout_s=int(MODEL_TIMEOUT_SHORT),
        )

    # ── Graph ──────────────────────────────────────────────────────────

    def _build_graph(self):
        checkpointer = MemorySaver()
        return create_react_agent(
            model=self._agent_llm,
            tools=self._tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )

    # ── Run ────────────────────────────────────────────────────────────

    def run(
        self,
        *,
        max_turns: int = 6,
        human_input_fn: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        self.state.pop("awaiting_user", None)
        self.state.pop("clarify_question", None)
        self.state.pop("finished", None)

        config = {"configurable": {"thread_id": self.task_id}}

        start_input = (
            f"任务ID：{self.task_id}\n"
            "请开始推进任务。\n"
            "用户需求：\n"
            f"{self.user_prompt or '生成一份报告'}"
        )

        # Helper: run one graph invocation and extract final text
        def _invoke_graph(input_text: str) -> str:
            state_json = safe_json(
                {k: v for k, v in self.state.items() if k not in ("awaiting_user", "clarify_question")},
                2600,
            )
            full_input = f"当前状态(state_json)：\n{state_json}\n\n---\n\n{input_text}"
            result = self._graph.invoke(
                {"messages": [HumanMessage(content=full_input)]},
                config=config,
            )
            msgs = result.get("messages", [])
            final = ""
            for msg in reversed(msgs if isinstance(msgs, list) else []):
                content = getattr(msg, "content", "")
                if isinstance(content, str) and content.strip():
                    final = content
                    break
            return final

        try:
            text = _invoke_graph(start_input)
            self.state["last_output"] = text

            # ── Handle ask_user ──────────────────────────────────────
            if self.state.get("awaiting_user"):
                question = str(self.state.get("clarify_question") or text or "").strip()
                if human_input_fn is None:
                    self.state["need_user"] = True
                    self.state["question"] = question
                    return dict(self.state)

                answer = str(human_input_fn(question) or "").strip()
                self.state["need_user"] = False
                self.state.pop("awaiting_user", None)
                self.state.pop("clarify_question", None)
                if answer:
                    self.state["user_clarifications"] = (
                        str(self.state.get("user_clarifications") or "") + "\n" + answer
                    ).strip()
                    resume_input = f"用户补充信息：\n{answer}\n\n请根据补充信息继续推进任务。"
                else:
                    resume_input = "用户未补充更多信息，请在当前信息基础上继续推进。"

                text = _invoke_graph(resume_input)
                self.state["last_output"] = text

                # Check for second ask_user
                if self.state.get("awaiting_user"):
                    if human_input_fn is None:
                        self.state["need_user"] = True
                        self.state["question"] = str(
                            self.state.get("clarify_question") or text or ""
                        ).strip()
                        return dict(self.state)
                    # Could handle another round, but for now just continue
                    self.state.pop("awaiting_user", None)
                    self.state.pop("clarify_question", None)

            self.state["finished"] = True

        except Exception as exc:
            logger.warning(f"agent_graph_error task={self.task_id} err={exc}")
            self.state["last_output"] = f"Agent error: {exc}"
            self.state["finished"] = True

        return dict(self.state)
