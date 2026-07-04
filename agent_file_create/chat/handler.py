import json
import logging
import random
import re
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from agent_file_create.chat.history import TaskChatMessageHistory
from agent_file_create.chat.intent import ChatIntent, classify_intent
from agent_file_create.prompts import (
    FOLLOWUPS_PROMPT,
    REWRITE_QUERY_PROMPT,
    SUMMARIZE_HISTORY_PROMPT,
    lobby_prompt,
    task_chat_prompt,
)
from agent_file_create.config import (
    CONTENT_API_ENDPOINT,
    CONTENT_API_KEY,
    CONTENT_API_STYLE,
    CONTENT_MODEL_NAME,
)
from agent_file_create.llm_factory import get_chat_model
from agent_file_create.rag import get_kb
from agent_file_create.rag.retriever import KnowledgeBaseRetriever
from agent_file_create.task.manager import TaskManager

logger = logging.getLogger(__name__)


class ChatHandler:
    def __init__(self, task_manager: Optional[TaskManager] = None, regenerate_fn: Any = None):
        self._task_manager = task_manager or TaskManager()
        self._regenerate_fn = regenerate_fn  # callable(task_id, mode) -> (bool, str)

        # Chat LLM — higher token budget for substantive replies
        self._chat_llm = get_chat_model(
            style=CONTENT_API_STYLE,
            model=CONTENT_MODEL_NAME,
            endpoint=CONTENT_API_ENDPOINT,
            api_key=CONTENT_API_KEY,
            temperature=0.3,
            max_tokens=800,
            timeout_s=120,
        )
        # Light LLM — for follow-ups, summaries, query rewriting (short outputs)
        self._short_llm = get_chat_model(
            style=CONTENT_API_STYLE,
            model=CONTENT_MODEL_NAME,
            endpoint=CONTENT_API_ENDPOINT,
            api_key=CONTENT_API_KEY,
            temperature=0.3,
            max_tokens=200,
            timeout_s=30,
        )

        _lobby_chain = lobby_prompt | self._chat_llm | StrOutputParser()
        _task_chain = task_chat_prompt | self._chat_llm | StrOutputParser()

        def _get_session_history(session_id: str):
            return TaskChatMessageHistory(session_id, self._task_manager)

        self._lobby_chain_with_history = RunnableWithMessageHistory(
            _lobby_chain,
            get_session_history=_get_session_history,
            input_messages_key="user_input",
            history_messages_key="history",
        )
        self._task_chain_with_history = RunnableWithMessageHistory(
            _task_chain,
            get_session_history=_get_session_history,
            input_messages_key="user_input",
            history_messages_key="history",
        )

    # ── helpers ─────────────────────────────────────────────────────────

    def _safe_str(self, obj: Any, max_len: int) -> str:
        if obj is None:
            return ""
        if isinstance(obj, str):
            s = obj.strip()
        else:
            s = json.dumps(obj, ensure_ascii=False)
        if len(s) > max_len:
            return s[:max_len] + "…"
        return s

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Extract keywords via jieba (semantic) + regex (technical terms)."""
        t = str(text or "").strip()
        if not t:
            return []
        words: list[str] = []
        try:
            import jieba
            for w in jieba.cut(t):
                w = w.strip()
                if len(w) >= 2 and not all(ch in "，。！？；：""''（）…— \t\n\r" for ch in w):
                    words.append(w)
        except ImportError:
            pass
        # Also extract technical tokens (English words, numbers, acronyms)
        tech = re.findall(r"[A-Za-z0-9_-]{2,}", t)
        # Deduplicate, cap at 18
        seen = set()
        out: list[str] = []
        for w in words + tech:
            wl = w.lower()
            if wl not in seen and len(w) >= 2:
                seen.add(wl)
                out.append(w)
        return out[:18]
        out: list[str] = []
        seen = set()
        for x in xs:
            k = x.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(k[:24])
            if len(out) >= 18:
                break
        return out

    def _split_md_sections(self, md: str) -> list[tuple[str, str]]:
        text = str(md or "")
        lines = text.splitlines()
        secs: list[tuple[str, str]] = []
        cur_title = ""
        buf: list[str] = []
        for line in lines:
            m = re.match(r"^\s*##\s+(.+?)\s*$", line)
            if m:
                if cur_title:
                    secs.append((cur_title, "\n".join(buf).strip()))
                cur_title = m.group(1).strip()
                buf = []
                continue
            if cur_title:
                buf.append(line)
        if cur_title:
            secs.append((cur_title, "\n".join(buf).strip()))
        return secs

    def _pick_snippets(
        self, query: str, md: str, *, max_sections: int = 3, max_chars: int = 900
    ) -> str:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return ""
        secs = self._split_md_sections(md)
        scored: list[tuple[int, str, str]] = []
        for title, body in secs:
            hay = (title + "\n" + body).lower()
            score = 0
            # Title match bonus: if any token appears in the section title, it's likely relevant
            title_lower = title.lower()
            for tok in q_tokens:
                tl = tok.lower()
                if tl and len(tl) >= 2 and tl in hay:
                    score += 1
                    if tl in title_lower:
                        score += 2  # extra weight for title match
            if score > 0:
                scored.append((score, title, body))
        scored.sort(key=lambda x: (-x[0], len(x[2])))
        out: list[str] = []
        for idx, (_, title, body) in enumerate(scored[:max_sections]):
            block = (f"## [{idx + 1}] {title}\n" + (body or "")).strip()
            if len(block) > max_chars:
                # Truncate at last sentence boundary
                cut = block[:max_chars]
                last_period = max(cut.rfind("。"), cut.rfind(".\n"), cut.rfind(".\r"))
                last_break = max(cut.rfind("\n\n"), cut.rfind("\n"))
                boundary = max(last_period, last_break)
                if boundary > max_chars * 0.6:  # only use boundary if it's not too early
                    block = cut[:boundary + 1] + "\n…"
                else:
                    block = cut + "…"
            out.append(block)
        return "\n\n".join(out).strip()

    def _load_task_text(self, task_id: str) -> tuple[str, str]:
        base = self._task_manager._result_dir / str(task_id)
        outline = ""
        content = ""
        try:
            p = base / "outline.md"
            if p.exists():
                outline = p.read_text(encoding="utf-8")
        except Exception:
            outline = ""
        try:
            p = base / "content.md"
            if p.exists():
                content = p.read_text(encoding="utf-8")
        except Exception:
            content = ""
        if (not outline) or (not content):
            try:
                from agent_file_create.db_service import (
                    get_db_connection,
                    get_latest_content_markdown,
                    get_latest_outline_markdown,
                    init_db,
                )
            except Exception:
                return outline, content
            conn = None
            try:
                conn = get_db_connection()
                init_db(conn)
                o2 = get_latest_outline_markdown(conn, task_id)
                c2 = get_latest_content_markdown(conn, task_id)
                outline = outline or o2
                content = content or c2
            except Exception:
                pass
            finally:
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass
        return outline, content

    _GEN_INTENT_PATTERNS = [
        "生成", "写", "帮我写", "帮我做", "帮我生成", "写一份", "写一个",
        "做一份", "做报告", "做文档", "出报告", "出文档",
    ]

    def _guide_task_creation(self, message: str) -> str | None:
        """When user expresses intent to generate a report but has no task yet,
        return a guided response instead of a generic chat reply."""
        m = (message or "").strip()
        if not m:
            return None
        if m.startswith("/"):
            return None
        # Check for generation intent keywords
        if not any(kw in m for kw in self._GEN_INTENT_PATTERNS):
            return None
        # Only trigger for reasonably specific requests (not just "写报告")
        if len(m) < 10:
            return None

        help_cmds = self._help_text()
        return (
            f"好的！我注意到你想要生成一份报告。按以下步骤开始：\n\n"
            f"**1️⃣ 上传材料**\n"
            f"将相关文件拖拽到左侧上传区域，或点击「选择文件」按钮。\n\n"
            f"**2️⃣（可选）选择模板**\n"
            f"在左侧上传区选择模板文件，或从已保存模板中挑选。\n\n"
            f"**3️⃣ 发送生成命令**\n"
            f"材料准备好后，发送：\n"
            f"`/gen {m[:120]}`\n\n"
            f"**💡 也可以先搜知识库**\n"
            f"如果左侧有知识库，发送 `/kb use <知识库名>` 选择知识库，\n"
            f"然后 `/kb ask <你的问题>` 查找相关资料。\n\n"
            f"---\n"
            f"其他可用指令：\n{help_cmds}"
        )

    def _help_text(self) -> str:
        return "\n".join(
            [
                "可用指令（在聊天框输入）：",
                "- /help：查看指令",
                "- /status：查看当前任务状态",
                "- /pause：暂停当前任务",
                "- /resume：继续当前任务",
                "- /cancel：取消当前任务",
                "- /regen [doc|all|<章节标题>]：重新生成（doc=重跑文档阶段；all=从抽取开始；章节标题=只重新生成该章节及其子节）",
                "- /prompt <文本>：更新需求（不自动重跑）",
                "- /prompt! <文本>：更新需求并立即 /regen doc",
                "- /template [task|default]：切换模板来源（task=使用本任务模板；default=使用全局默认模板）",
                "- /files：查看已上传文件",
                "- /templates：查看可用模板",
                "- /append：提示如何追加文件/模板到当前任务",
                "- /kb ask <问题>：在知识库中检索并回答",
                "- /kb list：列出知识库",
                "- /kb docs [kb]：查看知识库中已上传的文件",
                "- /kb use <kb>：选择知识库用于问答",
                "- /kb clear：取消知识库选择",
                "- /kb stats [kb]：查看知识库统计（文档数、chunk数）",
                "- /kb delete <kb> [doc_id]：删除知识库（指定 doc_id 则只删除该文档）",
                "",
                "结构化协议（前端/调用方可直接传 action）：",
                '{"task_id":"<id>", "message":"可选", "action":{"type":"pause|resume|cancel|status|help|regenerate|set_prompt|set_template_mode|list_files|list_templates|kb_list|kb_use|kb_clear|kb_stats|kb_delete"}}',
            ]
        ).strip()

    def _parse_chat_action(self, message: str, body_action: Any) -> Optional[dict]:
        if isinstance(body_action, dict) and body_action.get("type"):
            return {str(k): v for k, v in body_action.items()}
        m = str(message or "").strip()
        if not m:
            return None
        low = m.lower()
        if low.startswith("/"):
            parts = [x for x in m.strip().split(" ") if x.strip()]
            cmd = parts[0][1:].strip().lower()
            rest = m[len(parts[0]) :].strip() if parts else ""
            if cmd in {"help", "h"}:
                return {"type": "help"}
            if cmd in {"status", "st"}:
                return {"type": "status"}
            if cmd in {"pause"}:
                return {"type": "pause"}
            if cmd in {"resume", "continue"}:
                return {"type": "resume"}
            if cmd in {"cancel", "stop"}:
                return {"type": "cancel"}
            if cmd in {"regen", "regenerate"}:
                if len(parts) >= 2:
                    arg = parts[1].strip().lower()
                    # Only treat "all"/"full" as scope when they are the ONLY argument
                    if len(parts) == 2 and arg in {"all", "full"}:
                        return {"type": "regenerate", "scope": "all"}
                    else:
                        # Section name — join all remaining parts preserving original casing
                        section_name = " ".join(parts[1:]).strip()
                        return {"type": "regenerate", "scope": "section", "section": section_name}
                else:
                    return {"type": "regenerate", "scope": "doc"}
            if cmd in {"prompt", "prompt!"}:
                t = rest.strip()
                if not t:
                    return {"type": "help"}
                return {"type": "set_prompt", "prompt": t, "regen": cmd.endswith("!")}
            if cmd in {"template"}:
                arg = parts[1].strip().lower() if len(parts) >= 2 else ""
                if arg in {"default", "global"}:
                    return {"type": "set_template_mode", "mode": "default"}
                if arg in {"task", "local"}:
                    return {"type": "set_template_mode", "mode": "task"}
                return {"type": "help"}
            if cmd in {"files"}:
                return {"type": "list_files"}
            if cmd in {"templates"}:
                return {"type": "list_templates"}
            if cmd in {"append"}:
                return {"type": "append"}
            if cmd in {"kb"}:
                sub = (parts[1].strip().lower() if len(parts) >= 2 else "")
                if sub in {"list", "ls"}:
                    return {"type": "kb_list"}
                if sub in {"use"}:
                    name = (parts[2].strip() if len(parts) >= 3 else "").strip()
                    if not name:
                        return {"type": "help"}
                    return {"type": "kb_use", "kb": name}
                if sub in {"clear", "off"}:
                    return {"type": "kb_clear"}
                if sub == "stats":
                    name = (parts[2].strip() if len(parts) >= 3 else "default").strip() or "default"
                    return {"type": "kb_stats", "kb": name}
                if sub in {"ask", "query", "q"}:
                    question = " ".join(parts[2:]).strip() if len(parts) >= 3 else ""
                    if not question:
                        return {"type": "help"}
                    return {"type": "kb_ask", "question": question}
                if sub in {"docs", "files"}:
                    name = (parts[2].strip() if len(parts) >= 3 else "").strip()
                    return {"type": "kb_docs", "kb": name}
                if sub == "delete":
                    name = (parts[2].strip() if len(parts) >= 3 else "").strip()
                    if not name:
                        return {"type": "help"}
                    doc_id = (parts[3].strip() if len(parts) >= 4 else "").strip()
                    return {"type": "kb_delete", "kb": name, "doc_id": doc_id}
                return {"type": "help"}
            return {"type": "help"}
        if low in {"暂停", "pause"}:
            return {"type": "pause"}
        if low in {"继续", "恢复", "resume"}:
            return {"type": "resume"}
        if low in {"取消", "停止", "终止", "cancel"}:
            return {"type": "cancel"}
        if low in {"重新生成", "重做", "再生成", "regen"}:
            return {"type": "regenerate", "scope": "doc"}
        return None

    def _format_status(self, task_id: str) -> str:
        st = self._task_manager.read_status(task_id)
        meta = self._task_manager.read_task_meta(task_id)
        status = str(st.get("status") or "").strip() or "unknown"
        stage = str(st.get("stage") or "").strip()
        msg = str(st.get("message") or "").strip()
        total = int(st.get("total_files") or 0) if str(st.get("total_files") or "").strip() else 0
        done = int(st.get("done_files") or 0) if str(st.get("done_files") or "").strip() else 0
        mode = str(meta.get("template_mode") or "").strip() or "task"
        active_kb = str(meta.get("active_kb") or "").strip()
        fps = self._resolve_task_files(task_id)
        tps = self._task_manager.list_task_templates(task_id)
        parts = [f"task_id={task_id}", f"status={status}" + (f" stage={stage}" if stage else "")]
        if msg:
            parts.append("message=" + msg)
        if total or done:
            parts.append(f"progress={done}/{total}" if total else f"progress={done}")
        parts.append(f"files={len(fps)} templates={len(tps)} template_mode={mode} active_kb={active_kb or '-'}")
        return "\n".join(parts).strip()

    def _resolve_task_files(self, task_id: str) -> list[str]:
        meta = self._task_manager.read_task_meta(task_id)
        fps = meta.get("file_paths")
        if isinstance(fps, list) and all(isinstance(x, str) for x in fps) and fps:
            return [str(x) for x in fps if str(x)]
        return self._task_manager.list_task_upload_files(task_id)

    def _handle_chat_action(self, task_id: str, action: dict) -> str:
        typ = str(action.get("type") or "").strip().lower()
        if typ in {"help"}:
            return self._help_text()
        if typ in {"status"}:
            return self._format_status(task_id)

        if typ == "kb_list":
            items = get_kb().list_kb()
            if not items:
                return "暂无知识库。可调用 /api/kb/upload 上传文件入库。"
            return "知识库列表：\n" + "\n".join(["- " + str(x) for x in items[:50]])
        if typ == "kb_use":
            kb = str(action.get("kb") or "").strip()
            if not kb:
                return "用法：/kb use <kb>"
            self._task_manager.write_task_meta(task_id, {"active_kb": kb})
            return f"好，后面聊到相关问题时我会先去 {kb} 知识库里查一下。"
        if typ == "kb_clear":
            self._task_manager.write_task_meta(task_id, {"active_kb": ""})
            return "好，后面就不查知识库了，有问题再说。"
        if typ == "kb_ask":
            question = str(action.get("question") or "").strip()
            if not question:
                return "用法：/kb ask <你的问题>"
            return self._handle_kb_ask(task_id, question)
        if typ == "kb_docs":
            kb = str(action.get("kb") or "").strip()
            meta = self._task_manager.read_task_meta(task_id)
            if not kb:
                kb = str(meta.get("active_kb") or "").strip() or "default"
            try:
                docs = get_kb().list_docs(kb=kb)
                if not docs:
                    return f"知识库 {kb} 中还没有上传过文件。"
                lines = [f"知识库 {kb} 中的文件（共 {len(docs)} 个）："]
                for d in docs[:30]:
                    title = str(d.get("title") or d.get("doc_id") or "未知文件")
                    doc_type = str(d.get("doc_type") or "")
                    line = f"  · {title}"
                    if doc_type:
                        line += f" [{doc_type}]"
                    lines.append(line)
                return "\n".join(lines)
            except Exception as e:
                return f"获取文件列表失败：{str(e)[:180]}"
        if typ == "kb_stats":
            kb = str(action.get("kb") or "").strip() or "default"
            try:
                st = get_kb().kb_stats(kb=kb)
                return f"知识库 {kb}：文档数={st.get('doc_count',0)} chunk数={st.get('chunk_count',0)}"
            except Exception as e:
                return f"获取统计失败：{str(e)[:180]}"
        if typ == "kb_delete":
            kb = str(action.get("kb") or "").strip()
            doc_id = str(action.get("doc_id") or "").strip()
            if not kb:
                return "用法：/kb delete <kb> [doc_id]"
            try:
                if doc_id:
                    r = get_kb().delete_doc(kb=kb, doc_id=doc_id)
                    if r.get("ok"):
                        return f"已从知识库 {kb} 中删除文档 {doc_id}。"
                    return f"删除失败：{r.get('error', 'unknown')[:180]}"
                else:
                    r = get_kb().delete_kb(kb=kb)
                    if r.get("ok"):
                        return f"已删除知识库 {kb}。"
                    return f"删除失败：{r.get('error', 'unknown')[:180]}"
            except Exception as e:
                return f"删除失败：{str(e)[:180]}"

        if task_id == "lobby":
            return "当前未选择 task_id。可先上传材料生成任务，或输入 task_id 加载后再执行控制指令。"

        st = self._task_manager.read_status(task_id)
        stage = str(st.get("stage") or "").strip()

        if typ == "pause":
            self._task_manager.pause_task(task_id)
            self._task_manager.write_status(
                task_id, "paused", stage=stage or "pause",
                message="已暂停（发送 /resume 继续，/cancel 取消）", extra={}
            )
            return "先停一下，等你准备好了跟我说一声就行。"
        if typ == "resume":
            self._task_manager.resume_task(task_id)
            if str(st.get("status") or "") == "paused":
                self._task_manager.write_status(
                    task_id, "processing", stage=stage or "resume",
                    message="已继续执行…", extra={}
                )
            return "好，继续~"
        if typ == "cancel":
            self._task_manager.cancel_task(task_id)
            self._task_manager.write_status(
                task_id, "canceled", stage=stage or "cancel",
                message="已取消", extra={}
            )
            return "取消了，之前生成的内容还在，可以随时重新开始。"
        if typ == "list_files":
            fps = self._resolve_task_files(task_id)
            if not fps:
                return "未找到已上传文件。你可以点击「追加到当前任务」上传更多材料。"
            names = [Path(x).name for x in fps[:30]]
            tail = "…" if len(fps) > 30 else ""
            return "已上传文件：\n" + "\n".join(["- " + x for x in names]) + tail
        if typ == "list_templates":
            tps = self._task_manager.list_task_templates(task_id)
            if not tps:
                return "当前任务没有模板文件。你可以在上传时附带模板，或使用 /template default 使用全局默认模板。"
            names = [Path(x).name for x in tps[:30]]
            tail = "…" if len(tps) > 30 else ""
            return "可用模板：\n" + "\n".join(["- " + x for x in names]) + tail
        if typ == "set_template_mode":
            mode = str(action.get("mode") or "").strip().lower()
            if mode not in {"task", "default"}:
                return "template_mode 只能是 task 或 default。示例：/template task"
            self._task_manager.write_task_meta(task_id, {"template_mode": mode})
            return f"已切换模板模式：{mode}。下次生成/重新生成会按该模式渲染。"
        if typ == "set_prompt":
            p = str(action.get("prompt") or "").strip()
            if not p:
                return "prompt 不能为空。示例：/prompt 请生成一份面向管理层的报告"
            self._task_manager.write_task_meta(task_id, {"user_prompt": p})
            if bool(action.get("regen")):
                fps = self._resolve_task_files(task_id)
                tpl_override, tps = self._resolve_template_override(task_id)
                return "好的，需求已更新，正在重新生成。"
            return "好的，需求已更新。想重跑的话发 /regen doc 就行。"
        if typ == "append":
            return "\n".join(
                [
                    "追加文件/模板方式：",
                    "1) 在页面上选择要追加的文件/模板；",
                    "2) 点击「追加到当前任务」；",
                    "3) 系统会基于新旧材料重新生成。",
                    "",
                    "也可调用接口：POST /api/append（multipart/form-data），字段包含 task_id/files/templates/user_prompt。",
                ]
            ).strip()
        if typ == "regenerate":
            if self._task_manager.is_task_running(task_id):
                return "任务正在运行，无法重新生成。可先 /pause 或 /cancel。"
            scope = str(action.get("scope") or "doc").strip().lower()
            section_name = str(action.get("section") or "").strip()
            feedback = str(action.get("feedback") or "").strip()
            if self._regenerate_fn:
                ok, msg = self._regenerate_fn(task_id, scope if scope else "doc", section_name=section_name, feedback=feedback)
                return msg
            return "重新生成功能未启用，请通过 Web 界面操作（点击「追加到当前任务」）。"
        return self._help_text()

    def _resolve_template_override(self, task_id: str) -> tuple[Optional[str], list[str]]:
        meta = self._task_manager.read_task_meta(task_id)
        mode = str(meta.get("template_mode") or "").strip().lower()
        tps = self._task_manager.list_task_templates(task_id)
        if mode == "default":
            return None, tps
        if tps:
            return str((self._task_manager._result_dir / str(task_id) / "template").resolve()), tps
        return None, []

    # ── Clarify answer validation ────────────────────────────────────────

    def validate_clarify_answer(self, message: str, questions: list[str]) -> tuple[bool, str]:
        """Check if the user's reply is relevant to clarification questions.
        Returns (is_valid, warning_message). Warning is empty if valid."""
        m = (message or "").strip()
        if not m:
            return False, "请提供具体的补充信息后再提交，或回复「跳过」使用默认设置继续生成。"

        # Substantial answers are always accepted
        if len(m) >= 15:
            return True, ""

        # Extract keywords from questions
        q_chars = set()
        for q in (questions or [])[:4]:
            for ch in q:
                if ch.isalpha() or '一' <= ch <= '鿿':
                    q_chars.add(ch)

        # Heuristic: if very short answer shares keywords with questions, accept
        overlap = sum(1 for ch in m if ch in q_chars)
        if overlap >= 3:
            return True, ""

        return True, ""

    # ── Clarify phase (moved from web layer into handler) ─────────────────

    def _is_clarify_phase(self, st: dict) -> bool:
        """Check if task is waiting for clarify answer, robust against stage race conditions."""
        if str(st.get("status") or "") == "need_user" and str(st.get("stage") or "") == "clarify":
            return True
        has_questions = isinstance(st.get("clarify_questions"), list) and st.get("clarify_questions")
        has_answer = bool(str(st.get("clarify_answers") or "").strip()) or bool(st.get("clarify_skip"))
        return has_questions and not has_answer

    def _handle_clarify_answer(self, task_id: str, message: str, st: dict) -> str | None:
        """Process a clarify-phase answer. Returns reply string, or None if invalid.

        Handles: skip, valid long answer, valid keyword-overlapping short answer.
        """
        import time as _time
        m = message.strip()
        low = m.lower()
        if low in {"skip", "跳过", "略过", "不用了", "不需要"}:
            self._task_manager.write_status(task_id, "processing", stage="clarify",
                message="用户选择跳过澄清，继续生成…",
                extra={"clarify_answers": "", "clarify_skip": True,
                       "clarify_submitted_at": float(_time.time())})
            return "好嘞，先跳过。我先继续往下写，你觉得哪里不对再跟我说。"

        qs = st.get("clarify_questions") if isinstance(st.get("clarify_questions"), list) else []
        is_valid, warning = self.validate_clarify_answer(m, qs)
        if not is_valid:
            return None  # caller should return warning to user

        self._task_manager.write_status(task_id, "processing", stage="clarify",
            message="已收到用户补充信息，继续生成…",
            extra={"clarify_answers": m, "clarify_skip": False,
                   "clarify_submitted_at": float(_time.time())})
        return "记下了！我按你说的调整，写好了你看看。中间有什么想法随时说。"

    # ── Follow-up suggestions ────────────────────────────────────────────

    def _generate_followups(self, message: str, reply: str, task_id: str) -> str:
        """Generate 2-3 follow-up question suggestions after a reply. Returns a
        formatted markdown string, or empty string on failure / for action commands."""
        q = (message or "").strip()
        r = (reply or "").strip()
        # Skip for very short exchanges, commands, or error messages
        if not q or not r or len(r) < 30 or q.startswith("/") or r.startswith("❌"):
            return ""
        if "已暂停" in r or "已取消" in r or "已继续" in r or "已启动" in r:
            return ""

        # Determine report topics from task outline
        report_topics = "（无报告上下文）"
        if task_id and task_id != "lobby":
            outline, _ = self._load_task_text(task_id)
            if outline:
                # Extract section titles as topics
                topics = []
                for line in outline.splitlines()[:20]:
                    m = re.match(r"^##\s+(.+?)\s*$", line.strip())
                    if m:
                        topics.append(m.group(1).strip())
                if topics:
                    report_topics = "、".join(topics[:8])

        reply_summary = r[:500]

        try:
            chain = FOLLOWUPS_PROMPT | self._short_llm | StrOutputParser()
            raw = (chain.invoke({
                "question": q,
                "reply_summary": reply_summary,
                "report_topics": report_topics,
            }) or "").strip()
        except Exception:
            return ""

        # Parse lines starting with "- " or numbered
        lines = []
        for line in raw.splitlines():
            s = line.strip()
            s = re.sub(r"^\d+[.)、\s]+", "", s).strip()
            s = re.sub(r"^[-*]\s*", "", s).strip()
            if s and len(s) >= 6 and len(s) <= 80 and s != q[:50]:
                lines.append(s)
            if len(lines) >= 3:
                break

        if not lines:
            return ""

        return "\n".join([f"- {x}" for x in lines])

    # ── Summarization ───────────────────────────────────────────────────

    _SUMMARIZE_THRESHOLD = 16
    _SUMMARIZE_KEEP = 8
    _TOKEN_THRESHOLD = 2000        # estimated tokens before compressing
    _CHARS_PER_TOKEN = 1.5         # conservative for Chinese text

    def _maybe_summarize_history(self, task_id: str) -> None:
        """If chat history exceeds threshold (message count or token count),
        summarize oldest messages and truncate."""
        if task_id == "lobby":
            return
        history = self._task_manager.read_chat_history(task_id)

        # Check message-count threshold
        by_count = len(history) > self._SUMMARIZE_THRESHOLD

        # Check token-count threshold (defends against very long single messages)
        total_chars = sum(
            len(str(d.get("content") or "")) for d in history
        )
        est_tokens = total_chars / max(self._CHARS_PER_TOKEN, 0.5)
        by_tokens = est_tokens > self._TOKEN_THRESHOLD and len(history) >= 6

        if not (by_count or by_tokens):
            return

        older = history[: -self._SUMMARIZE_KEEP]
        if not older:
            return

        transcript = "\n".join(
            [f"[{d['role']}]: {d['content'][:300]}" for d in older if d.get("content")]
        )
        if not transcript.strip():
            return

        try:
            chain = SUMMARIZE_HISTORY_PROMPT | self._short_llm | StrOutputParser()
            new_summary = (
                chain.invoke({"transcript": transcript}) or ""
            ).strip()
        except Exception:
            return

        if not new_summary:
            return

        old_summary = self._task_manager.read_chat_summary(task_id)
        merged = (
            (old_summary + "\n" + new_summary).strip()
            if old_summary
            else new_summary
        )
        self._task_manager.write_chat_summary(task_id, merged[:2000])
        self._task_manager.truncate_chat_history(task_id, keep_last=self._SUMMARIZE_KEEP)

    # ── Query rewriting ─────────────────────────────────────────────────

    def _rewrite_query(self, message: str) -> str:
        """Expand a short user query into a retrieval-friendly form."""
        q = (message or "").strip()
        if len(q) < 10 or len(q) > 80:
            return q

        try:
            chain = REWRITE_QUERY_PROMPT | self._short_llm | StrOutputParser()
            rewritten = (
                chain.invoke({"question": q}) or ""
            ).strip()
            return rewritten[:200] if rewritten else q
        except Exception:
            return q

    @staticmethod
    def _is_complex_question(message: str) -> bool:
        """Heuristic: does this question benefit from chain-of-thought reasoning?"""
        q = (message or "").strip()
        if len(q) > 60:
            return True
        complex_markers = [
            "为什么", "如何", "怎么", "原因", "影响", "关系",
            "对比", "区别", "比较", "优劣", "优缺点", "分析",
            "vs", "VS", " versus ", "相比", "不同", "差异",
            "是否", "应该", "如果", "假设", "评估", "判断",
        ]
        return any(m in q for m in complex_markers)

    @staticmethod
    def _should_use_hyde(message: str) -> bool:
        """Only trigger HyDE when the question has clear reasoning / analysis intent.
        HyDE costs an extra LLM call to generate a hypothetical answer, so we avoid
        it for simple factual / short queries even when CoT is used."""
        q = (message or "").strip()
        if len(q) < 15:
            return False
        hyde_markers = [
            "为什么", "原因", "影响", "关系",
            "对比", "区别", "比较", "优劣", "优缺点", "分析",
            "vs", "VS", " versus ", "是否", "应该", "如果",
            "假设", "评估", "判断", "趋势", "归纳", "总结", "概括",
        ]
        return any(m in q for m in hyde_markers)

    # ── Modification intent keywords ──────────────────────────────────────

    _MODIFICATION_KEYWORDS = [
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

    # ── Trivial message detection ─────────────────────────────────────────

    _GREETINGS = {
        "在吗", "在不在", "在了吗", "在不在呀",
        "你好", "您好", "你好啊", "嗨", "hi", "hello", "hey",
        "早上好", "下午好", "晚上好", "晚安", "中午好",
    }
    _GREETING_REPLIES = [
        "嗨！有什么想生成的报告？或者想聊点别的也行。",
        "来啦~ 今天要写什么？直接把文件拖进来，我帮你弄。",
        "嘿！上传文件、搜知识库、直接提问，都行，看你想干嘛。",
        "哈喽！有新材料要出报告，还是想问问之前的内容？",
    ]
    _THANKS = {
        "我喜欢你", "我爱你", "爱你",
        "能叫我宝贝吗", "叫我宝贝",
        "你真棒", "太棒了", "厉害", "牛逼",
        "谢谢", "谢谢你", "感谢", "thx", "thanks", "thank you",
    }
    _THANKS_REPLIES = [
        "客气啥~ 还有什么需要调整的随时说。",
        "应该的！有哪里不满意再跟我说。",
        "有帮助就好。后续想改风格、加内容，说一声就行。",
    ]
    _FOLLOWUP_INTROS = [
        "这几个问题可能对你有用：",
        "顺着这个话题，你还想了解：",
        "如果还想深入看看，可以试试问这些：",
        "另外这几个点可能也值得关注：",
    ]

    @classmethod
    def _is_trivial_message(cls, message: str) -> Optional[str]:
        """Return a short reply if the message is a pure greeting or social
        chat that doesn't warrant a full LLM invocation. Otherwise None."""
        m = (message or "").strip()
        if not m:
            return "请输入消息。你可以提问、发送指令，或输入 /help 查看可用命令。"

        low = m.lower()

        # ── Single-char / meaningless input ──────────────────────────
        if len(m) <= 2 and low not in {"hi", "ok", "好", "行", "是", "对", "嗯", "有", "no", "go", "嗯嗯"}:
            return "请直接说出你的问题或需求，我会尽力帮你。\n\n💡 试试：/help 查看所有指令，或直接输入你想了解的内容。"

        # ── Interjections / testing noise ─────────────────────────────
        if low in {"喂", "喂喂", "有人吗", "在吗在吗", "test", "测试", "testing",
                   "？？", "？？？", "?", "??", "???", "。。。", "…"}:
            return "你好！有什么可以帮你的？你可以上传材料生成报告，或者直接在对话框中提问。\n\n💡 输入 /help 查看所有可用指令。"

        # ── Greetings (random variant) — checked BEFORE length filter ──
        if low in cls._GREETINGS:
            return random.choice(cls._GREETING_REPLIES)

        if low in cls._THANKS:
            return random.choice(cls._THANKS_REPLIES)

        if len(m) > 15:
            return None

        return None

    @classmethod
    def _detect_modification_intent(cls, message: str) -> bool:
        """Check if the message expresses intent to modify the report."""
        m = (message or "").strip()
        if not m or len(m) > 120:
            return False
        if m.startswith("/"):
            return False
        return any(kw in m for kw in cls._MODIFICATION_KEYWORDS)

    # ── Context builder (shared by chat_reply and chat_reply_stream) ─────

    def _build_context(
        self, message: str, task_id: str, *, intent: "ChatIntent | None" = None,
    ) -> tuple[Optional[str], dict[str, str]]:
        """Check for immediate replies; if none, build the chain input dict.

        Returns (immediate_text, chain_input).
        If immediate_text is not None the caller should return it directly.
        Otherwise chain_input is ready for the LCEL chain.

        The ``intent`` parameter controls context assembly:
        - QUESTION_REPORT → loads report snippets + KB (full context)
        - GENERAL_CHAT   → loads report snippets only (skip KB)
        - MODIFY_REPORT / CLARIFY / CONTROL → never reach this method
        """
        greeting_reply = self._is_trivial_message(message)
        if greeting_reply is not None:
            return greeting_reply, {}

        outline, content = self._load_task_text(task_id)
        summary = self._task_manager.read_chat_summary(task_id)
        st = self._task_manager.read_status(task_id)
        meta = self._task_manager.read_task_meta(task_id)
        active_kb = str(meta.get("active_kb") or "").strip()

        # ── Lobby mode ──────────────────────────────────────────────────
        if str(task_id) == "lobby" and st.get("status") == "unknown":
            if active_kb:
                try:
                    msg = str(message or "")
                    # For reasoning questions, expand via HyDE before retrieval
                    if self._should_use_hyde(msg):
                        search_query = get_kb()._hyde_expand(msg)
                    else:
                        search_query = self._rewrite_query(msg)
                    retriever = KnowledgeBaseRetriever(
                        kb=active_kb, knowledge_base=get_kb(), top_k=6, context_window=2
                    )
                    docs = retriever.invoke(search_query)
                    blocks: list[str] = []
                    used = 0
                    for doc in docs:
                        meta_d = doc.metadata
                        head = (
                            f"[kb={active_kb} doc={meta_d.get('doc_id','')}"
                            f" section={meta_d.get('section_path','')}"
                            f" score={float(meta_d.get('score',0)):.3f}]"
                        )
                        body = doc.page_content.strip()
                        if len(body) > 700:
                            body = body[:700] + "…"
                        block = (head + "\n" + body).strip()
                        if not block:
                            continue
                        if used + len(block) + 2 > 2200:
                            break
                        blocks.append(block)
                        used += len(block) + 2
                    kb_context = "\n\n".join(blocks).strip()
                    if kb_context:
                        enriched = (
                            f"知识库检索结果（请基于以下资料回答，注明信息来源）：\n\n{kb_context}\n\n"
                            f"用户问题：{self._safe_str(message, 1200) or '（空）'}"
                        )
                    else:
                        enriched = (
                            f"用户问题：{self._safe_str(message, 1200) or '（空）'}\n"
                            "（知识库中未找到相关内容，请如实告知用户）"
                        )
                    return None, {"user_input": enriched}
                except Exception as e:
                    return f"知识库检索失败：{str(e)[:180]}。请检查 EMBED_* 配置与 embedding 模型是否可用。", {}

            # ── Guided task creation: detect generation intent without a task ──
            guide = self._guide_task_creation(message)
            if guide:
                return guide, {}

            return None, {"user_input": self._safe_str(message, 1200) or "（空）"}

        # ── Task state checks ───────────────────────────────────────────
        status = str(st.get("status") or "").strip()
        stage = str(st.get("stage") or "").strip()

        if status == "paused":
            return (
                f"当前任务已暂停（task_id={task_id}）。发送 /resume 继续，/cancel 取消，/status 查看状态。",
                {},
            )

        if status in {"queued", "processing", "need_user"} and not (outline or content):
            total = int(st.get("total_files") or 0) if str(st.get("total_files") or "").strip() else 0
            done = int(st.get("done_files") or 0) if str(st.get("done_files") or "").strip() else 0
            msg = str(st.get("message") or "").strip()

            if status == "need_user" or stage == "clarify":
                qs = st.get("clarify_questions") if isinstance(st.get("clarify_questions"), list) else []
                qtxt = "\n".join([f"- {str(x)[:240]}" for x in qs[:6] if str(x).strip()]).strip()
                tail = "\n".join(
                    [
                        "你现在可以直接在对话里回复补充信息，我会把它当作澄清答案继续推进生成。",
                        "如果想跳过澄清，回复：跳过",
                    ]
                )
                return (
                    "\n\n".join(
                        [
                            f"当前任务需要补充信息（task_id={task_id}）。",
                            ("待回答问题：\n" + qtxt)
                            if qtxt
                            else "待回答问题：（系统未提供具体问题，可直接描述你的目标/受众/篇幅/风格/重点）",
                            tail,
                        ]
                    ).strip(),
                    {},
                )

            if stage == "extract":
                p = f"{done}/{total}" if total else str(done)
                base = f"当前任务正在解析材料（task_id={task_id}），进度 {p}。"
                if msg:
                    base += "\n状态：" + msg
                return (
                    "\n".join(
                        [
                            base,
                            "你可以边等边问：要生成什么类型报告更合适、希望的结构/风格、需要补充哪些材料等。",
                        ]
                    ).strip(),
                    {},
                )

            _WAITING = {
                "extract": "正在读你的文件，文件多的话会花点时间，不过我会并行处理，不会太慢。",
                "plan": "读完了，正在想怎么组织这份报告的结构。",
                "assess": "看了一下材料，可能需要问你几个问题。",
                "clarify": "需要你补充一些信息才能继续，你看一下上面的问题。",
                "enrich": "正在查资料补充内容，稍等~",
                "outline": "大纲正在生成中，写好了会先给你过目。",
                "content": "正在逐章写正文，写好的章节会陆续显示在右侧，你可以先看着。",
                "research": "正在查一些参考资料来充实内容。",
                "critic": "正在检查报告内容有没有问题。",
                "render": "写好了，正在生成最终文件（Word/PDF），马上就能下载。",
            }
            base = _WAITING.get(stage, f"当前任务正在处理（task_id={task_id} stage={stage}）。")
            if msg:
                base += "\n" + msg
            return ("\n".join([base, "你可以继续提问，我会结合当前进度给建议。"]).strip(), {})

        # ── Build context for task-chat mode ────────────────────────────
        # Intent-based filtering: skip expensive operations when not needed
        load_report = intent != ChatIntent.GENERAL_CHAT and intent != ChatIntent.KB_QUERY
        load_kb = intent in (ChatIntent.QUESTION_REPORT, ChatIntent.KB_QUERY)

        snippets = self._pick_snippets(message, content) if (content and load_report) else ""
        progress_hint = ""
        if status in {"queued", "processing", "need_user"}:
            progress_hint = "（提示：当前任务可能尚未完全生成，回答会基于现有内容，必要时我会提示缺失。）"

        kb_snippets = ""
        if active_kb and load_kb:
            try:
                msg = str(message or "")
                # For complex questions, expand via HyDE before retrieval
                if self._is_complex_question(msg):
                    search_query = get_kb()._hyde_expand(msg)
                else:
                    search_query = self._rewrite_query(msg)
                retriever = KnowledgeBaseRetriever(
                    kb=active_kb, knowledge_base=get_kb(), top_k=6, context_window=2
                )
                docs = retriever.invoke(search_query)
                blocks: list[str] = []
                used = 0
                for doc in docs:
                    meta = doc.metadata
                    head = (
                        f"[kb={active_kb} doc={meta.get('doc_id','')}"
                        f" section={meta.get('section_path','')}"
                        f" score={float(meta.get('score',0)):.3f}]"
                    )
                    body = doc.page_content.strip()
                    if len(body) > 700:
                        body = body[:700] + "…"
                    block = (head + "\n" + body).strip()
                    if not block:
                        continue
                    if used + len(block) + 2 > 2200:
                        break
                    blocks.append(block)
                    used += len(block) + 2
                kb_snippets = "\n\n".join(blocks).strip()
            except Exception:
                kb_snippets = ""

        context_blocks: list[str] = []

        if outline or content:
            # Full outline TOC (titles only) — gives LLM a bird's-eye view
            toc = ""
            if outline:
                toc_lines = []
                for line in str(outline).splitlines():
                    stripped = line.strip()
                    if re.match(r"^#{1,6}\s+", stripped):
                        toc_lines.append(stripped)
                if toc_lines:
                    toc = "报告章节一览：\n" + "\n".join(toc_lines[:40]) + "\n\n"

            context_blocks.append(
                "【可信度：高】已生成的报告内容（优先参考）：\n"
                + toc
                + ("与问题相关的正文摘录（编号 [N] 对应章节一览中的第 N 节）：\n" + self._safe_str(snippets, 2000) if snippets else "")
            )

        if kb_snippets:
            context_blocks.append(
                "【可信度：中】知识库检索片段（辅助参考，可能与报告内容不一致，以报告为准）：\n"
                + kb_snippets
            )

        if summary:
            context_blocks.append(
                "【可信度：低】对话摘要（长期记忆，仅供参考上下文）：\n"
                + self._safe_str(summary, 700)
            )

        context_text = "\n\n".join(context_blocks).strip() if context_blocks else "（暂无可用上下文）"

        # ── Multi-turn KB awareness: detect if user is following up on a prior KB search ──
        try:
            history = self._task_manager.read_chat_history(task_id)
            last_kb_q = ""
            last_kb_cites = ""
            for entry in reversed(history[-10:]):
                role = str(entry.get("role") or "")
                content = str(entry.get("content") or "")
                if role == "user" and content.startswith("[知识库:"):
                    last_kb_q = content
                    break
                if role == "assistant" and entry.get("_kb_citations"):
                    last_kb_cites = str(entry.get("_kb_citations") or "")
            if last_kb_q and any(kw in (message or "") for kw in ("那", "这个", "刚才", "上面", "之前", "继续", "还有", "别的", "其他")):
                progress_hint = (progress_hint or "") + (
                    "\n（用户可能在追问之前的KB检索结果。上一轮检索问题：" + last_kb_q[-200:] +
                    ("；检索命中文档：" + last_kb_cites[-200:] if last_kb_cites else "") + "）"
                )
        except Exception:
            pass

        chain_input = {
            "user_input": self._safe_str(message, 1200) or "（空）",
            "context_text": context_text,
            "progress_hint": progress_hint,
        }
        return None, chain_input

    # ── Public API ──────────────────────────────────────────────────────

    def _handle_modification_intent(self, message: str, task_id: str) -> str:
        """If the message expresses intent to modify the report, update
        user_prompt and return a suggestion suffix. Otherwise return ''."""
        if task_id == "lobby" or not self._detect_modification_intent(message):
            return ""

        outline, content = self._load_task_text(task_id)
        if not content and not outline:
            return ""

        try:
            meta = self._task_manager.read_task_meta(task_id)
            old_prompt = str(meta.get("user_prompt") or "").strip()
            append_msg = (message or "").strip()
            # Avoid duplicate appends
            if append_msg in old_prompt:
                return "\n\n---\n💡 你的需求中已包含类似要求，可发送 /regen doc 按最新需求重新生成。"
            new_prompt = (old_prompt + "\n" + append_msg).strip()
            self._task_manager.write_task_meta(task_id, {"user_prompt": new_prompt})
            return "\n\n---\n💡 已根据你的反馈更新生成需求。发送 /regen doc 即可按新要求重新生成报告。"
        except Exception:
            return ""

    def chat_reply(
        self, message: str, task_id: str, history: Optional[list[dict]] = None
    ) -> str:
        """Main chat entry point — classified intent routing.

        Intent routing order:
          1. CLARIFY_ANSWER → handle directly (no LLM)
          2. MODIFY_REPORT  → write to task_meta, quick reply (no LLM)
          3. CONTROL_TASK   → delegate to _handle_chat_action
          4. Fast-path immediate (greetings, status checks) in _build_context
          5. QUESTION_REPORT / GENERAL_CHAT → full LLM chain with filtered context
        """
        m = (message or "").strip()
        st = self._task_manager.read_status(task_id)

        # ── Intent classification ────────────────────────────────────
        is_clarify = self._is_clarify_phase(st) if task_id != "lobby" else False
        has_content = False
        if task_id != "lobby":
            try:
                outline, content = self._load_task_text(task_id)
                has_content = bool(outline or content)
            except Exception:
                has_content = False

        intent = classify_intent(m, task_status=st, is_clarify_phase=is_clarify,
                                 has_report_content=has_content)

        # ── Route: CLARIFY_ANSWER ─────────────────────────────────────
        if intent == ChatIntent.CLARIFY_ANSWER:
            reply = self._handle_clarify_answer(task_id, m, st)
            if reply is not None:
                return reply
            qs = st.get("clarify_questions") if isinstance(st.get("clarify_questions"), list) else []
            _, warning = self.validate_clarify_answer(m, qs)
            return warning

        # ── Route: KB_QUERY ───────────────────────────────────────────
        if intent == ChatIntent.KB_QUERY:
            return self._handle_kb_ask(task_id, m)

        # ── Route: MODIFY_REPORT ──────────────────────────────────────
        if intent == ChatIntent.MODIFY_REPORT:
            return self._handle_modification_quick(task_id, m)

        # ── Route: CONTROL_TASK (slash commands) ──────────────────────
        if intent == ChatIntent.CONTROL_TASK and m.startswith("/"):
            act = self._parse_chat_action(m, None)
            if act:
                return self._handle_chat_action(task_id, act)

        # ── Full LLM chain (QUESTION_REPORT / GENERAL_CHAT / lobby) ────
        immediate, chain_input = self._build_context(message, task_id, intent=intent)
        if immediate is not None:
            return immediate

        config = {"configurable": {"session_id": task_id}}
        if task_id == "lobby":
            raw = self._lobby_chain_with_history.invoke(chain_input, config=config)
        else:
            raw = self._task_chain_with_history.invoke(chain_input, config=config)

        t = (raw or "").strip()
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t).strip()
        t = re.sub(r"\s*```$", "", t).strip()
        if not t:
            return raw[:2000] or "当前对话生成失败，请稍后重试。"
        if t.startswith("{"):
            err_msg = ""
            try:
                obj = json.loads(t)
                if isinstance(obj, dict):
                    err_msg = str(obj.get("error") or "").strip()
            except Exception:
                err_msg = ""
            if err_msg:
                err_msg = err_msg[:160]
                return f"对话模型调用失败：{err_msg}。请检查 CONTENT_API_* 或 OPENAI_API_KEY 配置。"
            return t

        self._maybe_summarize_history(task_id)

        # Skip followups if retrieval was poor — KB results can't support meaningful followups
        _skip_followups = "检索到的内容与问题的相关度较低" in t
        followups = "" if _skip_followups else self._generate_followups(message, t, task_id)
        if followups:
            intro = random.choice(self._FOLLOWUP_INTROS)
            t = t.rstrip() + "\n\n---\n💡 " + intro + "\n" + followups

        return t

    def _handle_modification_quick(self, task_id: str, message: str) -> str:
        """Handle MODIFY_REPORT intent — fast path, no LLM invocation.

        Updates user_prompt in task_meta and returns a quick confirmation.
        """
        m = (message or "").strip()
        try:
            meta = self._task_manager.read_task_meta(task_id)
            old_prompt = str(meta.get("user_prompt") or "").strip()
            if m in old_prompt:
                return "💡 你的需求中已包含类似要求，可发送 /regen doc 按最新需求重新生成。"
            new_prompt = (old_prompt + "\n" + m).strip()
            self._task_manager.write_task_meta(task_id, {"user_prompt": new_prompt})
            logger.info("modify_intent task=%s msg_len=%d prompt_len=%d",
                        task_id, len(m), len(new_prompt))
            return "💡 已根据你的反馈更新生成需求。发送 /regen doc 即可按新要求重新生成报告。"
        except Exception as e:
            logger.warning("modify_intent_failed task=%s err=%s", task_id, e)
            return "未能更新需求，请稍后重试。"

    def _handle_kb_ask(self, task_id: str, question: str) -> str:
        """Handle /kb ask command — uses answer_smart() for full KB QA pipeline.

        This is the SAME chain used by the KB panel's /api/kb/query endpoint,
        ensuring consistent answer quality regardless of entry point.
        """
        # Determine which KB to use
        meta = self._task_manager.read_task_meta(task_id)
        kb_name = str(meta.get("active_kb") or "").strip()
        if not kb_name:
            try:
                available = get_kb().list_kb()
            except Exception:
                available = []
            if available:
                kb_name = available[0]
            else:
                return "当前没有可用的知识库。请在 KB 面板上传文档后重试，或使用 /kb use <名称> 选择知识库。"

        try:
            result = get_kb().answer_smart(kb=kb_name, question=question)
        except Exception as e:
            logger.warning("kb_ask_failed kb=%s err=%s", kb_name, e)
            return f"知识库检索失败：{str(e)[:180]}。请检查知识库是否正常。\n\n💡 提示：发送 /kb list 查看可用知识库。"

        # Format citations: group by document, show sections as sub-items
        citations_lines = []
        scores = [float(getattr(c, "score", 0) or 0) for c in (result.citations or [])[:5]]
        logger.info("kb_ask_citations scores=%s", [round(s, 4) for s in scores])
        by_doc: dict[str, list] = {}
        for c in (result.citations or [])[:5]:
            doc_label = (c.doc_name or c.doc_id or "未知文档")
            section = (c.section_path or "").strip()
            # Strip doc_name prefix from section_path if present
            for prefix in [doc_label + " / ", doc_label + "/ "]:
                if section.startswith(prefix):
                    section = section[len(prefix):]
                    break
            score = float(getattr(c, "score", 0) or 0)
            if len(section) > 40:
                section = section[:37] + "…"
            by_doc.setdefault(doc_label, []).append((section, score))
        for doc_label, items in by_doc.items():
            if len(by_doc) == 1:
                # Single document: just doc name + score, no need for chunk excerpts
                top_score = max(s for _, s in items)
                line = f"· {doc_label}"
                if top_score >= 0.01:
                    line += f" · 相关度 {top_score:.2f}"
                citations_lines.append(line)
            else:
                # Multiple documents: each gets its own line
                for sec, score in items:
                    line = f"· {doc_label}"
                    if sec:
                        line += f" · {sec}"
                    if score >= 0.01:
                        line += f" · 相关度 {score:.2f}"
                    citations_lines.append(line)

        reply = result.answer

        # ── Low-quality retrieval: add visible disclaimer ────────────
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        retrieval_poor = avg_score < 0.05 and max_score < 0.1
        if retrieval_poor:
            reply += (
                "\n\n---\n"
                "⚠️ 检索到的内容与问题的相关度较低（平均 {:.3f}），"
                "以上回答可能主要基于通用知识。建议上传相关文档获得更精准的答案。"
            ).format(avg_score)
        elif citations_lines:
            reply += "\n\n📚 来源\n" + "\n".join(citations_lines)

        # ── Write to chat history with structured citation info ──────
        try:
            cit_info = "; ".join(
                f"{c.doc_name or c.doc_id or ''}:{c.section_path or ''}"
                for c in (result.citations or [])[:3]
            ) if result.citations else ""
            self._task_manager.append_chat_history(task_id, [
                {"role": "user", "content": f"[知识库:{kb_name}] {question}"},
                {"role": "assistant", "content": reply, "_kb_citations": cit_info},
            ])
        except Exception:
            pass

        return reply

    def chat_reply_stream(
        self, message: str, task_id: str, history: Optional[list[dict]] = None
    ) -> Iterator[str]:
        """Streaming variant of chat_reply with intent routing.

        Mirrors chat_reply() routing:
          - CLARIFY / MODIFY / CONTROL → yield directly, no LLM
          - QUESTION / GENERAL → stream LLM tokens with filtered context
        """
        m = (message or "").strip()
        st = self._task_manager.read_status(task_id)

        # ── Intent classification ────────────────────────────────────
        is_clarify = self._is_clarify_phase(st) if task_id != "lobby" else False
        has_content = False
        if task_id != "lobby":
            try:
                outline, content = self._load_task_text(task_id)
                has_content = bool(outline or content)
            except Exception:
                has_content = False

        intent = classify_intent(m, task_status=st, is_clarify_phase=is_clarify,
                                 has_report_content=has_content)

        # ── Route: CLARIFY_ANSWER ─────────────────────────────────────
        if intent == ChatIntent.CLARIFY_ANSWER:
            reply = self._handle_clarify_answer(task_id, m, st)
            if reply is not None:
                yield reply
                return
            qs = st.get("clarify_questions") if isinstance(st.get("clarify_questions"), list) else []
            _, warning = self.validate_clarify_answer(m, qs)
            yield warning
            return

        # ── Route: KB_QUERY ───────────────────────────────────────────
        if intent == ChatIntent.KB_QUERY:
            yield self._handle_kb_ask(task_id, m)
            return

        # ── Route: MODIFY_REPORT ──────────────────────────────────────
        if intent == ChatIntent.MODIFY_REPORT:
            yield self._handle_modification_quick(task_id, m)
            return

        # ── Route: CONTROL_TASK ──────────────────────────────────────
        if intent == ChatIntent.CONTROL_TASK and m.startswith("/"):
            act = self._parse_chat_action(m, None)
            if act:
                yield self._handle_chat_action(task_id, act)
                return

        # ── Full LLM chain with intent-filtered context ────────────────
        immediate, chain_input = self._build_context(message, task_id, intent=intent)
        if immediate is not None:
            yield immediate
            return

        # Signal that context is ready, model is about to stream
        yield {"status": "streaming"}

        config = {"configurable": {"session_id": task_id}}
        chain = (
            self._lobby_chain_with_history
            if task_id == "lobby"
            else self._task_chain_with_history
        )

        full = ""
        followup_thread = None
        followup_result = [None]

        def _run_followups():
            try:
                followup_result[0] = self._generate_followups(message, full, task_id)
            except Exception:
                pass

        for chunk in chain.stream(chain_input, config=config):
            full += chunk
            yield chunk
            # Start follow-up generation in background once we have enough content
            if followup_thread is None and len(full) >= 300:
                followup_thread = threading.Thread(target=_run_followups, daemon=True)
                followup_thread.start()

        if not full.strip():
            yield "当前对话生成失败，请稍后重试。"

        # Follow-up suggestions (modification intent is already handled above)
        # Skip if retrieval was too poor to support meaningful followups
        _skip = "检索到的内容与问题的相关度较低" in full or "⚠️ 检索到的内容" in full
        followups = ""
        if not _skip and followup_thread is not None:
            followup_thread.join(timeout=10)
            followups = followup_result[0] or ""
        elif not _skip and len(full) >= 30:
            followups = self._generate_followups(message, full, task_id)

        if followups:
            intro = random.choice(self._FOLLOWUP_INTROS)
            suffix = "\n\n---\n💡 " + intro + "\n" + followups
            yield suffix

        self._maybe_summarize_history(task_id)
