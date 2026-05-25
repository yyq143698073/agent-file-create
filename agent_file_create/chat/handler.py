import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from agent_file_create.chat.history import TaskChatMessageHistory
from agent_file_create.chat.prompts import (
    CHECK_RELEVANCE_PROMPT,
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
from agent_file_create.rag.kb import KnowledgeBase
from agent_file_create.rag.retriever import KnowledgeBaseRetriever
from agent_file_create.task.manager import TaskManager

logger = logging.getLogger(__name__)

_KB = KnowledgeBase()


class ChatHandler:
    def __init__(self, task_manager: Optional[TaskManager] = None, regenerate_fn: Any = None):
        self._task_manager = task_manager or TaskManager()
        self._regenerate_fn = regenerate_fn  # callable(task_id, mode) -> (bool, str)

        self._shared_llm = get_chat_model(
            style=CONTENT_API_STYLE,
            model=CONTENT_MODEL_NAME,
            endpoint=CONTENT_API_ENDPOINT,
            api_key=CONTENT_API_KEY,
            temperature=0.3,
            max_tokens=420,
            timeout_s=120,
        )

        _lobby_chain = lobby_prompt | self._shared_llm | StrOutputParser()
        _task_chain = task_chat_prompt | self._shared_llm | StrOutputParser()

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

    def _split_questions(self, text: str) -> list[str]:
        out: list[str] = []
        cur = ""
        for line in (text or "").splitlines():
            s = line.strip()
            if not s:
                if cur:
                    out.append(cur)
                    cur = ""
                continue
            is_option = bool(re.match(r"^[A-Z][.)、\s]", s))
            if is_option and cur:
                cur += "\n" + s
                continue
            if cur:
                out.append(cur)
            s = re.sub(r"^[0-9]+[.)、\s]+", "", s).strip()
            s = re.sub(r"^[-*]\s+", "", s).strip()
            if s:
                cur = s[:240]
            if len(out) >= 6:
                break
        if cur and len(out) < 6:
            out.append(cur)
        if out:
            return out
        s = str(text or "").strip()
        return [s[:240]] if s else []

    def _tokenize(self, text: str) -> list[str]:
        xs = re.findall(r"[一-鿿A-Za-z0-9]{2,}", str(text or ""))
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
            for tok in q_tokens:
                if tok and tok in hay:
                    score += 1
            if score > 0:
                scored.append((score, title, body))
        scored.sort(key=lambda x: (-x[0], len(x[2])))
        out: list[str] = []
        for _, title, body in scored[:max_sections]:
            block = ("## " + title + "\n" + (body or "")).strip()
            if len(block) > max_chars:
                block = block[:max_chars] + "…"
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
                "- /kb list：列出知识库",
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
                arg = parts[1].strip().lower() if len(parts) >= 2 else ""
                if arg in {"all", "full", ""}:
                    scope = "all" if arg in {"all", "full"} else "doc"
                    return {"type": "regenerate", "scope": scope}
                else:
                    # Section name — join all remaining parts preserving original casing
                    section_name = " ".join(parts[1:]).strip()
                    return {"type": "regenerate", "scope": "section", "section": section_name}
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
            items = _KB.list_kb()
            if not items:
                return "暂无知识库。可调用 /api/kb/upload 上传文件入库。"
            return "知识库列表：\n" + "\n".join(["- " + str(x) for x in items[:50]])
        if typ == "kb_use":
            kb = str(action.get("kb") or "").strip()
            if not kb:
                return "用法：/kb use <kb>"
            self._task_manager.write_task_meta(task_id, {"active_kb": kb})
            return f"已选择知识库：{kb}。后续提问会优先使用该知识库检索。"
        if typ == "kb_clear":
            self._task_manager.write_task_meta(task_id, {"active_kb": ""})
            return "已取消知识库选择。"
        if typ == "kb_stats":
            kb = str(action.get("kb") or "").strip() or "default"
            try:
                st = _KB.kb_stats(kb=kb)
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
                    r = _KB.delete_doc(kb=kb, doc_id=doc_id)
                    if r.get("ok"):
                        return f"已从知识库 {kb} 中删除文档 {doc_id}。"
                    return f"删除失败：{r.get('error', 'unknown')[:180]}"
                else:
                    r = _KB.delete_kb(kb=kb)
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
            return "已暂停该任务。"
        if typ == "resume":
            self._task_manager.resume_task(task_id)
            if str(st.get("status") or "") == "paused":
                self._task_manager.write_status(
                    task_id, "processing", stage=stage or "resume",
                    message="已继续执行…", extra={}
                )
            return "已继续该任务。"
        if typ == "cancel":
            self._task_manager.cancel_task(task_id)
            self._task_manager.write_status(
                task_id, "canceled", stage=stage or "cancel",
                message="已取消", extra={}
            )
            return "已取消该任务。"
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
                return "已更新需求并重新生成。"
            return "已更新需求（未自动重跑）。如需重跑，请发送：/regen doc"
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
            if self._regenerate_fn:
                ok, msg = self._regenerate_fn(task_id, scope if scope else "doc", section_name=section_name)
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

        # For very short, potentially irrelevant answers, use quick LLM check
        qtext = "\n".join([f"- {q}" for q in (questions or [])[:4] if q.strip()])
        if not qtext:
            qtext = "（未提供具体问题）"
        try:
            chain = CHECK_RELEVANCE_PROMPT | self._shared_llm | StrOutputParser()
            result = (
                chain.invoke({
                    "clarify_question": qtext,
                    "user_reply": m,
                }) or ""
            ).strip().upper()
            if result and "NO" in result and "YES" not in result:
                return False, (
                    f"你的回复似乎与当前问题不太相关。请针对以下问题提供补充信息：\n{qtext}\n\n"
                    "如果不需要补充，回复：跳过"
                )
        except Exception:
            return True, ""

        return True, ""

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
            chain = FOLLOWUPS_PROMPT | self._shared_llm | StrOutputParser()
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
            chain = SUMMARIZE_HISTORY_PROMPT | self._shared_llm | StrOutputParser()
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
            chain = REWRITE_QUERY_PROMPT | self._shared_llm | StrOutputParser()
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

    @staticmethod
    def _is_trivial_message(message: str) -> Optional[str]:
        """Return a short reply if the message is a pure greeting or social
        chat that doesn't warrant a full LLM invocation. Otherwise None."""
        m = (message or "").strip()
        if not m or len(m) > 15:
            return None

        low = m.lower()

        if low in {
            "在吗", "在不在", "在了吗", "在不在呀",
            "你好", "您好", "你好啊", "嗨", "hi", "hello", "hey",
            "早上好", "下午好", "晚上好", "晚安", "中午好",
        }:
            return "你好！有什么可以帮你的？你可以上传材料生成报告，或者直接在对话框中提问。"

        if low in {
            "我喜欢你", "我爱你", "爱你",
            "能叫我宝贝吗", "叫我宝贝",
            "你真棒", "太棒了", "厉害", "牛逼",
            "谢谢", "谢谢你", "感谢", "thx", "thanks", "thank you",
        }:
            return "谢谢你的反馈！有什么关于文档生成或报告的问题，随时可以问我。"

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
        self, message: str, task_id: str
    ) -> tuple[Optional[str], dict[str, str]]:
        """Check for immediate replies; if none, build the chain input dict.

        Returns (immediate_text, chain_input).
        If immediate_text is not None the caller should return it directly.
        Otherwise chain_input is ready for the LCEL chain.
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
                        search_query = _KB._hyde_expand(msg)
                    else:
                        search_query = self._rewrite_query(msg)
                    retriever = KnowledgeBaseRetriever(
                        kb=active_kb, knowledge_base=_KB, top_k=6, context_window=2
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

            if stage:
                base = f"当前任务正在处理（task_id={task_id} stage={stage}）。"
            else:
                base = f"当前任务正在处理（task_id={task_id}）。"
            if msg:
                base += "\n状态：" + msg
            return ("\n".join([base, "你可以继续提问，我会结合当前进度给建议。"]).strip(), {})

        # ── Build context for task-chat mode ────────────────────────────
        snippets = self._pick_snippets(message, content) if content else ""
        progress_hint = ""
        if status in {"queued", "processing", "need_user"}:
            progress_hint = "（提示：当前任务可能尚未完全生成，回答会基于现有内容，必要时我会提示缺失。）"

        kb_snippets = ""
        if active_kb:
            try:
                msg = str(message or "")
                # For complex questions, expand via HyDE before retrieval
                if self._is_complex_question(msg):
                    search_query = _KB._hyde_expand(msg)
                else:
                    search_query = self._rewrite_query(msg)
                retriever = KnowledgeBaseRetriever(
                    kb=active_kb, knowledge_base=_KB, top_k=6, context_window=2
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
            context_blocks.append(
                "【可信度：高】已生成的报告内容（优先参考）：\n"
                + ("已生成的大纲：\n" + self._safe_str(outline, 1800) if outline else "")
                + ("\n\n与问题相关的正文摘录：\n" + self._safe_str(snippets, 2000) if snippets else "")
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
        immediate, chain_input = self._build_context(message, task_id)
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

        # Check for modification intent (higher priority, replaces follow-ups)
        mod_suffix = self._handle_modification_intent(message, task_id)
        if mod_suffix:
            t = t.rstrip() + mod_suffix
        else:
            followups = self._generate_followups(message, t, task_id)
            if followups:
                t = t.rstrip() + "\n\n---\n💡 你可能还想问：\n" + followups

        return t

    def chat_reply_stream(
        self, message: str, task_id: str, history: Optional[list[dict]] = None
    ) -> Iterator[str]:
        immediate, chain_input = self._build_context(message, task_id)
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

        # Check for modification intent (higher priority, replaces follow-ups)
        mod_suffix = self._handle_modification_intent(message, task_id)
        if mod_suffix:
            yield mod_suffix
        else:
            # Collect follow-up suggestions
            followups = ""
            if followup_thread is not None:
                followup_thread.join(timeout=10)
                followups = followup_result[0] or ""
            elif len(full) >= 30:
                followups = self._generate_followups(message, full, task_id)

            if followups:
                suffix = "\n\n---\n💡 你可能还想问：\n" + followups
                yield suffix

        self._maybe_summarize_history(task_id)
