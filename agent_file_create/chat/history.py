"""LangChain BaseChatMessageHistory backed by TaskManager.

Used with RunnableWithMessageHistory so that chat history is automatically
loaded before each LLM call and persisted after it completes.
"""

from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class TaskChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that reads/writes through TaskManager."""

    def __init__(self, task_id: str, task_manager) -> None:
        self._task_id = task_id
        self._tm = task_manager
        self._messages: list[BaseMessage] = []

        raw = self._tm.read_chat_history(task_id)
        if isinstance(raw, list):
            for d in raw:
                if not isinstance(d, dict):
                    continue
                role = str(d.get("role") or "").strip()
                content = str(d.get("content") or "")
                if not role or not content:
                    continue
                if role == "user":
                    self._messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    self._messages.append(AIMessage(content=content))

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        items: list[dict] = []
        for m in messages:
            if isinstance(m, HumanMessage):
                items.append({"role": "user", "content": str(m.content)})
            elif isinstance(m, AIMessage):
                items.append({"role": "assistant", "content": str(m.content)})
            else:
                items.append(
                    {"role": "assistant", "content": str(getattr(m, "content", ""))}
                )
            self._messages.append(m)
        if items:
            self._tm.append_chat_history(self._task_id, items)

    def clear(self) -> None:
        self._messages.clear()
