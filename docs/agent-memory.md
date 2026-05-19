# Agent 对话问答：上下文记忆管理

在文档生成智能体对话中，用户可能在生成过程中随时提问、补充需求、切换话题。需要解决以下问题：LLM 因历史过长丢失关键信息、新旧信息之间冲突、对话失去连贯性"忘记"之前共识。

本项目通过三层记忆架构解决上述问题：`RunnableWithMessageHistory` 自动持久化 + 双阈值自动摘要压缩 + 三级可信度上下文标注。

## 1. RunnableWithMessageHistory — 自动历史持久化

继承 `BaseChatMessageHistory`，以 `task_id` 为 `session_id` 将对话历史读写委托给 `TaskManager`，持久化到 `result/{task_id}/chat_history.json`。

```python
# chat/history.py
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

class TaskChatMessageHistory(BaseChatMessageHistory):
    """以 task_id 为 session_id，通过 TaskManager 读写对话历史"""

    def __init__(self, task_id: str, task_manager: TaskManager):
        self._task_id = task_id
        self._task_manager = task_manager

    @property
    def messages(self) -> list[BaseMessage]:
        raw = self._task_manager.read_chat_history(self._task_id)
        return [
            HumanMessage(content=item["content"]) if item["role"] == "user"
            else AIMessage(content=item["content"])
            for item in raw
        ]

    def add_messages(self, messages: list[BaseMessage]) -> None:
        items = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in messages
        ]
        self._task_manager.append_chat_history(self._task_id, items)
```

在 handler 中将其接入 `RunnableWithMessageHistory`：

```python
# chat/handler.py
self._task_chain_with_history = RunnableWithMessageHistory(
    task_chat_prompt | self._shared_llm | StrOutputParser(),
    get_session_history=self._get_session_history,
    input_messages_key="user_input",
    history_messages_key="history",
)
```

> [!Note]
> `RunnableWithMessageHistory` 接管了调用前后的历史读写：调用前自动加载历史消息注入 `MessagesPlaceholder`，调用后自动将本轮 user/assistant 消息追加写入历史存储。handler 代码无需手动拼接历史消息。

## 2. 自动摘要压缩

### 2.1 触发条件

采用**双重触发**机制，满足其一即执行压缩：

| 触发条件 | 阈值 | 说明 |
|---------|------|------|
| 消息数超标 | > 16 条 | 防止单轮短对话积累过多 |
| Token 数超标 | > 2000 token（且消息 ≥ 6 条） | 防止单条超长消息撑爆上下文窗口 |

> 提示：Token 估算采用保守策略——中文混合文本约 1.5 字符/token，2000 token ≈ 3000 字符。`len(history) >= 6` 的底线确保不会在对话刚开始时就触发压缩。

### 2.2 压缩策略

触发后，取最旧的超过保留阈值的消息，调用 LLM 压缩为不超过 150 字的摘要。摘要必须保留三项信息：

1. 用户首次提出的实质性提问（即第一个非问候、非闲聊的问题）
2. 双方达成的关键决策与偏好
3. 用户明确表示满意/不满意的内容

> [!important]
> 摘要指令中明确要求"禁止编造未发生的对话。如果无法确定某项信息，就不要写入摘要。"这是防幻觉护盾在摘要层的体现——宁可少记，不可乱记。

新摘要与已有旧摘要合并后写回 `TaskManager`，同时截断原始历史到最近 8 条。摘要以"可信度：低"标签注入上下文，LLM 知道这是压缩记忆而非原始对话，避免过度依赖。

## 3. 三级可信度上下文

在 `ChatHandler._build_context()` 中，为三种来源的信息分配明确的**可信度标签**，并在系统提示词中声明优先级：

**可信度：高 — 已生成的报告内容**
经过管线验证的结构化产出。包含已生成的大纲（截取前 1800 字符）和正文相关片段（截取前 2000 字符）。

**可信度：中 — KB 检索片段**
来自知识库的原始材料。提示词中明确标注"辅助参考，可能与报告内容不一致，以报告为准"。

**可信度：低 — 对话摘要**
长期压缩记忆，可能丢失细节。截取前 700 字符注入，避免在摘要不完整时产生误导。

```python
# chat/handler.py — _build_context()
if outline or content:
    context_blocks.append(
        "【可信度：高】已生成的报告内容（优先参考）：\n"
        + ("已生成的大纲：\n" + outline[:1800] if outline else "")
        + ("\n\n正文摘录：\n" + snippets[:2000] if snippets else "")
    )
if kb_snippets:
    context_blocks.append(
        "【可信度：中】知识库检索片段（辅助参考，"
        "可能与报告内容不一致，以报告为准）：\n" + kb_snippets
    )
if summary:
    context_blocks.append(
        "【可信度：低】对话摘要（长期记忆，仅供参考上下文）：\n" + summary[:700]
    )
```

> 提示：在系统提示词中明确声明"**信息优先级：已生成报告 > 知识库 > 对话记忆。当信息不一致时以可信度更高的来源为准。**"这使得 LLM 在面临矛盾信息时能做出正确判断。

## 4. 对话安全机制

### 4.1 问候/社交消息检测

`ChatHandler._is_trivial_message()` 在上下文构建入口处检查消息是否为纯问候（"在吗""你好""hi"）或社交内容（"谢谢""你真棒"）。命中后直接返回简短预设回复，**跳过完整的 LLM 调用**。

> [!Note]
> 最多匹配约 30 个预设表述，限制消息长度 ≤ 15 字符。超过 15 字符的消息即使含有问候词也不会被误判——长消息更可能包含实质性内容。

### 4.2 修改意图检测

`ChatHandler._detect_modification_intent()` 维护约 40 个中文修改意图关键词，覆盖增删改三类操作：

- **删减类**："太长""太短""精简""删除""去掉""缩减"
- **增加类**："增加""补充""添加""扩展""丰富"
- **调整类**："修改""调整""重点""侧重""优化""重写""换个"

检测到修改意图后，系统将用户反馈追加到 `task_meta` 的 `user_prompt` 字段（去重），并在回复末尾追加操作提示："已根据你的反馈更新生成需求。发送 /regen doc 即可按新要求重新生成。"

### 4.3 防幻觉护盾

系统提示词第 5 条规则（`TASK_CHAT_SYSTEM_TEMPLATE`）强制 LLM 对无法确认的历史信息诚实说明：

```
5) 【重要】当用户询问对话历史相关问题时（如"第一个问题是什么""之前聊过什么"），
   如果你无法从上下文中确认，必须诚实说"当前上下文已压缩，无法确认该信息"，
   禁止猜测或编造。不确定的信息不要说得斩钉截铁。
```

## 5. 任务控制事件

每个任务在创建时获得 `pause` 和 `cancel` 两个 `threading.Event`，支持用户在对话中通过 `/pause`、`/resume`、`/cancel` 命令实时控制生成任务的执行状态。

```python
# task/manager.py
_TASK_EVENTS: dict[str, dict[str, threading.Event]] = {}

class TaskManager:
    def pause_task(self, task_id: str) -> None:
        pause_ev, _ = _get_task_events(task_id)
        pause_ev.set()    # 任务线程在检查点处阻塞

    def resume_task(self, task_id: str) -> None:
        pause_ev, _ = _get_task_events(task_id)
        pause_ev.clear()  # 任务线程继续执行

    def cancel_task(self, task_id: str) -> None:
        _, cancel_ev = _get_task_events(task_id)
        cancel_ev.set()   # 任务线程在检查点处退出
```

## 6. 记忆管理的当前局限

| 维度 | 现状 | 可能改进 |
|------|------|---------|
| 历史持久化 | JSON 文件单机存储 | 可迁移到 SQLAlchemy 统一管理 |
| 摘要触发 | 消息数 > 16 或 token > 2000 | 可增加语义漂移检测（话题切换时触发） |
| 跨会话记忆 | 通过 task_id 隔离 | 同一用户不同 task 可共享长期偏好 |
| 历史任务对话 | 前端切换任务时自动加载 + 摘要横幅 | 可支持跨任务对话检索 |
| 用户偏好学习 | 摘要保留 + 修改意图自动更新 | 可构建结构化用户偏好模型 |

## 关键文件

| 文件 | 职责 |
|------|------|
| `chat/history.py` | `TaskChatMessageHistory` — LangChain 历史适配 |
| `chat/handler.py` | `ChatHandler._maybe_summarize_history()` — 摘要触发 |
| 同上 | `ChatHandler._is_trivial_message()` — 问候/社交检测 |
| 同上 | `ChatHandler._detect_modification_intent()` — 修改意图检测 |
| 同上 | `ChatHandler._build_context()` — 三级可信度上下文 |
| `chat/prompts.py` | `ChatPromptTemplate` — 系统提示词定义 |
| `task/manager.py` | `TaskManager` — 历史读写 + 摘要存储 + 控制事件 |
