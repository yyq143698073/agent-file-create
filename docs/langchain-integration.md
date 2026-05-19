# LangChain 框架集成与模块化架构

agent-file-create 从原始的 `call_llm()` 函数调用模式全面迁移到 LangChain 框架体系。本文档说明核心迁移路径和当前采用的架构模式。

> [!Note]
> 迁移过程中从 `llm_client.py`（原 713 行）移除了流式调用、异步调用等已被 LangChain 体系覆盖的函数，共计删除 292 行死代码，最终精简至 421 行。

## 1. 模块化目录设计

项目按功能域划分模块，遵循**单向依赖**原则：底层模块不依赖上层模块。

```
agent_file_create/
├── agent/          # 智能体（LangGraph ReAct + 7 Tools）
├── chat/           # 对话处理
│   ├── handler.py  # 对话控制器（编排层）
│   ├── history.py  # 历史持久化（BaseChatMessageHistory）
│   └── prompts.py  # Prompt 模板（ChatPromptTemplate）
├── config.py       # 全局配置入口（20+ 环境变量）
├── document/       # 文档生成管线（抽取→大纲→正文→渲染）
├── llm_client.py   # 低级 LLM 调用（HTTP 封装）
├── llm_factory.py  # 统一 LLM 工厂 + LRU 缓存
├── rag/            # RAG 知识库（独立子包，可单独复用）
│   ├── kb.py       # KnowledgeBase（对外 API）
│   ├── chunker.py  # 分块策略
│   ├── embedder.py # 向量化
│   ├── store.py    # 存储后端
│   ├── reranker.py # 重排序
│   ├── retriever.py    # LangChain BaseRetriever 封装
│   └── embeddings_lc.py # LangChain Embeddings 封装
├── task/           # 任务生命周期管理
│   └── manager.py  # TaskManager（状态 + 历史 + 控制事件）
└── web/            # FastAPI Web 服务
    └── server.py   # REST API 入口
```

### 模块依赖规则

```
web/server.py ─► chat/handler.py ─► rag/retriever.py ─► rag/kb.py ─► rag/store.py
       │                │                                        │
       │                ├► rag/kb.py（Lobby 直接检索）              ├► rag/chunker.py
       │                ├► llm_factory.py                          ├► rag/embedder.py
       │                └► task/manager.py                         └► rag/reranker.py
       │
       ├► task/manager.py
       ├► document/content_generator.py ─► llm_client.py
       └► agent/document_agent.py ─► document/ (extractor/outline/content/template)
```

> [!important]
> `rag/` 是独立子包，不依赖 `chat/`、`document/`、`web/` 模块。这意味着你可以将整个 `rag/` 目录复制到其他项目中单独使用。`chat/` 依赖 `rag/` 和 `task/`，但不依赖 `document/`。`web/` 是所有模块的组装层。

## 2. ChatPromptTemplate 替代 f-string Prompt

### 迁移前

```python
# 散落在 handler.py 各处的字符串拼接
system = f"你是一个文档生成助手。当前任务状态：{status}。大纲：{outline}"
# 容易出错：XSS、难以复用、无法使用 LangChain 消息占位符
```

### 迁移后

所有 Prompt 模板集中在 `chat/prompts.py` 管理：

```python
# chat/prompts.py — 集中管理所有 prompt 模板
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

LOBBY_SYSTEM_TEMPLATE = """\
你是一个文档生成与问答助手。用户可能尚未上传材料或选择任务。

你能做的事情：
- 指导用户上传材料（支持 PDF/Word/PPT/Excel/图片/文本），系统会自动抽取信息并生成报告。
- 回答关于报告类型、结构、风格的咨询。
- 管理知识库（/kb list|use|clear）用于辅助问答。

要求：
1) 只输出中文。2) 回答简洁，必要时用 3-6 条要点。
3) 如果用户询问操作步骤，给出具体可执行的指令（使用 /xxx 格式）。"""

lobby_prompt = ChatPromptTemplate.from_messages([
    ("system", LOBBY_SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}"),
])
```

> [!Note]
> `MessagesPlaceholder` 让历史消息由 `RunnableWithMessageHistory` 自动注入，无需手动拼接。变量通过 `chain.invoke({"user_input": ..., "context_text": ...})` 传入，类型安全。Prompt 模板与 handler 业务逻辑完全分离，修改 prompt 不涉及业务流程代码。

## 3. LCEL 链式组合

### 3.1 基本链

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"user_input": "你好", "context_text": "..."})
```

### 3.2 带历史管理的链

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

self._task_chain_with_history = RunnableWithMessageHistory(
    task_chat_prompt | self._shared_llm | StrOutputParser(),
    get_session_history=self._get_session_history,
    input_messages_key="user_input",
    history_messages_key="history",
)

# 调用时传入 config 指定 session_id
config = {"configurable": {"session_id": task_id}}
result = self._task_chain_with_history.invoke(chain_input, config=config)
```

`RunnableWithMessageHistory` 在调用前后自动完成三项操作：从 `TaskChatMessageHistory` 加载历史消息、将历史注入 `MessagesPlaceholder`、将本轮 user/assistant 消息写回存储。

## 4. Lobby KB 流式回答

Lobby 模式下知识库查询原先通过 `_KB.answer()` 直接返回完整答案（阻塞式，无流式）。现已改为检索 + 流式组合：

```python
# chat/handler.py — _build_context() lobby 模式
if active_kb:
    # 1. 用 KnowledgeBaseRetriever 检索文档
    retriever = KnowledgeBaseRetriever(kb=active_kb, knowledge_base=_KB, top_k=6)
    docs = retriever.invoke(search_query)

    # 2. 组装 KB 上下文片段
    kb_context = "\n\n".join(blocks).strip()

    # 3. 注入 user_input，通过 lobby chain 流式生成
    enriched = "知识库检索结果（请基于以下资料回答）：\n\n" \
               f"{kb_context}\n\n用户问题：{message}"
    return None, {"user_input": enriched}
```

> 提示：Lobby KB 不再返回预生成的静态文本，而是将检索结果注入 `user_input` 字段，通过 lobby chain 享受与其他对话相同的 SSE 流式输出体验。回答由 LLM 基于检索结果动态组织，质量更优。

## 5. LLM 工厂 + LRU 缓存

统一工厂函数消除散落各处的模型实例化：

```python
# llm_factory.py
from functools import lru_cache

@lru_cache(maxsize=32)
def get_chat_model(
    *,
    style: str,        # "openai" | "ollama"
    model: str,
    endpoint: str,
    api_key: str = "",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout_s: int = 120,
    stop_tuple: Tuple[str, ...] = (),
) -> BaseChatModel:
    """统一创建 ChatModel，按 (style, endpoint, model, ...) 做 LRU 缓存"""
```

> [!Note]
> 缓存键由调用参数组成，相同参数重复调用直接命中缓存，避免重复实例化。不同模块（chat、rag、document）使用各自的模型配置调用，互不冲突。`content_generator.py` 保留 `llm_client.py` 调用（历史兼容），其他模块统一使用此工厂。

## 6. LangChain RAG 抽象

### 6.1 BaseRetriever 封装

```python
# rag/retriever.py
class KnowledgeBaseRetriever(BaseRetriever):
    kb: str
    knowledge_base: Any
    top_k: int = 6

    def _get_relevant_documents(self, query: str) -> list[Document]:
        hits = self.knowledge_base.search(kb=self.kb, query=query, top_k=self.top_k)
        return [
            Document(
                page_content=h.content,
                metadata={"doc_id": h.doc_id, "section_path": h.section_path, "score": h.score}
            )
            for h in hits
        ]
```

### 6.2 Embeddings 封装

```python
# rag/embeddings_lc.py
class ChatchatEmbeddings(Embeddings):
    """LangChain 兼容的 Embeddings，背后调用 embed_texts()"""
    timeout_s: int = 60
    max_batch: int = 32

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embed_texts(texts, timeout_s=self.timeout_s, max_batch=self.max_batch)

    def embed_query(self, text: str) -> List[float]:
        vecs = embed_texts([text], timeout_s=self.timeout_s, max_batch=1)
        return vecs[0] if vecs else []
```

这些抽象使得 RAG 模块可以无缝接入 LangChain 的 `VectorStore`、`RetrievalQA` 等高层组件。

## 7. 配置管理

所有 LLM 相关配置统一收敛到 `config.py`，通过环境变量注入：

| 角色 | 配置项 | 默认值 |
|------|--------|--------|
| 大纲生成 | `OUTLINE_MODEL_NAME` | `deepseek-v4-flash` |
| 内容生成 | `CONTENT_MODEL_NAME` | `deepseek-v4-flash` |
| 嵌入 | `EMBED_MODEL_NAME` | `bge-m3` |
| 信息提取 | `EXTRACT_MODEL_NAME` | `qwen3.5:4b` |
| 视觉理解 | `VISION_MODEL_NAME` | `minicpm-v:8b` |
| 重排序 | `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` |

每个 LLM 调用点只需引用配置常量，不硬编码模型名。API endpoint、style、key 同理。
