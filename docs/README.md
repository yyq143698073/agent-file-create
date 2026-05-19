# agent-file-create — 文档生成智能体

agent-file-create 是一个基于大语言模型的多源信息文档生成系统。用户上传 PDF、Word、PPT、Excel、图片等格式材料，系统自动完成信息抽取、大纲规划、分章节正文生成与模板渲染，输出结构化的 Markdown/Word/PDF 报告。

> [!Note]
> 本项目当前处于原型阶段（v0.1.0），核心生成管线可完整运行，部分功能持续迭代中。

## 快速开始

### 安装依赖

```shell
pip install langchain langchain-core langchain-openai langgraph fastapi uvicorn
pip install pymupdf rapidocr-onnxruntime python-docx python-pptx openpyxl
pip install sentence-transformers
```

### 配置文件

复制项目后，通过环境变量配置模型参数：

```shell
# LLM 后端
export OPENAI_API_KEY="your-deepseek-key"
export CONTENT_MODEL_NAME="deepseek-v4-flash"

# 嵌入模型（通过 Ollama 本地部署）
export EMBED_MODEL_NAME="bge-m3"

# 数据库
export DB_PATH="result/app.db"
export KB_DB_PATH="result/kb.db"
```

> [!Note]
> 配置项完整列表见 [agent_file_create/config.py](../agent_file_create/config.py)。

### 启动服务

```shell
python -m agent_file_create --port 8123
```

启动后访问 `http://localhost:8123` 进入 Web 界面。

## 项目结构

```
agent-file-create/
├── agent_file_create/
│   ├── agent/              # LangGraph 文档智能体（ReAct + 7 Tools）
│   ├── chat/               # 对话处理（ChatPromptTemplate + RunnableWithMessageHistory）
│   │   ├── handler.py      # ChatHandler：上下文构建、摘要、查询改写
│   │   ├── history.py      # TaskChatMessageHistory（BaseChatMessageHistory）
│   │   └── prompts.py      # ChatPromptTemplate 定义
│   ├── config.py           # 全局配置（环境变量 + 默认值）
│   ├── document/           # 文档生成管线（抽取 → 大纲 → 正文 → 渲染）
│   │   ├── extractor.py      # 多格式文件信息抽取
│   │   ├── outline_generator.py  # 大纲生成与结构校验
│   │   ├── content_generator.py  # 章节生成、LLM 摘要衔接、终审检查
│   │   └── template_renderer.py  # docx/md/pdf 模板渲染
│   ├── llm_client.py       # 低级 LLM 调用（HTTP 封装）
│   ├── llm_factory.py      # 统一 LLM 工厂（get_chat_model + LRU 缓存）
│   ├── rag/                # RAG 知识库（独立子包，可单独复用）
│   │   ├── kb.py           # KnowledgeBase：摄入、搜索、回答
│   │   ├── chunker.py      # 7 级递归文本分块
│   │   ├── embedder.py     # 向量嵌入（Ollama / OpenAI 双后端）
│   │   ├── store.py        # 向量存储（SQLite / PostgreSQL + pgvector）
│   │   ├── reranker.py     # 重排序（Cross-encoder / LLM 回退）
│   │   ├── retriever.py    # LangChain BaseRetriever 封装
│   │   └── embeddings_lc.py # LangChain Embeddings 封装
│   ├── task/               # 任务生命周期管理
│   │   └── manager.py      # TaskManager：状态机、控制事件、持久化
│   └── web/                # FastAPI Web 服务
│       └── server.py       # REST API + SSE 流式对话
├── html/                   # 前端 SPA（零依赖）
├── template/               # 报告模板（docx/md/pdf）
├── result/                 # 生成结果输出目录
└── docs/                   # 技术文档
```

## 核心功能

### 文档生成管线

```
用户上传文件
      │
      ▼
并行信息抽取 ────── PDF（PyMuPDF+OCR）/ Word / PPT / Excel / 图片 / 文本
      │
      ▼
材料质评 + 用户澄清（信息不足时暂停等待）
      │
      ▼
大纲生成 ───────── PLANNER_MODEL，标题层级校验，最多 3 次自动重试
      │
      ▼
正文生成 ───────── 串行 H2（保证主线逻辑）+ 并行 H3（提速）
      │           每个 H2 完成后增量写入 → 前端实时预览
      ▼
终审检查 ───────── LLM 语义一致性 + 正则事实交叉比对
      │
      ▼
模板渲染 ───────── Markdown / Word / PDF
```

### 对话增强

生成过程中及完成后，对话系统提供以下能力：

- **三级可信度上下文**：已生成报告（高）> KB 检索（中）> 对话摘要（低），信息冲突时以高可信度来源为准
- **自动摘要压缩**：消息数超过 16 条或估算 token 超过 2000 时自动触发，摘要保留首个实质性提问和关键决策
- **路由判断**：简单问题走直接检索 + 回答（快），复杂推理问题走 HyDE + 思维链（准）
- **流式 KB 回答**：Lobby 模式知识库查询采用 LLM 流式生成，享受 SSE 实时输出体验
- **聊天与生成联动**：检测约 40 个中文修改意图关键词（如"太长了""多加数据""换个角度"），自动更新需求并建议 `/regen`
- **实时章节预览**：正文生成过程中每完成一个 H2 章节即增量写入 content.md，前端轮询拉取并渲染
- **单章节重生成**：支持 `/regen <章节标题>` 定向重新生成特定章节及其子节，不影响其他章节
- **问候/社交检测**：纯问候消息直接返回预设回复，不经过完整 LLM 调用
- **防幻觉护盾**：系统提示词要求对不确定的历史信息诚实说明，禁止猜测或编造

### RAG 知识增强

- **HyDE 假设文档嵌入**：让 LLM 先生成假设回答，用其专业风格向量替代用户口语查询做检索，桥接词汇鸿沟
- **思维链推理**：5 步推理 Prompt（问题理解→证据梳理→推理链条→最终回答→自我检查），强制标注 [1] [2] 来源
- **问题分解**：多角度对比问题自动拆解为 2-4 个子问题，分别检索后综合
- **三路混合检索**：向量检索 + 词汇检索 + BM25，RRF 融合排序，Cross-encoder 精排

## 核心依赖

| 组件 | 用途 |
|------|------|
| **LangChain Core** | ChatPromptTemplate、RunnableWithMessageHistory、BaseRetriever、StrOutputParser |
| **LangChain OpenAI / Ollama** | ChatOpenAI、ChatOllama 模型集成 |
| **LangGraph** | create_react_agent 构建文档智能体，MemorySaver 状态检查点 |
| **FastAPI + Uvicorn** | REST API + SSE 流式对话 |
| **PyMuPDF** | PDF 文本提取 + 页面渲染 |
| **RapidOCR** | 图片 OCR / PDF 扫描件文字识别 |
| **python-docx / python-pptx / openpyxl** | Office 文档结构化解析 |
| **FlagEmbedding** | BGE-M3 嵌入 + BGE-Reranker-v2-m3 重排序 |
| **PostgreSQL + pgvector** | 向量存储（生产环境，SQLite 用于本地开发） |

## 技术文档索引

本项目实现了以下技术方案，各方向有独立文档详细说明：

| 文档 | 内容 |
|------|------|
| [langchain-integration.md](langchain-integration.md) | LangChain 框架集成：ChatPromptTemplate 迁移、LCEL 链式组合、RunnableWithMessageHistory、LLM 工厂 + LRU 缓存、BaseRetriever/Embeddings 封装 |
| [agent-memory.md](agent-memory.md) | Agent 对话记忆管理：三层记忆架构、RunnableWithMessageHistory 自动持久化、双阈值摘要压缩、三级可信度上下文、修改意图检测、防幻觉护盾 |
| [rag-knowledge-base.md](rag-knowledge-base.md) | RAG 向量知识库：7 级递归分块、双后端向量化、SQLite/PG+pgvector 存储、三路混合检索+RRF 融合、Cross-encoder/LLM 重排序 |
| [long-text-generation.md](long-text-generation.md) | 长文生成质量保障：四层递进式幻觉防御、LLM 语义摘要驱动的章节逻辑衔接、终审一致性检查、正则结构化事实交叉比对、增量预览与定向重生成 |
| [reasoning-rag.md](reasoning-rag.md) | 推理性 RAG：HyDE 假设文档嵌入、5 步思维链回答、问题分解与综合、三级路由策略 |

## 架构设计文档

完整的系统架构设计请参阅 [architecture_design.md](architecture_design.md)（或 [architecture_design.docx](architecture_design.docx)），涵盖功能模块架构、系统架构和数据库表设计。
