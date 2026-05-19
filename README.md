# agent-file-create

一个“文档生成智能体”Demo：上传多模态文件 → 结构化抽取 → 生成大纲 → 生成正文 → 填充模板输出 md/docx/pdf → 入库（SQLite/PostgreSQL）。

## 运行（CLI）

```bash
python -m agent_file_create
```

把材料文件放到 `resource/` 后按提示选择。

## 运行（Web）

```bash
python agent_web.py
```

访问 `http://127.0.0.1:8000/`，可上传文件/模板并生成报告。

## 配置

全部通过环境变量配置（推荐），见 `config.py`。常用：

- `OLLAMA_HOST` / `MODEL_NAME`（用于本地多模态抽取）
- `OUTLINE_API_STYLE/OUTLINE_API_ENDPOINT/OUTLINE_MODEL_NAME`（大纲模型）
- `CONTENT_API_STYLE/CONTENT_API_ENDPOINT/CONTENT_MODEL_NAME`（正文模型）
- `DB_URL`（PostgreSQL 连接串；留空则用 SQLite `DB_PATH`）

## 知识库（RAG）

Web 端提供一个轻量知识库：上传文件入库（chunk + embedding + SQLite 存储），对话时可选知识库进行检索增强问答。

### PostgreSQL + pgvector

当 `KB_DB_URL`（或 `DB_URL`）为 PostgreSQL 连接串时，知识库会自动使用 pgvector 表结构，并尝试创建向量索引（HNSW/IVFFLAT）。数据库需要安装 `pgvector` 扩展（`create extension vector`）。

### 环境变量

- `KB_DB_PATH`：知识库存储路径（默认 `result/kb.db`）
- `KB_DB_URL`：知识库 PostgreSQL 连接串（默认复用 `DB_URL`；为空则使用 SQLite `KB_DB_PATH`）
- `KB_INDEX_TYPE`：向量索引类型（`hnsw`/`ivfflat`/`both`，默认 `hnsw`）
- `KB_HNSW_EF_SEARCH`：HNSW 查询参数（默认 40）
- `KB_IVFFLAT_PROBES`：IVFFLAT 查询参数（默认 10）
- `EMBED_API_STYLE`：embedding 风格（默认 `ollama`；可选 `openai`）
- `EMBED_MODEL_NAME`：embedding 模型（默认 `nomic-embed-text`）
- `EMBED_API_ENDPOINT`：embedding 端点（ollama 默认用 `OLLAMA_HOST`；openai 需要指向 base_url 或带 /v1/embeddings 的 endpoint）
- `EMBED_API_KEY`：openai embedding 的 key（ollama 可留空）

### 接口

- `POST /api/kb/upload`：multipart 上传文件入库，字段 `kb`（知识库名），文件字段 `files`
- `GET /api/kb/list`：列出知识库
- `POST /api/kb/query`：`{"kb":"xxx","question":"...","top_k":6}` 返回答案与引用片段

### 对话指令

- `/kb list`、`/kb use <kb>`、`/kb clear`

## 安装依赖

```bash
pip install -r requirements.txt
```
