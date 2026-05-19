# RAG 向量知识库：分块、向量化、检索与重排序

## 整体架构

```
文档摄入                                查询
   │                                     │
   ▼                                     ▼
_read_any_text()                   MD5 缓存检查
   │                                     │
   ▼                                     ▼
chunk_text()                       embed_texts(query)
   │                                     │
   ▼                              ┌──────┴──────┐
embed_texts(chunks)               │ 向量搜索    │ 词汇搜索
   │                              │ (cosine)   │ (LIKE %term%)
   ▼                              └──────┬──────┘
store.upsert_chunks()                    │
   │                              ┌──────┴──────┐
   ▼                              │  合并 + BM25 │
SQLite / PostgreSQL                └──────┬──────┘
+ pgvector                                │
                                   ┌──────┴──────┐
                                   │  RRF 融合   │
                                   │  (k=60)    │
                                   └──────┬──────┘
                                          │
                                   ┌──────┴──────┐
                                   │  重排序     │
                                   │  cross-enc  │
                                   └──────┬──────┘
                                          │
                                          ▼
                                   _ANSWER_PROMPT | LLM | StrOutputParser()
```

## 一、文本分块（chunker.py）

### 分块参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `target_chars` | 1200 | 目标块大小（字符） |
| `overlap_chars` | 120 | 块间重叠字符数 |
| `max_chunks` | 800 | 单文档最大块数 |

### 分块流程

```
原始文本
  │
  ├─ 1. 文本规范化
  │     \r\n → \n，3+ 连续换行 → 2，行尾空白去除
  │
  ├─ 2. 章节检测
  │     Markdown：解析 # ～ ###### 标题层级
  │     纯文本：启发式判断（3-30 字、无句末标点、50%+ CJK/Latin）
  │
  ├─ 3. 段落切分
  │     Markdown：按空行 / 代码块 / 表格边界切分
  │     纯文本：按 \n\n 段落边界切分
  │
  ├─ 4. 按目标大小组装
  │     段落逐个加入当前块，超过 target_chars 时 flush
  │     overlap 从块尾向前找语义边界（换行、句末标点）
  │
  └─ 5. 递归超长切分（7 级分隔符）
        ["\n\n", "\n",
         r"(?<=[。！？!?])(?=\S)",   # 中文句末
         r"(?<=[.!?])\s+(?=\S)",    # 英文句末
         r"(?<=[；;])(?=\S)",       # 分号
         r"(?<=[，,])(?=\S)",       # 逗号
         " "]                        # 空格
```

### chunk_id 格式

```
{filename}:{chunk_index}
# 例如："report.pdf:42"、"制度汇编.docx:7"
```

内容格式：`{section_path}\n{chunk_text}`，section_path 如 `"第三章 / 3.1 定义 / 3.1.2 术语"`。

### 父子块与上下文窗口检索

分块完成后，每 4 个连续子块绑定一个**父块 ID**（`{doc_id}:parent:{parent_idx}`）：

```python
# chunker.py
def _assign_parent_chunks(chunks: list[Chunk], doc_id: str, parent_size: int = 4) -> None:
    """将相邻子块编组，赋予共享的 parent_chunk_id"""
```

**检索时**（kb.py → `search_with_context()`），当某个子块命中查询后，自动拉取其所属文档的相邻块（窗口大小 `context_window=2`），让 LLM 获取更完整上下文：

```python
# kb.py
def search_with_context(self, *, kb, query, top_k=6, context_window=2, filters=None) -> list[Hit]:
    hits = self.search(kb=kb, query=query, top_k=top_k, filters=filters)
    # 为每个命中块拉取其相邻 chunk_index ± context_window 的兄弟块
    for h in hits:
        neighbors = self._fetch_neighbor_chunks(
            kb=kb, doc_id=h.doc_id,
            center_idx=h.chunk_index, window=context_window)
        for nh in neighbors:
            nh.meta["is_context"] = True
            nh.score *= 0.85  # 上下文块分数略微打折
            expanded[nh.chunk_id] = nh
    # 按 doc_id + chunk_index 排序后返回
    return sorted(expanded.values(), key=lambda x: (x.doc_id, x.chunk_index))
```

`KnowledgeBaseRetriever` 默认启用 `context_window=2`，调用 `search_with_context()` 替代普通 `search()`。

**好处**：不需要修改数据库 schema（无额外存储开销），在查询时按需拉取相邻块，保证 LLM 看到的片段有更好的前后文连贯性。

## 二、向量嵌入（embedder.py）

### 双后端支持

| 后端 | 模型 | API |
|------|------|-----|
| Ollama | `bge-m3` | `POST {OLLAMA_HOST}/api/embeddings` |
| OpenAI 兼容 | 可配置 | `POST {base}/v1/embeddings` |

### 批处理与重试

```python
def embed_texts(texts, *, timeout_s=60, max_batch=24):
    """
    将文本列表分批嵌入。
    - 默认 batch=24（Ollama 单次请求稳定上限）
    - 3 次重试，间隔 0.6 + 0.6 * attempt 秒
    """
```

摄入文件时固定 `max_batch=24`，搜索时 `max_batch=1`（单条查询）。

## 三、向量存储（store.py）

### 双后端自动选择

```python
def default_store():
    url = (KB_DB_URL or "").strip()
    if url.startswith("postgresql://") or url.startswith("postgres://"):
        return PostgresVectorStore(db_url=url)
    else:
        return SQLiteVectorStore(db_path=(url or KB_DB_PATH))
```

| 后端 | 适用场景 | 向量索引 |
|------|---------|---------|
| SQLiteVectorStore | 轻量部署、单机 | 无（Python 内存中计算 cosine） |
| PostgresVectorStore | 生产环境、大数据量 | pgvector：HNSW（m=16, ef=64）或 IVFFlat（lists=100, probes=10） |

### 数据表结构

```sql
-- kb_docs（文档元信息）
kb, id, title, source, doc_type, meta, updated_at

-- kb_chunks（分块 + 向量）
kb, doc_id, chunk_index, section_path, content, embedding, meta
```

### 查询缓存

```python
_QUERY_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_QUERY_CACHE_MAX = 128

# 查询文本 → MD5 → 缓存查找
# LRU 淘汰，最大 128 条
```

## 四、混合检索（kb.py → search）

### 三路召回

```python
def search(self, *, kb, query, top_k=8, filters=None) -> list[Hit]:
    cap = 160
    vec_cand = min(max(30, top_k * 10), cap)   # 向量召回数
    lex_cand = min(max(30, top_k * 10), cap)   # 词汇召回数

    # 1) 向量搜索：cosine 相似度
    vec_hits = store.similarity_search(kb=kb, query_embedding=qv, top_k=vec_cand)

    # 2) 词汇搜索：SQL LIKE %term% 匹配
    lex_hits = store.lexical_search(kb=kb, query=q, top_k=lex_cand)

    # 3) 合并 + BM25 重打分
    merged = 合并向量和词汇命中（按 chunk_id 去重）
    bm25_scores = BM25(query_terms, doc_terms, k1=1.2, b=0.75)
```

### BM25 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| k1 | 1.2 | 词频饱和参数 |
| b | 0.75 | 文档长度归一化 |
| 查询词最大数 | 40 | |
| 单文档词最大数 | 160 | |
| IDF 平滑 | `log((N-df+0.5)/(df+0.5)+1.0)` | |

### RRF 融合（Reciprocal Rank Fusion）

```python
k_rrf = 60.0

# 对每个 chunk 的三路排名求 RRF 得分
rrf_score = 1/(60 + vec_rank) + 1/(60 + bm25_rank) + 1/(60 + lex_rank)

# 按 RRF 得分降序排列，返回 top_k 结果
```

RRF 的优势是不需要校准各打分器的绝对分值，只关心相对排名。

## 五、重排序（reranker.py）

### 两级重排序策略

```
候选结果（top_k * 3 = 最多 80 个）
        │
        ▼
┌───────────────────────────┐
│ 1. Cross-encoder 重排序   │  ← 首选：BAAI/bge-reranker-v2-m3
│    逐对 (query, chunk)    │
│    compute_score(normalize=True)
└───────────┬───────────────┘
            │ 不可用 / 失败
            ▼
┌───────────────────────────┐
│ 2. LLM 重排序（回退）      │  ← 备选：LLM 排序 top-20 候选
│    prompt: "请按相关性排序" │
│    解析编号列表            │
└───────────┬───────────────┘
            │ 全部失败
            ▼
┌───────────────────────────┐
│ 3. 无重排序                │  ← 原始 RRF 排名截断
└───────────────────────────┘
```

### 后处理

- 每文档最多选 2 个块（diversity 控制）
- 相邻 chunk（index 差 ≤ 1）合并为 segment
- 每 snippet 截断到 900 字
- 总上下文上限 `max_context_chars=5200`

## 六、查询优化

### 查询改写（chat handler 层）

```python
def _rewrite_query(self, message: str) -> str:
    """短查询（10-80 字）扩展关键词和同义词"""
    prompt = (
        "将用户问题改写为一个适合知识库检索的查询短语，"
        "补充可能相关的关键词和同义词。只输出改写后的查询，不要解释。\n\n"
        f"用户问题：{q}\n\n查询："
    )
```

### HyDE 扩展（kb.py 层，复杂问题自动启用）

```python
def _hyde_expand(self, question: str) -> str:
    """生成假设回答，用其 embedding 做检索"""
    # 1. LLM 生成 3-5 句假设回答
    # 2. 用假设回答的 embedding 去搜索
    # 3. 桥接短查询与长文档之间的词汇鸿沟
```

### 问题分解（复杂推理问题）

```python
def _decompose_question(self, question: str) -> list[str]:
    """将复杂问题分解为 2-4 个子问题"""
    # 对对比类、多角度分析类问题有效
```

## 七、回答生成

```python
_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """\
你是企业知识库问答助手。请基于给定的检索片段回答问题。
规则：
1) 只根据检索片段回答；不要编造。
2) 如果片段不足以回答，明确说不确定。
3) 输出尽量简洁，必要时 3-6 条要点。
4) 末尾追加一行：依据：<引用编号（最多3条）>

知识库：{kb}
检索片段：{context}
用户问题：{question}
回答："""),
])

# LCEL 链
chain = _ANSWER_PROMPT | get_chat_model(...) | StrOutputParser()
answer = chain.invoke({"context": ctx, "question": q, "kb": kb})
```

## 关键文件

| 文件 | 职责 |
|------|------|
| `rag/chunker.py` | `chunk_text()` — 7 级递归分块 + `_assign_parent_chunks()` |
| `rag/embedder.py` | `embed_texts()` — 双后端嵌入 |
| `rag/store.py` | SQLiteVectorStore / PostgresVectorStore + `list_docs()` |
| `rag/kb.py` | `search()` / `search_with_context()` / `answer()` / `answer_with_reasoning()` / `_fetch_neighbor_chunks()` |
| `rag/reranker.py` | `rerank()` — cross-encoder + LLM 回退 |
| `rag/retriever.py` | `KnowledgeBaseRetriever` — LangChain BaseRetriever + context_window |
| `rag/embeddings_lc.py` | `ChatchatEmbeddings` — LangChain Embeddings |
