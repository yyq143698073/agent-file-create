# 推理性 RAG：HyDE、思维链与问题分解

标准 RAG 对于事实性查询（"报销流程是什么？"）效果好，但对于推理性质的自然语言问题（"为什么 A 方案比 B 方案更适合我们？"）表现不佳，原因有三：用户查询用词与知识库文档用词不匹配导致检索召回率低（词汇鸿沟）、LLM 只做"检索→摘要"而不做"检索→分析→推理→结论"、以及多角度问题需要不同方向的证据而单次检索难以覆盖。

本项目通过三个层次的机制解决上述问题：查询层的 HyDE 假设文档嵌入、推理层的思维链回答、综合层的问题分解与融合。

## 1. HyDE 假设文档嵌入

HyDE（Hypothetical Document Embeddings）的核心思路是：让 LLM 先生成一段"假设回答"，用知识库文档的专业语言风格撰写，再用这段假设回答的向量去做相似度检索。假设回答的专业用词与真实文档的用词更接近，因此其 embedding 能更准确地命中相关文档。

```
传统 RAG： query → embedding → search → results
HyDE RAG：  query → LLM 生成假设回答 → embedding → search → results
                            ↑
                   桥接词汇鸿沟的关键步骤
```

### 实现

```python
# rag/kb.py

_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个中文技术文档撰写助手。"),
    ("human", """\
请用3-5句话编写一段可能回答以下问题的文本段落。要求：
- 使用专业、正式的语气
- 包含可能的关键术语和概念
- 模拟知识库文档的风格
- 只输出文本段落，不要解释或标注

问题：{question}

假设回答："""),
])

def _hyde_expand(self, question: str) -> str:
    """生成假设回答，用其 embedding 做检索"""
    q = str(question or "").strip()
    if len(q) < 10:
        return q  # 太短的问题不值得 HyDE
    chain = _HYDE_PROMPT | self._get_answer_llm() | StrOutputParser()
    hypothetical = chain.invoke({"question": q}).strip()
    return hypothetical[:600] if len(hypothetical) >= 15 else q
```

> [!Note]
> HyDE 有触发条件：短于 10 字的问题不触发（太短的查询写不出有意义的假设回答），生成的假设回答短于 15 字符时回退到原始查询。

## 2. 思维链回答（Chain-of-Thought）

设计 5 步推理 Prompt，强制 LLM 进行显式推理并标注来源：

```python
_ANSWER_COT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个中文助手，擅长逐步推理和严谨分析。"),
    ("human", """\
你是企业知识库问答助手。请基于给定的检索片段，通过逐步推理回答问题。

推理步骤：
1) 问题理解：用自己的话复述问题的核心要点和隐含假设。
2) 证据梳理：从检索片段中逐条列出相关证据，标注来源编号（如 [1] [2]）。
3) 推理链条：基于证据进行逐步推理。若多个证据之间存在关联
   （因果、对比、递进等），请明确说明推理路径。
4) 最终回答：给出简洁的最终答案（3-6 条要点）。
5) 自我检查：逐条核查最终回答中的每个论断——是否有对应的检索片段支撑？
   无支撑的推断请明确标注为「（推测）」或「（材料未覆盖）」。

知识库：{kb}
检索片段：{context}
用户问题：{question}

推理过程："""),
])
```

### 与传统回答的对比

| 维度 | 传统 `answer()` | 思维链 `answer_with_reasoning()` |
|------|----------------|-------------------------------|
| Prompt 步骤 | 直接回答 | 5 步推理 |
| 证据引用 | 可选 | 强制标注 [1] [2] |
| 推理路径 | 隐式 | 显式说明（因果/对比/递进） |
| 自我验证 | 无 | 步骤 5 逐条核查 |
| 不确定性标注 | 无 | "（推测）""（材料未覆盖）" |
| LLM 调用 | 1 次 | 1 次（但 prompt 更长，推理 token 更多） |

## 3. 问题分解与综合

对于比较类、多角度分析类问题（"从成本、性能、可维护性三个角度对比 A 和 B"），一次检索难以覆盖所有方面。系统首先判断是否需要分解：

```python
_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个中文问答系统分析师。"),
    ("human", """\
判断以下问题是否需要分解为子问题来回答。如果需要，请分解为2-4个子问题，每个子问题一行。
如果问题本身很简单、不需要分解，只回复：SIMPLE

问题：{question}

分解结果："""),
])
```

如需分解，每个子问题独立走完整的检索 + 回答流程，最后通过综合 prompt 融合各子问题的分析结果，要求标注不同证据之间的关联（因果、对比、互补），并在存在矛盾时指出并给出最可能的结论。

### 数据流

```
复杂问题
  │
  ├─ 分解为子问题
  │    ├─ 子问题1 → HyDE → search → answer
  │    ├─ 子问题2 → HyDE → search → answer
  │    └─ 子问题3 → HyDE → search → answer
  │
  └─ 综合（融合 + 标注关联 + 处理矛盾）
       │
       ▼
     最终回答
```

## 4. 路由策略

在 chat handler 层通过 `_is_complex_question()` 进行三级路由判断。简单问题（少于 60 字且无推理关键词）走 `answer()` 直接检索 + 简洁回答，复杂推理问题（含"为什么""对比""分析"等标记）走 `answer_with_reasoning()`（HyDE + 5 步思维链），多角度/对比类问题走 `decompose_and_answer()`（分解各角度 → 分别检索 → 综合）。

```python
@staticmethod
def _is_complex_question(message: str) -> bool:
    q = message.strip()
    if len(q) > 60:
        return True
    complex_markers = [
        "为什么", "如何", "怎么", "原因", "影响", "关系",
        "对比", "区别", "比较", "优劣", "优缺点", "分析",
        "vs", "是否", "应该", "如果", "假设", "评估", "判断",
    ]
    return any(m in q for m in complex_markers)
```

> 提示：Lobby 模式的 KB 查询不再通过 `_KB.answer()` 返回预生成的静态答案，而是将检索结果注入 `user_input`，通过 lobby chain 流式生成回答，享受 SSE 流式输出体验。

## 5. 当前局限与改进方向

| 维度 | 现状 | 可能改进 |
|------|------|---------|
| HyDE | 单次假设回答 | 可迭代：检索→发现→修正假设→再检索 |
| 思维链 | 5 步直链 | 可加入分支推理（if-then 路径） |
| 问题分解 | 线性分解 | 可加入子问题间的依赖图 |
| 多跳推理 | 不支持 | 可实现 A→B→C 链式检索 |
| 知识图谱 | 无 | 可构建实体关系图谱辅助推理 |

## 6. 关键文件

| 文件 | 职责 |
|------|------|
| `rag/kb.py` | `_HYDE_PROMPT` / `_hyde_expand()` — 假设文档嵌入 |
| 同上 | `_ANSWER_COT_PROMPT` / `answer_with_reasoning()` — 思维链回答 |
| 同上 | `_DECOMPOSE_PROMPT` / `_decompose_question()` — 问题分解 |
| 同上 | `decompose_and_answer()` — 分解 + 检索 + 综合 |
| `chat/handler.py` | `ChatHandler._is_complex_question()` — 路由判断 |
| 同上 | `_build_context()` — HyDE 改写 + 检索集成 |
