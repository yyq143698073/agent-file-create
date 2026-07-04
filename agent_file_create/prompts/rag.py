"""Prompt templates and public data types extracted from kb.py.

These are stateless module-level constants that can be imported freely
without creating a KnowledgeBase instance.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate


# ── Public data types ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Citation:
    doc_id: str
    chunk_id: str
    section_path: str
    score: float
    snippet: str
    doc_name: str = ""


@dataclass(frozen=True)
class Answer:
    kb: str
    question: str
    answer: str
    citations: list[Citation]


# ── Answer generation prompts ─────────────────────────────────────────────────

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个中文知识库问答助手。"),
        (
            "human",
            """\
你是企业知识库问答助手。请基于检索片段回答用户问题。

回答策略（按检索质量分层）：
- 片段与问题高度相关、覆盖充分 → 以片段为主要依据，直接回答。
- 片段提供了部分线索但不完整 → 先用自己的知识给出框架性、定义性解答，再引用片段中的具体内容作为例证或补充。
- 片段与问题几乎不相关或分数极低 → 依赖自身知识回答，明确标注"以下回答基于通用知识，知识库中未找到直接相关内容"。
- 绝对不要机械复述片段原文——需要归纳、提炼，给用户一个完整的答案而非碎片信息。

输出要求：
1) 先给出核心结论或定义（1-3 句），再展开要点（3-6 条）。
2) 对于"应该怎样""什么是""如何定义"等框架性问题，优先用通用知识构建回答框架，检索内容作为参考案例。
3) 末尾追加一行：依据：<引用编号或doc_id（最多3条，未命中写"未命中"）>。
4) 如果检索片段来自同一篇文档，不要逐条堆砌——合并相关要点。

知识库：
{kb}

检索片段：
{context}

用户问题：
{question}

回答：""",
        ),
    ]
)


ANSWER_COT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个中文文档处理助手。请展示推理过程。"),
        (
            "human",
            """\
你是企业知识库问答助手。请基于给定的检索片段，通过逐步推理回答问题。

推理步骤：
1) 问题理解：用自己的话复述问题的核心要点和隐含假设。
2) 证据梳理：从检索片段中逐条列出相关证据，标注来源编号（如 [1] [2]）。
3) 推理链条：基于证据进行逐步推理。若多个证据之间存在关联（因果、对比、递进等），请明确说明推理路径。
4) 最终回答：给出简洁的最终答案（3-6 条要点）。
5) 自我检查：逐条核查最终回答中的每个论断——是否有对应的检索片段支撑？无支撑的推断请明确标注为「（推测）」或「（材料未覆盖）」。

规则：
- 只根据检索片段回答；不要编造。
- 如果片段不足以回答，在步骤 4 明确说不确定，并在步骤 5 建议补充哪些文档。
- 末尾追加一行：依据：<引用编号（最多3条）>；若无法定位写"依据：未命中"。

知识库：
{kb}

检索片段：
{context}

用户问题：
{question}

推理过程：""",
        ),
    ]
)


# ── HyDE (Hypothetical Document Embeddings) prompt ───────────────────────────

HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个中文文档处理助手。"),
        (
            "human",
            """\
请用3-5句话编写一段可能回答以下问题的文本段落。要求：
- 使用专业、正式的语气
- 包含可能的关键术语和概念
- 模拟知识库文档的风格
- 只输出文本段落，不要解释或标注

问题：{question}

假设回答：""",
        ),
    ]
)


# ── Question decomposition prompt ────────────────────────────────────────────

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个中文文档处理助手。"),
        (
            "human",
            """\
判断以下问题是否需要分解为子问题来回答。如果需要，请分解为2-4个子问题，每个子问题一行。
如果问题本身很简单、不需要分解，只回复：SIMPLE

问题：{question}

分解结果：""",
        ),
    ]
)


# ── Query Rewriting prompt ────────────────────────────────────────────────────

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个中文文档处理助手。"),
        (
            "human",
            """\
将以下用户口语化问题改写为一个更精确、更适合知识库检索的查询。

规则：
- 补全代词和省略的主语，例如"那个政策"→指明具体政策名
- 将口语化表达转为书面语，例如"咋报销"→"费用报销流程"
- 保留所有关键信息，不添加用户未提及的内容
- 只输出改写后的查询，不要任何解释

原始问题：{question}

改写查询：""",
        ),
    ]
)


# ── Multi-Query expansion prompt ──────────────────────────────────────────────

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个中文文档处理助手。"),
        (
            "human",
            """\
为以下问题生成 {n} 个不同角度的检索查询，提高从知识库中找到相关文档的概率。

规则：
- 从不同表述方式、不同关键词组合、不同粒度（宏观/微观）生成变体
- 包含同义词替换，例如"预算"可替换为"资金分配""财务计划"
- 每个查询一行，不要编号，不要解释

原始问题：{question}

{n}个查询变体：""",
        ),
    ]
)


# ── Step-Back prompting ───────────────────────────────────────────────────────

STEPBACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个中文文档处理助手。"),
        (
            "human",
            """\
为以下具体问题生成一个更高层次、更宽泛的背景问题，用于检索广泛的背景知识。

规则：
- 从具体细节中抽象出更高层的概念或原则
- 背景问题应帮助理解原始问题所处的上下文
- 只输出一个背景问题，不要解释

具体问题：{question}

背景问题：""",
        ),
    ]
)


# ── Query Routing / Classification prompt ─────────────────────────────────────

QUERY_ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个中文文档处理助手。只输出指定的标签。"),
        (
            "human",
            """\
将以下问题分类为以下之一：
- fact_lookup：查找单个事实、数字、定义
- comparison：比较两个或多个事物的异同
- summary：要求总结某个主题或文档的要点
- multi_document：需要综合多份文档的信息才能回答
- how_to：询问操作步骤或方法

只输出标签名称，不要解释。

问题：{question}

分类：""",
        ),
    ]
)


# ── Metadata filter extraction prompt ─────────────────────────────────────────

METADATA_FILTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个中文文档处理助手。只输出JSON。"),
        (
            "human",
            """\
从用户问题中提取隐含的过滤条件，用于缩小知识库检索范围。

可提取的字段：
- doc_type：文档类型，如"制度""规范""报告""FAQ""合同"
- source：文档来源关键词，如文件名或部门名
- time_range：时间范围，如"2024""2023-2024""近三年"

只输出JSON，如果没有可提取的条件输出空对象{{}}。

问题：{question}

JSON过滤条件：""",
        ),
    ]
)
