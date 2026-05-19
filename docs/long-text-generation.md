# 长文生成：幻觉抑制与语义逻辑链保持

生成多章节深度报告（典型 3-10 个 H2 章节，每章数百字）时，面临两个核心挑战：LLM 在没有材料支撑时编造具体数字、机构名、人名、年份（幻觉），以及章节之间缺乏因果/递进关系，读起来像独立短文拼贴而非完整报告（逻辑断裂）。

本项目从 Prompt 约束、结构化处理、终审审查和事实核查四个层次构建了递进式防御体系，并通过 LLM 语义摘要替代传统的硬截断来维护章节间的逻辑衔接。

## 1. 四层递进式幻觉防御

### 1.1 第 1 层：Prompt 内嵌约束

在 `_build_section_prompt()` 中，每条生成指令都内嵌了反幻觉约束：

```python
parts = [
    "核心指令：",
    "1) 拒绝复读：严禁直接复制粘贴参考材料中的长句，用全新语言重构核心观点。",
    "2) 逻辑流与衔接：用因果、递进、转折等连接词建立段落间关系。",
    "3) 场景化扩写：解释数据和结论的业务含义，但场景必须是材料中有线索支撑的。",
    "4) 降低幻觉：不要编造具体的数字、机构名、人名、年份。"
       "材料中明确出现的数值可以引用，但要标注「据材料显示」等限定语。"
       "不确定就说「相关数据暂缺」。",
    "5) 溯源要求：每个关键论断后，用「（据<材料简称>）」标注信息来源。"
       "如多个材料共同支撑，标注「（综合多份材料）」。"
       "无材料支撑的推论，标注「（分析推测）」。",
]
```

> [!important]
> 第五条（溯源标注）是关键约束——它强制 LLM 在输出时自我区分"事实"和"推测"，相当于内置了一层轻量的事实核查。LLM 每次做出论断时必须判断它是否有材料依据，并在文本中显式标注。

### 1.2 第 2 层：结构化处理指令

```python
"结构化内容处理：",
"- 如果材料中包含表格数据，用小段落描述关键趋势，不要逐行罗列。",
"- 如果涉及多方案对比，用「相比之下」「与之相反」等短语体现对比关系。",
"- 如果本节的论点需要数据支撑但材料中数据不足，说明「材料中暂缺该维度数据」而不是编造。",
```

### 1.3 第 3 层：终审一致性检查

生成全篇后，运行一次独立的 LLM 审查，检查四个维度的质量问题：

1. 相邻章节之间是否存在逻辑断裂或跳跃？
2. 不同章节是否存在相互矛盾的陈述？
3. 是否存在材料中无依据的具体数字、人名、机构名、年份？
4. 章节之间的术语使用是否一致？

审查通过则直接返回原文；若发现问题，在对应章节后插入 HTML 注释形式的核查标记（如 `<!-- 核查提示：第 5 章中"35%增长率"在材料中未找到出处 -->`），不破坏 Markdown 渲染但可在原文中搜索定位。

### 1.4 第 4 层：正则结构化事实交叉比对

除了 LLM 语义审查，还加入了基于正则表达式的确定性事实提取，在不增加外部 NER 依赖的情况下实现轻量事实核查：

```python
def _extract_facts_from_materials(multimodal_digest: str) -> dict:
    """返回三类结构化数据：numbers、entities、years"""
```

生成全篇后，`_cross_check_facts()` 将报告中的数字、机构名、年份与材料中的事实集合逐项交叉比对，标记不在材料中的条目。与第 3 层的 LLM 审查结果合并后以 HTML 注释形式插入原文。

### 防御层次总览

```
用户材料 ───────────────────────────────────► 生成报告
   │                                               │
   ├─ 第1层（Prompt 内嵌，生成时）                     │
   ├─ 第2层（结构化处理，推导时）                       │
   ├─ 第3层（LLM 全篇终审，完成后）◄───────────────────┘
   └─ 第4层（正则事实核查，核查时）
```

## 2. 语义逻辑链保持

### 2.1 LLM 摘要替代硬截断

迁移前采用固定字数硬截断（180 字符），可能卡在句子中间丢失语义。迁移后改用 LLM 摘要：

```python
_SECTION_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """\
将以下报告章节内容压缩为一段不超过200字的摘要，重点提取：
1) 核心论点与结论
2) 涉及的关键实体、数据、概念
3) 本章节在报告逻辑链中的角色（是铺垫、论证、对比、还是总结）

章节标题：{title}
章节内容：{content}

摘要："""),
])

def _llm_summarize_for_next(section_title: str, content: str) -> str:
    chain = _SECTION_SUMMARY_PROMPT | llm | StrOutputParser()
    summary = chain.invoke({"title": section_title, "content": content[:2000]})
    return summary[:280]
```

> 提示：第三条（逻辑角色标注）是保持整体逻辑弧线的关键。例如"本章为背景铺垫，介绍了 AI 行业的市场规模"→ 下一章知道应基于此展开分析；"本章通过对比 A/B 方案得出结论"→ 下一章知道对比已完成，应进入建议/展望。

### 2.2 串行 H2 + 并行 H3 生成策略

```python
def generate_full_content_parallel(outline, multimodal_results, user_prompt):
    # Phase 1: H2 章节串行生成（保证主线逻辑不跳跃）
    rolling_summary = ""
    for h2_task in h2_tasks:
        body = generate_section_content(..., rolling_summary, ...)
        rolling_summary = _llm_summarize_for_next(title, body)
        _flush_partial()  # 每个 H2 完成后增量写入 content.md

    # Phase 2: H3 子节并行生成（共享父 H2 摘要，提高速度）
    with ThreadPoolExecutor(max_workers=4) as pool:
        for h3_task in h3_tasks:
            parent_summary = h2_summaries[parent_h2_idx]
            future = pool.submit(generate_section_content, ..., parent_summary, ...)
    _flush_partial()

    # 生成完成后统一审查
    return _final_coherence_review(raw_output, multimodal_digest)
```

> [!Note]
> 串行 H2 保证主线逻辑不跳跃，并行 H3 提高生成速度。两者结合的混合策略在质量和效率之间取得了平衡。

### 2.3 实时增量写入

每个 H2 章节完成后调用 `_flush_partial()`，将已完成章节组装后写入 `result/<task_id>/content.md`。前端每 2 秒轮询检测到 `stage=document` 时拉取该文件，实现生成过程中的实时预览。未完成章节标记为"（生成中…）"，让用户清楚知道哪些部分还在生成。

### 2.4 单章节定向重新生成

支持通过 `/regen <章节标题>` 命令只重新生成特定章节及其子节。流程为：解析 content.md 按 `##` 标题切分为章节块 → 模糊匹配目标章节（精确匹配优先，找不到则按字符交集打分取最佳）→ 确定子节范围 → 构建父章节摘要作为上下文 → 并行重新生成目标章节和子节 → 将新内容替换回 content.md 对应位置。

## 3. 关键文件

| 文件 | 职责 |
|------|------|
| `document/content_generator.py` | `_build_section_prompt()` — Prompt 约束 + 溯源指令 |
| 同上 | `_llm_summarize_for_next()` — LLM 摘要衔接 |
| 同上 | `_final_coherence_review()` — 终审 + 结构化事实核查 |
| 同上 | `_extract_facts_from_materials()` / `_cross_check_facts()` — 正则事实交叉比对 |
| 同上 | `generate_full_content_parallel()` — 串行 H2 + 并行 H3 + 增量写入 |
| 同上 | `regenerate_section()` — 单章节定向重生成 |
| 同上 | `_flush_partial()` — 实时增量写入 content.md |
