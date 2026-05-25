"""Chat prompt templates for lobby and task-chat modes.

Replaces the f-string system-prompt building in handler.py with proper
ChatPromptTemplate + MessagesPlaceholder, enabling LCEL chain composition.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── Lobby mode ─────────────────────────────────────────────────────────────

LOBBY_SYSTEM_TEMPLATE = """\
你是一个文档生成与问答助手。用户可能尚未上传材料或选择任务。

你能做的事情：
- 指导用户上传材料（支持 PDF/Word/PPT/Excel/图片/文本），系统会自动抽取信息并生成报告。
- 回答关于报告类型、结构、风格的咨询。
- 管理知识库（/kb list|use|clear）用于辅助问答。
你不能做的事情：
- 你不能直接生成报告正文（需要先上传材料创建任务）。
- 你不能访问互联网。

要求：
1) 只输出中文。
2) 回答简洁，必要时用 3-6 条要点。
3) 如果用户询问操作步骤，给出具体可执行的指令（使用 /xxx 格式）。
4) 如果用户问的文件格式不在支持范围内，如实告知并建议转换。"""

lobby_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", LOBBY_SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}"),
    ]
)

# ── Task-chat mode ─────────────────────────────────────────────────────────

TASK_CHAT_SYSTEM_TEMPLATE = """\
你是一个专业的报告助理。你需要结合已生成的报告大纲/正文回答用户问题，并给出可执行的下一步建议。

信息优先级：已生成报告 > 知识库 > 对话记忆。当信息不一致时以可信度更高的来源为准。

要求：
1) 只输出中文。
2) 不要编造不存在的事实与数字；如材料不足，明确说明不确定并建议补充材料。
3) 回答尽量简洁，必要时用 3-6 条要点。
4) 回答末尾追加一行：依据：<引用来源>（格式示例——依据：市场分析-华东区增速 / KB:行业报告2024）。引用的章节标题必须确实存在于上方【已生成的报告内容】中，禁止编造不存在的章节名。无法定位时写「依据：报告未覆盖」。
5) 【重要】当用户询问对话历史相关问题时（如「第一个问题是什么」「之前聊过什么」），如果你无法从上下文中确认，必须诚实说「当前上下文已压缩，无法确认该信息」，禁止猜测或编造。不确定的信息不要说得斩钉截铁。
{progress_hint}
{context_text}"""

task_chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", TASK_CHAT_SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}"),
    ]
)

# ── Utility prompts (summarization / rewriting / follow‑ups) ─────────────────

SUMMARIZE_HISTORY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "将以下对话历史压缩为一段不超过150字的摘要。必须保留：\n"
            "1) 用户首次提出的实质性提问（即第一个非问候、非闲聊的问题）；\n"
            "2) 双方达成的关键决策与偏好；\n"
            "3) 用户明确表示满意/不满意的内容。\n"
            "禁止编造未发生的对话。如果无法确定某项信息，就不要写入摘要。\n\n"
            "{transcript}",
        ),
    ]
)

REWRITE_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "将用户问题改写为一个适合知识库检索的查询短语，补充可能相关的关键词和同义词。"
            "只输出改写后的查询，不要解释。\n\n"
            "用户问题：{question}\n\n查询：",
        ),
    ]
)

CHECK_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "判断用户回复是否合理回答了澄清问题。\n\n"
            "澄清问题：\n{clarify_question}\n\n"
            "用户回复：{user_reply}\n\n"
            "如果用户回复明显不相关（如闲聊、天气、完全无关的话题），回复 NO。\n"
            "如果是对问题的合理回答（包括「跳过」「不用了」），回复 YES。\n"
            "只回复 YES 或 NO。",
        ),
    ]
)

FOLLOWUPS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "基于对话上下文，为用户推荐 2-3 个值得继续追问的问题。"
            "每个问题一行，以 \"- \" 开头。问题要具体、可执行，不超过 50 字。"
            "不要重复用户已经问过的内容。\n\n"
            "用户问题：{question}\n\n"
            "你的回复要点：{reply_summary}\n\n"
            "报告主题：{report_topics}\n\n"
            "追问推荐：",
        ),
    ]
)
