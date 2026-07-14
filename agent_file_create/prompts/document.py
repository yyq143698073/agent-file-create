"""Extraction prompt builders — ChatPromptTemplate style.

Replaces legacy string‑concatenation with proper LangChain templates.
"""

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate


def _fuse_ocr_text(ocr_text: str, max_lines: int = 25) -> str:
    """Trim and deduplicate OCR text for injection into the prompt.

    Keep both the beginning and the end of long OCR outputs, because
    page-bottom answers such as addresses / signatures / contact blocks
    are often critical for DocVQA-style tasks.
    """
    lines = [l.strip() for l in (ocr_text or "").splitlines() if l.strip()]
    seen: set[str] = set()
    uniq_all: list[str] = []
    for line in lines:
        if len(line) < 3:
            continue
        key = line[:60]
        if key in seen:
            continue
        seen.add(key)
        uniq_all.append(line)
    if len(uniq_all) <= max_lines:
        return "\n".join(uniq_all)

    head_n = max_lines // 2
    tail_n = max_lines - head_n
    uniq = uniq_all[:head_n] + uniq_all[-tail_n:]
    return "\n".join(uniq)


def _is_qwen_vl(model_name: Optional[str]) -> bool:
    name = str(model_name or "").strip().lower()
    return "qwen2.5vl" in name or "qwen-vl" in name


# ── Few‑shot example (shared by all content types) ───────────────────────────

_EXAMPLE = (
    '正确输出示例（仅供参考格式，内容需根据实际材料填写）：\n'
    '{"title":"2024年Q3销售数据报告","keywords":["销售数据","Q3","同比增长"],'
    '"summary":"本报告汇总了2024年第三季度各区域销售情况，整体同比增长12.3%，'
    '其中华东区增速最快达18.7%。","key_points":["整体销售额同比增长12.3%",'
    '"华东区增速领先达18.7%","西南区持平略降0.8%","线上渠道占比提升至41%"],'
    '"data":[{"区域":"华东","增长率":"18.7%"},{"区域":"华南","增长率":"9.2%"},'
    '{"区域":"西南","增长率":"-0.8%"}],"conclusion":"Q3整体表现超出预期，'
    '华东区线上渠道贡献显著，建议Q4加大华东线上投放。","prediction":"预计Q4增速可维持10%以上"}'
)


def _make_extract_template(system_parts: list[str]) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate from shared base + type‑specific instructions."""
    base = [
        "你是一个多模态信息抽取助手。",
        "请严格按以下 JSON 结构输出，不要输出任何额外文本，不要用 markdown 代码块。",
        'JSON Schema: {"title":str,"keywords":[str],"summary":str,"key_points":[str],"data":(list|dict|[str]),"conclusion":str,"prediction":str}',
        "要求：",
        "1) 只输出合法 JSON；字段必须齐全（prediction 若无依据可为空字符串）。",
        "2) 中文输出。",
        "3) 不要编造具体数值；材料中明确出现的数字可以引用，看不清就写不确定。",
        "4) title 不超过 30 字；summary 150-300 字；key_points 3-7 条，每条不超过 40 字。",
    ]
    system = "\n".join(base + [_EXAMPLE] + system_parts).strip()
    system = system.replace("{", "{{").replace("}", "}}")
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{content}"),
    ])


def build_extract_prompt(
    content_type: str,
    *,
    ocr_text: Optional[str] = None,
    model_name: Optional[str] = None,
) -> ChatPromptTemplate:
    """Return a ChatPromptTemplate for the given content type.

    The template has one variable: ``{content}`` — inject the material
    text via ``template.invoke({"content": text}).to_string()``.
    When *ocr_text* is provided it is baked into the system prompt.
    """
    ct = (content_type or "text").strip().lower()
    qwen_vl_mode = _is_qwen_vl(model_name)

    # ── Chart / data‑visualisation image ─────────────────────────────
    if ct in {"chart", "image_chart"}:
        parts = [
            "这是一张图表/数据可视化图片。",
            "重点：识别图表类型（柱状图/折线图/饼图等）、标题、坐标轴含义、数据系列、趋势与对比关系。",
            "data 字段：输出表格化结构（list[dict]），每行一条数据记录，包含类别名和数值。",
        ]
        if ocr_text:
            clean = _fuse_ocr_text(ocr_text)
            parts += [
                "【OCR 预识别文字（用于获取精确数值与标签，优先于视觉估算）】",
                clean,
                "请结合图片视觉信息对 OCR 文字进行纠错和结构化，用 OCR 中的精确数值替代视觉估算。",
            ]
        else:
            parts += ["请基于图片视觉信息提取所有可见标签和数值。"]
        return _make_extract_template(parts)

    # ── Document / screenshot image ──────────────────────────────────
    if ct in {"image", "screenshot", "handwriting", "ppt_image"}:
        parts = [
            "这是一张文档/截图/手写图片。",
            "重点：识别页面结构、标题层级、段落要点、列表项。不要遗漏边栏或页脚的关键信息。",
            "data 字段：若出现流程/表格/层级结构，尽量 struct 化输出（list[dict]）。",
            "若图片是表单或票据，请优先抽取字段名-字段值配对；若图片是表格，请保留表头、行列关系和关键单元格内容。",
            "请尽量按自然阅读顺序组织结果，避免把相邻栏位或上下两行错误拼接。",
        ]
        if qwen_vl_mode:
            parts += [
                "这是精确抽取任务，不是泛化总结任务。先抽取答案、地址、机构名、人名、日期、号码、字段值，再写摘要。",
                "若页面包含地址、电话、传真、邮箱、网址、组织名称、收发件人、日期、页码、表单字段，请原样保留，不要改写成宽泛概述。",
                "summary 必须是事实性摘要，优先覆盖可核对的实体和字段，不要先写广告主题、宣传口号或空泛主旨。",
                "data 字段优先输出精确片段，推荐使用 list[dict]，如 [{\"field\":\"FAX NUMBER\",\"value\":\"(336)335-7392\"}] 或 [{\"answer\":\"1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036\"}]。",
                "如果 OCR 中已经有疑似答案或字段值，优先复用原文，不要翻译、扩写或润色。",
            ]
        if ocr_text:
            clean = _fuse_ocr_text(ocr_text)
            parts += [
                "【OCR 预识别文字（文字内容以此为准，精确度高于视觉识别）】",
                clean,
                "请基于 OCR 文字进行结构化提取，结合图片视觉信息判断段落层级和阅读顺序。",
                "OCR 中的文字是准确的，不要随意改写或省略其中的人名、数字、术语。",
            ]
            if qwen_vl_mode:
                parts += [
                    "先在 OCR 中定位疑似答案和字段，再结合版面确认；对于地址、编号、电话号码、日期，尽量逐字保留。",
                    "若材料像问答文档或表单封面，请把最关键的可核对答案放进 data 的前几项。",
                ]
        else:
            parts += ["请基于图片视觉信息提取所有可见文字和结构化内容。"]
        return _make_extract_template(parts)

    # ── PDF ──────────────────────────────────────────────────────────
    if ct in {"pdf"}:
        return _make_extract_template([
            "重点：识别文档标题、章节层级、核心论点与结论。",
            "data 字段：用于提取重要清单/表格/公式/关键定义。",
            "注意：材料可能包含正文和嵌入图片的 OCR 结果，请整合两者信息。",
            "若材料中包含表格，请优先保留表头、行列关系、计量单位和关键数值，不要把整表压成纯段落描述。",
            "若材料中包含表单，请把字段名和值以键值对形式保留在 data 中。",
        ])

    # ── DOCX / PPTX ──────────────────────────────────────────────────
    if ct in {"docx", "pptx", "doc", "ppt"}:
        return _make_extract_template([
            "重点：识别文档标题、章节层级、核心论点与结论。",
            "data 字段：用于提取重要清单/表格/关键定义。",
            "注意：材料是从文档结构化提取的内容（包含段落、表格和图片 OCR 结果），请基于此进行提取。",
            "优先保留标题层级、表格表头、以及字段名-字段值关系，不要将结构化信息全部压缩为摘要。",
        ])

    # ── Excel ────────────────────────────────────────────────────────
    if ct in {"excel", "xlsx"}:
        return _make_extract_template([
            "重点：从表格中提取关键指标、统计摘要（均值/最大最小/分布）、异常点。",
            "data 字段：输出前 3-10 行的结构化预览或关键指标表。",
            "请保留 sheet 之间的重要差异，优先复用原始列名；不要随意改写列标题。",
        ])

    # ── Plain text / unknown ─────────────────────────────────────────
    return _make_extract_template([
        "重点：提取标题、摘要、要点、结论与趋势预测。",
        "如果内容中含 Markdown/CSV/表格文本，请保留原始表头和关键行列关系；如果含键值对，请保留字段和值。",
    ])
