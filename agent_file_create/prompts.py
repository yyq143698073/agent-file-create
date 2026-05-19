from typing import Optional


def _fuse_ocr_text(ocr_text: str, max_lines: int = 25) -> str:
    """Trim and deduplicate OCR text for injection into the prompt."""
    lines = [l.strip() for l in (ocr_text or "").splitlines() if l.strip()]
    seen: set[str] = set()
    uniq: list[str] = []
    for line in lines:
        if len(line) < 3:
            continue
        key = line[:60]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(line)
        if len(uniq) >= max_lines:
            break
    return "\n".join(uniq)


def build_extract_prompt(content_type: str, *, ocr_text: Optional[str] = None) -> str:
    """Build extraction prompt for a given content type.

    When *ocr_text* is provided the prompt instructs the model to use it as
    the authoritative text source while the image provides visual context.
    """
    ct = (content_type or "text").strip().lower()

    # ── shared base ──────────────────────────────────────────────
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

    # ── few-shot example ─────────────────────────────────────────
    few_shot = (
        '正确输出示例（仅供参考格式，内容需根据实际材料填写）：\n'
        '{"title":"2024年Q3销售数据报告","keywords":["销售数据","Q3","同比增长"],'
        '"summary":"本报告汇总了2024年第三季度各区域销售情况，整体同比增长12.3%，'
        '其中华东区增速最快达18.7%。","key_points":["整体销售额同比增长12.3%",'
        '"华东区增速领先达18.7%","西南区持平略降0.8%","线上渠道占比提升至41%"],'
        '"data":[{"区域":"华东","增长率":"18.7%"},{"区域":"华南","增长率":"9.2%"},'
        '{"区域":"西南","增长率":"-0.8%"}],"conclusion":"Q3整体表现超出预期，'
        '华东区线上渠道贡献显著，建议Q4加大华东线上投放。","prediction":"预计Q4增速可维持10%以上"}'
    )

    # ── chart / data-visualisation image ─────────────────────────
    if ct in {"chart", "image_chart"}:
        body = [
            "这是一张图表/数据可视化图片。",
            "重点：识别图表类型（柱状图/折线图/饼图等）、标题、坐标轴含义、数据系列、趋势与对比关系。",
            "data 字段：输出表格化结构（list[dict]），每行一条数据记录，包含类别名和数值。",
        ]
        if ocr_text:
            clean = _fuse_ocr_text(ocr_text)
            body += [
                "",
                "【OCR 预识别文字（用于获取精确数值与标签，优先于视觉估算）】",
                clean,
                "",
                "请结合图片视觉信息对 OCR 文字进行纠错和结构化，用 OCR 中的精确数值替代视觉估算。",
            ]
        else:
            body += ["请基于图片视觉信息提取所有可见标签和数值。"]

    # ── document / screenshot image ──────────────────────────────
    elif ct in {"image", "screenshot", "handwriting", "ppt_image"}:
        body = [
            "这是一张文档/截图/手写图片。",
            "重点：识别页面结构、标题层级、段落要点、列表项。不要遗漏边栏或页脚的关键信息。",
            "data 字段：若出现流程/表格/层级结构，尽量 struct 化输出（list[dict]）。",
        ]
        if ocr_text:
            clean = _fuse_ocr_text(ocr_text)
            body += [
                "",
                "【OCR 预识别文字（文字内容以此为准，精确度高于视觉识别）】",
                clean,
                "",
                "请基于 OCR 文字进行结构化提取，结合图片视觉信息判断段落层级和阅读顺序。",
                "OCR 中的文字是准确的，不要随意改写或省略其中的人名、数字、术语。",
            ]
        else:
            body += ["请基于图片视觉信息提取所有可见文字和结构化内容。"]

    # ── PDF ──────────────────────────────────────────────────────
    elif ct in {"pdf"}:
        body = [
            "重点：识别文档标题、章节层级、核心论点与结论。",
            "data 字段：用于提取重要清单/表格/公式/关键定义。",
            "注意：以下文本可能包含正文和嵌入图片的 OCR 结果，请整合两者信息。",
        ]

    # ── DOCX / PPTX ──────────────────────────────────────────────
    elif ct in {"docx", "pptx", "doc", "ppt"}:
        body = [
            "重点：识别文档标题、章节层级、核心论点与结论。",
            "data 字段：用于提取重要清单/表格/关键定义。",
            "注意：以下文本是从文档结构化提取的内容（包含段落、表格和图片 OCR 结果），请基于此进行提取。",
        ]

    # ── Excel ────────────────────────────────────────────────────
    elif ct in {"excel", "xlsx"}:
        body = [
            "重点：从表格中提取关键指标、统计摘要（均值/最大最小/分布）、异常点。",
            "data 字段：输出前 3-10 行的结构化预览或关键指标表。",
        ]

    # ── plain text / unknown ─────────────────────────────────────
    else:
        body = ["重点：提取标题、摘要、要点、结论与趋势预测。"]

    parts = base + [few_shot] + body
    return "\n".join(parts).strip()


def build_planner_prompt(goal: str) -> str:
    return "\n".join(
        [
            "你是一个文档生成智能体的规划器。",
            "你需要根据当前 goal 和 state，选择下一步要调用的工具并给出参数。",
            "只输出 JSON，不要输出多余文本。",
            "可选工具：",
            '1) extract_files {"preprocess": bool}',
            '2) generate_outline {}',
            '3) generate_content {}',
            '4) render_templates {}',
            '5) finish {"result":"ok"}',
            "规则：",
            "1) 必须一步步推进，不能跳过 extract_files。",
            "2) 如果已存在 outline/content，就不要重复生成，除非 state.force_regen 为 true。",
            "3) JSON 格式：{\"tool\":\"...\",\"args\":{...}}",
            "",
            "goal:",
            goal.strip(),
        ]
    ).strip()
