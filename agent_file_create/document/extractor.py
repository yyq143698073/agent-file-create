import json
import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from agent_file_create.config import (
    EXTRACT_API_ENDPOINT,
    EXTRACT_API_KEY,
    EXTRACT_API_STYLE,
    EXTRACT_MODEL_NAME,
    IMAGE_JPEG_QUALITY,
    IMAGE_MAX_LONG_EDGE,
    OCR_AUX_ENGINE,
    OCR_AUX_FORM_MARKERS,
    OCR_AUX_MODE,
    OCR_AUX_SCORE_THRESHOLD,
    OCR_ENABLED,
    PDF_MAX_PAGES_VISION,
    PDF_OCR_THRESHOLD_X,
    PDF_OCR_THRESHOLD_Y,
    VISION_MODEL_NAME,
)
from agent_file_create.llm_client import call_llm
from agent_file_create.preprocessor import (
    choose_better_extraction,
    compute_quality_metrics,
    deduplicate_analysis_results,
    easyocr_image,
    easyocr_image_with_boxes,
    extract_key_value_candidates,
    extract_form_kv_by_layout,
    extract_form_kv_by_reading_order,
    extract_form_kv_by_text_sections,
    extract_docx_structured,
    extract_pdf_embedded_images,
    extract_pdf_tables_detailed,
    extract_pdf_tables_structured,
    extract_table_structure_from_image,
    extract_pdf_text_fast,
    extract_pptx_structured,
    merge_ocr_texts,
    ocr_image,
    ocr_image_with_boxes,
    parse_text_table_preview,
    preprocess_image_path,
    preprocess_text,
    read_text_file,
    score_ocr_text_quality,
    render_pdf_pages,
    render_pdf_pages,
)
from agent_file_create.prompts import build_extract_prompt

logger = logging.getLogger(__name__)

# Extraction result cache keyed by (file_hash, preprocess_flag)
_CACHE: dict[str, dict] = {}


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _dedup_ocr_items_by_position(
    items: list[dict],
    dist_threshold: float = 8.0,
) -> list[dict]:
    """Deduplicate OCR items that share similar text and centre positions.

    When multiple OCR variants detect the same text block the clustering
    step can be skewed — keep only the highest-confidence instance per cluster.
    """
    if not items:
        return []

    usable = [
        it
        for it in items
        if str(it.get("text") or "").strip()
        and it.get("cx") is not None
        and it.get("cy") is not None
    ]
    if len(usable) <= 1:
        return usable

    # Sort by confidence score descending (higher score = keep)
    sorted_items = sorted(usable, key=lambda x: float(x.get("score") or 0.0), reverse=True)
    kept: list[dict] = []

    for item in sorted_items:
        cx = float(item["cx"])
        cy = float(item["cy"])
        text_lower = str(item.get("text") or "").strip().lower()

        # Check if this item is a near-duplicate of any already-kept item
        duplicate = False
        for k in kept:
            kx = float(k["cx"])
            ky = float(k["cy"])
            ktext = str(k.get("text") or "").strip().lower()

            if abs(cx - kx) <= dist_threshold and abs(cy - ky) <= dist_threshold:
                # Same position — if text is similar, it's a duplicate
                if text_lower == ktext or (
                    len(text_lower) >= 4
                    and len(ktext) >= 4
                    and (
                        text_lower in ktext
                        or ktext in text_lower
                    )
                ):
                    duplicate = True
                    break
        if not duplicate:
            kept.append(item)

    return kept


def _extract_table_via_vision(
    image_b64: str,
    ocr_text: str,
    ocr_tables: list[dict],
) -> dict | None:
    """Ask the vision LLM to extract a table as markdown from the image.

    Returns a minimal extraction result dict with ``_structured_tables``
    populated from the vision model's output, or ``None`` on failure.
    """
    # Build a hint from OCR-detected table dimensions
    hint = ""
    if ocr_tables:
        t0 = ocr_tables[0]
        dims = t0.get("dimensions", {})
        hint = (
            f"\n(OCR 预检测到约 {dims.get('rows', '?')} 行 × {dims.get('cols', '?')} 列的表格，"
            "请以视觉为准进行修正。)"
        )

    ocr_hint = ""
    if ocr_text:
        ocr_hint = f"\n参考 OCR 文本（可能含错）：\n{ocr_text[:3000]}"

    prompt = (
        "将图片中的表格提取为 markdown 格式。只输出 markdown 表格，不要任何解释、JSON 或其他文字。\n"
        "要求：\n"
        "1) 保留所有单元格原文，不要总结或改写。\n"
        "2) 正确识别表头行和数据行。\n"
        "3) 空单元格保留为空（| |）。\n"
        "4) 如果图片中没有表格，输出空字符串。"
        + hint + ocr_hint
    )

    try:
        raw = call_llm(
            prompt,
            images_base64=[image_b64],
            timeout_s=90,
            api_style=EXTRACT_API_STYLE,
            model_name=VISION_MODEL_NAME,
        )
        table_md = str(raw or "").strip()
        # Strip code fences
        table_md = re.sub(r"^```(?:markdown|md|html)?\s*", "", table_md, flags=re.I).strip()
        table_md = re.sub(r"\s*```$", "", table_md).strip()

        if not table_md or "|" not in table_md:
            return None

        return {
            "title": "表格数据",
            "summary": "表格结构化提取结果",
            "key_points": [],
            "content_type": "image",
            "_structured_tables": [
                {
                    "page": 0,
                    "table_index": 1,
                    "caption": "",
                    "headers": [],
                    "rows": [],
                    "dimensions": {"rows": 0, "cols": 0},
                    "markdown": table_md,
                    "validation": {"method": "vision_llm"},
                }
            ],
            "_has_tables": True,
            "_ocr_text": _head_tail_text(ocr_text, max_chars=3200) if ocr_text else "",
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def clean_json_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```json\s*", "", t, flags=re.I).strip()
    t = re.sub(r"^```\s*", "", t).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    s = t.find("{")
    e = t.rfind("}")
    if s != -1 and e != -1 and e > s:
        return t[s : e + 1].strip()
    return t


def parse_model_json(text: str) -> dict:
    cleaned = clean_json_text(text)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
        return {"error": "not_dict", "raw_output": cleaned[:2000]}
    except Exception as e:
        return {"error": f"json_parse_failed:{str(e)[:120]}", "raw_output": cleaned[:2000]}


def _retry_json_parse(raw: str, prompt: str, content_type: str, timeout_s: int, model_name: str, api_endpoint: str, api_key: str, api_style: str) -> dict:
    """Retry JSON parsing by asking the LLM to fix its output."""
    obj = parse_model_json(raw)
    if "error" not in obj:
        return obj

    for _ in range(2):
        fix_prompt = (
            "你之前的输出无法解析为合法 JSON。请严格只输出 JSON，不要任何其他文字。\n\n"
            f"原始输出：\n{raw[:3000]}\n\n"
            f"解析错误：{obj.get('error', 'unknown')}"
        )
        try:
            fixed_raw = call_llm(fix_prompt, timeout_s=timeout_s, api_style=api_style, api_endpoint=api_endpoint, api_key=api_key, model_name=model_name)
            obj = parse_model_json(str(fixed_raw))
            if "error" not in obj:
                obj["_json_retry"] = True
                return obj
        except Exception:
            break

    return obj


def _head_tail_text(text: str, max_chars: int = 2400) -> str:
    s = str(text or "").strip()
    if len(s) <= max_chars:
        return s
    head_n = max_chars // 2
    tail_n = max_chars - head_n
    return s[:head_n].rstrip() + "\n...\n" + s[-tail_n:].lstrip()


def _extract_address_candidates(text: str, max_items: int = 3) -> list[str]:
    candidates: list[str] = []
    lines = [re.sub(r"\s+", " ", str(line or "")).strip() for line in str(text or "").splitlines()]
    for line in lines:
        if len(line) < 12:
            continue
        lower = line.lower()
        if not re.search(r"\d{3,}", line):
            continue
        if not any(token in lower for token in ("st", "street", "ave", "avenue", "road", "rd", "blvd", "drive", "lane", "washington", "suite")):
            continue
        fixed = re.sub(r"([,.;:])(?=\S)", r"\1 ", line)
        fixed = re.sub(r"\b([A-Z])\.\s*([A-Z])\.", r"\1. \2.", fixed)
        fixed = re.sub(r"\b([NSEW])\.\s*([NSEW])\.\s+([A-Z])", r"\1. \2., \3", fixed)
        fixed = re.sub(r"\s+", " ", fixed).strip(" ,")
        if fixed not in candidates:
            candidates.append(fixed)
        if len(candidates) >= max_items:
            break
    return candidates


def _normalize_form_field_text(text: str, *, is_field: bool) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip(" -:;,")
    if not s:
        return ""
    s = re.sub(r"[_/\\|-]+", " ", s)
    s = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", s)
    s = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", s)
    upper = re.sub(r"\s+", " ", s.upper()).strip()
    if is_field:
        return upper.strip(" -:;,")
    normalized = re.sub(r"\s+", " ", s).strip()
    return normalized


def _normalize_form_kv_pairs(pairs: list[dict], max_items: int = 16) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in pairs or []:
        field = _normalize_form_field_text(item.get("field", ""), is_field=True)
        value = _normalize_form_field_text(item.get("value", ""), is_field=False)
        if not field or not value:
            continue
        token = (field.lower(), value.lower())
        if token in seen:
            continue
        seen.add(token)
        merged.append({"field": field, "value": value})
        if len(merged) >= max_items:
            break
    return merged


def _extract_form_slots_from_ocr(ocr_text: str) -> list[dict[str, str]]:
    source = _head_tail_text(ocr_text, max_chars=6000)
    if not source:
        return []
    prompt = (
        "你是表单字段抽取助手。请从下面 OCR 文本中抽取英文原字段名和原字段值。"
        "只输出合法 JSON，不要额外文字。"
        'JSON Schema: {"slots":[{"field":str,"value":str}]}\n'
        "要求：\n"
        "1) field 必须尽量保留英文原标签，不翻译。\n"
        "2) value 必须尽量保留原文，不润色。\n"
        "3) 优先抽取 TO, FROM, DATE, FAX NUMBER, PHONE NUMBER, NUMBER OF PAGES INCLUDING COVER SHEET, "
        "SENDER PHONE NUMBER, SPECIAL INSTRUCTIONS, NOTE，以及带长标题的业务段落字段。\n"
        "4) 不要编造，不存在就不要输出。\n\n"
        f"OCR 文本：\n{source}"
    )
    try:
        raw = call_llm(
            prompt,
            timeout_s=60,
            api_style=EXTRACT_API_STYLE,
            api_endpoint=EXTRACT_API_ENDPOINT,
            api_key=EXTRACT_API_KEY,
            model_name=EXTRACT_MODEL_NAME,
        )
        obj = parse_model_json(str(raw))
        slots = obj.get("slots", []) if isinstance(obj, dict) else []
        if not isinstance(slots, list):
            return []
        return _normalize_form_kv_pairs(slots, max_items=16)
    except Exception:
        return []


def _collect_low_quality_ocr_fragments(ocr_text: str, ocr_items: list[dict[str, Any]], max_items: int = 10) -> list[str]:
    fragments: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        s = re.sub(r"\s+", " ", str(text or "")).strip()
        if not s:
            return
        key = s.lower()
        if key in seen:
            return
        seen.add(key)
        fragments.append(s)

    for line in str(ocr_text or "").splitlines():
        s = re.sub(r"\s+", " ", str(line or "")).strip()
        if len(s) < 4:
            continue
        compact = re.sub(r"\s+", "", s)
        special_ratio = len(re.findall(r"[^A-Za-z0-9\s:：,./()\-&$%#@]", s)) / float(len(s) or 1)
        weird_case = bool(re.search(r"[A-Z][a-z]+[A-Z]|[a-z]+[A-Z]{2,}", s))
        if (len(compact) >= 10 and " " not in s and ":" not in s and "：" not in s) or special_ratio > 0.15 or weird_case:
            _add(s)
        if len(fragments) >= max_items:
            return fragments[:max_items]

    for item in ocr_items or []:
        try:
            score = float(item.get("score")) if item.get("score") is not None else None
        except Exception:
            score = None
        text = str(item.get("text") or "").strip()
        if text and score is not None and score < 0.70:
            _add(text)
        if len(fragments) >= max_items:
            break
    return fragments[:max_items]


def _repair_ocr_text_with_vision(ocr_text: str, suspicious_lines: list[str], image_b64: str) -> str:
    source = _head_tail_text(ocr_text, max_chars=5000)
    suspects = "\n".join(f"- {line}" for line in suspicious_lines[:10])
    prompt = (
        "你是 OCR 纠错助手。下面是从文档图片中识别出来的 OCR 文本，其中部分字段名、编号、日期、金额、地址可能有误。"
        "请结合图片视觉内容和上下文，对明显错误的 OCR 行进行纠正。\n"
        "要求：\n"
        "1) 只输出纠正后的纯文本，不要解释，不要输出 JSON。\n"
        "2) 不要凭空补内容；无法确认时保留原文。\n"
        "3) 尽量保留原始换行结构。\n\n"
        f"疑似低质量片段：\n{suspects}\n\n"
        f"OCR 原文：\n{source}"
    )
    try:
        repaired = call_llm(
            prompt,
            images_base64=[image_b64],
            timeout_s=75,
            api_style=EXTRACT_API_STYLE,
            model_name=VISION_MODEL_NAME,
        )
        text = str(repaired or "").strip()
        text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.I).strip()
        text = re.sub(r"\s*```$", "", text).strip()
        return text
    except Exception:
        return ""


def _detect_field_type(field: str, value: str) -> str:
    f = str(field or "").strip().upper()
    v = str(value or "").strip()
    lower = v.lower()
    digits = re.sub(r"\D", "", v)
    if "@" in v and re.search(r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", v):
        return "EMAIL"
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", lower) or re.search(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}", v):
        return "DATE"
    if any(token in f for token in ("PHONE", "FAX", "TEL")) or len(digits) >= 7:
        return "PHONE"
    if re.search(r"[$€¥£]|(?:^|\s)\d[\d,]*(?:\.\d+)?(?:\s|$)", v) and any(token in f for token in ("PRICE", "AMOUNT", "TOTAL", "COST")):
        return "AMOUNT"
    if re.search(r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", v):
        return "EMAIL"
    if re.search(r"\d{2,}.*\b(?:street|st|road|rd|ave|avenue|blvd|drive|lane|suite|boulevard|路|街)\b", lower):
        return "ADDRESS"
    if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}", v) and not re.search(r"\d", v):
        return "NAME"
    return "TEXT"


def _annotate_form_fields(pairs: list[dict[str, str]]) -> list[dict[str, str]]:
    annotated: list[dict[str, str]] = []
    for item in pairs or []:
        field = str(item.get("field") or "").strip()
        value = str(item.get("value") or "").strip()
        if not field or not value:
            continue
        annotated.append({"field": field, "value": value, "type": _detect_field_type(field, value)})
    return annotated


def _group_form_fields_by_region(ocr_items: list[dict[str, Any]], pairs: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    usable = [item for item in (ocr_items or []) if str(item.get("text") or "").strip()]
    if not usable or not pairs:
        return {}

    doc_top = min(float(item.get("top") or 0.0) for item in usable)
    doc_bottom = max(float(item.get("bottom") or 0.0) for item in usable)
    total_height = max(1.0, doc_bottom - doc_top)
    groups = {"header": [], "body": [], "footer": []}

    for pair in pairs:
        field = str(pair.get("field") or "").strip()
        field_norm = re.sub(r"[^a-z0-9]+", "", field.lower())
        matched = None
        for item in usable:
            text = str(item.get("text") or "").strip().lower()
            text_norm = re.sub(r"[^a-z0-9]+", "", text)
            if field_norm and (field_norm in text_norm or text_norm in field_norm):
                matched = item
                break
        cy = float(matched.get("cy") or doc_top) if matched else doc_top
        ratio = (cy - doc_top) / total_height
        if ratio <= 0.30:
            groups["header"].append(pair)
        elif ratio >= 0.72:
            groups["footer"].append(pair)
        else:
            groups["body"].append(pair)
    return {key: value for key, value in groups.items() if value}


def _repair_required_fields(
    obj: dict,
    *,
    source_text: str,
    content_type: str,
    timeout_s: int,
    model_name: str,
    api_endpoint: str,
    api_key: str,
    api_style: str,
) -> dict:
    """Ask the LLM for a compact补抽 only when required fields are missing."""
    metrics = compute_quality_metrics(obj)
    missing = metrics.get("missing_required", [])
    if not missing:
        return obj

    follow_prompt = (
        "你之前返回的抽取结果缺少必填字段。"
        "请严格只输出合法 JSON，并补全以下字段："
        f"{', '.join(missing)}。\n\n"
        f"内容类型：{content_type}\n"
        "必须保持已有字段语义一致，无法确定时保守概括，不要编造。\n\n"
        f"已有 JSON：\n{json.dumps(obj, ensure_ascii=False)[:3000]}\n\n"
        f"原始材料：\n{source_text[:5000]}"
    )
    try:
        repaired_raw = call_llm(
            follow_prompt,
            timeout_s=timeout_s,
            api_style=api_style,
            api_endpoint=api_endpoint,
            api_key=api_key,
            model_name=model_name,
        )
        repaired = parse_model_json(str(repaired_raw))
        if "error" in repaired:
            return obj
        for key, value in obj.items():
            repaired.setdefault(key, value)
        repaired["_schema_repaired"] = True
        return repaired
    except Exception:
        return obj


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------


def _detect_type(file_path: str) -> str:
    suf = Path(file_path).suffix.lower().lstrip(".")
    if suf in {"png", "jpg", "jpeg", "webp", "gif", "bmp"}:
        return "image"
    if suf in {"pdf"}:
        return "pdf"
    if suf in {"txt", "md"}:
        return "text"
    if suf in {"xlsx", "xls"}:
        return "excel"
    if suf in {"docx"}:
        return "docx"
    if suf in {"pptx", "ppt"}:
        return "pptx"
    return "text"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_from_file(file_path: str, *, preprocess: bool = True) -> dict:
    import hashlib

    p = Path(file_path)

    # Check cache using content hash
    try:
        file_hash = hashlib.md5(p.read_bytes()).hexdigest()
    except Exception:
        file_hash = ""
    if file_hash:
        cache_key = f"{file_hash}:{int(preprocess)}"
        if cache_key in _CACHE:
            cached = dict(_CACHE[cache_key])
            cached["_cached"] = True
            return cached
    else:
        cache_key = ""

    result = _extract_from_file(p, preprocess)
    if cache_key and "error" not in result:
        _CACHE[cache_key] = dict(result)
    return result


def deduplicate_extracted_results(results: list[dict]) -> list[dict]:
    return deduplicate_analysis_results(results)


# ---------------------------------------------------------------------------
# Per-type extraction
# ---------------------------------------------------------------------------


def _extract_from_file(p: Path, preprocess: bool) -> dict:
    ct = _detect_type(str(p))
    name = p.name

    try:
        if ct == "image":
            return _extract_image(p, preprocess)

        if ct == "pdf":
            return _extract_pdf(p, preprocess)

        if ct == "excel":
            return _extract_excel(p)

        if ct == "docx":
            return _extract_docx_pptx(p, "docx")

        if ct == "pptx":
            return _extract_docx_pptx(p, "pptx")

        # Plain text / unknown
        text = read_text_file(str(p))
        prompt = build_extract_prompt(ct).invoke({"content": preprocess_text(text)}).to_string()
        raw = call_llm(prompt, timeout_s=120, api_style=EXTRACT_API_STYLE, api_endpoint=EXTRACT_API_ENDPOINT, api_key=EXTRACT_API_KEY, model_name=EXTRACT_MODEL_NAME)
        obj = _retry_json_parse(str(raw), prompt, ct, 120, EXTRACT_MODEL_NAME, EXTRACT_API_ENDPOINT, EXTRACT_API_KEY, EXTRACT_API_STYLE)
        table_preview = parse_text_table_preview(text)
        if table_preview:
            existing_data = obj.get("data")
            if not existing_data:
                obj["data"] = table_preview
            elif isinstance(existing_data, list):
                merged = []
                for item in existing_data + table_preview:
                    if item not in merged:
                        merged.append(item)
                obj["data"] = merged[:12]
            obj["_has_tables"] = True
            obj["_table_preview"] = json.dumps(table_preview[:3], ensure_ascii=False)
        obj["content_type"] = ct
        return obj

    except Exception as e:
        logger.warning(f"extract_failed file={name} err={str(e)[:200]}")
        return {"error": str(e), "content_type": ct}


# ---------------------------------------------------------------------------
# Image extraction (OCR + vision LLM hybrid)
# ---------------------------------------------------------------------------


def _extract_image(p: Path, preprocess: bool) -> dict:
    import base64

    ct = "image"
    raw_bytes = Path(p).read_bytes()
    if preprocess:
        img_bytes = preprocess_image_path(
            str(p),
            max_long_edge=IMAGE_MAX_LONG_EDGE,
            jpeg_quality=IMAGE_JPEG_QUALITY,
            profile="adaptive",
        )
        light_bytes = preprocess_image_path(
            str(p),
            max_long_edge=IMAGE_MAX_LONG_EDGE,
            jpeg_quality=IMAGE_JPEG_QUALITY,
            profile="light",
        )
        strong_bytes = preprocess_image_path(
            str(p),
            max_long_edge=IMAGE_MAX_LONG_EDGE,
            jpeg_quality=IMAGE_JPEG_QUALITY,
            profile="strong",
        )
    else:
        img_bytes = raw_bytes
        light_bytes = raw_bytes
        strong_bytes = raw_bytes

    b64 = base64.b64encode(img_bytes).decode("utf-8")

    # OCR first for accurate text
    ocr_text = ""
    raw_ocr_text = ""
    processed_ocr_text = ""
    strong_ocr_text = ""
    easy_ocr_text = ""
    raw_ocr_items = []
    processed_ocr_items = []
    strong_ocr_items = []
    easy_ocr_items = []
    best_variant = "raw"
    if OCR_ENABLED:
        raw_ocr_text = ocr_image(raw_bytes)
        processed_ocr_text = ocr_image(light_bytes)
        strong_ocr_text = ocr_image(strong_bytes)
        raw_ocr_items = ocr_image_with_boxes(raw_bytes)
        processed_ocr_items = ocr_image_with_boxes(light_bytes)
        strong_ocr_items = ocr_image_with_boxes(strong_bytes)

        ocr_variants = {
            "raw": raw_ocr_text,
            "light": processed_ocr_text,
            "strong": strong_ocr_text,
        }
        best_variant = max(ocr_variants.items(), key=lambda item: score_ocr_text_quality(item[1] or ""))[0]
        sorted_variants = sorted(
            ocr_variants.items(),
            key=lambda item: score_ocr_text_quality(item[1] or ""),
            reverse=True,
        )
        ocr_text = ""
        for _, variant_text in sorted_variants:
            ocr_text = merge_ocr_texts(ocr_text, variant_text, max_lines=80) if ocr_text else str(variant_text or "").strip()

        # EasyOCR is slower on CPU, so only use it as a supplement for form-like or noisy scans.
        rapid_best_score = score_ocr_text_quality(ocr_variants.get(best_variant, "") or "")
        rapid_best_text = str(ocr_variants.get(best_variant, "") or "")
        rapid_form_markers = len(
            re.findall(
                r"(?i)\b(to|from|date|fax|phone|number|address|note|sender|recipient|policy|claim|account|market|brand|manufacturer)\b|[:：]",
                rapid_best_text,
            )
        )
        aux_enabled = OCR_AUX_ENGINE == "easyocr" and OCR_AUX_MODE != "off"
        aux_should_run = OCR_AUX_MODE == "always" or (
            OCR_AUX_MODE == "auto"
            and (rapid_form_markers >= OCR_AUX_FORM_MARKERS or rapid_best_score < OCR_AUX_SCORE_THRESHOLD)
        )
        if aux_enabled and aux_should_run:
            easy_ocr_text = easyocr_image(raw_bytes)
            easy_ocr_items = easyocr_image_with_boxes(raw_bytes)
            if easy_ocr_text:
                ocr_text = merge_ocr_texts(ocr_text, easy_ocr_text, max_lines=100)

    combined_ocr_items = raw_ocr_items + processed_ocr_items + strong_ocr_items + easy_ocr_items
    suspicious_ocr_lines = _collect_low_quality_ocr_fragments(ocr_text, combined_ocr_items)
    corrected_ocr_text = ""
    if ocr_text and suspicious_ocr_lines and _env_enabled("Q1_ENABLE_LLM_OCR_FIX", True):
        corrected_ocr_text = _repair_ocr_text_with_vision(ocr_text, suspicious_ocr_lines, b64)
        if corrected_ocr_text:
            corrected_score = score_ocr_text_quality(corrected_ocr_text)
            original_score = score_ocr_text_quality(ocr_text)
            if corrected_score >= original_score - 2:
                ocr_text = merge_ocr_texts(corrected_ocr_text, ocr_text, max_lines=120)

    # ── Table structure extraction from image (OCR spatial analysis) ────
    structured_tables: list[dict] = []
    if _env_enabled("Q1_ENABLE_TABLE_STRUCT", True) and combined_ocr_items:
        deduped_items = _dedup_ocr_items_by_position(combined_ocr_items)
        all_items = deduped_items  # default — may be augmented below

        # For small images, optionally merge 2× upscaled OCR items
        try:
            from PIL import Image as PILImage
            _tw, _th = PILImage.open(BytesIO(raw_bytes)).size
            if min(_tw, _th) < 300 and _env_enabled("Q1_TABLE_SMALL_UPSCALE", False):
                up_buf = BytesIO()
                PILImage.open(BytesIO(raw_bytes)).resize(
                    (_tw * 2, _th * 2), getattr(PILImage, "LANCZOS", PILImage.BICUBIC)
                ).save(up_buf, format="PNG")
                up_items = ocr_image_with_boxes(up_buf.getvalue())
                for it in up_items:
                    it["cx"] = float(it.get("cx", 0)) / 2.0
                    it["cy"] = float(it.get("cy", 0)) / 2.0
                    for _k in ("left", "right", "top", "bottom"):
                        if it.get(_k) is not None:
                            it[_k] = float(it[_k]) / 2.0
                    it["width"] = float(it.get("width", 0)) / 2.0
                    it["height"] = float(it.get("height", 0)) / 2.0
                all_items = _dedup_ocr_items_by_position(deduped_items + up_items)
        except Exception:
            pass

        structured_tables = extract_table_structure_from_image(all_items, image_bytes=raw_bytes)

    # Fast eval mode: skip vision LLM entirely — for table-only evaluation.
    # Only enable this when evaluating table datasets (PubTabNet), not forms.
    if _env_enabled("Q1_EVAL_FAST_TABLE", False):
        obj = {
            "title": "表格数据" if structured_tables else (ocr_text[:80] if ocr_text else "图片"),
            "summary": "表格结构化提取结果" if structured_tables else (ocr_text[:200] if ocr_text else ""),
            "key_points": [],
            "content_type": "image",
            "_structured_tables": structured_tables,
            "_has_tables": bool(structured_tables),
            "_ocr_text": _head_tail_text(ocr_text, max_chars=3200) if ocr_text else "",
        }
        return obj

    image_task = "请分析上传的图片"
    if suspicious_ocr_lines:
        image_task += "\n以下 OCR 片段可能有识别错误，请结合视觉内容纠正后再抽取：\n- " + "\n- ".join(suspicious_ocr_lines[:8])
    prompt = build_extract_prompt(
        ct,
        ocr_text=ocr_text if ocr_text else None,
        model_name=VISION_MODEL_NAME,
    ).invoke({"content": image_task}).to_string()
    raw = call_llm(prompt, images_base64=[b64], timeout_s=120, api_style=EXTRACT_API_STYLE, model_name=VISION_MODEL_NAME)

    obj = _retry_json_parse(str(raw), prompt, ct, 120, VISION_MODEL_NAME, "", "", EXTRACT_API_STYLE)
    obj = _repair_required_fields(
        obj,
        source_text=ocr_text or "图片视觉内容",
        content_type=ct,
        timeout_s=90,
        model_name=VISION_MODEL_NAME,
        api_endpoint="",
        api_key="",
        api_style=EXTRACT_API_STYLE,
    )

    # Form-like documents respond better to an OCR-first structured extraction pass.
    form_markers = len(
        re.findall(
            r"(?i)\b(to|from|date|fax|phone|number|address|note|sender|recipient|policy|claim|account)\b|[:：]",
            ocr_text or "",
        )
    )
    variant_items = {
        "raw": raw_ocr_items,
        "light": processed_ocr_items,
        "strong": strong_ocr_items,
    }
    selected_ocr_items = variant_items.get(best_variant, [])
    backup_ocr_items: list[dict] = []
    for key, items in variant_items.items():
        if key != best_variant:
            backup_ocr_items.extend(items)
    if easy_ocr_items:
        backup_ocr_items.extend(easy_ocr_items)
    layout_kv_pairs = extract_form_kv_by_layout(selected_ocr_items)
    if len(layout_kv_pairs) < 3 and backup_ocr_items:
        merged_layout = layout_kv_pairs[:]
        for item in extract_form_kv_by_layout(backup_ocr_items):
            if item not in merged_layout:
                merged_layout.append(item)
        layout_kv_pairs = merged_layout[:12]
    section_kv_pairs = extract_form_kv_by_text_sections(ocr_text, max_items=10)
    reading_kv_pairs = extract_form_kv_by_reading_order(ocr_text, max_items=12)
    address_candidates = _extract_address_candidates(ocr_text, max_items=3)
    slot_kv_pairs = _extract_form_slots_from_ocr(ocr_text) if form_markers >= 6 else []
    raw_kv_pairs = []
    for item in slot_kv_pairs + layout_kv_pairs + reading_kv_pairs + section_kv_pairs + extract_key_value_candidates(ocr_text):
        if item not in raw_kv_pairs:
            raw_kv_pairs.append(item)
    kv_pairs = _normalize_form_kv_pairs(raw_kv_pairs, max_items=16)
    typed_kv_pairs = _annotate_form_fields(kv_pairs)
    grouped_fields = _group_form_fields_by_region(selected_ocr_items or backup_ocr_items, typed_kv_pairs)
    if ocr_text and (form_markers >= 6 or len(kv_pairs) >= 3):
        ocr_focus_text = _head_tail_text(ocr_text, max_chars=5000)
        text_prompt = build_extract_prompt(ct, ocr_text=ocr_text).invoke({"content": ocr_focus_text}).to_string()
        text_raw = call_llm(
            text_prompt,
            timeout_s=90,
            api_style=EXTRACT_API_STYLE,
            api_endpoint=EXTRACT_API_ENDPOINT,
            api_key=EXTRACT_API_KEY,
            model_name=EXTRACT_MODEL_NAME,
        )
        text_obj = _retry_json_parse(
            str(text_raw),
            text_prompt,
            ct,
            90,
            EXTRACT_MODEL_NAME,
            EXTRACT_API_ENDPOINT,
            EXTRACT_API_KEY,
            EXTRACT_API_STYLE,
        )
        obj = choose_better_extraction(obj, text_obj)

    obj["content_type"] = "image"
    if structured_tables:
        obj["_structured_tables"] = structured_tables
        obj["_has_tables"] = True
    if ocr_text:
        obj["_ocr_text"] = _head_tail_text(ocr_text, max_chars=3200)
        if raw_ocr_text:
            obj["_raw_ocr_preview"] = raw_ocr_text[:500]
        if processed_ocr_text:
            obj["_processed_ocr_preview"] = processed_ocr_text[:500]
        if strong_ocr_text:
            obj["_strong_ocr_preview"] = strong_ocr_text[:500]
        if easy_ocr_text:
            obj["_easyocr_preview"] = easy_ocr_text[:500]
        if corrected_ocr_text:
            obj["_corrected_ocr_preview"] = corrected_ocr_text[:500]
        if suspicious_ocr_lines:
            obj["_ocr_suspect_preview"] = suspicious_ocr_lines[:8]
        obj["_selected_ocr_variant"] = best_variant if OCR_ENABLED else ""
        if layout_kv_pairs:
            obj["_layout_kv_preview"] = layout_kv_pairs[:5]
        if section_kv_pairs:
            obj["_section_kv_preview"] = section_kv_pairs[:5]
        if reading_kv_pairs:
            obj["_reading_kv_preview"] = reading_kv_pairs[:5]
        if slot_kv_pairs:
            obj["_slot_kv_preview"] = slot_kv_pairs[:5]
        if address_candidates:
            obj["_address_preview"] = address_candidates[:3]
        if kv_pairs:
            existing = obj.get("data")
            if not existing:
                obj["data"] = raw_kv_pairs[:20] or kv_pairs
            elif isinstance(existing, list):
                merged = []
                for item in raw_kv_pairs + kv_pairs + existing:
                    if item not in merged:
                        merged.append(item)
                obj["data"] = merged[:28]
            elif isinstance(existing, dict):
                raw_fields = existing.get("raw_fields")
                if not isinstance(raw_fields, list):
                    existing["raw_fields"] = raw_kv_pairs[:12]
                else:
                    merged_raw = []
                    for item in raw_kv_pairs + raw_fields:
                        if item not in merged_raw:
                            merged_raw.append(item)
                    existing["raw_fields"] = merged_raw[:12]
                existing["normalized_fields"] = kv_pairs[:12]
                if typed_kv_pairs:
                    existing["typed_fields"] = typed_kv_pairs[:12]
                if grouped_fields:
                    existing["field_groups"] = grouped_fields
                existing["reading_order_fields"] = reading_kv_pairs[:12]
            if address_candidates:
                existing_data = obj.get("data")
                addr_items = [{"answer": item} for item in address_candidates]
                if not existing_data:
                    obj["data"] = addr_items
                elif isinstance(existing_data, list):
                    merged = []
                    for item in addr_items + existing_data:
                        if item not in merged:
                            merged.append(item)
                    obj["data"] = merged[:16]
                elif isinstance(existing_data, dict):
                    existing_data["address_candidates"] = address_candidates[:3]
            kv_summary = "；".join(
                f"{item['field']}:{item['value']}"
                for item in kv_pairs[:5]
                if len(str(item.get("field") or "")) <= 40 and len(str(item.get("value") or "")) <= 80
            )
            current_summary = str(obj.get("summary") or "").strip()
            if not current_summary:
                obj["summary"] = kv_summary
            elif kv_summary and kv_summary not in current_summary:
                obj["summary"] = (current_summary + "；关键信息：" + kv_summary)[:400]
    if typed_kv_pairs:
        obj["_typed_fields_preview"] = typed_kv_pairs[:6]
    if grouped_fields:
        obj["_field_groups_preview"] = grouped_fields
    return obj


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


def _extract_pdf(p: Path, preprocess: bool) -> dict:
    import base64

    # Phase 1: text layer via PyMuPDF (faster + more accurate than PyPDF2)
    text_content = extract_pdf_text_fast(str(p))
    table_struct_enabled = _env_enabled("Q1_ENABLE_TABLE_STRUCT", True)
    structured_tables = extract_pdf_tables_detailed(str(p)) if table_struct_enabled else []
    table_content = extract_pdf_tables_structured(str(p)) if table_struct_enabled else ""

    # Phase 2: embedded images OCR
    ocr_content = ""
    if OCR_ENABLED:
        threshold = (PDF_OCR_THRESHOLD_X, PDF_OCR_THRESHOLD_Y)
        ocr_content = extract_pdf_embedded_images(str(p), threshold=threshold)

    if text_content or ocr_content or table_content:
        # Combine text layer and embedded images OCR for the text LLM
        combined = []
        if text_content:
            combined.append(text_content)
        if table_content:
            combined.append("--- 检测到的表格结构 ---\n" + table_content)
        if ocr_content:
            combined.append("--- 嵌入图片 OCR 结果 ---\n" + ocr_content)
        content = "\n\n".join(combined)

        prompt = build_extract_prompt("pdf").invoke({"content": content}).to_string()
        raw = call_llm(prompt, timeout_s=120, api_style=EXTRACT_API_STYLE, api_endpoint=EXTRACT_API_ENDPOINT, api_key=EXTRACT_API_KEY, model_name=EXTRACT_MODEL_NAME)
        obj = _retry_json_parse(str(raw), prompt, "pdf", 120, EXTRACT_MODEL_NAME, EXTRACT_API_ENDPOINT, EXTRACT_API_KEY, EXTRACT_API_STYLE)
        obj = _repair_required_fields(
            obj,
            source_text=content,
            content_type="pdf",
            timeout_s=90,
            model_name=EXTRACT_MODEL_NAME,
            api_endpoint=EXTRACT_API_ENDPOINT,
            api_key=EXTRACT_API_KEY,
            api_style=EXTRACT_API_STYLE,
        )
        obj["content_type"] = "pdf"
        obj["_extraction_method"] = "text_layer"
        if ocr_content:
            obj["_has_ocr"] = True
        if structured_tables:
            obj["_has_tables"] = True
            obj["_table_preview"] = table_content[:800]
            obj["_structured_tables"] = structured_tables[:6]
        return obj

    # Phase 3: no text layer — render pages, OCR each, then vision LLM
    pages = render_pdf_pages(str(p), max_pages=PDF_MAX_PAGES_VISION)
    if not pages:
        return {"error": "pdf_unreadable", "content_type": "pdf"}

    b64s = [base64.b64encode(x).decode("utf-8") for x in pages]

    # OCR each rendered page
    ocr_parts: list[str] = []
    if OCR_ENABLED:
        for page_bytes in pages:
            page_ocr = ocr_image(page_bytes)
            if page_ocr:
                ocr_parts.append(page_ocr)

    ocr_combined = "\n\n".join(ocr_parts) if ocr_parts else ""
    prompt = build_extract_prompt(
        "image", ocr_text=ocr_combined if ocr_combined else None
    ).invoke({"content": "（PDF 无法抽取文本层，请基于页面截图和 OCR 预识别结果进行理解。）"}).to_string()

    raw = call_llm(prompt, images_base64=b64s, timeout_s=180, api_style=EXTRACT_API_STYLE, model_name=VISION_MODEL_NAME)
    obj = _retry_json_parse(str(raw), prompt, "pdf", 180, VISION_MODEL_NAME, "", "", EXTRACT_API_STYLE)
    obj = _repair_required_fields(
        obj,
        source_text=ocr_combined or "PDF 页面截图",
        content_type="pdf",
        timeout_s=120,
        model_name=VISION_MODEL_NAME,
        api_endpoint="",
        api_key="",
        api_style=EXTRACT_API_STYLE,
    )
    obj["content_type"] = "pdf"
    obj["_extraction_method"] = "vision"
    if ocr_combined:
        obj["_has_ocr"] = True
    if structured_tables:
        obj["_has_tables"] = True
        obj["_table_preview"] = table_content[:800]
        obj["_structured_tables"] = structured_tables[:6]
    return obj


# ---------------------------------------------------------------------------
# Excel extraction (unchanged logic)
# ---------------------------------------------------------------------------


def _extract_excel(p: Path) -> dict:
    try:
        import pandas as pd
    except Exception:
        txt = "（缺少 pandas，无法解析 excel）"
        prompt = build_extract_prompt("excel").invoke({"content": txt}).to_string()
        raw = call_llm(prompt, timeout_s=60, api_style=EXTRACT_API_STYLE, api_endpoint=EXTRACT_API_ENDPOINT, api_key=EXTRACT_API_KEY, model_name=EXTRACT_MODEL_NAME)
        obj = parse_model_json(str(raw))
        obj["content_type"] = "excel"
        return obj

    sheets = pd.read_excel(str(p), sheet_name=None)
    preview_parts: list[str] = []
    merged_preview: list[dict] = []
    for sheet_name, df in list(sheets.items())[:5]:
        head = df.head(10).to_dict(orient="records")
        merged_preview.extend(head[:5])
        preview_parts.append(
            f"[sheet={sheet_name}] columns={list(df.columns)} rows={len(df)}\n"
            + json.dumps(head, ensure_ascii=False)
        )
    prompt = build_extract_prompt("excel").invoke({"content": "\n\n".join(preview_parts)}).to_string()
    raw = call_llm(prompt, timeout_s=120, api_style=EXTRACT_API_STYLE, api_endpoint=EXTRACT_API_ENDPOINT, api_key=EXTRACT_API_KEY, model_name=EXTRACT_MODEL_NAME)
    obj = _retry_json_parse(str(raw), prompt, "excel", 120, EXTRACT_MODEL_NAME, EXTRACT_API_ENDPOINT, EXTRACT_API_KEY, EXTRACT_API_STYLE)
    obj = _repair_required_fields(
        obj,
        source_text="\n\n".join(preview_parts),
        content_type="excel",
        timeout_s=90,
        model_name=EXTRACT_MODEL_NAME,
        api_endpoint=EXTRACT_API_ENDPOINT,
        api_key=EXTRACT_API_KEY,
        api_style=EXTRACT_API_STYLE,
    )
    obj["content_type"] = "excel"
    obj["data_preview"] = merged_preview[:10]
    return obj


# ---------------------------------------------------------------------------
# DOCX / PPTX structured extraction
# ---------------------------------------------------------------------------


def _extract_docx_pptx(p: Path, ct: str) -> dict:
    if ct == "docx":
        text = extract_docx_structured(str(p))
    else:
        text = extract_pptx_structured(str(p))

    if not text:
        # Fallback: read raw bytes as text
        text = read_text_file(str(p))

    prompt = build_extract_prompt(ct).invoke({"content": text}).to_string()
    raw = call_llm(prompt, timeout_s=120, api_style=EXTRACT_API_STYLE, api_endpoint=EXTRACT_API_ENDPOINT, api_key=EXTRACT_API_KEY, model_name=EXTRACT_MODEL_NAME)
    obj = _retry_json_parse(str(raw), prompt, ct, 120, EXTRACT_MODEL_NAME, EXTRACT_API_ENDPOINT, EXTRACT_API_KEY, EXTRACT_API_STYLE)
    obj = _repair_required_fields(
        obj,
        source_text=text,
        content_type=ct,
        timeout_s=90,
        model_name=EXTRACT_MODEL_NAME,
        api_endpoint=EXTRACT_API_ENDPOINT,
        api_key=EXTRACT_API_KEY,
        api_style=EXTRACT_API_STYLE,
    )
    obj["content_type"] = ct
    return obj


# ---------------------------------------------------------------------------
# A/B comparison
# ---------------------------------------------------------------------------


def ab_extract(file_path: str) -> dict:
    a = extract_from_file(file_path, preprocess=False)
    b = extract_from_file(file_path, preprocess=True)
    qa = compute_quality_metrics(a)
    qb = compute_quality_metrics(b)
    chosen = choose_better_extraction(a, b)
    return {"file": Path(file_path).name, "a": qa, "b": qb, "chosen": chosen}
