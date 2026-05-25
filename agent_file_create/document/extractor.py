import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from agent_file_create.config import (
    EXTRACT_API_ENDPOINT,
    EXTRACT_API_KEY,
    EXTRACT_API_STYLE,
    EXTRACT_MODEL_NAME,
    IMAGE_JPEG_QUALITY,
    IMAGE_MAX_LONG_EDGE,
    OCR_ENABLED,
    PDF_MAX_PAGES_VISION,
    PDF_OCR_THRESHOLD_X,
    PDF_OCR_THRESHOLD_Y,
    VISION_MODEL_NAME,
)
from agent_file_create.llm_client import call_llm
from agent_file_create.preprocessor import (
    compute_quality_metrics,
    extract_docx_structured,
    extract_pdf_embedded_images,
    extract_pdf_text_fast,
    extract_pptx_structured,
    ocr_image,
    preprocess_image_path,
    preprocess_text,
    read_text_file,
    render_pdf_pages,
)
from agent_file_create.prompts import build_extract_prompt

logger = logging.getLogger(__name__)

# Extraction result cache keyed by (file_hash, preprocess_flag)
_CACHE: dict[str, dict] = {}

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

    if preprocess:
        img_bytes = preprocess_image_path(str(p), max_long_edge=IMAGE_MAX_LONG_EDGE, jpeg_quality=IMAGE_JPEG_QUALITY)
    else:
        img_bytes = Path(p).read_bytes()

    b64 = base64.b64encode(img_bytes).decode("utf-8")

    # OCR first for accurate text
    ocr_text = ""
    if OCR_ENABLED:
        ocr_text = ocr_image(img_bytes)

    prompt = build_extract_prompt(ct, ocr_text=ocr_text if ocr_text else None).invoke({"content": "请分析上传的图片"}).to_string()
    raw = call_llm(prompt, images_base64=[b64], timeout_s=120, api_style=EXTRACT_API_STYLE, model_name=VISION_MODEL_NAME)

    obj = _retry_json_parse(str(raw), prompt, ct, 120, VISION_MODEL_NAME, "", "", EXTRACT_API_STYLE)
    obj["content_type"] = "image"
    if ocr_text:
        obj["_ocr_text"] = ocr_text[:500]
    return obj


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


def _extract_pdf(p: Path, preprocess: bool) -> dict:
    import base64

    # Phase 1: text layer via PyMuPDF (faster + more accurate than PyPDF2)
    text_content = extract_pdf_text_fast(str(p))

    # Phase 2: embedded images OCR
    ocr_content = ""
    if OCR_ENABLED:
        threshold = (PDF_OCR_THRESHOLD_X, PDF_OCR_THRESHOLD_Y)
        ocr_content = extract_pdf_embedded_images(str(p), threshold=threshold)

    if text_content or ocr_content:
        # Combine text layer and embedded images OCR for the text LLM
        combined = []
        if text_content:
            combined.append(text_content)
        if ocr_content:
            combined.append("--- 嵌入图片 OCR 结果 ---\n" + ocr_content)
        content = "\n\n".join(combined)

        prompt = build_extract_prompt("pdf").invoke({"content": content}).to_string()
        raw = call_llm(prompt, timeout_s=120, api_style=EXTRACT_API_STYLE, api_endpoint=EXTRACT_API_ENDPOINT, api_key=EXTRACT_API_KEY, model_name=EXTRACT_MODEL_NAME)
        obj = _retry_json_parse(str(raw), prompt, "pdf", 120, EXTRACT_MODEL_NAME, EXTRACT_API_ENDPOINT, EXTRACT_API_KEY, EXTRACT_API_STYLE)
        obj["content_type"] = "pdf"
        obj["_extraction_method"] = "text_layer"
        if ocr_content:
            obj["_has_ocr"] = True
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
    obj["content_type"] = "pdf"
    obj["_extraction_method"] = "vision"
    if ocr_combined:
        obj["_has_ocr"] = True
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

    df = pd.read_excel(str(p))
    head = df.head(10).to_dict(orient="records")
    info = f"columns={list(df.columns)} rows={len(df)}"
    prompt = build_extract_prompt("excel").invoke({"content": info + "\n\ndata_preview:\n" + json.dumps(head, ensure_ascii=False)}).to_string()
    raw = call_llm(prompt, timeout_s=120, api_style=EXTRACT_API_STYLE, api_endpoint=EXTRACT_API_ENDPOINT, api_key=EXTRACT_API_KEY, model_name=EXTRACT_MODEL_NAME)
    obj = _retry_json_parse(str(raw), prompt, "excel", 120, EXTRACT_MODEL_NAME, EXTRACT_API_ENDPOINT, EXTRACT_API_KEY, EXTRACT_API_STYLE)
    obj["content_type"] = "excel"
    obj["data_preview"] = head
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
    chosen = b if (qb.get("filled_fields", 0) >= qa.get("filled_fields", 0)) else a
    return {"file": Path(file_path).name, "a": qa, "b": qb, "chosen": chosen}
