import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

_OCR_INSTANCE: Optional[Any] = None


def get_ocr(use_cuda: bool = True) -> Any:
    """Return a cached RapidOCR instance (paddle GPU preferred, onnxruntime fallback)."""
    global _OCR_INSTANCE
    if _OCR_INSTANCE is not None:
        return _OCR_INSTANCE
    try:
        from rapidocr_paddle import RapidOCR

        _OCR_INSTANCE = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR

        _OCR_INSTANCE = RapidOCR()
    return _OCR_INSTANCE


def ocr_image(image_data: bytes) -> str:
    """Run OCR on raw image bytes, return joined text lines."""
    try:
        ocr = get_ocr()
        from PIL import Image

        img = Image.open(BytesIO(image_data))
        # RapidOCR accepts image path, numpy array, or PIL Image
        result, _ = ocr(img)
        if not result:
            return ""
        lines = []
        for item in result:
            text = (item[1] or "").strip()
            if text:
                lines.append(text)
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"ocr_failed err={str(e)[:160]}")
        return ""


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def preprocess_text(text: str, max_chars: int = 8000) -> str:
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()
    if len(s) > max_chars:
        return s[:max_chars] + "…"
    return s


def read_text_file(file_path: str, max_chars: int = 12000) -> str:
    p = Path(file_path)
    data = p.read_bytes()
    try:
        return preprocess_text(data.decode("utf-8"), max_chars=max_chars)
    except UnicodeDecodeError:
        pass
    for enc in ("gb18030", "latin-1"):
        try:
            return preprocess_text(data.decode(enc), max_chars=max_chars)
        except Exception:
            continue
    return preprocess_text(data.decode("utf-8", errors="ignore"), max_chars=max_chars)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def preprocess_image_path(
    file_path: str,
    max_long_edge: int = 2048,
    jpeg_quality: int = 85,
) -> bytes:
    try:
        from PIL import Image, ImageEnhance, ImageOps
    except Exception:
        return Path(file_path).read_bytes()

    try:
        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        w, h = img.size
        long_edge = max(w, h)
        if long_edge > max_long_edge:
            ratio = max_long_edge / long_edge
            new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
            img = img.resize(new_size, Image.LANCZOS)

        try:
            from PIL import ImageStat

            stat = ImageStat.Stat(img)
            r_std = stat.stddev[0]
            if r_std < 50:
                img = ImageEnhance.Contrast(img).enhance(1.2)
                img = ImageEnhance.Sharpness(img).enhance(1.1)
        except Exception:
            img = ImageEnhance.Contrast(img).enhance(1.2)
            img = ImageEnhance.Sharpness(img).enhance(1.1)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"preprocess_image_failed err={str(e)[:160]}")
        return Path(file_path).read_bytes()


# ---------------------------------------------------------------------------
# PDF (PyMuPDF)
# ---------------------------------------------------------------------------


def extract_pdf_text_fast(file_path: str, max_chars: int = 20000) -> str:
    """Extract text layer from PDF using PyMuPDF (fitz)."""
    try:
        import fitz
    except Exception:
        return _extract_pdf_text_pypdf2_fallback(file_path, max_chars)

    try:
        doc = fitz.open(str(file_path))
        parts: list[str] = []
        total = 0
        for page in doc:
            try:
                t = page.get_text("text") or ""
            except Exception:
                t = ""
            if t:
                parts.append(t)
                total += len(t)
                if total >= max_chars:
                    break
        doc.close()
        return preprocess_text("\n".join(parts), max_chars=max_chars)
    except Exception as e:
        logger.warning(f"pdf_text_extract_fitz_failed err={str(e)[:160]}")
        return _extract_pdf_text_pypdf2_fallback(file_path, max_chars)


def _extract_pdf_text_pypdf2_fallback(file_path: str, max_chars: int = 20000) -> str:
    """Fallback: extract text using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
    except Exception:
        return ""

    try:
        reader = PdfReader(str(file_path))
        parts: list[str] = []
        for page in reader.pages[:30]:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                parts.append(t)
            if sum(len(x) for x in parts) >= max_chars:
                break
        return preprocess_text("\n".join(parts), max_chars=max_chars)
    except Exception as e:
        logger.warning(f"pdf_text_extract_pypdf2_failed err={str(e)[:160]}")
        return ""


def extract_pdf_embedded_images(
    file_path: str,
    threshold: tuple[float, float] = (0.6, 0.6),
) -> str:
    """Extract embedded images from PDF pages and OCR them.

    Skips images whose width/page-width or height/page-height ratios
    fall below *threshold* (avoids OCR-ing small decorative images).
    """
    try:
        import fitz
    except Exception:
        return ""

    try:
        doc = fitz.open(str(file_path))
        ocr_parts: list[str] = []
        for page in doc:
            page_w = page.rect.width
            page_h = page.rect.height
            for img_info in page.get_image_info(xrefs=True):
                w = img_info.get("width", 0)
                h = img_info.get("height", 0)
                if page_w > 0 and w / page_w < threshold[0]:
                    continue
                if page_h > 0 and h / page_h < threshold[1]:
                    continue
                try:
                    xref = img_info["xref"]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    # Handle rotation
                    if page.rotation != 0:
                        from PIL import Image

                        im = Image.open(BytesIO(image_bytes))
                        if page.rotation == 90:
                            im = im.transpose(Image.ROTATE_270)
                        elif page.rotation == 180:
                            im = im.transpose(Image.ROTATE_180)
                        elif page.rotation == 270:
                            im = im.transpose(Image.ROTATE_90)
                        buf = BytesIO()
                        im.save(buf, format="PNG")
                        image_bytes = buf.getvalue()
                    ocr_text = ocr_image(image_bytes)
                    if ocr_text:
                        ocr_parts.append(ocr_text)
                except Exception:
                    continue
        doc.close()
        return "\n".join(ocr_parts)
    except Exception as e:
        logger.warning(f"pdf_embedded_images_failed err={str(e)[:160]}")
        return ""


def render_pdf_pages(file_path: str, max_pages: int = 6) -> list[bytes]:
    """Render PDF pages as PNG images at 180 DPI (up to *max_pages* pages).

    Uses adaptive sampling for long documents: first 3 + samples from middle + last 2.
    """
    try:
        import fitz
    except Exception:
        return []

    try:
        doc = fitz.open(str(file_path))
        total = doc.page_count
        if total == 0:
            doc.close()
            return []

        # Adaptive page selection for long documents
        if total <= max_pages:
            indices = list(range(total))
        elif total <= max_pages * 2:
            indices = list(range(max_pages))
        else:
            # First N pages + samples from middle + last N pages
            front = min(3, max_pages // 3)
            back = min(2, max_pages // 3)
            mid_count = max_pages - front - back
            if mid_count > 0 and total > front + back:
                step = max(1, (total - front - back) // mid_count)
                mid_indices = list(range(front, total - back, step))[:mid_count]
            else:
                mid_indices = []
            indices = list(range(front)) + mid_indices + list(range(total - back, total))
            indices = list(dict.fromkeys(indices))  # deduplicate while preserving order

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _render(i: int) -> tuple[int, bytes]:
            page = doc.load_page(i)
            return i, page.get_pixmap(dpi=180).tobytes("png")

        results: dict[int, bytes] = {}
        workers = min(len(indices), 4)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_render, i): i for i in indices}
            for fut in as_completed(futures):
                idx, data = fut.result()
                results[idx] = data
        doc.close()
        return [results[i] for i in sorted(results)]
    except Exception as e:
        logger.warning(f"pdf_render_failed err={str(e)[:160]}")
        return []


# ---------------------------------------------------------------------------
# DOCX structured extraction
# ---------------------------------------------------------------------------


def extract_docx_structured(file_path: str, max_chars: int = 20000) -> str:
    """Extract text, tables, and embedded images from a .docx file.

    Uses python-docx to walk paragraphs and tables in document order,
    and OCRs embedded images found in paragraphs.
    """
    try:
        from docx import Document
        from docx.oxml.ns import qn
    except Exception:
        return ""

    try:
        doc = Document(str(file_path))
        parts: list[str] = []

        for block in doc.element.body:
            tag = block.tag.split("}")[-1] if "}" in block.tag else block.tag

            if tag == "p":
                # Paragraph
                para = _paragraph_for_element(doc, block)
                if para is None:
                    continue
                text = para.text.strip()
                if text:
                    parts.append(text)
                # Check for embedded images in this paragraph
                images_text = _extract_images_from_paragraph(para, doc)
                if images_text:
                    parts.append(images_text)

            elif tag == "tbl":
                # Table
                table_text = _extract_table_text(block)
                if table_text:
                    parts.append(table_text)

        result = "\n\n".join(parts)
        return preprocess_text(result, max_chars=max_chars)
    except Exception as e:
        logger.warning(f"docx_structured_failed err={str(e)[:160]}")
        return ""


def _paragraph_for_element(doc, element) -> Any:
    """Find the Paragraph object matching an XML element."""
    from docx.text.paragraph import Paragraph

    for para in doc.paragraphs:
        if para._element is element:
            return para
    return None


def _extract_images_from_paragraph(para, doc) -> str:
    """Extract and OCR embedded images in a paragraph."""
    try:
        from docx.oxml.ns import qn
    except Exception:
        return ""

    nsmap = {"pic": "http://schemas.openxmlformats.org/drawingml/2006/picture"}
    pics = para._element.xpath(".//pic:pic", namespaces=nsmap)
    if not pics:
        return ""

    # Build rId → image part lookup
    image_parts: dict[str, Any] = {}
    for rel in doc.part.rels.values():
        if "image" in (rel.reltype or ""):
            image_parts[rel.rId] = rel.target_part

    texts: list[str] = []
    for pic in pics:
        blips = pic.xpath(".//*[local-name()='blip']")
        for blip in blips:
            embed = blip.get(qn("r:embed"))
            if not embed or embed not in image_parts:
                continue
            try:
                image_bytes = image_parts[embed].blob
                ocr_text = ocr_image(image_bytes)
                if ocr_text:
                    texts.append(ocr_text)
            except Exception:
                continue
    return "\n".join(texts) if texts else ""


def _extract_table_text(tbl_element) -> str:
    """Extract text from a table XML element."""
    try:
        from docx.oxml.ns import qn
    except Exception:
        return ""

    rows: list[str] = []
    for row in tbl_element.findall(qn("w:tr")):
        cells: list[str] = []
        for cell in row.findall(qn("w:tc")):
            cell_texts: list[str] = []
            for p in cell.findall(qn("w:p")):
                t = "".join(t_node.text or "" for t_node in p.findall(".//" + qn("w:t")))
                if t.strip():
                    cell_texts.append(t.strip())
            cells.append(" ".join(cell_texts))
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows) if rows else ""


# ---------------------------------------------------------------------------
# PPTX structured extraction
# ---------------------------------------------------------------------------


def extract_pptx_structured(file_path: str, max_chars: int = 20000) -> str:
    """Extract text and embedded images from a .pptx file.

    Uses python-pptx to walk shapes on each slide (sorted positionally),
    and OCRs embedded images found in picture shapes or groups.
    """
    try:
        from pptx import Presentation
        from pptx.shapes.picture import Picture
        from pptx.shapes.group import GroupShapes
        from pptx.enum.shapes import MSO_SHAPE_TYPE
    except Exception:
        return ""

    try:
        prs = Presentation(str(file_path))
        parts: list[str] = []

        for slide in prs.slides:
            slide_parts: list[str] = []

            # Sort shapes top-to-bottom, left-to-right
            shapes_sorted = sorted(
                list(slide.shapes),
                key=lambda s: (getattr(s, "top", 0) or 0, getattr(s, "left", 0) or 0),
            )

            for shape in shapes_sorted:
                shape_text = _extract_pptx_shape_text(shape)
                if shape_text:
                    slide_parts.append(shape_text)

            if slide_parts:
                parts.append("\n".join(slide_parts))

        result = "\n\n".join(parts)
        return preprocess_text(result, max_chars=max_chars)
    except Exception as e:
        logger.warning(f"pptx_structured_failed err={str(e)[:160]}")
        return ""


def _extract_pptx_shape_text(shape) -> str:
    """Extract text from a single PPTX shape (text, table, picture, group)."""
    from pptx.shapes.picture import Picture
    from pptx.shapes.group import GroupShapes
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    results: list[str] = []

    # Text frame
    if shape.has_text_frame:
        for para in shape.text_frame.paragraphs:
            t = para.text.strip()
            if t:
                results.append(t)

    # Table
    if shape.has_table:
        table = shape.table
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            results.append(" | ".join(cells))

    # Embedded image (picture)
    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
        try:
            image_bytes = shape.image.blob
            ocr_text = ocr_image(image_bytes)
            if ocr_text:
                results.append(ocr_text)
        except Exception:
            pass

    # Group shape — recurse
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        try:
            for child in shape.shapes:
                child_text = _extract_pptx_shape_text(child)
                if child_text:
                    results.append(child_text)
        except Exception:
            pass

    return "\n".join(results) if results else ""


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def compute_quality_metrics(result: Any) -> dict:
    fields = ["title", "keywords", "summary", "key_points", "data", "conclusion", "prediction"]
    filled = 0
    for f in fields:
        v = result.get(f) if isinstance(result, dict) else None
        ok = False
        if isinstance(v, str):
            ok = bool(v.strip())
        elif isinstance(v, list) or isinstance(v, dict):
            ok = bool(v)
        if ok:
            filled += 1
    return {"filled_fields": filled, "field_ratio": filled / float(len(fields) or 1)}
