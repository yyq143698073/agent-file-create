import logging
import re
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def build_content_dict(markdown_content: str) -> Dict[str, str]:
    lines = (markdown_content or "").splitlines()
    out: Dict[str, str] = {}
    cur = None
    buf: list[str] = []
    for line in lines:
        m2 = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if m2:
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur = m2.group(1).strip()
            buf = []
            continue
        if cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()

    for k in list(out.keys()):
        if out.get(k):
            continue
        start = f"## {k}"
        block: list[str] = []
        in_block = False
        for line in lines:
            if line.strip() == start:
                in_block = True
                continue
            if in_block and re.match(r"^\s*##\s+", line):
                break
            if in_block:
                if line.strip().startswith("### "):
                    block.append(line.strip())
                elif block:
                    block.append(line)
        if block:
            out[k] = "\n".join(block).strip()

    title = ""
    for line in lines:
        m1 = re.match(r"^\s*#\s+(.+?)\s*$", line)
        if m1:
            title = m1.group(1).strip()
            break
    if title:
        out["title"] = title
    return out


def render_markdown_template(template_path: str, content_dict: Dict[str, str], output_path: str) -> None:
    tp = Path(template_path)
    rendered = tp.read_text(encoding="utf-8")
    for k, v in (content_dict or {}).items():
        rendered = rendered.replace("{{" + str(k) + "}}", str(v))
    Path(output_path).write_text(rendered, encoding="utf-8")


def render_word_template(template_path: str, content_dict: Dict[str, str], output_path: str) -> None:
    try:
        from docxtpl import DocxTemplate

        doc = DocxTemplate(template_path)
        doc.render(content_dict or {})
        doc.save(output_path)
        return
    except Exception:
        pass

    try:
        from docx import Document
    except Exception as e:
        raise RuntimeError("缺少 docxtpl 或 python-docx 依赖，无法生成 docx") from e

    doc = Document(template_path)
    mapping = {str(k): str(v) for k, v in (content_dict or {}).items()}
    for para in doc.paragraphs:
        for k, v in mapping.items():
            token = "{{" + k + "}}"
            if token in para.text:
                for run in para.runs:
                    if token in run.text:
                        run.text = run.text.replace(token, v)
    doc.save(output_path)


def render_pdf_template(template_path: str, content_dict: Dict[str, str], output_path: str) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
    except Exception as e:
        raise RuntimeError("缺少 reportlab 依赖，无法生成 pdf") from e

    c = canvas.Canvas(output_path, pagesize=A4)
    w, h = A4
    x = 46
    y = h - 60

    font_name = "Helvetica"
    try:
        base_dir = Path(__file__).resolve().parent.parent.parent
        simhei = base_dir / "resource" / "SimHei.ttf"
        if simhei.exists():
            pdfmetrics.registerFont(TTFont("SimHei", str(simhei)))
            font_name = "SimHei"
    except Exception:
        font_name = "Helvetica"

    c.setFont(font_name, 16)
    title = str((content_dict or {}).get("title") or "").strip()
    if title:
        c.drawString(x, y, title)
        y -= 26

    c.setFont(font_name, 11)
    for k, v in (content_dict or {}).items():
        if k == "title":
            continue
        txt = str(v or "").strip()
        if not txt:
            continue
        head = str(k).strip()
        c.setFont(font_name, 12)
        c.drawString(x, y, head)
        y -= 18
        c.setFont(font_name, 11)
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                y -= 10
                continue
            if y <= 60:
                c.showPage()
                y = h - 60
                c.setFont(font_name, 11)
            c.drawString(x, y, line[:120])
            y -= 14
        y -= 10

    c.save()


def render_template(template_path: str, content_dict: Dict[str, str], output_path: str) -> None:
    suf = Path(template_path).suffix.lower().lstrip(".")
    if suf == "md":
        render_markdown_template(template_path, content_dict, output_path)
    elif suf == "docx":
        render_word_template(template_path, content_dict, output_path)
    elif suf == "pdf":
        render_pdf_template(template_path, content_dict, output_path)
    else:
        render_markdown_template(template_path, content_dict, output_path)