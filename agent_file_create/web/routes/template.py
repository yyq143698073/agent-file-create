"""Template management routes — built-in marketplace and custom CRUD.

Extracted from web/server.py.
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from agent_file_create.web._utils import get_base_dir, result_dir, sanitize_filename

logger = logging.getLogger(__name__)

router = APIRouter(tags=["template"])

_CUSTOM_TEMPLATE_DIR = result_dir() / "template" / "custom"
_BUILTIN_TEMPLATES_DIR = get_base_dir() / "resource" / "templates"


@router.get("/api/template/variables")
def api_template_variables():
    """Return metadata about all available template placeholder variables."""
    return {
        "system_variables": [
            {"key": "title", "label": "文档标题", "description": "从大纲自动提取的文档主标题（# 开头）"},
            {"key": "task_id", "label": "任务ID", "description": "当前任务的8位唯一标识符"},
            {"key": "document_outline", "label": "文档大纲", "description": "完整的 Markdown 大纲内容"},
            {"key": "document_content", "label": "文档正文", "description": "完整的 Markdown 正文内容"},
        ],
        "section_variables_note": "章节级变量（如 {{背景分析}}、{{核心内容}}）由大纲的 ## 二级标题动态生成。你可以在模板中预先写入预期的章节变量名，生成时系统会自动匹配替换。",
    }


def api_template_custom_list():
    """List all user-created custom templates with metadata."""
    try:
        _CUSTOM_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    items = []
    if _CUSTOM_TEMPLATE_DIR.exists():
        from agent_file_create.document.template_renderer import _scan_md_placeholders
        for p in sorted(_CUSTOM_TEMPLATE_DIR.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                vars_set = _scan_md_placeholders(str(p))
                items.append({
                    "name": p.name,
                    "stripped": p.stem,
                    "size": p.stat().st_size,
                    "modified_at": p.stat().st_mtime,
                    "variable_count": len(vars_set),
                })
            except Exception:
                pass
    return {"templates": items}


def api_template_custom_get(name: str):
    """Get content of a single custom template by filename."""
    safe = sanitize_filename(name)
    fp = _CUSTOM_TEMPLATE_DIR / safe
    if not fp.exists():
        raise HTTPException(404, f"模板不存在: {safe}")
    from agent_file_create.document.template_renderer import _scan_md_placeholders
    content = fp.read_text(encoding="utf-8")
    return {
        "name": safe,
        "content": content,
        "variables": sorted(_scan_md_placeholders(str(fp))),
    }


async def api_template_custom_save(request: Request):
    """Create or update a custom template."""
    body = await request.json()
    name = str(body.get("name") or "").strip()
    content = str(body.get("content") or "")
    if not name:
        raise HTTPException(400, "模板名称不能为空")
    safe = sanitize_filename(name)
    if not safe.lower().endswith(".md"):
        safe += ".md"
    try:
        _CUSTOM_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    fp = _CUSTOM_TEMPLATE_DIR / safe
    try:
        fp.write_text(content, encoding="utf-8")
    except Exception as e:
        raise HTTPException(500, f"保存失败: {str(e)[:200]}")
    return {"name": safe, "ok": True}


def api_template_custom_delete(name: str):
    """Delete a custom template."""
    safe = sanitize_filename(name)
    fp = _CUSTOM_TEMPLATE_DIR / safe
    if not fp.exists():
        raise HTTPException(404, f"模板不存在: {safe}")
    try:
        fp.unlink()
    except Exception as e:
        raise HTTPException(500, f"删除失败: {str(e)[:200]}")
    return {"name": safe, "ok": True}


async def api_template_custom_use(request: Request):
    """Copy a custom template to a task's template directory."""
    body = await request.json()
    name = str(body.get("name") or "").strip()
    task_id = str(body.get("task_id") or "").strip()
    if not name or not task_id:
        raise HTTPException(400, "name 和 task_id 不能为空")
    safe = sanitize_filename(name)
    if not safe.lower().endswith(".md"):
        safe += ".md"
    src = _CUSTOM_TEMPLATE_DIR / safe
    if not src.exists():
        raise HTTPException(404, f"模板不存在: {safe}")
    dest_dir = result_dir() / task_id / "template"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    dst = dest_dir / safe
    try:
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception as e:
        raise HTTPException(500, f"复制失败: {str(e)[:200]}")
    return {"name": safe, "task_id": task_id, "ok": True}


def api_template_builtin_list():
    """List all built-in report templates with metadata."""
    items = []
    if _BUILTIN_TEMPLATES_DIR.exists():
        for md_path in sorted(_BUILTIN_TEMPLATES_DIR.glob("*.md")):
            meta_path = md_path.with_suffix("").with_name(md_path.stem + "_meta.json")
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            try:
                from agent_file_create.document.template_renderer import _scan_md_placeholders
                vars_set = _scan_md_placeholders(str(md_path))
            except Exception:
                vars_set = set()
            items.append({
                "name": md_path.name,
                "id": md_path.stem,
                "size": md_path.stat().st_size,
                "variables": sorted(vars_set),
                "description": meta.get("description", ""),
                "category": meta.get("category", "通用"),
                "suggested_prompt": meta.get("suggested_prompt", ""),
            })
    return {"templates": items}


def api_template_builtin_get(name: str):
    """Get a single built-in template by name."""
    safe = sanitize_filename(name)
    if not safe.endswith(".md"):
        safe += ".md"
    fp = _BUILTIN_TEMPLATES_DIR / safe
    if not fp.exists():
        raise HTTPException(404, f"内置模板不存在: {safe}")
    meta = {}
    meta_path = _BUILTIN_TEMPLATES_DIR / (fp.stem + "_meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    content = fp.read_text(encoding="utf-8")
    from agent_file_create.document.template_renderer import _scan_md_placeholders
    return {
        "name": safe,
        "content": content,
        "variables": sorted(_scan_md_placeholders(str(fp))),
        "description": meta.get("description", ""),
        "category": meta.get("category", "通用"),
        "suggested_prompt": meta.get("suggested_prompt", ""),
    }


async def api_template_builtin_use(request: Request):
    """Copy a built-in template to a task for use."""
    body = await request.json()
    task_id = str(body.get("task_id") or "").strip()
    name = str(body.get("name") or "").strip()
    if not task_id or not name:
        raise HTTPException(400, "task_id 和 name 不能为空")
    safe = sanitize_filename(name)
    if not safe.endswith(".md"):
        safe += ".md"
    src = _BUILTIN_TEMPLATES_DIR / safe
    if not src.exists():
        raise HTTPException(404, f"内置模板不存在: {safe}")
    dest_dir = result_dir() / task_id / "template"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / safe
    dst.write_bytes(src.read_bytes())
    return {"name": safe, "task_id": task_id, "ok": True}
