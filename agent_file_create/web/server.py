"""FastAPI application — wires routes and serves static files.

All route implementations live in web/routes/*.py; this module only does
app creation, route registration, and server startup.
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from agent_file_create.logging_config import setup_logging
from agent_file_create.rag import get_kb
from agent_file_create.web._kb_routes import init_kb_routes, router as kb_router
from agent_file_create.web._utils import get_base_dir, html_dir, result_dir
from agent_file_create.web.routes.interact import (
    api_chat,
    api_chat_stream,
    api_clarify,
    api_gen,
    api_satisfaction,
    api_section_edit,
    api_section_save,
    api_status,
    api_stream,
    api_tasks,
    api_versions,
    api_versions_delete,
    api_versions_redo,
    api_versions_select,
    chat_history,
    save_chat_message,
)
from agent_file_create.web.routes.template import (
    api_template_builtin_get,
    api_template_builtin_list,
    api_template_builtin_use,
    api_template_custom_delete,
    api_template_custom_get,
    api_template_custom_list,
    api_template_custom_save,
    api_template_custom_use,
    api_template_variables,
)
from agent_file_create.web.routes.upload import api_append, api_upload

logger = logging.getLogger(__name__)

# ── App creation ──────────────────────────────────────────────────────────────

app = FastAPI(title="agent-file-create", version="1.0.0")

# KB routes (pre-built router)
init_kb_routes(get_kb)
app.include_router(kb_router)

# Static files
_html = html_dir()
_result = result_dir()

if _html.exists():
    app.mount("/static", StaticFiles(directory=str(_html)), name="static")

# ── Route registration ───────────────────────────────────────────────────────

# Upload
app.post("/api/upload")(api_upload)
app.post("/api/append")(api_append)
app.post("/api/gen")(api_gen)

# Chat & interaction
app.post("/api/clarify")(api_clarify)
app.post("/api/satisfaction")(api_satisfaction)
app.post("/api/chat")(api_chat)
app.post("/api/chat/stream")(api_chat_stream)
app.get("/api/chat/history")(chat_history)
app.post("/api/chat/history/save")(save_chat_message)

# Versions
app.get("/api/versions")(api_versions)
app.post("/api/versions/select")(api_versions_select)
app.post("/api/versions/delete")(api_versions_delete)
app.post("/api/versions/redo")(api_versions_redo)

# Sections
app.post("/api/section/save")(api_section_save)
app.post("/api/section/edit")(api_section_edit)

# Templates
app.get("/api/template/variables")(api_template_variables)
app.get("/api/template/custom/list")(api_template_custom_list)
app.get("/api/template/custom/{name:path}")(api_template_custom_get)
app.post("/api/template/custom/save")(api_template_custom_save)
app.delete("/api/template/custom/{name:path}")(api_template_custom_delete)
app.post("/api/template/custom/use")(api_template_custom_use)
app.get("/api/template/builtin/list")(api_template_builtin_list)
app.get("/api/template/builtin/{name}")(api_template_builtin_get)
app.post("/api/template/builtin/use")(api_template_builtin_use)

# Status / stream / tasks
app.get("/api/status")(api_status)
app.get("/api/stream")(api_stream)
app.get("/api/tasks")(api_tasks)

# ── Static file serving ──────────────────────────────────────────────────────

if _result.exists():
    app.mount("/result", StaticFiles(directory=str(_result)), name="result")

if _html.exists():
    @app.get("/")
    @app.get("/index.html")
    async def serve_index():
        index = _html / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return PlainTextResponse("Not Found", status_code=404)


# ── Public API ────────────────────────────────────────────────────────────────

application = app


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    setup_logging()
    logger.info(f"web_listen http://{host}:{port}/")
    uvicorn.run(app, host=host, port=int(port), log_level="warning")
