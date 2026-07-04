"""FastAPI application — wires routes and serves static files.

All route implementations live in web/routes/*.py; this module only does
app creation, route registration, and server startup.
"""

import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from agent_file_create.config import validate_config
from agent_file_create.logging_config import setup_logging
from agent_file_create.rag import get_kb
from agent_file_create.web._kb_routes import init_kb_routes, router as kb_router
from agent_file_create.web._utils import get_base_dir, html_dir, result_dir
from agent_file_create.web.routes.interact import router as interact_router
from agent_file_create.web.routes.upload import router as upload_router

logger = logging.getLogger(__name__)

# ── App creation ──────────────────────────────────────────────────────────────

app = FastAPI(title="agent-file-create", version="1.0.0")

# ── Routers (pre-built APIRouters) ────────────────────────────────────────────

init_kb_routes(get_kb)
app.include_router(kb_router)
app.include_router(interact_router)
app.include_router(upload_router)

# Template routes (complex path patterns, registered manually)
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

app.get("/api/template/variables")(api_template_variables)
app.get("/api/template/custom/list")(api_template_custom_list)
app.get("/api/template/custom/{name:path}")(api_template_custom_get)
app.post("/api/template/custom/save")(api_template_custom_save)
app.delete("/api/template/custom/{name:path}")(api_template_custom_delete)
app.post("/api/template/custom/use")(api_template_custom_use)
app.get("/api/template/builtin/list")(api_template_builtin_list)
app.get("/api/template/builtin/{name}")(api_template_builtin_get)
app.post("/api/template/builtin/use")(api_template_builtin_use)

# ── Health check ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check endpoint for Docker / load balancer probes.

    Returns 200 with basic service info when the app is running.
    A deep-health variant (querying LLM and DB) is available via ?deep=true.
    """
    from fastapi import Query

    deep = False  # default — fast ping
    try:
        from agent_file_create.config import MODEL_NAME, EMBED_MODEL_NAME
    except Exception:
        MODEL_NAME = EMBED_MODEL_NAME = "unknown"

    resp: dict = {"status": "ok", "version": "1.0.0"}

    # Optional deep check: test DB connectivity and LLM availability
    if deep:
        try:
            from agent_file_create.db_service import get_db_connection
            conn = get_db_connection()
            conn.close()
            resp["database"] = "ok"
        except Exception as e:
            resp["database"] = f"error: {e}"

        try:
            from agent_file_create.llm_factory import get_chat_model
            llm = get_chat_model(style="ollama", model=MODEL_NAME)
            # Ping with minimal token generation
            llm.invoke("ping")
            resp["llm"] = "ok"
        except Exception as e:
            resp["llm"] = f"unavailable: {e}"
    else:
        resp["database"] = "not_checked"
        resp["llm"] = "not_checked"

    return resp


# ── Static file serving ──────────────────────────────────────────────────────

_html = html_dir()
_result = result_dir()

if _result.exists():
    app.mount("/result", StaticFiles(directory=str(_result)), name="result")

if _html.exists():
    app.mount("/static", StaticFiles(directory=str(_html)), name="static")

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
    import signal
    import threading
    import time as _time

    import uvicorn

    setup_logging()

    # ── Prompt versions ──────────────────────────────────────────────────
    try:
        from agent_file_create.prompts import get_prompt_info
        pv = get_prompt_info()
        logger.info("prompt_versions total=%d", pv["total"])
        logger.debug("prompt_versions detail=%s", pv["versions"])
    except Exception:
        pass

    # ── Configuration validation ─────────────────────────────────────────
    config_errors = validate_config()
    if config_errors:
        for err in config_errors:
            logger.error("config_error: %s", err)
        logger.error("startup_aborted due to %d configuration errors", len(config_errors))
        sys.exit(1)

    # ── Graceful shutdown ──────────────────────────────────────────────
    _shutting_down = False

    def _handle_shutdown(signum, frame):
        nonlocal _shutting_down
        if _shutting_down:
            return
        _shutting_down = True
        sig_name = signal.Signals(signum).name
        logger.warning("received_signal signal=%s shutting_down", sig_name)

        # Cancel all running tasks
        try:
            from agent_file_create.task.manager import TaskManager
            tm = TaskManager()
            for tid in list(tm.list_running_tasks()):
                try:
                    tm.cancel_task(tid)
                    logger.info("shutdown_cancelled task=%s", tid)
                except Exception:
                    pass
        except Exception:
            pass

        # Brief grace period for threads to finish
        _time.sleep(2)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    logger.info("web_listen http://%s:%s/", host, port)
    uvicorn.run(app, host=host, port=int(port), log_level="warning")
