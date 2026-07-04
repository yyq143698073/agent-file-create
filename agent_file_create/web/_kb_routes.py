"""Knowledge Base management API routes — extracted from server.py."""

import logging
import re
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kb", tags=["knowledge_base"])

# Shared reference — set by server.py at startup
_kb_ref = None


def init_kb_routes(kb_getter):
    """Called by server.py to inject the KB singleton getter."""
    global _kb_ref
    _kb_ref = kb_getter


def _get_kb():
    return _kb_ref() if _kb_ref else None


def _sanitize_filename(name: str) -> str:
    import re
    n = (name or "").strip()
    n = n.replace("\\", "/").split("/")[-1]
    n = re.sub(r"[^0-9A-Za-z一-鿿._-]+", "_", n)
    n = n.strip("._")
    return n or "upload"


def _result_dir():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent.parent / "result"


@router.get("/list")
def kb_list():
    items = _get_kb().list_kb()
    return {"kbs": items}


@router.post("/create")
async def kb_create(request: Request):
    """Register a new (empty) knowledge base."""
    body = await request.json()
    name = str(body.get("kb") or body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "kb 名称不能为空")
    if not re.fullmatch(r"[a-zA-Z0-9_一-鿿-]+", name):
        raise HTTPException(400, "kb 名称只能包含中英文、数字、下划线和连字符")
    try:
        _get_kb().register_kb(kb=name)
        return {"ok": True, "kb": name}
    except Exception as e:
        raise HTTPException(500, str(e)[:240])


@router.get("/docs")
def kb_docs(kb: str = Query("default")):
    docs = _get_kb().list_docs(kb=kb)
    return {"kb": kb, "docs": docs}


@router.post("/query")
async def kb_query(request: Request):
    body = await request.json()
    kb = str(body.get("kb") or "").strip() or "default"
    question = str(body.get("question") or body.get("message") or "").strip()
    if not question:
        raise HTTPException(400, "question 不能为空")
    try:
        top_k = int(body.get("top_k") or 6)
    except Exception:
        top_k = 6
    filters = body.get("filters") if isinstance(body.get("filters"), dict) else None
    try:
        ans = _get_kb().answer(kb=kb, question=question, top_k=top_k, filters=filters)
        cits = [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "section_path": c.section_path, "score": float(c.score), "snippet": c.snippet, "doc_name": c.doc_name} for c in ans.citations]

        # ── Write to chat history if task_id is provided ──────────────
        task_id = str(body.get("task_id") or "").strip()
        if task_id:
            try:
                from agent_file_create.task.manager import TaskManager
                tm = TaskManager()
                cits_text = "; ".join(
                    f"{c.doc_name}·{c.section_path}" for c in (ans.citations or [])[:3]
                    if c.doc_name or c.section_path
                )
                history_entry = ans.answer
                if cits_text:
                    history_entry += f"\n\n📚 来源：{cits_text}"
                tm.append_chat_history(task_id, [
                    {"role": "user", "content": f"[知识库:{kb}] {question}"},
                    {"role": "assistant", "content": history_entry},
                ])
            except Exception:
                pass  # best-effort, don't fail the query

        return {"kb": ans.kb, "question": ans.question, "answer": ans.answer, "citations": cits}
    except Exception as e:
        raise HTTPException(500, str(e)[:240])


@router.post("/upload")
async def kb_upload(
    files: list[UploadFile] = File(...),
    kb: str = Form("default"),
    doc_type: str = Form(""),
):
    kb = kb.strip() or "default"
    doc_type = doc_type.strip()
    if not files:
        raise HTTPException(400, "未收到文件")

    base = _result_dir() / "kb" / kb / "uploads"
    base.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for f in files:
        name = _sanitize_filename(f.filename or "upload")
        fp = base / (uuid.uuid4().hex[:8] + "_" + name)
        data = await f.read()
        logger.info("kb_upload_file kb=%s file=%s size=%d", kb, name, len(data))
        try:
            fp.write_bytes(data)
        except Exception:
            results.append({"file": name, "ok": False, "error": "write_failed"})
            continue
        try:
            r = _get_kb().ingest_file(kb=kb, file_path=str(fp), doc_id=name, title=name, source=str(fp), doc_type=doc_type)
            r["file"] = name
            results.append(r)
            if not r.get("ok"):
                logger.warning("kb_ingest_failed file=%s err=%s", name, r.get("error", "?"))
        except Exception as e:
            logger.error("kb_ingest_exception file=%s err=%s", name, str(e)[:200])
            results.append({"file": name, "ok": False, "error": str(e)[:240]})
    logger.info("kb_upload_done kb=%s files=%d ok=%d", kb, len(files), sum(1 for r in results if r.get("ok")))
    return {"kb": kb, "results": results}


@router.post("/delete")
async def kb_delete(request: Request):
    body = await request.json()
    kb = str(body.get("kb") or "").strip()
    doc_id = str(body.get("doc_id") or "").strip()
    if not kb:
        raise HTTPException(400, "kb 不能为空")
    if doc_id:
        r = _get_kb().delete_doc(kb=kb, doc_id=doc_id)
    else:
        r = _get_kb().delete_kb(kb=kb)
    if not r.get("ok"):
        raise HTTPException(500, str(r.get("error") or "delete_failed")[:240])
    return r


@router.get("/stats")
def kb_stats(kb: str = "default"):
    if not kb.strip():
        raise HTTPException(400, "kb 不能为空")
    try:
        return _get_kb().kb_stats(kb=kb.strip())
    except Exception as e:
        raise HTTPException(500, str(e)[:240])


@router.post("/reembed")
async def kb_reembed(request: Request):
    """Re-embed all chunks with empty vectors in a KB. Use after repairing embedding service."""
    body = await request.json()
    kb = str(body.get("kb") or "").strip() or "default"
    doc_id = str(body.get("doc_id") or "").strip() or None
    try:
        result = _get_kb().reembed_kb(kb=kb, doc_id=doc_id)
        return result
    except Exception as e:
        raise HTTPException(500, str(e)[:240])

@router.post("/health")
def kb_health():
    return _get_kb().check_embed_health()
