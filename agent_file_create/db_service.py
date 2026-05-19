import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

from agent_file_create.config import DB_PATH, DB_URL

logger = logging.getLogger(__name__)


def _dialect() -> str:
    url = (DB_URL or "").strip().lower()
    if url.startswith("postgresql://") or url.startswith("postgres://"):
        return "postgres"
    return "sqlite"


def get_db_connection():
    if _dialect() == "postgres":
        import psycopg2

        return psycopg2.connect(DB_URL)
    base = Path(__file__).resolve().parent.parent
    db_path = base / str(DB_PATH or "result/app.db")
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    conn = sqlite3.connect(str(db_path))
    return conn


def init_db(conn) -> None:
    d = _dialect()
    cur = conn.cursor()
    if d == "postgres":
        cur.execute(
            """
            create table if not exists document_tasks(
              id text primary key,
              title text,
              document_type text,
              user_prompt text,
              status text,
              created_at double precision,
              updated_at double precision,
              output_dir text,
              meta_json text
            )
            """
        )
        cur.execute(
            """
            create table if not exists document_outlines(
              id text primary key,
              task_id text,
              outline_markdown text,
              outline_tree_json text,
              created_at double precision
            )
            """
        )
        cur.execute(
            """
            create table if not exists outline_sections(
              id text primary key,
              outline_id text,
              task_id text,
              level integer,
              title text,
              parent_title text,
              order_index integer
            )
            """
        )
        cur.execute(
            """
            create table if not exists document_contents(
              id text primary key,
              task_id text,
              markdown_content text,
              created_at double precision,
              meta_json text
            )
            """
        )
        cur.execute(
            """
            create table if not exists rendered_outputs(
              id text primary key,
              task_id text,
              file_path text,
              created_at double precision
            )
            """
        )
    else:
        cur.execute(
            """
            create table if not exists document_tasks(
              id text primary key,
              title text,
              document_type text,
              user_prompt text,
              status text,
              created_at real,
              updated_at real,
              output_dir text,
              meta_json text
            )
            """
        )
        cur.execute(
            """
            create table if not exists document_outlines(
              id text primary key,
              task_id text,
              outline_markdown text,
              outline_tree_json text,
              created_at real
            )
            """
        )
        cur.execute(
            """
            create table if not exists outline_sections(
              id text primary key,
              outline_id text,
              task_id text,
              level integer,
              title text,
              parent_title text,
              order_index integer
            )
            """
        )
        cur.execute(
            """
            create table if not exists document_contents(
              id text primary key,
              task_id text,
              markdown_content text,
              created_at real,
              meta_json text
            )
            """
        )
        cur.execute(
            """
            create table if not exists rendered_outputs(
              id text primary key,
              task_id text,
              file_path text,
              created_at real
            )
            """
        )
    conn.commit()


def create_task(conn, *, task_id: str, title: str, document_type: str, user_prompt: str, status: str, output_dir: str, meta: Optional[dict] = None, now_ts: float = 0.0) -> None:
    import time

    ts = float(now_ts or time.time())
    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    cur = conn.cursor()
    if _dialect() == "postgres":
        cur.execute(
            "insert into document_tasks(id,title,document_type,user_prompt,status,created_at,updated_at,output_dir,meta_json) values(%s,%s,%s,%s,%s,%s,%s,%s,%s) on conflict (id) do update set updated_at=excluded.updated_at",
            (task_id, title, document_type, user_prompt, status, ts, ts, output_dir, meta_json),
        )
    else:
        cur.execute(
            "insert or replace into document_tasks(id,title,document_type,user_prompt,status,created_at,updated_at,output_dir,meta_json) values(?,?,?,?,?,?,?,?,?)",
            (task_id, title, document_type, user_prompt, status, ts, ts, output_dir, meta_json),
        )
    conn.commit()


def update_task_status(conn, task_id: str, status: str) -> None:
    import time

    ts = float(time.time())
    cur = conn.cursor()
    if _dialect() == "postgres":
        cur.execute("update document_tasks set status=%s, updated_at=%s where id=%s", (status, ts, task_id))
    else:
        cur.execute("update document_tasks set status=?, updated_at=? where id=?", (status, ts, task_id))
    conn.commit()


def update_task_title(conn, task_id: str, title: str) -> None:
    import time

    ts = float(time.time())
    cur = conn.cursor()
    if _dialect() == "postgres":
        cur.execute("update document_tasks set title=%s, updated_at=%s where id=%s", (title, ts, task_id))
    else:
        cur.execute("update document_tasks set title=?, updated_at=? where id=?", (title, ts, task_id))
    conn.commit()


def save_outline(conn, *, task_id: str, outline_markdown: str, outline_sections: list[dict]) -> str:
    import time, uuid

    oid = uuid.uuid4().hex[:12]
    ts = float(time.time())
    tree_json = json.dumps(outline_sections or [], ensure_ascii=False)
    cur = conn.cursor()
    if _dialect() == "postgres":
        cur.execute(
            "insert into document_outlines(id,task_id,outline_markdown,outline_tree_json,created_at) values(%s,%s,%s,%s,%s)",
            (oid, task_id, outline_markdown, tree_json, ts),
        )
        for idx, sec in enumerate(outline_sections or []):
            sid = uuid.uuid4().hex[:12]
            cur.execute(
                "insert into outline_sections(id,outline_id,task_id,level,title,parent_title,order_index) values(%s,%s,%s,%s,%s,%s,%s)",
                (sid, oid, task_id, int(sec.get("level") or 0), str(sec.get("title") or ""), "", int(idx)),
            )
    else:
        cur.execute(
            "insert into document_outlines(id,task_id,outline_markdown,outline_tree_json,created_at) values(?,?,?,?,?)",
            (oid, task_id, outline_markdown, tree_json, ts),
        )
        for idx, sec in enumerate(outline_sections or []):
            sid = uuid.uuid4().hex[:12]
            cur.execute(
                "insert into outline_sections(id,outline_id,task_id,level,title,parent_title,order_index) values(?,?,?,?,?,?,?)",
                (sid, oid, task_id, int(sec.get("level") or 0), str(sec.get("title") or ""), "", int(idx)),
            )
    conn.commit()
    return oid


def save_content(conn, *, task_id: str, markdown_content: str, meta: Optional[dict] = None) -> str:
    import time, uuid

    cid = uuid.uuid4().hex[:12]
    ts = float(time.time())
    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    cur = conn.cursor()
    if _dialect() == "postgres":
        cur.execute(
            "insert into document_contents(id,task_id,markdown_content,created_at,meta_json) values(%s,%s,%s,%s,%s)",
            (cid, task_id, markdown_content, ts, meta_json),
        )
    else:
        cur.execute(
            "insert into document_contents(id,task_id,markdown_content,created_at,meta_json) values(?,?,?,?,?)",
            (cid, task_id, markdown_content, ts, meta_json),
        )
    conn.commit()
    return cid


def save_rendered_outputs(conn, *, task_id: str, outputs: list[str]) -> None:
    import time, uuid

    ts = float(time.time())
    cur = conn.cursor()
    for op in outputs or []:
        rid = uuid.uuid4().hex[:12]
        if _dialect() == "postgres":
            cur.execute("insert into rendered_outputs(id,task_id,file_path,created_at) values(%s,%s,%s,%s)", (rid, task_id, op, ts))
        else:
            cur.execute("insert into rendered_outputs(id,task_id,file_path,created_at) values(?,?,?,?)", (rid, task_id, op, ts))
    conn.commit()


def task_exists(conn, task_id: str) -> bool:
    cur = conn.cursor()
    try:
        if _dialect() == "postgres":
            cur.execute("select 1 from document_tasks where id=%s limit 1", (task_id,))
        else:
            cur.execute("select 1 from document_tasks where id=? limit 1", (task_id,))
        row = cur.fetchone()
        return bool(row)
    except Exception:
        return False


def get_latest_outline_markdown(conn, task_id: str) -> str:
    cur = conn.cursor()
    try:
        if _dialect() == "postgres":
            cur.execute("select outline_markdown from document_outlines where task_id=%s order by created_at desc limit 1", (task_id,))
        else:
            cur.execute("select outline_markdown from document_outlines where task_id=? order by created_at desc limit 1", (task_id,))
        row = cur.fetchone()
        if not row:
            return ""
        return str(row[0] or "")
    except Exception:
        return ""


def get_latest_content_markdown(conn, task_id: str) -> str:
    cur = conn.cursor()
    try:
        if _dialect() == "postgres":
            cur.execute("select markdown_content from document_contents where task_id=%s order by created_at desc limit 1", (task_id,))
        else:
            cur.execute("select markdown_content from document_contents where task_id=? order by created_at desc limit 1", (task_id,))
        row = cur.fetchone()
        if not row:
            return ""
        return str(row[0] or "")
    except Exception:
        return ""
