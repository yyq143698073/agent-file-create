import json
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
import re

import numpy as np

from agent_file_create.config import DB_URL, KB_DB_PATH, KB_DB_URL, KB_HNSW_EF_SEARCH, KB_INDEX_TYPE, KB_IVFFLAT_PROBES


def _cosine_batch(query_vec: list[float], embeddings: list[list[float]]) -> np.ndarray:
    """Compute cosine similarity between *query_vec* and each embedding in *embeddings*.

    Uses numpy for ~50-100x speedup over pure-Python loops.
    Returns a 1-D array of scores aligned with *embeddings*.
    """
    if not embeddings:
        return np.array([], dtype=np.float64)
    q = np.asarray(query_vec, dtype=np.float64)
    # Stack all embeddings into a 2-D array (n_embeddings, dim)
    M = np.asarray(embeddings, dtype=np.float64)  # (N, D)
    # Cosine = dot(q, M[i]) / (||q|| * ||M[i]||)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return np.zeros(len(embeddings), dtype=np.float64)
    m_norms = np.linalg.norm(M, axis=1)  # (N,)
    # Avoid division by zero for zero-vectors
    m_norms = np.where(m_norms == 0, 1.0, m_norms)
    scores = np.dot(M, q) / (m_norms * q_norm)
    return scores.astype(np.float64)


def _cosine(a: list[float], b: list[float]) -> float:
    """Single-pair cosine similarity — kept for backward compatibility."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(dot) / float(math.sqrt(na) * math.sqrt(nb))


def _safe_json_obj(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        obj = json.loads(str(raw))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"


@dataclass(frozen=True)
class Hit:
    kb: str
    doc_id: str
    chunk_id: str
    chunk_index: int
    section_path: str
    content: str
    score: float
    meta: dict
    parent_chunk_id: str = ""


class SQLiteVectorStore:
    def register_kb(self, kb: str) -> None:
        """Register a new (empty) KB so it appears in list_kb()."""
        import time as _time
        conn = self._conn()
        try:
            ts = float(_time.time())
            cur = conn.cursor()
            cur.execute(
                "insert or ignore into kb_docs(id,kb,title,source,meta_json,updated_at) "
                "values(?,?,?,?,?,?)",
                (f"__registry__{kb}", kb, kb, "", "{}", ts),
            )
            conn.commit()
        finally:
            conn.close()

    def __init__(self, *, db_path: str | None = None) -> None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        p = str(db_path or KB_DB_PATH or "result/kb.db").strip()
        self.db_path = str((base_dir / p).resolve()) if not Path(p).is_absolute() else p
        # Per-instance thread-local for connection reuse — must be set before
        # _init_schema() which calls _conn().
        self._tlocal = threading.local()
        self._ensure_dir()
        self._init_schema()

    def _ensure_dir(self) -> None:
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _conn(self):
        """Return a thread-local SQLite connection with WAL mode enabled.

        Reuses the connection across calls on the same thread.  If a legacy
        caller closes the connection, it is transparently re-created on the
        next access.
        """
        conn = getattr(self._tlocal, 'conn', None)
        if conn is not None:
            try:
                conn.execute("SELECT 1")
            except sqlite3.ProgrammingError:
                conn = None  # was closed externally → recreate
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-8000")  # 8 MB page cache
            self._tlocal.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                create table if not exists kb_chunks(
                  id text primary key,
                  kb text,
                  doc_id text,
                  chunk_index integer,
                  section_path text,
                  content text,
                  embedding_json text,
                  meta_json text,
                  parent_chunk_id text,
                  created_at real
                )
                """
            )
            try:
                cur.execute("alter table kb_chunks add column parent_chunk_id text")
            except Exception:
                pass
            cur.execute("create index if not exists idx_kb_chunks_kb on kb_chunks(kb)")
            cur.execute("create index if not exists idx_kb_chunks_doc on kb_chunks(kb, doc_id)")
            cur.execute("create index if not exists idx_kb_chunks_section on kb_chunks(kb, section_path)")
            cur.execute(
                """
                create table if not exists kb_docs(
                  id text primary key,
                  kb text,
                  title text,
                  source text,
                  meta_json text,
                  updated_at real
                )
                """
            )
            cur.execute("create index if not exists idx_kb_docs_kb on kb_docs(kb)")
            conn.commit()
        finally:
            conn.close()

    def list_kb(self) -> list[str]:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("select distinct kb from kb_docs order by kb")
            rows = cur.fetchall() or []
            return [str(r[0]) for r in rows if r and str(r[0]).strip()]
        finally:
            conn.close()

    def upsert_document(self, *, kb: str, doc_id: str, title: str, source: str, meta: dict | None = None) -> None:
        conn = self._conn()
        try:
            ts = float(time.time())
            cur = conn.cursor()
            cur.execute(
                "insert or replace into kb_docs(id,kb,title,source,meta_json,updated_at) values(?,?,?,?,?,?)",
                (doc_id, kb, title, source, _safe_json(meta or {}), ts),
            )
            conn.commit()
        finally:
            conn.close()

    def get_chunks_with_empty_embedding(self, *, kb: str, doc_id: str | None = None) -> list[dict]:
        """SQLite: return chunks whose embedding_json is empty/null."""
        conn = self._conn()
        try:
            cur = conn.cursor()
            if doc_id:
                cur.execute(
                    "select id,kb,doc_id,chunk_index,section_path,content,embedding_json,meta_json "
                    "from kb_chunks where kb=? and doc_id=? and (embedding_json='[]' or embedding_json is null or embedding_json='')",
                    (kb, str(doc_id)),
                )
            else:
                cur.execute(
                    "select id,kb,doc_id,chunk_index,section_path,content,embedding_json,meta_json "
                    "from kb_chunks where kb=? and (embedding_json='[]' or embedding_json is null or embedding_json='')",
                    (kb,),
                )
            rows = cur.fetchall() or []
            return [
                {"chunk_id": str(r[0] or ""), "kb": str(r[1] or ""), "doc_id": str(r[2] or ""),
                 "chunk_index": int(r[3] or 0), "section_path": str(r[4] or ""),
                 "content": str(r[5] or ""), "embedding_json": str(r[6] or ""),
                 "meta_json": str(r[7] or "")}
                for r in rows
            ]
        finally:
            conn.close()

    def update_chunk_embedding(self, *, kb: str, chunk_id: str, embedding: list[float]) -> None:
        """SQLite: update a single chunk's embedding_json."""
        import json as _json
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("update kb_chunks set embedding_json=? where id=?", (_json.dumps(embedding), str(chunk_id)))
            conn.commit()
        finally:
            conn.close()

    def delete_doc_chunks(self, *, kb: str, doc_id: str) -> None:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("delete from kb_chunks where kb=? and doc_id=?", (kb, doc_id))
            conn.commit()
        finally:
            conn.close()

    def delete_document(self, *, kb: str, doc_id: str) -> None:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("delete from kb_docs where kb=? and id=?", (kb, doc_id))
            cur.execute("delete from kb_chunks where kb=? and doc_id=?", (kb, doc_id))
            conn.commit()
        finally:
            conn.close()

    def delete_kb(self, *, kb: str) -> None:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("delete from kb_docs where kb=?", (kb,))
            cur.execute("delete from kb_chunks where kb=?", (kb,))
            conn.commit()
        finally:
            conn.close()

    def get_chunks_by_doc_id(self, *, kb: str, doc_id: str) -> list[Hit]:
        """Fetch all chunks for a document by doc_id (no embedding needed)."""
        kb = str(kb or "").strip() or "default"
        did = str(doc_id or "").strip()
        if not did:
            return []
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "select id,kb,doc_id,chunk_index,section_path,content,meta_json from kb_chunks where kb=? and doc_id=? order by chunk_index",
                (kb, did),
            )
            rows = cur.fetchall() or []
        finally:
            conn.close()
        hits: list[Hit] = []
        for r in rows:
            meta = _safe_json_obj(r[6])
            hits.append(Hit(
                kb=str(r[1] or ""), doc_id=str(r[2] or ""), chunk_id=str(r[0] or ""),
                chunk_index=int(r[3] or 0), section_path=str(r[4] or ""),
                content=str(r[5] or ""), score=0.0, meta=meta or {},
            ))
        return hits

    def get_parent_context(self, *, kb: str, parent_chunk_id: str) -> list[Hit]:
        """Fetch all chunks sharing the same parent_chunk_id.

        Used for parent-document backtracking: when a child chunk is retrieved,
        expand to the full parent paragraph for richer LLM context.
        """
        kb = str(kb or "").strip() or "default"
        pid = str(parent_chunk_id or "").strip()
        if not pid:
            return []
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "select id,kb,doc_id,chunk_index,section_path,content,meta_json from kb_chunks where kb=? and parent_chunk_id=? order by chunk_index",
                (kb, pid),
            )
            rows = cur.fetchall() or []
        finally:
            conn.close()
        hits: list[Hit] = []
        for r in rows:
            meta = _safe_json_obj(r[6])
            hits.append(Hit(
                kb=str(r[1] or ""), doc_id=str(r[2] or ""), chunk_id=str(r[0] or ""),
                chunk_index=int(r[3] or 0), section_path=str(r[4] or ""),
                content=str(r[5] or ""), score=0.0, meta=meta or {},
            ))
        return hits

    def list_docs(self, *, kb: str) -> list[dict]:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "select d.id, d.title, d.source, d.meta_json,"
                " (select count(*) from kb_chunks c where c.kb=d.kb and c.doc_id=d.id) as cnt"
                " from kb_docs d where d.kb=? order by d.updated_at desc",
                (kb,),
            )
            rows = cur.fetchall() or []
            out: list[dict] = []
            for r in rows:
                meta = _safe_json_obj(r[3])
                out.append({
                    "doc_id": str(r[0] or ""),
                    "title": str(r[1] or ""),
                    "source": str(r[2] or ""),
                    "chunk_count": int(r[4] or 0),
                    "doc_type": str((meta or {}).get("doc_type") or ""),
                    "file_ext": str((meta or {}).get("file_ext") or ""),
                })
            return out
        finally:
            conn.close()

    def kb_stats(self, *, kb: str) -> dict:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("select count(distinct id) from kb_docs where kb=? and id not like '__registry__%'", (kb,))
            docs = int((cur.fetchone() or [0])[0] or 0)
            cur.execute("select count(*) from kb_chunks where kb=?", (kb,))
            chunks = int((cur.fetchone() or [0])[0] or 0)
            return {"kb": kb, "doc_count": docs, "chunk_count": chunks}
        finally:
            conn.close()

    def upsert_chunks(
        self,
        *,
        kb: str,
        doc_id: str,
        chunks: Iterable[dict],
    ) -> int:
        kb = str(kb or "").strip() or "default"
        doc_id = str(doc_id or "").strip() or "doc"
        items = list(chunks or [])
        if not items:
            return 0
        conn = self._conn()
        ts = float(time.time())
        rows: list[tuple] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            cid = str(it.get("chunk_id") or "").strip()
            if not cid:
                continue
            content = str(it.get("content") or "").strip()
            content = content.replace("\x00", "")
            emb = it.get("embedding")
            emb_json = _safe_json(emb if isinstance(emb, list) else [])
            meta_json = _safe_json(it.get("meta") if isinstance(it.get("meta"), dict) else {})
            pid = str(it.get("parent_chunk_id") or "").strip()
            section_path = str(it.get("section_path") or "").replace("\x00", "")
            rows.append((
                cid, kb, doc_id,
                int(it.get("chunk_index") or 0),
                section_path, content, emb_json, meta_json, pid, ts,
            ))
        if rows:
            conn.executemany(
                "insert or replace into kb_chunks(id,kb,doc_id,chunk_index,section_path,content,embedding_json,meta_json,parent_chunk_id,created_at) values(?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
        return len(items)

    def similarity_search(
        self,
        *,
        kb: str,
        query_embedding: list[float],
        top_k: int = 8,
        doc_id: Optional[str] = None,
        filters: Optional[dict] = None,
    ) -> list[Hit]:
        kb = str(kb or "").strip() or "default"
        top_k = max(1, int(top_k or 0))
        conn = self._conn()
        if doc_id:
            rows = conn.execute(
                "select id,kb,doc_id,chunk_index,section_path,content,embedding_json,meta_json from kb_chunks where kb=? and doc_id=?",
                (kb, str(doc_id)),
            ).fetchall() or []
        else:
            rows = conn.execute(
                "select id,kb,doc_id,chunk_index,section_path,content,embedding_json,meta_json from kb_chunks where kb=?",
                (kb,),
            ).fetchall() or []

        # Apply post-query filters
        if filters:
            _json_mod = json
            filtered_rows = []
            for r in rows:
                try:
                    meta = _json_mod.loads(str(r[7] or "{}")) if r[7] else {}
                except Exception:
                    meta = {}
                ok = True
                if filters.get("doc_type") and str(meta.get("doc_type") or "") != str(filters["doc_type"]):
                    ok = False
                if filters.get("source") and str(meta.get("source") or "") != str(filters["source"]):
                    ok = False
                if filters.get("section_path"):
                    sp = str(r[5] or "")
                    if str(filters["section_path"]) not in sp:
                        ok = False
                if ok:
                    filtered_rows.append(r)
            rows = filtered_rows

        if not rows:
            return []

        # ── Fast path: numpy batch cosine ────────────────────────────
        # Deserialize all embeddings once into a list, compute all scores
        # in a single numpy call, then argsort + slice top_k.
        embeddings: list[list[float]] = []
        row_meta: list[tuple] = []  # (r[0]..r[5], meta_dict)
        for r in rows:
            try:
                emb = json.loads(r[6] or "[]")
            except Exception:
                emb = []
            if not isinstance(emb, list) or not emb:
                continue
            vec = []
            ok_vec = True
            for v in emb:
                try:
                    vec.append(float(v))
                except Exception:
                    ok_vec = False
                    break
            if not ok_vec or not vec:
                continue
            embeddings.append(vec)
            try:
                meta = json.loads(r[7] or "{}")
                meta = meta if isinstance(meta, dict) else {}
            except Exception:
                meta = {}
            row_meta.append((r[0], r[1], r[2], r[3], r[4], r[5], meta))

        if not embeddings:
            return []

        scores = _cosine_batch(query_embedding, embeddings)
        if top_k >= len(scores):
            top_indices = list(range(len(scores)))
        else:
            # Use argpartition for O(N) top-k instead of full sort
            top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        hits: list[Hit] = []
        for idx in top_indices:
            score = float(scores[idx])
            r = row_meta[idx]
            hits.append(Hit(
                kb=str(r[1] or ""),
                doc_id=str(r[2] or ""),
                chunk_id=str(r[0] or ""),
                chunk_index=int(r[3] or 0),
                section_path=str(r[4] or ""),
                content=str(r[5] or ""),
                score=score,
                meta=r[6],
            ))
        return hits

    def lexical_search(
        self,
        *,
        kb: str,
        query: str,
        top_k: int = 20,
        filters: Optional[dict] = None,
    ) -> list[Hit]:
        kb = str(kb or "").strip() or "default"
        q = str(query or "").strip()
        if not q:
            return []
        xs = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", q)
        terms: list[str] = []
        seen = set()
        for x in xs:
            k = x.lower()
            if k in seen:
                continue
            seen.add(k)
            terms.append(k)
            if len(terms) >= 12:
                break
        if not terms:
            return []

        where = ["kb=?"]
        params: list[Any] = [kb]
        f = filters if isinstance(filters, dict) else {}
        if isinstance(f.get("doc_id"), str) and f.get("doc_id").strip():
            where.append("doc_id=?")
            params.append(f.get("doc_id").strip())
        if isinstance(f.get("section_path"), str) and f.get("section_path").strip():
            where.append("section_path=?")
            params.append(f.get("section_path").strip())

        like = []
        for t in terms:
            like.append("content like ?")
            params.append("%" + t + "%")
        where.append("(" + " or ".join(like) + ")")
        sql = (
            "select id,kb,doc_id,chunk_index,section_path,content,meta_json from kb_chunks where "
            + " and ".join(where)
            + " limit ?"
        )
        params.append(int(max(50, int(top_k or 0) * 30)))

        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            rows = cur.fetchall() or []
        finally:
            conn.close()

        hits: list[Hit] = []
        for r in rows:
            content = str(r[5] or "")
            if not content.strip():
                continue
            try:
                meta = json.loads(r[6] or "{}")
            except Exception:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            if isinstance(f.get("doc_type"), str) and f.get("doc_type").strip():
                if str((meta or {}).get("doc_type") or "").strip() != f.get("doc_type").strip():
                    continue
            if isinstance(f.get("source"), str) and f.get("source").strip():
                if str((meta or {}).get("source") or "").strip() != f.get("source").strip():
                    continue
            cnt = 0
            low = content.lower()
            for t in terms:
                if t and (t in low):
                    cnt += 1
            score = float(cnt) / float(len(terms) or 1)
            hits.append(
                Hit(
                    kb=str(r[1] or ""),
                    doc_id=str(r[2] or ""),
                    chunk_id=str(r[0] or ""),
                    chunk_index=int(r[3] or 0),
                    section_path=str(r[4] or ""),
                    content=content,
                    score=score,
                    meta=meta,
                )
            )

        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[: max(1, int(top_k or 0))]


def _is_postgres_url(url: str) -> bool:
    u = str(url or "").strip().lower()
    return u.startswith("postgresql://") or u.startswith("postgres://")


def _vec_literal(vec: list[float]) -> Optional[str]:
    if not isinstance(vec, list) or not vec:
        return None  # NULL in SQL — rows with empty embeddings skip vector search
    parts: list[str] = []
    for v in vec:
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError("vector_contains_non_finite")
        parts.append(str(fv))
    return "[" + ",".join(parts) + "]"


class PostgresVectorStore:
    def register_kb(self, kb: str) -> None:
        """Register a new (empty) KB so it appears in list_kb()."""
        import json as _json
        from datetime import datetime, timezone
        conn = self._conn()
        try:
            now = datetime.now(timezone.utc)
            cur = conn.cursor()
            cur.execute(
                "insert into kb_docs(id,kb,title,source,doc_type,meta,updated_at) "
                "values(%s,%s,%s,%s,%s,%s,%s) on conflict do nothing",
                (f"__registry__{kb}", kb, kb, "", "", _json.dumps({}), now),
            )
            conn.commit()
        finally:
            conn.close()

    def __init__(self, *, db_url: str | None = None, index_type: str | None = None) -> None:
        self.db_url = str(db_url or KB_DB_URL or DB_URL or "").strip()
        if not _is_postgres_url(self.db_url):
            raise RuntimeError("DB_URL 不是 PostgreSQL 连接串")
        self.index_type = str(index_type or KB_INDEX_TYPE or "hnsw").strip().lower() or "hnsw"
        self._init_schema()

    def _conn(self):
        try:
            import psycopg2
        except Exception as e:
            raise RuntimeError("缺少 psycopg2 依赖，请 pip install psycopg2-binary") from e
        return psycopg2.connect(self.db_url)

    def _exec(self, sql: str, params: tuple | None = None) -> None:
        conn = self._conn()
        try:
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(sql, params or None)
        finally:
            conn.close()

    def _init_schema(self) -> None:
        try:
            self._exec("create extension if not exists vector")
        except Exception:
            pass

        self._exec(
            """
            create table if not exists kb_docs(
              kb text not null,
              id text not null,
              title text,
              source text,
              doc_type text,
              meta jsonb,
              updated_at timestamptz,
              primary key (kb, id)
            )
            """
        )
        self._exec("create index if not exists idx_kb_docs_kb on kb_docs(kb)")

        self._exec(
            """
            create table if not exists kb_chunks(
              id text primary key,
              kb text not null,
              doc_id text not null,
              chunk_index integer,
              section_path text,
              content text,
              embedding vector,
              source text,
              title text,
              doc_type text,
              meta jsonb,
              parent_chunk_id text,
              created_at timestamptz default now()
            )
            """
        )
        try:
            self._exec("alter table kb_chunks add column if not exists parent_chunk_id text")
        except Exception:
            pass
        self._exec("create index if not exists idx_kb_chunks_kb on kb_chunks(kb)")
        self._exec("create index if not exists idx_kb_chunks_doc on kb_chunks(kb, doc_id)")
        self._exec("create index if not exists idx_kb_chunks_section on kb_chunks(kb, section_path)")
        self._exec("create index if not exists idx_kb_chunks_source on kb_chunks(kb, source)")

        idx = self.index_type
        if idx in {"hnsw", "ivfflat", "both"}:
            try:
                if idx in {"hnsw", "both"}:
                    self._exec(
                        "create index if not exists idx_kb_chunks_emb_hnsw on kb_chunks using hnsw (embedding vector_cosine_ops) with (m=16, ef_construction=64)"
                    )
            except Exception:
                pass
            try:
                if idx in {"ivfflat", "both"}:
                    self._exec(
                        "create index if not exists idx_kb_chunks_emb_ivfflat on kb_chunks using ivfflat (embedding vector_cosine_ops) with (lists=100)"
                    )
                    self._exec("analyze kb_chunks")
            except Exception:
                pass

        # Clean up empty-vector rows from previous bug where _vec_literal returned '[]'
        try:
            self._exec("update kb_chunks set embedding = null where embedding = '[]'::vector")
        except Exception:
            pass

    def list_kb(self) -> list[str]:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("select distinct kb from kb_docs order by kb")
            rows = cur.fetchall() or []
            return [str(r[0]) for r in rows if r and str(r[0]).strip()]
        finally:
            conn.close()

    def upsert_document(self, *, kb: str, doc_id: str, title: str, source: str, meta: dict | None = None) -> None:
        kb = str(kb or "").strip() or "default"
        did = str(doc_id or "").strip() or "doc"
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                insert into kb_docs(kb,id,title,source,doc_type,meta,updated_at)
                values(%s,%s,%s,%s,%s,%s::jsonb,now())
                on conflict (kb,id) do update set
                  title=excluded.title,
                  source=excluded.source,
                  doc_type=excluded.doc_type,
                  meta=excluded.meta,
                  updated_at=excluded.updated_at
                """,
                (kb, did, str(title or ""), str(source or ""), str((meta or {}).get("doc_type") or ""), json.dumps(meta or {}, ensure_ascii=False)),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_doc_chunks(self, *, kb: str, doc_id: str) -> None:
        kb = str(kb or "").strip() or "default"
        did = str(doc_id or "").strip() or "doc"
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("delete from kb_chunks where kb=%s and doc_id=%s", (kb, did))
            conn.commit()
        finally:
            conn.close()

    def get_chunks_with_empty_embedding(self, *, kb: str, doc_id: str | None = None) -> list[dict]:
        """Postgres: return chunks whose embedding is null or zero-dimensional."""
        conn = self._conn()
        try:
            cur = conn.cursor()
            if doc_id:
                cur.execute(
                    "select id,kb,doc_id,chunk_index,section_path,content,meta "
                    "from kb_chunks where kb=%s and doc_id=%s and (embedding is null or vector_dims(embedding) = 0)",
                    (kb, str(doc_id)),
                )
            else:
                cur.execute(
                    "select id,kb,doc_id,chunk_index,section_path,content,meta "
                    "from kb_chunks where kb=%s and (embedding is null or vector_dims(embedding) = 0)",
                    (kb,),
                )
            rows = cur.fetchall() or []
            return [
                {"chunk_id": str(r[0] or ""), "kb": str(r[1] or ""), "doc_id": str(r[2] or ""),
                 "chunk_index": int(r[3] or 0), "section_path": str(r[4] or ""),
                 "content": str(r[5] or ""), "meta": r[6] if r[6] else {}}
                for r in rows
            ]
        finally:
            conn.close()

    def update_chunk_embedding(self, *, kb: str, chunk_id: str, embedding: list[float]) -> None:
        """Postgres: update a single chunk's embedding vector."""
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("update kb_chunks set embedding=%s::vector where id=%s", (embedding, str(chunk_id)))
            conn.commit()
        finally:
            conn.close()

    def delete_document(self, *, kb: str, doc_id: str) -> None:
        kb = str(kb or "").strip() or "default"
        did = str(doc_id or "").strip() or "doc"
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("delete from kb_docs where kb=%s and id=%s", (kb, did))
            cur.execute("delete from kb_chunks where kb=%s and doc_id=%s", (kb, did))
            conn.commit()
        finally:
            conn.close()

    def delete_kb(self, *, kb: str) -> None:
        kb = str(kb or "").strip() or "default"
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("delete from kb_docs where kb=%s", (kb,))
            cur.execute("delete from kb_chunks where kb=%s", (kb,))
            conn.commit()
        finally:
            conn.close()

    def get_chunks_by_doc_id(self, *, kb: str, doc_id: str) -> list[Hit]:
        """Fetch all chunks for a document by doc_id (no embedding needed)."""
        kb = str(kb or "").strip() or "default"
        did = str(doc_id or "").strip()
        if not did:
            return []
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "select id,kb,doc_id,chunk_index,section_path,content,meta from kb_chunks where kb=%s and doc_id=%s order by chunk_index",
                (kb, did),
            )
            rows = cur.fetchall() or []
        finally:
            conn.close()
        hits: list[Hit] = []
        for r in rows:
            meta = r[6] if isinstance(r[6], dict) else {}
            hits.append(Hit(
                kb=str(r[1] or ""), doc_id=str(r[2] or ""), chunk_id=str(r[0] or ""),
                chunk_index=int(r[3] or 0), section_path=str(r[4] or ""),
                content=str(r[5] or ""), score=0.0, meta=meta,
            ))
        return hits

    def get_parent_context(self, *, kb: str, parent_chunk_id: str) -> list[Hit]:
        """Fetch all chunks sharing the same parent_chunk_id (Postgres version)."""
        kb = str(kb or "").strip() or "default"
        pid = str(parent_chunk_id or "").strip()
        if not pid:
            return []
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "select id,kb,doc_id,chunk_index,section_path,content,meta from kb_chunks where kb=%s and parent_chunk_id=%s order by chunk_index",
                (kb, pid),
            )
            rows = cur.fetchall() or []
        finally:
            conn.close()
        hits: list[Hit] = []
        for r in rows:
            meta = r[6] if isinstance(r[6], dict) else {}
            hits.append(Hit(
                kb=str(r[1] or ""), doc_id=str(r[2] or ""), chunk_id=str(r[0] or ""),
                chunk_index=int(r[3] or 0), section_path=str(r[4] or ""),
                content=str(r[5] or ""), score=0.0, meta=meta,
            ))
        return hits

    def list_docs(self, *, kb: str) -> list[dict]:
        kb = str(kb or "").strip() or "default"
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "select d.id, d.title, d.source, d.doc_type, d.meta,"
                " (select count(*) from kb_chunks c where c.kb=d.kb and c.doc_id=d.id) as cnt"
                " from kb_docs d where d.kb=%s order by d.updated_at desc",
                (kb,),
            )
            rows = cur.fetchall() or []
            out: list[dict] = []
            for r in rows:
                doc_type = str(r[3] or "")
                if not doc_type and isinstance(r[4], dict):
                    doc_type = str((r[4]).get("doc_type") or "")
                out.append({
                    "doc_id": str(r[0] or ""),
                    "title": str(r[1] or ""),
                    "source": str(r[2] or ""),
                    "chunk_count": int(r[5] or 0),
                    "doc_type": doc_type,
                    "file_ext": str((r[4] or {}).get("file_ext") or "") if isinstance(r[4], dict) else "",
                })
            return out
        finally:
            conn.close()

    def kb_stats(self, *, kb: str) -> dict:
        kb = str(kb or "").strip() or "default"
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("select count(distinct id) from kb_docs where kb=%s and id not like '__registry__%%'", (kb,))
            docs = int((cur.fetchone() or [0])[0] or 0)
            cur.execute("select count(*) from kb_chunks where kb=%s", (kb,))
            chunks = int((cur.fetchone() or [0])[0] or 0)
            return {"kb": kb, "doc_count": docs, "chunk_count": chunks}
        finally:
            conn.close()

    def upsert_chunks(
        self,
        *,
        kb: str,
        doc_id: str,
        chunks: Iterable[dict],
    ) -> int:
        kb = str(kb or "").strip() or "default"
        doc_id = str(doc_id or "").strip() or "doc"
        items = list(chunks or [])
        if not items:
            return 0
        conn = self._conn()
        try:
            cur = conn.cursor()
            for it in items:
                if not isinstance(it, dict):
                    continue
                cid = str(it.get("chunk_id") or "").strip()
                if not cid:
                    continue
                content = str(it.get("content") or "").strip()
                content = content.replace("\x00", "")  # strip NUL bytes (OCR artefacts)
                emb = it.get("embedding")
                vec = _vec_literal(emb if isinstance(emb, list) else [])
                meta = it.get("meta") if isinstance(it.get("meta"), dict) else {}
                section_path = str(it.get("section_path") or "").replace("\x00", "")
                cur.execute(
                    """
                    insert into kb_chunks(id,kb,doc_id,chunk_index,section_path,content,embedding,source,title,doc_type,meta,created_at)
                    values(%s,%s,%s,%s,%s,%s,%s::vector,%s,%s,%s,%s::jsonb,now())
                    on conflict (id) do update set
                      kb=excluded.kb,
                      doc_id=excluded.doc_id,
                      chunk_index=excluded.chunk_index,
                      section_path=excluded.section_path,
                      content=excluded.content,
                      embedding=excluded.embedding,
                      source=excluded.source,
                      title=excluded.title,
                      doc_type=excluded.doc_type,
                      meta=excluded.meta
                    """,
                    (
                        cid,
                        kb,
                        doc_id,
                        int(it.get("chunk_index") or 0),
                        section_path,
                        content,
                        vec,
                        str((meta or {}).get("source") or ""),
                        str((meta or {}).get("title") or ""),
                        str((meta or {}).get("doc_type") or ""),
                        json.dumps(meta or {}, ensure_ascii=False),
                    ),
                )
            conn.commit()
            return len(items)
        finally:
            conn.close()

    def similarity_search(
        self,
        *,
        kb: str,
        query_embedding: list[float],
        top_k: int = 8,
        doc_id: Optional[str] = None,
        filters: Optional[dict] = None,
    ) -> list[Hit]:
        kb = str(kb or "").strip() or "default"
        top_k = max(1, int(top_k or 0))
        qv = _vec_literal(query_embedding)
        if qv is None:
            return []  # empty query embedding → skip vector search
        where = ["kb=%s", "embedding is not null", "vector_dims(embedding) > 0"]
        params: list[Any] = [kb]
        if doc_id:
            where.append("doc_id=%s")
            params.append(str(doc_id))
        f = filters if isinstance(filters, dict) else {}
        if isinstance(f.get("doc_id"), str) and f.get("doc_id").strip():
            where.append("doc_id=%s")
            params.append(f.get("doc_id").strip())
        if isinstance(f.get("source"), str) and f.get("source").strip():
            where.append("source=%s")
            params.append(f.get("source").strip())
        if isinstance(f.get("doc_type"), str) and f.get("doc_type").strip():
            where.append("doc_type=%s")
            params.append(f.get("doc_type").strip())
        if isinstance(f.get("section_path"), str) and f.get("section_path").strip():
            where.append("section_path=%s")
            params.append(f.get("section_path").strip())

        sql = (
            "select id,kb,doc_id,chunk_index,section_path,content,meta,(embedding <=> %s::vector) as dist "
            + "from kb_chunks where "
            + " and ".join(where)
            + " order by embedding <=> %s::vector asc limit %s"
        )
        args = [qv, *params, qv, int(top_k)]
        conn = self._conn()
        try:
            cur = conn.cursor()
            try:
                if self.index_type in {"hnsw", "both"}:
                    # Adaptive ef_search: scale with requested top_k
                    # Rule of thumb: ef_search ≥ top_k; higher values trade speed for recall
                    if int(top_k) <= 10:
                        ef = 30
                    elif int(top_k) <= 50:
                        ef = int(KB_HNSW_EF_SEARCH or 64)
                    elif int(top_k) <= 160:
                        ef = 120
                    else:
                        ef = max(200, int(top_k) * 3 // 2)
                    cur.execute("set local hnsw.ef_search = %s", (ef,))
            except Exception:
                pass
            try:
                if self.index_type in {"ivfflat", "both"}:
                    cur.execute("set local ivfflat.probes = %s", (int(KB_IVFFLAT_PROBES or 10),))
            except Exception:
                pass
            cur.execute(sql, tuple(args))
            rows = cur.fetchall() or []
        finally:
            conn.close()

        hits: list[Hit] = []
        for r in rows:
            dist = float(r[7] or 0.0)
            score = 1.0 - dist
            meta = r[6] if isinstance(r[6], dict) else {}
            if not isinstance(meta, dict):
                try:
                    meta = json.loads(str(r[6] or "{}"))
                    meta = meta if isinstance(meta, dict) else {}
                except Exception:
                    meta = {}
            hits.append(
                Hit(
                    kb=str(r[1] or ""),
                    doc_id=str(r[2] or ""),
                    chunk_id=str(r[0] or ""),
                    chunk_index=int(r[3] or 0),
                    section_path=str(r[4] or ""),
                    content=str(r[5] or ""),
                    score=float(score),
                    meta=meta,
                )
            )
        return hits

    def lexical_search(
        self,
        *,
        kb: str,
        query: str,
        top_k: int = 20,
        filters: Optional[dict] = None,
    ) -> list[Hit]:
        kb = str(kb or "").strip() or "default"
        q = str(query or "").strip()
        if not q:
            return []
        xs = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", q)
        terms: list[str] = []
        seen = set()
        for x in xs:
            k = x.lower()
            if k in seen:
                continue
            seen.add(k)
            terms.append(k)
            if len(terms) >= 12:
                break
        if not terms:
            return []

        where = ["kb=%s"]
        params: list[Any] = [kb]
        f = filters if isinstance(filters, dict) else {}
        if isinstance(f.get("doc_id"), str) and f.get("doc_id").strip():
            where.append("doc_id=%s")
            params.append(f.get("doc_id").strip())
        if isinstance(f.get("source"), str) and f.get("source").strip():
            where.append("source=%s")
            params.append(f.get("source").strip())
        if isinstance(f.get("doc_type"), str) and f.get("doc_type").strip():
            where.append("doc_type=%s")
            params.append(f.get("doc_type").strip())
        if isinstance(f.get("section_path"), str) and f.get("section_path").strip():
            where.append("section_path=%s")
            params.append(f.get("section_path").strip())

        like = []
        for t in terms:
            like.append("content ilike %s")
            params.append("%" + t + "%")
        where.append("(" + " or ".join(like) + ")")
        sql = (
            "select id,kb,doc_id,chunk_index,section_path,content,meta from kb_chunks where "
            + " and ".join(where)
            + " limit %s"
        )
        params.append(int(max(50, int(top_k or 0) * 30)))

        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            rows = cur.fetchall() or []
        finally:
            conn.close()

        hits: list[Hit] = []
        for r in rows:
            content = str(r[5] or "")
            if not content.strip():
                continue
            meta = r[6] if isinstance(r[6], dict) else {}
            if not isinstance(meta, dict):
                try:
                    meta = json.loads(str(r[6] or "{}"))
                    meta = meta if isinstance(meta, dict) else {}
                except Exception:
                    meta = {}
            cnt = 0
            low = content.lower()
            for t in terms:
                if t and (t in low):
                    cnt += 1
            score = float(cnt) / float(len(terms) or 1)
            hits.append(
                Hit(
                    kb=str(r[1] or ""),
                    doc_id=str(r[2] or ""),
                    chunk_id=str(r[0] or ""),
                    chunk_index=int(r[3] or 0),
                    section_path=str(r[4] or ""),
                    content=content,
                    score=score,
                    meta=meta,
                )
            )

        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[: max(1, int(top_k or 0))]


def default_store():
    url = str(KB_DB_URL or DB_URL or "").strip()
    if _is_postgres_url(url):
        return PostgresVectorStore(db_url=url)
    return SQLiteVectorStore()
