import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_TASK_LOCK = threading.Lock()
_TASK_THREADS: dict[str, threading.Thread] = {}
_TASK_EVENTS: dict[str, dict[str, threading.Event]] = {}


def _get_task_events(task_id: str) -> tuple[threading.Event, threading.Event]:
    tid = str(task_id or "").strip() or "lobby"
    with _TASK_LOCK:
        ev = _TASK_EVENTS.get(tid)
        if not isinstance(ev, dict):
            ev = {}
            _TASK_EVENTS[tid] = ev
        if "pause" not in ev:
            ev["pause"] = threading.Event()
        if "cancel" not in ev:
            ev["cancel"] = threading.Event()
        return ev["pause"], ev["cancel"]


def _set_task_thread(task_id: str, th: threading.Thread | None) -> None:
    tid = str(task_id or "").strip() or "lobby"
    with _TASK_LOCK:
        if th is None:
            _TASK_THREADS.pop(tid, None)
            return
        _TASK_THREADS[tid] = th


def _task_running(task_id: str) -> bool:
    tid = str(task_id or "").strip() or "lobby"
    with _TASK_LOCK:
        th = _TASK_THREADS.get(tid)
    return bool(th and th.is_alive())


class TaskManager:
    _status_locks: dict[str, threading.Lock] = {}
    _status_locks_lock = threading.Lock()

    def __init__(self, result_dir: Optional[Path] = None):
        self._result_dir = result_dir or Path(__file__).resolve().parent.parent.parent / "result"

    @classmethod
    def _get_status_lock(cls, task_id: str) -> threading.Lock:
        with cls._status_locks_lock:
            if task_id not in cls._status_locks:
                cls._status_locks[task_id] = threading.Lock()
            return cls._status_locks[task_id]

    def _status_path(self, task_id: str) -> Path:
        return self._result_dir / str(task_id) / "status.json"

    def _task_meta_path(self, task_id: str) -> Path:
        return self._result_dir / str(task_id) / "task_meta.json"

    def _analysis_results_path(self, task_id: str) -> Path:
        return self._result_dir / str(task_id) / "analysis_results.json"

    def _chat_history_path(self, task_id: str) -> Path:
        return self._result_dir / str(task_id) / "chat_history.json"

    def _chat_summary_path(self, task_id: str) -> Path:
        return self._result_dir / str(task_id) / "chat_summary.txt"

    def _task_upload_files_path(self, task_id: str) -> Path:
        return self._result_dir / str(task_id) / "uploads"

    def _task_templates_path(self, task_id: str) -> Path:
        return self._result_dir / str(task_id) / "template"

    def normalize_task_id(self, task_id: str) -> str:
        tid = str(task_id or "").strip()
        if not tid:
            return ""
        if len(tid) > 64:
            return ""
        if not re.fullmatch(r"[0-9A-Za-z_-]+", tid):
            return ""
        return tid

    def generate_task_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def write_status(
        self,
        task_id: str,
        status: str,
        *,
        stage: str = "",
        message: str = "",
        extra: Optional[dict] = None,
    ) -> None:
        with self._get_status_lock(task_id):
            base = self._result_dir / str(task_id)
            try:
                base.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            payload: dict[str, Any] = {}
            try:
                p = self._status_path(task_id)
                if p.exists():
                    old = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(old, dict):
                        payload.update(old)
            except Exception:
                payload = {}
            payload.update(
                {
                    "task_id": str(task_id),
                    "status": str(status),
                    "stage": str(stage),
                    "message": str(message),
                    "updated_at": float(time.time()),
                }
            )
            if extra:
                payload.update(extra)
            try:
                self._status_path(task_id).write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
            except Exception:
                return

    def read_status(self, task_id: str) -> dict:
        p = self._status_path(task_id)
        if not p.exists():
            return {"task_id": str(task_id), "status": "unknown"}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"task_id": str(task_id), "status": "unknown"}

    def write_task_meta(self, task_id: str, meta: dict) -> None:
        base = self._result_dir / str(task_id)
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        payload = self.read_task_meta(task_id)
        if isinstance(meta, dict):
            payload.update(meta)
        payload["task_id"] = str(task_id)
        payload["updated_at"] = float(time.time())
        try:
            self._task_meta_path(task_id).write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            return

    def read_task_meta(self, task_id: str) -> dict:
        p = self._task_meta_path(task_id)
        if not p.exists():
            return {}
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def write_analysis_results(self, task_id: str, results: list[dict]) -> None:
        base = self._result_dir / str(task_id)
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            self._analysis_results_path(task_id).write_text(
                json.dumps(results or [], ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            return

    def read_analysis_results(self, task_id: str) -> list[dict]:
        p = self._analysis_results_path(task_id)
        if not p.exists():
            return []
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            return obj if isinstance(obj, list) else []
        except Exception:
            return []

    def read_chat_history(self, task_id: str) -> list[dict]:
        p = self._chat_history_path(task_id)
        if not p.exists():
            return []
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                out = []
                for it in obj[-50:]:
                    if not isinstance(it, dict):
                        continue
                    role = str(it.get("role") or "").strip()
                    content = str(it.get("content") or "").strip()
                    if role in {"user", "assistant"} and content:
                        out.append({"role": role, "content": content})
                return out
            return []
        except Exception:
            return []

    def read_chat_summary(self, task_id: str) -> str:
        p = self._chat_summary_path(task_id)
        if not p.exists():
            return ""
        try:
            return (p.read_text(encoding="utf-8") or "").strip()[:2000]
        except Exception:
            return ""

    def write_chat_summary(self, task_id: str, text: str) -> None:
        base = self._result_dir / str(task_id)
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            self._chat_summary_path(task_id).write_text(
                (text or "").strip(), encoding="utf-8"
            )
        except Exception:
            return

    def truncate_chat_history(self, task_id: str, keep_last: int = 8) -> None:
        """Keep only the last N messages, drop older ones (typically after summarization)."""
        old = self.read_chat_history(task_id)
        trimmed = old[-max(1, int(keep_last or 0)):]
        try:
            self._chat_history_path(task_id).write_text(
                json.dumps(trimmed, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            return

    def append_chat_history(self, task_id: str, items: list[dict]) -> None:
        base = self._result_dir / str(task_id)
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        old = self.read_chat_history(task_id)
        for it in items or []:
            if not isinstance(it, dict):
                continue
            role = str(it.get("role") or "").strip()
            content = str(it.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            old.append({"role": role, "content": content[:2000]})
        old = old[-50:]
        try:
            self._chat_history_path(task_id).write_text(
                json.dumps(old, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            return

    def list_task_upload_files(self, task_id: str) -> list[str]:
        base = self._task_upload_files_path(task_id)
        if not base.exists():
            return []
        out: list[str] = []
        for p in sorted([x for x in base.iterdir() if x.is_file()]):
            if p.name.startswith("."):
                continue
            out.append(str(p))
        return out

    def list_task_templates(self, task_id: str) -> list[str]:
        base = self._task_templates_path(task_id)
        if not base.exists():
            return []
        out: list[str] = []
        for p in sorted([x for x in base.iterdir() if x.is_file()]):
            suf = p.suffix.lower().lstrip(".")
            if suf not in {"md", "docx", "pdf"}:
                continue
            out.append(str(p))
        return out

    def collect_downloads(self, task_id: str) -> dict:
        base = self._result_dir / str(task_id)
        out = {"task_id": str(task_id), "files": []}
        if not base.exists():
            return out
        for p in sorted([x for x in base.iterdir() if x.is_file()]):
            if p.name.startswith("~$"):
                continue
            out["files"].append(
                {"name": p.name, "url": f"/result/{task_id}/{p.name}", "size": int(p.stat().st_size)}
            )
        return out

    # ── Version Management ─────────────────────────────────────────────

    def _versions_dir(self, task_id: str) -> Path:
        return self._result_dir / str(task_id) / "versions"

    def _versions_meta_path(self, task_id: str) -> Path:
        return self._versions_dir(task_id) / "versions.json"

    def save_version(self, task_id: str, version_type: str, version_num: int, content: str, feedback: str = "") -> None:
        """Save a version snapshot to result/<task_id>/versions/<type>_v<N>.md"""
        vdir = self._versions_dir(task_id)
        try:
            vdir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        # Save content file
        fname = f"{version_type}_v{version_num}.md"
        try:
            (vdir / fname).write_text(content or "", encoding="utf-8")
        except Exception:
            pass
        # Update versions.json metadata
        meta = self._read_versions_meta(task_id)
        key = f"{version_type}_versions"
        existing = list(meta.get(key) or [])
        # Check if this version already recorded
        found = False
        for item in existing:
            if item.get("version") == version_num:
                item["feedback"] = feedback
                item["ts"] = time.time()
                found = True
                break
        if not found:
            existing.append({
                "version": version_num,
                "feedback": feedback,
                "ts": time.time(),
                "selected": False,
            })
        meta[key] = existing
        try:
            self._versions_meta_path(task_id).write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def _read_versions_meta(self, task_id: str) -> dict:
        p = self._versions_meta_path(task_id)
        if not p.exists():
            return {}
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def list_versions(self, task_id: str, version_type: str) -> list[dict]:
        """List all versions of a type with metadata and content."""
        meta = self._read_versions_meta(task_id)
        key = f"{version_type}_versions"
        items = list(meta.get(key) or [])
        vdir = self._versions_dir(task_id)
        for item in items:
            vnum = item.get("version", 0)
            fname = f"{version_type}_v{vnum}.md"
            fp = vdir / fname
            if fp.exists():
                try:
                    item["content"] = fp.read_text(encoding="utf-8")
                except Exception:
                    item["content"] = ""
            else:
                item["content"] = ""
        return sorted(items, key=lambda x: x.get("version", 0))

    def delete_version(self, task_id: str, version_type: str, version_num: int) -> bool:
        """Delete a single version file and its metadata entry."""
        vdir = self._versions_dir(task_id)
        fname = f"{version_type}_v{version_num}.md"
        fp = vdir / fname
        try:
            if fp.exists():
                fp.unlink()
        except Exception:
            pass
        meta = self._read_versions_meta(task_id)
        key = f"{version_type}_versions"
        items = list(meta.get(key) or [])
        meta[key] = [it for it in items if it.get("version") != version_num]
        try:
            self._versions_meta_path(task_id).write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass
        return True

    def select_version(self, task_id: str, version_type: str, version_num: int) -> bool:
        """Mark a version as selected, and copy its content to the main file."""
        vdir = self._versions_dir(task_id)
        fname = f"{version_type}_v{version_num}.md"
        fp = vdir / fname
        if not fp.exists():
            return False

        # Update metadata
        meta = self._read_versions_meta(task_id)
        key = f"{version_type}_versions"
        items = list(meta.get(key) or [])
        for item in items:
            item["selected"] = (item.get("version") == version_num)
        meta[key] = items
        try:
            self._versions_meta_path(task_id).write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

        # Copy content to main file
        base = self._result_dir / str(task_id)
        target = base / f"{version_type}.md"
        try:
            target.write_text(fp.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            return False
        return True

    # ── Control ────────────────────────────────────────────────────────

    def is_task_running(self, task_id: str) -> bool:
        return _task_running(task_id)

    def start_task(
        self,
        task_id: str,
        target: Callable,
        *args,
        daemon: bool = True,
        **kwargs,
    ) -> bool:
        if self.is_task_running(task_id):
            return False
        pause_ev, cancel_ev = _get_task_events(task_id)
        pause_ev.clear()
        cancel_ev.clear()
        th = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=daemon)
        _set_task_thread(task_id, th)
        th.start()
        return True

    def pause_task(self, task_id: str) -> None:
        pause_ev, _ = _get_task_events(task_id)
        pause_ev.set()

    def resume_task(self, task_id: str) -> None:
        pause_ev, _ = _get_task_events(task_id)
        pause_ev.clear()

    def cancel_task(self, task_id: str) -> None:
        _, cancel_ev = _get_task_events(task_id)
        cancel_ev.set()

    def get_control_events(self, task_id: str) -> tuple[threading.Event, threading.Event]:
        return _get_task_events(task_id)

    def wait_for_clarify(self, task_id: str, *, timeout_s: int = 1800) -> tuple[str, bool]:
        t0 = time.time()
        while time.time() - t0 <= float(timeout_s or 0):
            st = self.read_status(task_id)
            if str(st.get("status") or "") in {"canceled", "cancelled"}:
                return "", True
            ans = str(st.get("clarify_answers") or "").strip()
            if ans:
                return ans, False
            if bool(st.get("clarify_skip")):
                return "", True
            time.sleep(1.0)
        return "", True

    def wait_for_satisfaction(self, task_id: str, stage: str, *, timeout_s: int = 1800) -> dict:
        """Wait for user satisfaction response. Returns dict with satisfied/feedback/scope/selected_version."""
        t0 = time.time()
        while time.time() - t0 <= float(timeout_s or 0):
            st = self.read_status(task_id)
            if str(st.get("status") or "") in {"canceled", "cancelled"}:
                return {"satisfied": True, "feedback": "", "scope": "outline", "selected_version": None}
            # Check for satisfaction response keys
            satisfied = st.get(f"{stage}_satisfied")
            if satisfied is not None:
                result = {
                    "satisfied": bool(satisfied),
                    "feedback": str(st.get("satisfaction_feedback") or "").strip(),
                    "scope": str(st.get("regeneration_scope") or "outline").strip(),
                }
                if st.get("selected_version"):
                    result["selected_version"] = st.get("selected_version")
                return result
            time.sleep(1.0)
        # Timeout: assume satisfied
        return {"satisfied": True, "feedback": "", "scope": "outline", "selected_version": None}