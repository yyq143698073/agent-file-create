import os
from pathlib import Path

# Auto-load .env from project root (fallback defaults, overridden by real env vars)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.5:4b").strip()
API_GENERATE_ENDPOINT = os.getenv("API_GENERATE_ENDPOINT", f"{OLLAMA_HOST.rstrip('/')}/api/generate").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

OUTLINE_API_STYLE = os.getenv("OUTLINE_API_STYLE", "openai").strip()
OUTLINE_MODEL_NAME = os.getenv("OUTLINE_MODEL_NAME", "deepseek-v4-flash").strip()
OUTLINE_API_ENDPOINT = os.getenv("OUTLINE_API_ENDPOINT", "https://api.deepseek.com/v1/chat/completions").strip()
OUTLINE_API_KEY = os.getenv("OUTLINE_API_KEY", "").strip() or OPENAI_API_KEY

CONTENT_API_STYLE = os.getenv("CONTENT_API_STYLE", "openai").strip()
CONTENT_MODEL_NAME = os.getenv("CONTENT_MODEL_NAME", "deepseek-v4-flash").strip()
CONTENT_API_ENDPOINT = os.getenv("CONTENT_API_ENDPOINT", "https://api.deepseek.com/v1/chat/completions").strip()
CONTENT_API_KEY = os.getenv("CONTENT_API_KEY", "").strip() or OPENAI_API_KEY

PLANNER_API_STYLE = os.getenv("PLANNER_API_STYLE", "").strip()
PLANNER_MODEL_NAME = os.getenv("PLANNER_MODEL_NAME", "").strip()
PLANNER_API_ENDPOINT = os.getenv("PLANNER_API_ENDPOINT", "").strip()
PLANNER_API_KEY = os.getenv("PLANNER_API_KEY", "").strip() or OPENAI_API_KEY

OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT", "https://api.deepseek.com/v1/chat/completions").strip()
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "deepseek-v4-pro").strip()
DB_URL = os.getenv("DB_URL", "").strip()
DB_PATH = os.getenv("DB_PATH", "result/app.db").strip()

KB_DB_URL = os.getenv("KB_DB_URL", "").strip()
KB_DB_PATH = os.getenv("KB_DB_PATH", "result/kb.db").strip()
KB_INDEX_TYPE = os.getenv("KB_INDEX_TYPE", "hnsw").strip()
KB_HNSW_EF_SEARCH = int(os.getenv("KB_HNSW_EF_SEARCH", "64"))
KB_IVFFLAT_PROBES = int(os.getenv("KB_IVFFLAT_PROBES", "10"))

EMBED_API_STYLE = os.getenv("EMBED_API_STYLE", "ollama").strip()
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "bge-m3:latest").strip()
EMBED_API_ENDPOINT = os.getenv("EMBED_API_ENDPOINT", "").strip()
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "").strip()

VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "minicpm-v:8b").strip()

EXTRACT_API_STYLE = os.getenv("EXTRACT_API_STYLE", "ollama").strip()
EXTRACT_MODEL_NAME = os.getenv("EXTRACT_MODEL_NAME", "").strip() or MODEL_NAME
EXTRACT_API_ENDPOINT = os.getenv("EXTRACT_API_ENDPOINT", "").strip()
EXTRACT_API_KEY = os.getenv("EXTRACT_API_KEY", "").strip()

IMAGE_MAX_LONG_EDGE = int(os.getenv("IMAGE_MAX_LONG_EDGE", "2048"))
IMAGE_JPEG_QUALITY = int(os.getenv("IMAGE_JPEG_QUALITY", "85"))

OCR_ENABLED = str(os.getenv("OCR_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "y", "on"}
PDF_OCR_THRESHOLD_X = float(os.getenv("PDF_OCR_THRESHOLD_X", "0.6"))
PDF_OCR_THRESHOLD_Y = float(os.getenv("PDF_OCR_THRESHOLD_Y", "0.6"))
PDF_MAX_PAGES_VISION = int(os.getenv("PDF_MAX_PAGES_VISION", "8"))

MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "60"))
MODEL_TIMEOUT_SHORT = int(os.getenv("MODEL_TIMEOUT_SHORT", "60"))
MODEL_TIMEOUT_LONG = int(os.getenv("MODEL_TIMEOUT_LONG", "180"))

MAX_WORKERS_DEFAULT = int(os.getenv("MAX_WORKERS_DEFAULT", "8"))

RERANK_ENABLED = str(os.getenv("RERANK_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "y", "on"}
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3").strip()
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "20"))
RERANK_FINAL_K = int(os.getenv("RERANK_FINAL_K", "8"))

# ── Version retention ──
VERSION_MAX_RETENTION = int(os.getenv("VERSION_MAX_RETENTION", "20"))

# ── Content / prompt limits (centralized from scattered magic numbers) ──
SOURCE_DIGEST_MAX_CHARS = int(os.getenv("SOURCE_DIGEST_MAX_CHARS", "3000"))
CONTENT_PREVIEW_CHARS = int(os.getenv("CONTENT_PREVIEW_CHARS", "4000"))
CHAT_MSG_MAX_CHARS = int(os.getenv("CHAT_MSG_MAX_CHARS", "2000"))
SECTION_BODY_MAX_CHARS = int(os.getenv("SECTION_BODY_MAX_CHARS", "800"))
ERROR_MSG_MAX_CHARS = int(os.getenv("ERROR_MSG_MAX_CHARS", "240"))
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "120"))
CHAT_HISTORY_MAX_MSGS = int(os.getenv("CHAT_HISTORY_MAX_MSGS", "50"))
RERANK_CHUNK_MAX_CHARS = int(os.getenv("RERANK_CHUNK_MAX_CHARS", "400"))

# ── Graph execution limits ──
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "100"))

# ── Quality gate thresholds ─────────────────────────────────────────────────
EVAL_MIN_FAITHFULNESS = float(os.getenv("EVAL_MIN_FAITHFULNESS", "0.6"))
EVAL_MIN_COMPLETENESS = float(os.getenv("EVAL_MIN_COMPLETENESS", "0.5"))
EVAL_AUTO_RETRY = str(os.getenv("EVAL_AUTO_RETRY", "true")).strip().lower() in {"1", "true", "yes", "y", "on"}


# ── Startup validation ────────────────────────────────────────────────────

def validate_config() -> list[str]:
    """Validate configuration at startup. Returns list of error messages.

    Call this from main() entry points before starting any work.
    An empty list means all checks passed.
    """
    errors: list[str] = []

    # ── Content LLM (primary generation path) ──
    if CONTENT_API_STYLE in ("openai",):
        if not CONTENT_API_KEY:
            errors.append("CONTENT_API_KEY is required when CONTENT_API_STYLE=openai")
    if not CONTENT_MODEL_NAME:
        errors.append("CONTENT_MODEL_NAME is required (cannot be empty)")

    # ── Outline LLM ──
    if not OUTLINE_MODEL_NAME:
        errors.append("OUTLINE_MODEL_NAME is required (cannot be empty)")

    # ── Embedding ──
    _valid_embed_styles = {"ollama", "openai", ""}
    if EMBED_API_STYLE not in _valid_embed_styles:
        errors.append(
            f"EMBED_API_STYLE='{EMBED_API_STYLE}' is invalid. "
            f"Valid values: {sorted(_valid_embed_styles)}"
        )

    # ── Planner LLM (optional but warn if unset) ──
    if not PLANNER_API_STYLE:
        import logging
        _log = logging.getLogger(__name__)
        _log.warning(
            "PLANNER_API_STYLE is empty — planner features (RAG / skill selection) "
            "will fall back to CONTENT_API_STYLE"
        )

    # ── Numeric range checks ──
    if not (1 <= MAX_WORKERS_DEFAULT <= 64):
        errors.append(
            f"MAX_WORKERS_DEFAULT={MAX_WORKERS_DEFAULT} is out of range (1-64)"
        )
    if not (1 <= KB_HNSW_EF_SEARCH <= 1024):
        errors.append(
            f"KB_HNSW_EF_SEARCH={KB_HNSW_EF_SEARCH} is out of range (1-1024)"
        )
    for _name, _val in (
        ("MODEL_TIMEOUT", MODEL_TIMEOUT),
        ("MODEL_TIMEOUT_SHORT", MODEL_TIMEOUT_SHORT),
        ("MODEL_TIMEOUT_LONG", MODEL_TIMEOUT_LONG),
    ):
        if not (5 <= _val <= 600):
            errors.append(f"{_name}={_val} is out of range (5-600)")

    return errors
