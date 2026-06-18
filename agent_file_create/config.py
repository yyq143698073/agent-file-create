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
