"""Entry point for the web server.

Usage:
    python -m agent_file_create.web
    python agent_web.py (from project root)
"""

import sys
from pathlib import Path

from agent_file_create.web.server import run

__all__ = ["run"]

if __name__ == "__main__":
    from agent_file_create.config import validate_config
    errors = validate_config()
    if errors:
        print("配置错误：")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # Warn if checkpoints DB is growing too large
    _checkpoints_db = Path(__file__).resolve().parent.parent.parent / "result" / "checkpoints.db"
    if _checkpoints_db.exists():
        _size_mb = _checkpoints_db.stat().st_size / (1024 * 1024)
        if _size_mb > 100:
            print(f"⚠️  checkpoints.db 大小 {_size_mb:.0f} MB，建议清理已完成任务的旧 checkpoint。")

    # ── Embedding health check ────────────────────────────────────────
    try:
        from agent_file_create.rag import get_kb
        health = get_kb().check_embed_health()
        if not health.get("ok"):
            err = health.get("error", "未知错误")
            dims = health.get("dim", "?")
            print(f"❌ 嵌入服务异常：{err}")
            print(f"   模型：{health.get('model', '?')}  维度：{dims}")
            print(f"   知识库检索和上传功能将不可用。请检查：")
            print(f"   1. Ollama 是否已启动：ollama serve")
            print(f"   2. bge-m3 模型是否已拉取：ollama pull bge-m3:latest")
            print(f"   继续启动服务（KB 功能不可用）...")
        else:
            print(f"✅ 嵌入服务正常  model={health.get('model','?')} dim={health.get('dim','?')}")
    except Exception as e:
        print(f"⚠️  无法检查嵌入服务：{e}")

    run()
