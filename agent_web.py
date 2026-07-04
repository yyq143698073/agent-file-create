"""Web server entry point — run with: python agent_web.py  or  python -m agent_file_create.web

Starts the FastAPI server on http://127.0.0.1:8000 by default.
"""
import sys
from pathlib import Path

from agent_file_create.web.server import run

if __name__ == "__main__":
    from agent_file_create.config import validate_config
    errors = validate_config()
    if errors:
        print("配置错误：")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # Checkpoints DB warning
    _cp = Path(__file__).resolve().parent / "result" / "checkpoints.db"
    if _cp.exists() and _cp.stat().st_size > 100 * 1024 * 1024:
        print(f"checkpoints.db >100MB, suggest cleaning old checkpoints")

    # Embedding health check
    try:
        from agent_file_create.rag import get_kb
        health = get_kb().check_embed_health()
        if health.get("ok"):
            print(f"embedding OK model={health.get('model','?')} dim={health.get('dim','?')}")
        else:
            print(f"embedding DOWN: {health.get('error','?')}")
    except Exception as e:
        print(f"embedding check failed: {e}")

    run()
