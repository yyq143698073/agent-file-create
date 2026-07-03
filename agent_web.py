"""Web server entry point — run with: python agent_web.py  or  python -m agent_file_create.web

Starts the FastAPI server on http://127.0.0.1:8000 by default.
"""
from agent_file_create.web.server import run

if __name__ == "__main__":
    run()
