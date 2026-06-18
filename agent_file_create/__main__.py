"""Package entry point — ``python -m agent_file_create`` launches the CLI.

Use ``python -m agent_file_create.web`` for the web server.
"""
from agent_file_create.cli import main

if __name__ == "__main__":
    main()
