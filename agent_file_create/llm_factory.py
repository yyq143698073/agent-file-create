"""Re-export of get_chat_model from llm_client (merged module).

Kept for backward compatibility — all LLM instantiation now lives in llm_client.py.
"""

from agent_file_create.llm_client import get_chat_model  # noqa: F401

__all__ = ["get_chat_model"]
