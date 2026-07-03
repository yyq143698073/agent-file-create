"""RAG (Retrieval-Augmented Generation) module.

Provides KnowledgeBase singleton access, retrieval, chunking, embedding,
and reranking utilities for document question-answering.
"""

__all__ = ["kb", "get_kb"]

from agent_file_create.rag import kb

# ── KnowledgeBase singleton ──────────────────────────────────────────────
# Lazy-initialized singleton so that KB initialization (which may load
# embedding models, connect to vector stores, etc.) is deferred until
# the first actual use, avoiding costly startup when KB is not needed.

_kb_instance = None

def get_kb():
    """Return the module-level KnowledgeBase singleton, initializing it on first call."""
    global _kb_instance
    if _kb_instance is None:
        from agent_file_create.rag.kb import KnowledgeBase
        _kb_instance = KnowledgeBase()
    return _kb_instance
