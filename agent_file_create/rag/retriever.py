"""LangChain BaseRetriever wrapper around KnowledgeBase.search().

Enables using the KB with standard LangChain RAG chains such as
create_stuff_documents_chain, create_retrieval_chain, etc.
"""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class KnowledgeBaseRetriever(BaseRetriever):
    """LangChain retriever backed by a KnowledgeBase instance.

    Usage:
        retriever = KnowledgeBaseRetriever(kb="default", knowledge_base=kb)
        docs = retriever.invoke("your question")
    """

    kb: str = "default"
    knowledge_base: object  # KnowledgeBase — typed as object to avoid circular import
    top_k: int = 8
    context_window: int = 2  # neighbor chunks to include per hit
    filters: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # Use context-window search for richer retrieval
        if hasattr(self.knowledge_base, "search_with_context") and self.context_window > 0:
            hits = self.knowledge_base.search_with_context(
                kb=self.kb, query=query, top_k=self.top_k,
                context_window=self.context_window, filters=self.filters,
            )
        else:
            hits = self.knowledge_base.search(
                kb=self.kb, query=query, top_k=self.top_k, filters=self.filters,
            )
        out: list[Document] = []
        for h in hits:
            meta: dict = {
                "doc_id": h.doc_id,
                "chunk_id": h.chunk_id,
                "section_path": h.section_path,
                "score": float(h.score),
                "kb": h.kb,
            }
            if isinstance(h.meta, dict):
                meta.update({str(k): v for k, v in h.meta.items()})
            out.append(Document(page_content=h.content, metadata=meta))
        return out
