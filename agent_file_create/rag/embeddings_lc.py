"""LangChain Embeddings wrapper around the project's embed_texts().

Enables using the local embedding pipeline with LangChain VectorStore,
retrievers, and other components that accept an Embeddings instance.
"""

from typing import List

from langchain_core.embeddings import Embeddings

from agent_file_create.rag.embedder import embed_texts


class ChatchatEmbeddings(Embeddings):
    """LangChain-compatible embeddings backed by embed_texts().

    Usage:
        emb = ChatchatEmbeddings(timeout_s=60, max_batch=32)
        vectors = emb.embed_documents(["text one", "text two"])
        query_vec = emb.embed_query("search query")
    """

    timeout_s: int = 60
    max_batch: int = 32

    class Config:
        arbitrary_types_allowed = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embed_texts(texts, timeout_s=self.timeout_s, max_batch=self.max_batch)

    def embed_query(self, text: str) -> List[float]:
        vecs = embed_texts([text], timeout_s=self.timeout_s, max_batch=1)
        if vecs:
            return vecs[0]
        return []
