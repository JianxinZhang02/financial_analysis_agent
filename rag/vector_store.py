from __future__ import annotations

import math
from typing import Any

from ingestion.pipeline import read_chunks, rebuild_index
from ingestion.schema import Chunk
from model.factory import SimpleEmbeddings, embed_model
from utils.config_handler import rag_cof


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    numerator = sum(x * y for x, y in zip(a, b))
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return numerator / denom if denom else 0.0


class InMemoryVectorStore:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.embedding_model = embed_model
        self.embedding_error: str | None = None
        self.embeddings = self._embed_documents([chunk.text for chunk in chunks]) if chunks else []

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.embedding_model.embed_documents(texts)
        except Exception as exc:
            self.embedding_error = str(exc)
            self.embedding_model = SimpleEmbeddings()
            return self.embedding_model.embed_documents(texts)

    def _embed_query(self, query: str) -> list[float]:
        try:
            return self.embedding_model.embed_query(query)
        except Exception as exc:
            self.embedding_error = str(exc)
            self.embedding_model = SimpleEmbeddings()
            self.embeddings = self.embedding_model.embed_documents([chunk.text for chunk in self.chunks])
            return self.embedding_model.embed_query(query)

    def search(self, query: str, top_k: int = 8) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        query_embedding = self._embed_query(query)
        scored = [
            (chunk, cosine_similarity(query_embedding, embedding))
            for chunk, embedding in zip(self.chunks, self.embeddings)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(chunk, score) for chunk, score in scored[:top_k] if score > 0]


class VectorStoreService:
    """Compatibility wrapper around the new dependency-light vector index."""

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or read_chunks()
        self.vector_store = InMemoryVectorStore(self.chunks)

    def get_retriever(self) -> "VectorStoreService":
        return self

    def invoke(self, query: str) -> list[Any]:
        return [chunk for chunk, _ in self.vector_store.search(query, top_k=int(rag_cof.get("retriever_k", 8)))]

    def search(self, query: str, top_k: int = 8) -> list[tuple[Chunk, float]]:
        return self.vector_store.search(query, top_k)

    def load_doucments(self) -> None:
        rebuild_index()
        self.chunks = read_chunks()
        self.vector_store = InMemoryVectorStore(self.chunks)
