from __future__ import annotations

from ingestion.pipeline import read_chunks
from ingestion.schema import Chunk
from rag.bm25_store import BM25Store
from rag.context_compressor import ContextCompressor
from rag.reranker import LocalReranker
from rag.vector_store import VectorStoreService
from utils.config_handler import rag_cof


class HybridRetriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or read_chunks()
        self.vector_store = VectorStoreService(self.chunks)
        self.bm25_store = BM25Store(self.chunks)
        self.reranker = LocalReranker()
        self.compressor = ContextCompressor()

    def retrieve(self, query: str, top_k: int | None = None, top_n: int | None = None) -> list[dict]:
        top_k = top_k or int(rag_cof.get("retriever_k", 8))
        top_n = top_n or int(rag_cof.get("rerank_top_n", 6))

        dense_hits = self.vector_store.search(query, top_k=top_k)
        bm25_hits = self.bm25_store.search(query, top_k=top_k)
        merged: dict[str, dict] = {}

        for chunk, score in dense_hits:
            merged.setdefault(chunk.chunk_id, {"chunk": chunk, "dense_score": 0.0, "bm25_score": 0.0})
            merged[chunk.chunk_id]["dense_score"] = max(merged[chunk.chunk_id]["dense_score"], score)

        max_bm25 = max([score for _, score in bm25_hits], default=1.0)
        for chunk, score in bm25_hits:
            merged.setdefault(chunk.chunk_id, {"chunk": chunk, "dense_score": 0.0, "bm25_score": 0.0})
            merged[chunk.chunk_id]["bm25_score"] = max(merged[chunk.chunk_id]["bm25_score"], score / max_bm25)

        return self.reranker.rerank(query, list(merged.values()), top_n=top_n)

    def retrieve_evidence(self, query: str, top_k: int | None = None, top_n: int | None = None):
        candidates = self.retrieve(query, top_k=top_k, top_n=top_n)
        return self.compressor.compress(query, candidates)
