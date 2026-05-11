from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ingestion.pipeline import read_chunks
from ingestion.schema import Chunk
from rag.bm25_store import BM25Store
from rag.context_compressor import ContextCompressor
from rag.query_filters import infer_metadata_filter, normalize_query_for_metadata_filter
from rag.reranker import LocalReranker
from rag.vector_store import VectorStoreService
from utils.config_handler import rag_cof
from utils.logger_handler import log_stage, safe_preview


class RetrievalStep(ABC):
    @abstractmethod
    def run(self, query: str, candidates: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
        pass


class DenseSearchStep(RetrievalStep):
    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store

    def run(self, query: str, candidates: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
        top_k = context.get("top_k", 8)
        metadata_filter = context.get("metadata_filter")
        search_query = context.get("search_query", query)
        hits = self.vector_store.search(search_query, top_k=top_k, metadata_filter=metadata_filter)
        context["dense_hits"] = hits
        for chunk, score in hits:
            entry = context["merged"].setdefault(chunk.chunk_id, {"chunk": chunk, "dense_score": 0.0, "bm25_score": 0.0})
            entry["dense_score"] = max(entry["dense_score"], score)
        return list(context["merged"].values())


class BM25SearchStep(RetrievalStep):
    def __init__(self, bm25_store: BM25Store):
        self.bm25_store = bm25_store

    def run(self, query: str, candidates: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
        top_k = context.get("top_k", 8)
        metadata_filter = context.get("metadata_filter")
        search_query = context.get("search_query", query)
        hits = self.bm25_store.search(search_query, top_k=top_k, metadata_filter=metadata_filter)
        context["bm25_hits"] = hits
        max_bm25 = max([score for _, score in hits], default=0.0) or 1.0
        for chunk, score in hits:
            entry = context["merged"].setdefault(chunk.chunk_id, {"chunk": chunk, "dense_score": 0.0, "bm25_score": 0.0})
            entry["bm25_score"] = max(entry["bm25_score"], score / max_bm25)
        return list(context["merged"].values())


class RerankStep(RetrievalStep):
    def __init__(self, reranker: LocalReranker):
        self.reranker = reranker

    def run(self, query: str, candidates: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
        top_n = context.get("top_n", 6)
        return self.reranker.rerank(context.get("search_query", query), candidates, top_n=top_n)


class CompressStep(RetrievalStep):
    def __init__(self, compressor: ContextCompressor):
        self.compressor = compressor

    def run(self, query: str, candidates: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
        cards = self.compressor.compress(query, candidates)
        context["evidence_cards"] = cards
        return candidates


class RetrievalPipeline:
    def __init__(self, steps: list[RetrievalStep] | None = None):
        self.steps = steps or []

    def add_step(self, step: RetrievalStep) -> "RetrievalPipeline":
        self.steps.append(step)
        return self

    def execute(self, query: str, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        context = context or {}
        context.setdefault("merged", {})
        candidates: list[dict[str, Any]] = []
        for step in self.steps:
            candidates = step.run(query, candidates, context)
        return candidates


class HybridRetriever:
    """Backward-compatible facade over RetrievalPipeline."""

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or read_chunks()
        self.vector_store = VectorStoreService(self.chunks)
        self.bm25_store = BM25Store(self.chunks)
        self.reranker = LocalReranker()
        self.compressor = ContextCompressor()
        self.pipeline = self._build_default_pipeline()

    def _build_default_pipeline(self) -> RetrievalPipeline:
        return RetrievalPipeline([
            DenseSearchStep(self.vector_store),
            BM25SearchStep(self.bm25_store),
            RerankStep(self.reranker),
            CompressStep(self.compressor),
        ])

    def retrieve(self, query: str, top_k: int | None = None, top_n: int | None = None) -> list[dict]:
        with log_stage("rag.retrieve", query=safe_preview(query)) as stage:
            top_k = top_k or int(rag_cof.get("retriever_k", 8))
            top_n = top_n or int(rag_cof.get("rerank_top_n", 6))
            metadata_filter = infer_metadata_filter(query)
            search_query = normalize_query_for_metadata_filter(query, metadata_filter)
            context = {
                "top_k": top_k,
                "top_n": top_n,
                "metadata_filter": metadata_filter,
                "search_query": search_query,
                "merged": {},
            }
            self.pipeline.execute(query, context)
            reranked = context.get("reranked", list(context.get("merged", {}).values()))
            stage.add_done_fields(
                top_k=top_k,
                top_n=top_n,
                metadata_filter=metadata_filter or None,
                dense_hits=len(context.get("dense_hits", [])),
                bm25_hits=len(context.get("bm25_hits", [])),
                merged=len(context.get("merged", {})),
                reranked=len(reranked),
            )
            return reranked

    def retrieve_evidence(self, query: str, top_k: int | None = None, top_n: int | None = None):
        context = {
            "top_k": top_k or int(rag_cof.get("retriever_k", 8)),
            "top_n": top_n or int(rag_cof.get("rerank_top_n", 6)),
            "merged": {},
        }
        metadata_filter = infer_metadata_filter(query)
        context["metadata_filter"] = metadata_filter
        context["search_query"] = normalize_query_for_metadata_filter(query, metadata_filter)
        self.pipeline.execute(query, context)
        return context.get("evidence_cards", [])
