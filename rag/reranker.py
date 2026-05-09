from __future__ import annotations

from rag.bm25_store import tokenize
from utils.config_handler import rag_cof


def _score_weight(name: str, default: float) -> float:
    try:
        return float(rag_cof.get(name, default))
    except (TypeError, ValueError):
        return default


class LocalReranker:
    def score(self, query: str, text: str, metadata: dict | None = None) -> float:
        metadata = metadata or {}
        query_terms = set(tokenize(query))
        text_terms = set(tokenize(text))
        if not query_terms or not text_terms:
            lexical = 0.0
        else:
            lexical = len(query_terms & text_terms) / len(query_terms)
        authority = 0.1 if metadata.get("doc_type") in {"annual_report", "research_report", "financial_table"} else 0.0
        freshness = 0.05 if "2024" in text or "2025" in text else 0.0
        return min(1.0, lexical + authority + freshness)

    def rerank(self, query: str, candidates: list[dict], top_n: int = 6) -> list[dict]:
        dense_weight = _score_weight("dense_weight", 0.35)
        bm25_weight = _score_weight("bm25_weight", 0.35)
        rerank_weight = _score_weight("rerank_weight", 0.30)

        for candidate in candidates:
            chunk = candidate["chunk"]
            candidate["rerank_score"] = self.score(query, chunk.text, chunk.metadata | {"doc_type": chunk.doc_type})
            candidate["final_score"] = (
                dense_weight * candidate.get("dense_score", 0.0)
                + bm25_weight * candidate.get("bm25_score", 0.0)
                + rerank_weight * candidate.get("rerank_score", 0.0)
            )
        candidates.sort(key=lambda item: item["final_score"], reverse=True)
        return candidates[:top_n]
