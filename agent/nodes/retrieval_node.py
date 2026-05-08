from __future__ import annotations

from agent.state import FinancialAgentState
from rag.hybrid_retriever import HybridRetriever


_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def retrieval_node(state: FinancialAgentState) -> dict:
    retriever = _get_retriever()
    sub_queries = state.get("sub_queries") or [state.get("user_query", "")]
    cards = []
    for query in sub_queries:
        cards.extend(retriever.retrieve_evidence(query))

    seen = set()
    unique_cards = []
    for card in sorted(cards, key=lambda item: item.score, reverse=True):
        if card.chunk_id not in seen:
            unique_cards.append(card)
            seen.add(card.chunk_id)
    return {
        "evidence_cards": [card.to_dict() for card in unique_cards[:8]],
        "retrieved_docs": [card.metadata for card in unique_cards[:8]],
    }
