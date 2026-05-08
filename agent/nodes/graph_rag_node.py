from __future__ import annotations

from agent.state import FinancialAgentState
from graph_rag.graph_retriever import GraphRetriever


_graph_retriever: GraphRetriever | None = None


def _get_graph_retriever() -> GraphRetriever:
    global _graph_retriever
    if _graph_retriever is None:
        _graph_retriever = GraphRetriever()
    return _graph_retriever


def graph_rag_node(state: FinancialAgentState) -> dict:
    retriever = _get_graph_retriever()
    return {"graph_relations": retriever.retrieve(state.get("user_query", ""))}
