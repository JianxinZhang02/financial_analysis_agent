from rag.hybrid_retriever import HybridRetriever


def test_hybrid_retrieval_returns_evidence():
    cards = HybridRetriever().retrieve_evidence("示例科技2024年现金流")
    assert cards
    assert cards[0].source_file
    assert cards[0].chunk_id
