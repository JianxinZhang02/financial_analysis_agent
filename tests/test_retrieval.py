from rag.hybrid_retriever import HybridRetriever
from rag.hybrid_retriever import RerankStep


def test_hybrid_retrieval_returns_evidence():
    cards = HybridRetriever().retrieve_evidence("示例科技2024年现金流")
    assert cards
    assert cards[0].source_file
    assert cards[0].chunk_id


class FakeReranker:
    def rerank(self, query, candidates, top_n=6):
        return candidates[:top_n]


def test_rerank_step_stores_reranked_candidates_in_context():
    candidates = [{"rank": index} for index in range(5)]
    context = {"top_n": 3}

    reranked = RerankStep(FakeReranker()).run("query", candidates, context)

    assert reranked == candidates[:3]
    assert context["reranked"] == candidates[:3]


class FakePipeline:
    def execute(self, query, context):
        context["dense_hits"] = [object(), object()]
        context["bm25_hits"] = [object()]
        context["merged"] = {str(index): {"rank": index} for index in range(5)}
        context["reranked"] = [{"rank": index} for index in range(context["top_n"])]


def test_hybrid_retrieve_returns_context_reranked_top_n():
    retriever = object.__new__(HybridRetriever)
    retriever.pipeline = FakePipeline()

    result = HybridRetriever.retrieve(retriever, "腾讯2024年现金流", top_k=4, top_n=3)

    assert result == [{"rank": 0}, {"rank": 1}, {"rank": 2}]
