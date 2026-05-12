from __future__ import annotations

from ingestion.schema import Chunk
from rag.reranker import LocalReranker


def _candidate(chunk_id: str, text: str, dense_score: float, bm25_score: float) -> dict:
    return {
        "chunk": Chunk(
            chunk_id=chunk_id,
            doc_id=f"doc-{chunk_id}",
            text=text,
            source_file=f"{chunk_id}.txt",
            doc_type="annual_report",
            metadata={},
        ),
        "dense_score": dense_score,
        "bm25_score": bm25_score,
    }


class FakeCrossEncoder:
    def __init__(self, scores: list[float]):
        self.scores = scores
        self.pairs: list[tuple[str, str]] = []
        self.batch_size: int | None = None

    def predict(self, pairs, batch_size=None, show_progress_bar=False):
        self.pairs = list(pairs)
        self.batch_size = batch_size
        return self.scores


class FailingCrossEncoder:
    def predict(self, pairs, batch_size=None, show_progress_bar=False):
        raise RuntimeError("model unavailable")


def test_cross_encoder_reranker_scores_merged_candidates():
    candidates = [
        _candidate("dense-only", "腾讯经营现金流承压", dense_score=0.7, bm25_score=0.0),
        _candidate("bm25-only", "腾讯自由现金流改善", dense_score=0.0, bm25_score=1.0),
        _candidate("both", "腾讯现金流质量稳定", dense_score=0.6, bm25_score=0.8),
    ]
    model = FakeCrossEncoder(scores=[-1.0, 3.0, 0.5])
    reranker = LocalReranker(provider="local_cross_encoder", model_name="fake-model", cross_encoder=model)

    reranked = reranker.rerank("腾讯现金流", candidates, top_n=2)

    assert model.pairs == [
        ("腾讯现金流", "腾讯经营现金流承压"),
        ("腾讯现金流", "腾讯自由现金流改善"),
        ("腾讯现金流", "腾讯现金流质量稳定"),
    ]
    assert [item["chunk"].chunk_id for item in reranked] == ["bm25-only", "both"]
    assert all(item["rerank_provider"] == "cross_encoder" for item in reranked)


def test_reranker_fallback_uses_best_retrieval_score_not_missing_score_penalty():
    candidates = [
        _candidate("dense-only", "腾讯现金流质量", dense_score=0.7, bm25_score=0.0),
        _candidate("bm25-only", "腾讯现金流质量", dense_score=0.0, bm25_score=0.7),
    ]
    reranker = LocalReranker(
        provider="local_cross_encoder",
        model_name="fake-model",
        cross_encoder=FailingCrossEncoder(),
    )

    reranked = reranker.rerank("腾讯现金流", candidates, top_n=2)
    scores = {item["chunk"].chunk_id: item["final_score"] for item in reranked}

    assert abs(scores["dense-only"] - scores["bm25-only"]) < 1e-9
    assert all(item["rerank_provider"] == "local_fallback" for item in reranked)
    assert all(item["final_score"] > 0 for item in reranked)
