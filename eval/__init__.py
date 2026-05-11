from __future__ import annotations

from rag.hybrid_retriever import HybridRetriever, RetrievalPipeline
from rag.citation import EvidenceCard


def evaluate_retrieval_precision(query: str, expected_sources: list[str], retriever: HybridRetriever | None = None) -> dict:
    retriever = retriever or HybridRetriever()
    cards = retriever.retrieve_evidence(query)
    if not cards:
        return {'query': query, 'precision': 0.0, 'recall': 0.0, 'cards': 0}
    retrieved_sources = set(card.source_file for card in cards)
    expected_set = set(expected_sources)
    hits = len(retrieved_sources & expected_set)
    precision = hits / len(retrieved_sources) if retrieved_sources else 0.0
    recall = hits / len(expected_set) if expected_set else 0.0
    return {'query': query, 'precision': round(precision, 3), 'recall': round(recall, 3), 'cards': len(cards)}


def evaluate_citation_completeness(cards: list[EvidenceCard] | list[dict]) -> dict:
    from rag.citation import has_valid_citation
    if cards and isinstance(cards[0], dict):
        cards = [EvidenceCard.from_dict(card) for card in cards]
    total = len(cards)
    valid = sum(1 for card in cards if has_valid_citation(card))
    return {'total': total, 'valid': valid, 'completeness': round(valid / total, 3) if total else 0.0}


class GoldenDataset:
    def __init__(self):
        self.entries: list[dict] = []

    def add(self, query: str, expected_sources: list[str], expected_answer_keywords: list[str]) -> None:
        self.entries.append({
            'query': query,
            'expected_sources': expected_sources,
            'expected_answer_keywords': expected_answer_keywords,
        })

    def load_from_jsonl(self, path: str) -> None:
        import json
        from pathlib import Path
        target = Path(path)
        if not target.exists():
            return
        with target.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    self.entries.append(entry)

    def save_to_jsonl(self, path: str) -> None:
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open('w', encoding='utf-8') as f:
            for entry in self.entries:
                f.write(json.dumps(entry, ensure_ascii=False) + chr(10))

    def run_eval(self, retriever: HybridRetriever | None = None) -> list[dict]:
        retriever = retriever or HybridRetriever()
        results = []
        for entry in self.entries:
            r = evaluate_retrieval_precision(entry['query'], entry.get('expected_sources', []), retriever)
            results.append(r)
        return results
