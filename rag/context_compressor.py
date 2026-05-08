from __future__ import annotations

import re

from ingestion.schema import Chunk
from rag.bm25_store import tokenize
from rag.citation import EvidenceCard


NOISE_PATTERNS = ["免责声明", "重要声明", "目录", "本报告仅供"]


class ContextCompressor:
    def __init__(self, max_evidence_chars: int = 420):
        self.max_evidence_chars = max_evidence_chars

    def compress(self, query: str, candidates: list[dict]) -> list[EvidenceCard]:
        cards: list[EvidenceCard] = []
        query_terms = set(tokenize(query))
        for candidate in candidates:
            chunk: Chunk = candidate["chunk"]
            if any(pattern in chunk.text[:80] for pattern in NOISE_PATTERNS):
                continue
            evidence = self._select_relevant_sentences(query_terms, chunk.text)
            if not evidence:
                continue
            cards.append(
                EvidenceCard(
                    claim=self._claim_from_evidence(evidence),
                    evidence=evidence,
                    source_file=chunk.metadata.get("file_name", chunk.source_file),
                    page_number=chunk.page_start,
                    chunk_id=chunk.chunk_id,
                    score=float(candidate.get("final_score", 0.0)),
                    confidence=min(0.95, 0.5 + float(candidate.get("final_score", 0.0)) / 2),
                    metadata={
                        "section_path": chunk.section_path,
                        "doc_type": chunk.doc_type,
                        "doc_id": chunk.doc_id,
                    },
                )
            )
        return cards

    def _select_relevant_sentences(self, query_terms: set[str], text: str) -> str:
        sentences = [s.strip() for s in re.split(r"(?<=[。！？；;])\s*|\n+", text) if s.strip()]
        metadata_prefixes = (
            "#",
            "文档类型",
            "研究机构",
            "发布日期",
            "公司：",
            "股票代码",
            "会议日期",
            "报告期",
            "研究机构",
        )
        sentences = [
            sentence
            for sentence in sentences
            if not any(sentence.startswith(prefix) for prefix in metadata_prefixes)
        ]
        if not sentences:
            return ""
        scored = []
        for sentence in sentences:
            terms = set(tokenize(sentence))
            overlap = len(query_terms & terms)
            has_number = bool(re.search(r"\d", sentence))
            has_financial_metric = bool(
                re.search(r"营收|营业收入|净利润|毛利率|现金流|自由现金流|净现比|应收账款|市盈率|估值|风险", sentence)
            )
            score = overlap + (0.5 if has_number else 0.0) + (1.0 if has_financial_metric else 0.0)
            scored.append((sentence, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        selected = [sentence for sentence, score in scored[:3] if score > 0]
        if not selected:
            selected = [sentences[0]]
        evidence = " ".join(selected)
        return evidence[: self.max_evidence_chars]

    def _claim_from_evidence(self, evidence: str) -> str:
        return evidence.split("。")[0].strip()[:160]
