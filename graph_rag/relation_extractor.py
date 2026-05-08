from __future__ import annotations

from ingestion.schema import Chunk
from graph_rag.entity_extractor import extract_entities


def extract_relations(chunks: list[Chunk]) -> list[dict]:
    relations: list[dict] = []
    for chunk in chunks:
        entities = extract_entities(chunk.text)
        companies = entities.get("companies", [])
        metrics = entities.get("metrics", [])
        for company in companies:
            for metric in metrics:
                relations.append(
                    {
                        "head": company,
                        "relation": "披露指标",
                        "tail": metric,
                        "source_file": chunk.metadata.get("file_name", chunk.source_file),
                        "page_number": chunk.page_start,
                        "chunk_id": chunk.chunk_id,
                    }
                )
        if "云资源成本" in chunk.text and "毛利率" in chunk.text:
            relations.append(
                {
                    "head": "云资源成本",
                    "relation": "影响",
                    "tail": "毛利率",
                    "source_file": chunk.metadata.get("file_name", chunk.source_file),
                    "page_number": chunk.page_start,
                    "chunk_id": chunk.chunk_id,
                }
            )
    return relations
