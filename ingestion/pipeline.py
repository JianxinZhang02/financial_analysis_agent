from __future__ import annotations

import json
import os
from pathlib import Path

from ingestion.chunkers.semantic_chunker import SemanticChunker
from ingestion.chunkers.structure_chunker import StructureAwareChunker
from ingestion.loaders.image_loader import load_image_file
from ingestion.loaders.pdf_loader import load_pdf_file
from ingestion.loaders.table_loader import load_csv_file
from ingestion.loaders.text_loader import load_text_file
from ingestion.parsers.financial_pdf_parser import clean_financial_text
from ingestion.schema import Chunk, SourceDocument
from utils.config_handler import rag_cof
from utils.path_tool import get_abs_path


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".csv", ".png", ".jpg", ".jpeg"}


def discover_files(data_path: str | None = None) -> list[str]:
    root = Path(get_abs_path(data_path or rag_cof["data_path"]))
    if not root.exists():
        return []
    return [
        str(path)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def load_documents(path: str) -> list[SourceDocument]:
    suffix = Path(path).suffix.lower()
    if suffix in {".txt", ".md"}:
        docs = load_text_file(path)
    elif suffix == ".pdf":
        docs = load_pdf_file(path)
    elif suffix == ".csv":
        docs = load_csv_file(path)
    elif suffix in {".png", ".jpg", ".jpeg"}:
        docs = load_image_file(path)
    else:
        docs = []
    return [clean_financial_text(doc) for doc in docs]


def build_chunks(data_path: str | None = None) -> list[Chunk]:
    files = discover_files(data_path)
    docs: list[SourceDocument] = []
    for file in files:
        docs.extend(load_documents(file))

    structured_docs = [doc for doc in docs if doc.doc_type in {"annual_report", "research_report", "financial_table", "pdf"}]
    semantic_docs = [doc for doc in docs if doc not in structured_docs]

    structure_chunker = StructureAwareChunker(
        max_chars=int(rag_cof.get("chunk_max_chars", 1200)),
        overlap_chars=int(rag_cof.get("chunk_overlap_chars", 120)),
    )
    semantic_chunker = SemanticChunker(max_chars=int(rag_cof.get("chunk_max_chars", 1200)))

    chunks = structure_chunker.split(structured_docs)
    chunks.extend(semantic_chunker.split(semantic_docs))
    return chunks


def write_chunks(chunks: list[Chunk], output_path: str | None = None) -> str:
    target = Path(get_abs_path(output_path or rag_cof["processed_path"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return str(target)


def read_chunks(path: str | None = None) -> list[Chunk]:
    target = Path(get_abs_path(path or rag_cof["processed_path"]))
    if not target.exists():
        chunks = build_chunks()
        write_chunks(chunks, str(target))
        return chunks
    chunks: list[Chunk] = []
    with target.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks


def rebuild_index(data_path: str | None = None, output_path: str | None = None) -> str:
    chunks = build_chunks(data_path)
    return write_chunks(chunks, output_path)


if __name__ == "__main__":
    output = rebuild_index()
    print(f"Indexed chunks written to {output}")
