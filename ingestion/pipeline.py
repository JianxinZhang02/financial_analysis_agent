from __future__ import annotations

import json
from pathlib import Path

from ingestion.chunk_cache import ChunkCache, file_md5
from ingestion.chunkers.markdown_chunker import MarkdownHierarchyChunker
from ingestion.chunkers.semantic_chunker import SemanticChunker
from ingestion.loaders.image_loader import load_image_file
from ingestion.loaders.pdf_loader import load_pdf_file
from ingestion.loaders.table_loader import load_csv_file
from ingestion.loaders.text_loader import load_text_file
from ingestion.metadata_registry import (
    enrich_documents_from_registry,
    is_registered_document,
    registered_document_paths,
    registry_row_fingerprint,
    should_skip_registry_file,
)
from ingestion.parsers.financial_pdf_parser import clean_financial_text
from ingestion.schema import Chunk, SourceDocument
from utils.chinese_text import normalize_zh_for_retrieval
from utils.config_handler import rag_cof
from utils.logger_handler import logger
from utils.progress import progress_bar, set_progress_detail
from utils.path_tool import get_abs_path


SUPPORTED_SUFFIXES = {".txt", ".md", ".html", ".htm", ".pdf", ".csv", ".png", ".jpg", ".jpeg"}


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _scan_supported_files(root: Path) -> list[str]:
    return sorted(
        str(path)
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not should_skip_registry_file(str(path))
    )


def discover_files(data_path: str | None = None) -> list[str]:
    active_data_path = data_path or rag_cof["data_path"]
    root = Path(get_abs_path(active_data_path))
    if not root.exists():
        return []
    if _as_bool(rag_cof.get("require_document_registry"), default=True):
        if _as_bool(rag_cof.get("auto_register_local_files"), default=False):
            from ingestion.registry_sync import sync_registry_from_data_path

            sync_registry_from_data_path(active_data_path, SUPPORTED_SUFFIXES)
        files = registered_document_paths(active_data_path, SUPPORTED_SUFFIXES)
        all_files = _scan_supported_files(root)
        unregistered_count = sum(1 for file in all_files if not is_registered_document(file))
        if unregistered_count:
            logger.info(f"Skipped {unregistered_count} unregistered file(s). Add them to document_registry.csv to ingest.")
        return files
    return _scan_supported_files(root)


def load_documents(path: str) -> list[SourceDocument]:
    suffix = Path(path).suffix.lower()
    if suffix in {".txt", ".md", ".html", ".htm"}:
        docs = load_text_file(path)
    elif suffix == ".pdf":
        docs = load_pdf_file(path)
    elif suffix == ".csv":
        docs = load_csv_file(path)
    elif suffix in {".png", ".jpg", ".jpeg"}:
        docs = load_image_file(path)
    else:
        docs = []
    docs = enrich_documents_from_registry(path, docs)
    cleaned_docs = [clean_financial_text(doc) for doc in docs]
    for doc in cleaned_docs:
        mode = doc.metadata.get("text_normalization", "")
        normalized_text, status = normalize_zh_for_retrieval(doc.text, mode)
        doc.text = normalized_text
        if mode:
            doc.metadata["normalization_status"] = status
    return cleaned_docs


def _chunk_cache_signature() -> str:
    payload = {
        "schema": "financial_pipeline_v2",
        "markdown_chunker": "hierarchy_recursive_v1",
        "max_chars": int(rag_cof.get("chunk_max_chars", 1200)),
        "overlap_chars": int(rag_cof.get("chunk_overlap_chars", 120)),
        "normalization": "registry_driven_opencc_t2s",
    }
    return json.dumps(payload, sort_keys=True)


def _short_path(path: str, root: str | None = None, max_chars: int = 80) -> str:
    file_path = Path(path)
    if root:
        try:
            text = str(file_path.relative_to(Path(root)))
        except ValueError:
            text = file_path.name
    else:
        text = file_path.name
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars + 3 :]


def chunk_documents(docs: list[SourceDocument]) -> list[Chunk]:
    markdown_docs = [
        doc
        for doc in docs
        if doc.doc_type in {"annual_report", "research_report", "financial_table", "pdf"}
        or doc.metadata.get("parser_profile")
    ]
    markdown_doc_ids = {id(doc) for doc in markdown_docs}
    semantic_docs = [doc for doc in docs if id(doc) not in markdown_doc_ids]

    markdown_chunker = MarkdownHierarchyChunker(
        max_chars=int(rag_cof.get("chunk_max_chars", 1200)),
        overlap_chars=int(rag_cof.get("chunk_overlap_chars", 120)),
    )
    semantic_chunker = SemanticChunker(max_chars=int(rag_cof.get("chunk_max_chars", 1200)))

    chunks = markdown_chunker.split(markdown_docs)
    chunks.extend(semantic_chunker.split(semantic_docs))
    return chunks


def build_chunks(data_path: str | None = None, use_cache: bool = True, show_progress: bool = True) -> list[Chunk]:
    files = discover_files(data_path)
    data_root = get_abs_path(data_path or rag_cof["data_path"])
    cache = ChunkCache(signature=_chunk_cache_signature()) if use_cache else None
    chunks: list[Chunk] = []
    cache_hits = 0
    cache_misses = 0

    file_progress = progress_bar(files, desc="Build chunks", unit="file", total=len(files), enabled=show_progress)
    for file in file_progress:
        display_name = _short_path(file, data_root)
        set_progress_detail(file_progress, f"checking {display_name}")
        source_hash = file_md5(file)
        metadata_hash = registry_row_fingerprint(file)
        cached_chunks = cache.get(file, source_hash, metadata_hash) if cache else None
        if cached_chunks is not None:
            cache_hits += 1
            chunks.extend(cached_chunks)
            set_progress_detail(file_progress, f"cache hit {display_name} ({len(cached_chunks)} chunks)")
            continue

        cache_misses += 1
        set_progress_detail(file_progress, f"parsing {display_name}")
        file_chunks = chunk_documents(load_documents(file))
        for chunk in file_chunks:
            chunk.metadata["source_file_hash"] = source_hash
            if metadata_hash:
                chunk.metadata["registry_metadata_hash"] = metadata_hash
            chunk.metadata["chunk_cache_signature"] = _chunk_cache_signature()
        if cache:
            cache.set(file, source_hash, metadata_hash, file_chunks)
        chunks.extend(file_chunks)
        set_progress_detail(file_progress, f"processed {display_name} ({len(file_chunks)} chunks)")

    if cache:
        cache.prune(files)
        cache.save()
        logger.info(
            f"Chunk cache finished: files={len(files)}, hits={cache_hits}, misses={cache_misses}, chunks={len(chunks)}"
        )
    return chunks


def write_chunks(chunks: list[Chunk], output_path: str | None = None, show_progress: bool = True) -> str:
    target = Path(get_abs_path(output_path or rag_cof["processed_path"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        chunk_progress = progress_bar(
            chunks,
            desc="Write chunks",
            unit="chunk",
            total=len(chunks),
            enabled=show_progress and len(chunks) >= 200,
        )
        for chunk in chunk_progress:
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


def rebuild_index(
    data_path: str | None = None,
    output_path: str | None = None,
    build_vector: bool = True,
    force_vector: bool = False,
) -> str:
    chunks = build_chunks(data_path, show_progress=True)
    written_path = write_chunks(chunks, output_path, show_progress=True)
    if build_vector:
        from rag.vector_store import VectorStoreService

        service = VectorStoreService(chunks)
        if force_vector:
            service.build_from_chunks(chunks, force=True)
        else:
            service.sync_from_chunks(chunks)
    return written_path


if __name__ == "__main__":
    output = rebuild_index()
    print(f"Indexed chunks written to {output}")
