from __future__ import annotations

import hashlib
from pathlib import Path

from ingestion.schema import SourceDocument


def load_pdf_file(path: str) -> list[SourceDocument]:
    """Load page-level PDF text with optional dependencies."""

    file_path = Path(path)
    doc_id_prefix = hashlib.md5(str(file_path.resolve()).encode("utf-8")).hexdigest()

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        docs: list[SourceDocument] = []
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            docs.append(
                SourceDocument(
                    doc_id=f"{doc_id_prefix}-p{idx}",
                    source_file=str(file_path),
                    text=text,
                    doc_type="pdf",
                    page_number=idx,
                    metadata={"file_name": file_path.name},
                )
            )
        return docs
    except Exception:
        return [
            SourceDocument(
                doc_id=doc_id_prefix,
                source_file=str(file_path),
                text=(
                    f"[PDF占位] 当前环境未安装 pypdf，无法解析 {file_path.name}。"
                    "请安装依赖后重新运行 ingestion.pipeline。"
                ),
                doc_type="pdf_unparsed",
                page_number=1,
                metadata={"file_name": file_path.name, "parse_status": "missing_pdf_dependency"},
            )
        ]
