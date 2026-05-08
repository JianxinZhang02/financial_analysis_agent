from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ingestion.schema import SourceDocument


def detect_doc_type(path: str, text: str) -> str:
    name = os.path.basename(path).lower()
    if "annual" in name or "年报" in name or "财报" in name:
        return "annual_report"
    if "research" in name or "研报" in name or "深度" in name:
        return "research_report"
    if "call" in name or "纪要" in name or "conference" in name:
        return "conference_call"
    if "news" in name or "新闻" in name:
        return "financial_news"
    return "text"


def load_text_file(path: str) -> list[SourceDocument]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    doc_id = hashlib.md5(str(file_path.resolve()).encode("utf-8")).hexdigest()
    return [
        SourceDocument(
            doc_id=doc_id,
            source_file=str(file_path),
            text=text,
            doc_type=detect_doc_type(str(file_path), text),
            page_number=1,
            metadata={"file_name": file_path.name},
        )
    ]
