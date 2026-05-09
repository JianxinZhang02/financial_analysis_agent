from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.path_tool import get_abs_path


COMPANY_REGISTRY = "data/raw/financial_reports/company_registry.csv"
DOCUMENT_REGISTRY = "data/raw/financial_reports/document_registry.csv"
REGISTRY_FIELDS = [
    "doc_id",
    "company_id",
    "ticker",
    "company_name",
    "doc_type",
    "report_period",
    "publish_date",
    "source_url",
    "local_path",
    "file_ext",
    "language",
    "is_core_document",
    "parse_status",
    "hash",
    "source_priority",
    "text_normalization",
    "cninfo_detail_url",
]

COMPANY_ALIASES = {
    "百度": "BIDU_baidu",
    "baidu": "BIDU_baidu",
    "腾讯": "0700.HK_tencent",
    "tencent": "0700.HK_tencent",
    "美团": "3690.HK_meituan",
    "meituan": "3690.HK_meituan",
    "快手": "1024.HK_kuaishou",
    "kuaishou": "1024.HK_kuaishou",
}


def _read_csv(path: str) -> tuple[list[str], list[dict[str, str]]]:
    target = Path(get_abs_path(path))
    if not target.exists():
        return [], []
    with target.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: str, fields: list[str], rows: list[dict[str, str]]) -> None:
    target = Path(get_abs_path(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _load_companies() -> dict[str, dict[str, str]]:
    _, rows = _read_csv(COMPANY_REGISTRY)
    return {row["company_id"]: row for row in rows if row.get("company_id")}


def _infer_company_id(file_name: str) -> str:
    lowered = file_name.lower()
    for alias, company_id in COMPANY_ALIASES.items():
        if alias.lower() in lowered:
            return company_id
    return ""


def _infer_report_period(file_name: str) -> str:
    match = re.search(r"(20\d{2})", file_name)
    return f"FY{match.group(1)}" if match else ""


def _default_text_normalization(language: str) -> str:
    return "traditional_to_simplified" if language == "zh-Hant" else ""


def _canonical_local_path(company_id: str, report_period: str, doc_type: str, file_ext: str) -> str:
    year = report_period.replace("FY", "").split("_")[0]
    doc_id = f"{company_id.split('_')[0]}_{report_period.lower()}_{doc_type}"
    return f"data/raw/financial_reports/{company_id}/{year}/{doc_type}/{doc_id}.{file_ext}"


def _ensure_fields(fields: list[str], rows: list[dict[str, str]]) -> list[str]:
    merged = list(fields)
    for field in REGISTRY_FIELDS:
        if field not in merged:
            merged.append(field)
    for row in rows:
        for field in merged:
            row.setdefault(field, "")
    return merged


def register(args: argparse.Namespace) -> dict[str, str]:
    source_path = Path(args.file)
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    companies = _load_companies()
    company_id = args.company_id or _infer_company_id(source_path.name)
    if company_id not in companies:
        raise ValueError("company_id is required or could not be inferred from file name.")
    company = companies[company_id]

    report_period = args.report_period or _infer_report_period(source_path.name)
    if not report_period:
        raise ValueError("report_period is required, for example FY2024.")

    doc_type = args.doc_type
    file_ext = source_path.suffix.lower().lstrip(".")
    target_local_path = args.local_path or _canonical_local_path(company_id, report_period, doc_type, file_ext)
    target_path = Path(get_abs_path(target_local_path))
    if args.keep_path:
        target_path = source_path
        target_local_path = _relative_to_project(target_path)
    elif source_path.resolve() != target_path.resolve():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and not args.replace:
            raise FileExistsError(f"{target_path} already exists. Use --replace to overwrite it.")
        shutil.copy2(source_path, target_path)

    fields, rows = _read_csv(DOCUMENT_REGISTRY)
    fields = _ensure_fields(fields or REGISTRY_FIELDS, rows)

    doc_id = args.doc_id or f"{company['ticker']}_{report_period.lower()}_{doc_type}".replace(".", "")
    existing_index = next((idx for idx, row in enumerate(rows) if row.get("doc_id") == doc_id), None)
    if existing_index is not None and not args.replace:
        raise ValueError(f"doc_id already exists in registry: {doc_id}. Use --replace to update it.")

    row = {field: "" for field in fields}
    row.update(
        {
            "doc_id": doc_id,
            "company_id": company_id,
            "ticker": company.get("ticker", ""),
            "company_name": company.get("company_name", ""),
            "doc_type": doc_type,
            "report_period": report_period,
            "publish_date": args.publish_date or "",
            "source_url": args.source_url or "local_file",
            "local_path": target_local_path.replace("\\", "/"),
            "file_ext": file_ext,
            "language": args.language,
            "is_core_document": "true" if args.core else "false",
            "parse_status": "downloaded",
            "hash": _file_md5(target_path),
            "source_priority": args.source_priority,
            "text_normalization": args.text_normalization or _default_text_normalization(args.language),
            "cninfo_detail_url": args.cninfo_detail_url or "",
        }
    )

    if existing_index is None:
        rows.append(row)
    else:
        rows[existing_index] = row
    _write_csv(DOCUMENT_REGISTRY, fields, rows)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a local financial document into document_registry.csv.")
    parser.add_argument("--file", required=True, help="Local PDF/TXT/CSV/HTML file to register.")
    parser.add_argument("--company-id", default="", help="Company id in company_registry.csv. Can be inferred for common names.")
    parser.add_argument("--doc-type", default="annual_report")
    parser.add_argument("--report-period", default="", help="For example FY2024 or FY2024_Q4.")
    parser.add_argument("--publish-date", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--language", default="zh-Hans")
    parser.add_argument("--text-normalization", default="")
    parser.add_argument("--source-priority", default="local")
    parser.add_argument("--cninfo-detail-url", default="")
    parser.add_argument("--doc-id", default="")
    parser.add_argument("--local-path", default="", help="Optional registry local_path. Defaults to canonical company/year/doc_type path.")
    parser.add_argument("--keep-path", action="store_true", help="Register the file at its current path instead of copying it.")
    parser.add_argument("--replace", action="store_true", help="Overwrite existing target file or registry row.")
    parser.add_argument("--core", action="store_true", default=True)
    result = register(parser.parse_args())
    print(f"registered {result['doc_id']} -> {result['local_path']}")
