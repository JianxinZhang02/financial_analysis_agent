from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.path_tool import get_abs_path


COMPANY_REGISTRY = "data/raw/financial_reports/company_registry.csv"
DOCUMENT_REGISTRY = "data/raw/financial_reports/document_registry.csv"

COMPANY_FIELDS = [
    "company_id",
    "ticker",
    "company_name",
    "market",
    "sector",
    "subsector",
    "size_bucket",
    "listing_status",
    "ir_url",
]
DOCUMENT_FIELDS = [
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
]
FIRST_BATCH_COMPANIES = {
    "0700.HK_tencent",
    "BABA_alibaba",
    "BIDU_baidu",
    "3690.HK_meituan",
    "NTES_netease",
    "PDD_pdd",
    "1024.HK_kuaishou",
    "9626.HK_bilibili",
}
CORE_DOC_TYPES = {
    "annual_report",
    "interim_report",
    "quarterly_results",
    "investor_presentation",
    "earnings_call",
    "structured_financials",
}
SIZE_BUCKETS = {"large", "mid", "small"}


def _read_csv(path: str) -> list[dict[str, str]]:
    with Path(get_abs_path(path)).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _assert_fields(rows: list[dict[str, str]], fields: list[str], name: str) -> None:
    if not rows:
        raise AssertionError(f"{name} is empty")
    missing = [field for field in fields if field not in rows[0]]
    if missing:
        raise AssertionError(f"{name} missing fields: {missing}")


def validate() -> dict[str, int]:
    companies = _read_csv(COMPANY_REGISTRY)
    documents = _read_csv(DOCUMENT_REGISTRY)
    _assert_fields(companies, COMPANY_FIELDS, COMPANY_REGISTRY)
    _assert_fields(documents, DOCUMENT_FIELDS, DOCUMENT_REGISTRY)

    if len(companies) < 15:
        raise AssertionError("company registry must contain at least 15 companies")

    company_ids = {row["company_id"] for row in companies}
    unknown_doc_companies = {row["company_id"] for row in documents} - company_ids
    if unknown_doc_companies:
        raise AssertionError(f"document registry references unknown companies: {unknown_doc_companies}")

    invalid_buckets = {row["size_bucket"] for row in companies} - SIZE_BUCKETS
    if invalid_buckets:
        raise AssertionError(f"invalid size_bucket values: {invalid_buckets}")

    invalid_core_types = {
        row["doc_type"]
        for row in documents
        if row["is_core_document"].lower() == "true" and row["doc_type"] not in CORE_DOC_TYPES
    }
    if invalid_core_types:
        raise AssertionError(f"invalid core document types: {invalid_core_types}")

    annual_coverage: dict[str, set[str]] = defaultdict(set)
    for row in documents:
        if row["doc_type"] == "annual_report":
            annual_coverage[row["company_id"]].add(row["report_period"])
        for required in ["doc_id", "company_id", "doc_type", "report_period", "source_url", "local_path"]:
            if not row.get(required):
                raise AssertionError(f"{row.get('doc_id', '<unknown>')} missing {required}")

    missing_coverage = {
        company_id: sorted({"FY2022", "FY2023", "FY2024"} - annual_coverage.get(company_id, set()))
        for company_id in FIRST_BATCH_COMPANIES
        if {"FY2022", "FY2023", "FY2024"} - annual_coverage.get(company_id, set())
    }
    if missing_coverage:
        raise AssertionError(f"first batch annual report coverage missing: {missing_coverage}")

    existing_files = sum(1 for row in documents if Path(get_abs_path(row["local_path"])).exists())
    pending_files = sum(1 for row in documents if row["parse_status"] == "pending_download")
    return {
        "companies": len(companies),
        "documents": len(documents),
        "existing_files": existing_files,
        "pending_download": pending_files,
    }


if __name__ == "__main__":
    result = validate()
    print("financial dataset registry ok:", result)
