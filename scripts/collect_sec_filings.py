from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.path_tool import get_abs_path


DOCUMENT_REGISTRY = "data/raw/financial_reports/document_registry.csv"
SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_SUBMISSIONS_ARCHIVE_URL = "https://data.sec.gov/submissions/{name}"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"


def _headers() -> dict[str, str]:
    user_agent = os.getenv("SEC_USER_AGENT", "financial-analysis-agent contact@example.com")
    return {"User-Agent": user_agent}


def _fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, path: Path) -> bytes:
    request = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def _read_registry(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with Path(get_abs_path(path)).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _write_registry(path: str, fields: list[str], rows: list[dict[str, str]]) -> None:
    with Path(get_abs_path(path)).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sec_ticker_map() -> dict[str, int]:
    data = _fetch_json(SEC_TICKER_URL)
    return {item["ticker"].upper(): int(item["cik_str"]) for item in data.values()}


def _filing_records(filings: dict[str, list[str]]) -> list[dict[str, str]]:
    forms = filings.get("form", [])
    return [
        {
            "form": form,
            "reportDate": filings.get("reportDate", [""] * len(forms))[idx],
            "filingDate": filings.get("filingDate", [""] * len(forms))[idx],
            "accessionNumber": filings.get("accessionNumber", [""] * len(forms))[idx],
            "primaryDocument": filings.get("primaryDocument", [""] * len(forms))[idx],
        }
        for idx, form in enumerate(forms)
    ]


def _all_filing_records(submissions: dict[str, Any]) -> list[dict[str, str]]:
    records = _filing_records(submissions.get("filings", {}).get("recent", {}))
    for archive in submissions.get("filings", {}).get("files", []):
        name = archive.get("name")
        if not name:
            continue
        records.extend(_filing_records(_fetch_json(SEC_SUBMISSIONS_ARCHIVE_URL.format(name=name))))
    return records


def _annual_filing_for_year(records: list[dict[str, str]], fiscal_year: int) -> dict[str, str] | None:
    exact_candidates: list[dict[str, str]] = []
    filing_year_candidates: list[dict[str, str]] = []
    for record in records:
        if record["form"] != "20-F":
            continue
        if record["reportDate"].startswith(str(fiscal_year)):
            exact_candidates.append(record)
        elif not record["reportDate"] and record["filingDate"].startswith(str(fiscal_year)):
            filing_year_candidates.append(record)

    candidates = exact_candidates or filing_year_candidates
    candidates.sort(key=lambda item: item["filingDate"], reverse=True)
    return candidates[0] if candidates else None


def collect(registry_path: str, dry_run: bool = False, sleep_seconds: float = 0.2) -> dict[str, int]:
    fields, rows = _read_registry(registry_path)
    cik_by_ticker = _sec_ticker_map()
    stats = {"downloaded": 0, "skipped": 0, "missing": 0}

    for row in rows:
        ticker = row.get("ticker", "").upper()
        if row.get("doc_type") != "annual_report" or row.get("file_ext") not in {"html", "htm"}:
            stats["skipped"] += 1
            continue
        if row.get("parse_status") not in {"pending_download", "download_failed", ""}:
            stats["skipped"] += 1
            continue
        if ticker not in cik_by_ticker:
            stats["missing"] += 1
            row["parse_status"] = "sec_ticker_not_found"
            continue

        fiscal_year = int(row["report_period"].replace("FY", ""))
        cik = cik_by_ticker[ticker]
        submissions = _fetch_json(SEC_SUBMISSIONS_URL.format(cik=cik))
        filing = _annual_filing_for_year(_all_filing_records(submissions), fiscal_year)
        if not filing:
            stats["missing"] += 1
            row["parse_status"] = "sec_20f_not_found"
            continue

        accession = filing["accessionNumber"].replace("-", "")
        primary_doc = filing["primaryDocument"]
        url = SEC_ARCHIVE_URL.format(cik=str(cik), accession=accession, primary_doc=primary_doc)
        target = Path(get_abs_path(row["local_path"]))
        print(f"{row['doc_id']}: {url} -> {target}")
        if not dry_run:
            content = _download(url, target)
            row["source_url"] = url
            row["publish_date"] = row.get("publish_date") or filing["filingDate"]
            row["hash"] = hashlib.md5(content).hexdigest()
            row["parse_status"] = "downloaded"
        stats["downloaded"] += 1
        time.sleep(sleep_seconds)

    if not dry_run:
        _write_registry(registry_path, fields, rows)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SEC 20-F annual reports declared in document_registry.csv.")
    parser.add_argument("--registry", default=DOCUMENT_REGISTRY)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()
    print(collect(args.registry, dry_run=args.dry_run, sleep_seconds=args.sleep))
