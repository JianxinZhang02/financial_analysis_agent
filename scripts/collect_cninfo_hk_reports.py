from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.path_tool import get_abs_path


DOCUMENT_REGISTRY = "data/raw/financial_reports/document_registry.csv"
CNINFO_QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC_BASE = "https://static.cninfo.com.cn/"
CNINFO_DETAIL_URL = (
    "https://www.cninfo.com.cn/new/disclosure/detail?"
    "stockCode={sec_code}&announcementId={announcement_id}&orgId={org_id}&announcementTime={announcement_time}"
)
EXTRA_REGISTRY_FIELDS = ["source_priority", "text_normalization"]

CNINFO_TARGETS = {
    "0700.HK_tencent": {"sec_code": "00700", "search_name": "腾讯控股"},
    "3690.HK_meituan": {"sec_code": "03690", "search_name": "美团"},
    "1024.HK_kuaishou": {"sec_code": "01024", "search_name": "快手-W"},
}
EXCLUDE_TITLE_TERMS = (
    "摘要",
    "补充",
    "更正",
    "通函",
    "月报",
    "中期",
    "季度",
    "股份变动",
    "股份購回",
    "股份购回",
    "奖励",
    "獎勵",
    "授出",
    "董事会",
    "委任",
    "股东大会",
    "股東大會",
)


def _headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 financial-analysis-agent/0.1",
        "Referer": "https://www.cninfo.com.cn/new/fulltextSearch?notautosubmit=&keyWord=&searchType=0",
        "Origin": "https://www.cninfo.com.cn",
        "X-Requested-With": "XMLHttpRequest",
    }


def _read_registry(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with Path(get_abs_path(path)).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _write_registry(path: str, fields: list[str], rows: list[dict[str, str]]) -> None:
    with Path(get_abs_path(path)).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_registry_fields(fields: list[str], rows: list[dict[str, str]]) -> list[str]:
    updated_fields = list(fields)
    for field in EXTRA_REGISTRY_FIELDS:
        if field not in updated_fields:
            updated_fields.append(field)
    for row in rows:
        for field in updated_fields:
            row.setdefault(field, "")
    return updated_fields


def _post_form(url: str, data: dict[str, str]) -> dict[str, Any]:
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={**_headers(), "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _strip_html(value: Any) -> str:
    text = html.unescape(str(value or ""))
    return re.sub(r"<[^>]+>", "", text).strip()


def _normalize_sec_code(value: Any) -> str:
    raw = re.sub(r"\D", "", str(value or ""))
    return raw.zfill(5) if raw else ""


def _publish_date_from_cninfo_ms(value: Any) -> str:
    try:
        timestamp_ms = int(value)
    except (TypeError, ValueError):
        return ""
    china_tz = timezone(timedelta(hours=8))
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=china_tz).date().isoformat()


def _search_keys(target: dict[str, str], year: int) -> list[str]:
    return [
        f"{target['search_name']} {year} 年报",
        f"{target['search_name']} {year} 年度报告",
        f"{target['search_name']} 年报",
        f"{target['search_name']} 年度报告",
        f"{target['sec_code']} {year} 年报",
        f"{target['sec_code']} 年报",
    ]


def _find_annual_report(target: dict[str, str], year: int, sec_code: str, page_size: int = 30) -> dict[str, Any] | None:
    seen_ids: set[str] = set()
    announcements: list[dict[str, Any]] = []
    for search_key in _search_keys(target, year):
        try:
            items = _query_announcements_once(search_key, page_size)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"  search key skipped ({search_key}): {exc}")
            continue
        for item in items:
            announcement_id = str(item.get("announcementId") or item.get("adjunctUrl") or "")
            if announcement_id in seen_ids:
                continue
            seen_ids.add(announcement_id)
            announcements.append(item)
        match = _select_annual_report(announcements, sec_code, year)
        if match:
            return match
    return None


def _query_announcements_once(search_key: str, page_size: int) -> list[dict[str, Any]]:
    data = {
        "pageNum": "1",
        "pageSize": str(page_size),
        "column": "hke",
        "tabName": "fulltext",
        "plate": "",
        "stock": "",
        "searchkey": search_key,
        "secid": "",
        "category": "",
        "trade": "",
        "seDate": "",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    payload = _post_form(CNINFO_QUERY_URL, data)
    return list(payload.get("announcements") or [])


def _title_matches_year(title: str, year: int) -> bool:
    compact_title = re.sub(r"\s+", "", title)
    accepted_patterns = (
        f"{year}年报",
        f"{year}年年报",
        f"{year}年度报告",
        f"年报{year}",
        f"年度报告{year}",
    )
    return any(pattern in compact_title for pattern in accepted_patterns)


def _is_target_annual_report(item: dict[str, Any], sec_code: str, year: int) -> bool:
    title = _strip_html(item.get("announcementTitle"))
    if _normalize_sec_code(item.get("secCode")) != sec_code:
        return False
    if not _title_matches_year(title, year):
        return False
    if any(term in title for term in EXCLUDE_TITLE_TERMS):
        return False
    return bool(item.get("adjunctUrl"))


def _select_annual_report(items: list[dict[str, Any]], sec_code: str, year: int) -> dict[str, Any] | None:
    matches = [item for item in items if _is_target_annual_report(item, sec_code, year)]
    matches.sort(key=lambda item: int(item.get("announcementTime") or 0), reverse=True)
    return matches[0] if matches else None


def _static_url(adjunct_url: str) -> str:
    return urllib.parse.urljoin(CNINFO_STATIC_BASE, adjunct_url.lstrip("/"))


def _detail_url(item: dict[str, Any], sec_code: str) -> str:
    return CNINFO_DETAIL_URL.format(
        sec_code=sec_code,
        announcement_id=item.get("announcementId", ""),
        org_id=item.get("orgId", ""),
        announcement_time=item.get("announcementTime", ""),
    )


def _download(url: str, path: Path) -> bytes:
    request = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(request, timeout=180) as response:
        content = response.read()
    if not content.startswith(b"%PDF"):
        raise ValueError(f"downloaded content does not look like a PDF: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def _iter_target_rows(
    rows: list[dict[str, str]],
    company_ids: set[str],
    years: set[int],
    replace_existing: bool,
) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        if row.get("company_id") not in company_ids:
            continue
        if row.get("doc_type") != "annual_report":
            continue
        if row.get("file_ext") != "pdf":
            continue
        try:
            year = int(row.get("report_period", "").replace("FY", ""))
        except ValueError:
            continue
        if year not in years:
            continue
        if not replace_existing and row.get("parse_status") not in {"", "pending_download", "download_failed"}:
            continue
        selected.append(row)
    return selected


def collect(
    registry_path: str,
    company_ids: set[str] | None = None,
    years: set[int] | None = None,
    dry_run: bool = False,
    replace_existing: bool = False,
    mark_missing: bool = False,
    sleep_seconds: float = 0.3,
) -> dict[str, int]:
    fields, rows = _read_registry(registry_path)
    fields = _ensure_registry_fields(fields, rows)

    selected_company_ids = company_ids or set(CNINFO_TARGETS)
    selected_years = years or {2022, 2023, 2024}
    unknown_company_ids = selected_company_ids - set(CNINFO_TARGETS)
    if unknown_company_ids:
        raise ValueError(f"CNInfo target mapping missing for: {sorted(unknown_company_ids)}")

    stats = {"downloaded": 0, "skipped": 0, "missing": 0, "failed": 0}
    target_rows = _iter_target_rows(rows, selected_company_ids, selected_years, replace_existing)
    stats["skipped"] = len(rows) - len(target_rows)

    for row in target_rows:
        company_id = row["company_id"]
        target = CNINFO_TARGETS[company_id]
        sec_code = target["sec_code"]
        fiscal_year = int(row["report_period"].replace("FY", ""))

        print(f"search {company_id} FY{fiscal_year}: {target['search_name']} {fiscal_year} 年报")
        try:
            match = _find_annual_report(target, fiscal_year, sec_code)
        except Exception as exc:
            stats["failed"] += 1
            if not dry_run:
                row["parse_status"] = "cninfo_query_failed"
            print(f"  query failed: {exc}")
            continue

        if not match:
            stats["missing"] += 1
            if not dry_run and mark_missing:
                row["parse_status"] = "cninfo_annual_report_not_found"
            print("  no exact annual report match")
            continue

        title = _strip_html(match.get("announcementTitle"))
        source_url = _static_url(str(match["adjunctUrl"]))
        target_path = Path(get_abs_path(row["local_path"]))
        detail_url = _detail_url(match, sec_code)
        print(f"  match: {title} -> {source_url}")
        print(f"  save:  {target_path}")
        if dry_run:
            stats["downloaded"] += 1
            continue

        try:
            content = _download(source_url, target_path)
        except Exception as exc:
            stats["failed"] += 1
            row["parse_status"] = "download_failed"
            print(f"  download failed: {exc}")
            continue

        row["source_url"] = source_url
        row["publish_date"] = _publish_date_from_cninfo_ms(match.get("announcementTime"))
        row["language"] = "zh-Hant"
        row["source_priority"] = "primary"
        row["text_normalization"] = "traditional_to_simplified"
        row["parse_status"] = "downloaded"
        row["hash"] = hashlib.md5(content).hexdigest()
        row["cninfo_detail_url"] = detail_url
        stats["downloaded"] += 1
        time.sleep(sleep_seconds)

    if not dry_run:
        if "cninfo_detail_url" not in fields:
            fields.append("cninfo_detail_url")
            for row in rows:
                row.setdefault("cninfo_detail_url", "")
        _write_registry(registry_path, fields, rows)
    return stats


def _parse_years(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HK annual report PDFs from CNInfo into document_registry.csv paths.")
    parser.add_argument("--registry", default=DOCUMENT_REGISTRY)
    parser.add_argument("--company", action="append", choices=sorted(CNINFO_TARGETS), help="Company id to download. Repeatable.")
    parser.add_argument("--years", default="2022,2023,2024", help="Comma-separated fiscal years.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--mark-missing", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    selected_companies = set(args.company) if args.company else None
    print(
        collect(
            args.registry,
            company_ids=selected_companies,
            years=_parse_years(args.years),
            dry_run=args.dry_run,
            replace_existing=args.replace_existing,
            mark_missing=args.mark_missing,
            sleep_seconds=args.sleep,
        )
    )
