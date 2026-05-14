import csv
from collections import defaultdict
from pathlib import Path

from ingestion.pipeline import discover_files


ROOT = Path(__file__).resolve().parents[1]
COMPANY_REGISTRY = ROOT / "data" / "raw" / "financial_reports" / "company_registry.csv"
DOCUMENT_REGISTRY = ROOT / "data" / "raw" / "financial_reports" / "document_registry.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_company_registry_scope():
    rows = _read_csv(COMPANY_REGISTRY)
    assert len(rows) >= 15
    assert {row["size_bucket"] for row in rows} == {"large", "mid", "small"}
    assert all(row["sector"] == "china_internet_ai_software" for row in rows)
    assert not any("福田" in row["company_name"] or "Foton" in row["company_name"] for row in rows)


def test_document_registry_first_batch_annual_coverage():
    rows = _read_csv(DOCUMENT_REGISTRY)
    first_batch = {
        "0700.HK_tencent",
        "BABA_alibaba",
        "BIDU_baidu",
        "3690.HK_meituan",
        "NTES_netease",
        "PDD_pdd",
        "1024.HK_kuaishou",
        "9626.HK_bilibili",
    }
    coverage = defaultdict(set)
    for row in rows:
        if row["doc_type"] == "annual_report":
            coverage[row["company_id"]].add(row["report_period"])
        for required in ["doc_id", "company_id", "doc_type", "report_period", "source_url", "local_path", "industry"]:
            assert row[required]

    for company_id in first_batch:
        assert {"FY2022", "FY2023", "FY2024"} <= coverage[company_id]


def test_ingestion_skips_financial_report_registries():
    files = discover_files("data/raw/financial_reports")
    names = {Path(path).name for path in files}
    assert "company_registry.csv" not in names
    assert "document_registry.csv" not in names
