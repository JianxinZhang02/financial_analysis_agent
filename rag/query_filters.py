from __future__ import annotations

import re


COMPANY_ALIASES = {
    "0700.HK_tencent": ("腾讯", "腾讯控股", "Tencent"),
    "3690.HK_meituan": ("美团", "美团-W", "Meituan"),
    "1024.HK_kuaishou": ("快手", "快手-W", "Kuaishou"),
    "BABA_alibaba": ("阿里", "阿里巴巴", "Alibaba"),
    "BIDU_baidu": ("百度", "Baidu"),
    "NTES_netease": ("网易", "NetEase"),
    "PDD_pdd": ("拼多多", "PDD"),
    "9626.HK_bilibili": ("哔哩哔哩", "B站", "Bilibili"),
}


def infer_metadata_filter(query: str) -> dict[str, str]:    # 这里是旧版本 之前的那个!!!
    metadata_filter: dict[str, str] = {}
    lowered = query.lower()

    for company_id, aliases in COMPANY_ALIASES.items():
        if any(alias.lower() in lowered for alias in aliases):
            metadata_filter["company_id"] = company_id
            break

    year_match = re.search(r"(20\d{2})\s*年?", query)
    if year_match:
        metadata_filter["report_period"] = f"FY{year_match.group(1)}"

    return metadata_filter


def normalize_query_for_metadata_filter(query: str, metadata_filter: dict[str, str] | None) -> str:     # 感觉这个有点抽象好像没用啊，甚至是负作用
    """Remove terms already enforced by metadata filters before lexical/vector search."""

    if not metadata_filter:
        return query

    normalized = query
    company_id = metadata_filter.get("company_id")
    if company_id:
        for alias in COMPANY_ALIASES.get(company_id, ()):
            normalized = re.sub(re.escape(alias), " ", normalized, flags=re.I)

    report_period = metadata_filter.get("report_period", "")
    year_match = re.match(r"FY(20\d{2})", report_period)
    if year_match:
        normalized = re.sub(rf"{year_match.group(1)}\s*年?", " ", normalized)

    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or query


def matches_metadata_filter(metadata: dict, metadata_filter: dict[str, str] | None) -> bool:
    if not metadata_filter:
        return True
    return all(str(metadata.get(key, "")) == value for key, value in metadata_filter.items())


def to_chroma_filter(metadata_filter: dict[str, str] | None) -> dict | None:
    if not metadata_filter:
        return None
    if len(metadata_filter) == 1:
        return dict(metadata_filter)
    return {"$and": [{key: value} for key, value in metadata_filter.items()]}
