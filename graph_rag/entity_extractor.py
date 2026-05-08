from __future__ import annotations

import re


FINANCIAL_METRICS = ["营收", "营业收入", "净利润", "毛利率", "现金流", "自由现金流", "市盈率", "ROE", "资产负债率"]


def extract_entities(text: str) -> dict[str, list[str]]:
    companies = sorted(
        set(re.findall(r"[\u4e00-\u9fffA-Za-z]{2,20}(?:科技|股份|银行|证券|集团|公司)", text))
    )
    tickers = sorted(set(re.findall(r"\b\d{6}\.(?:SZ|SH|BJ)\b", text.upper())))
    metrics = [metric for metric in FINANCIAL_METRICS if metric in text]
    years = sorted(set(re.findall(r"20\d{2}年?", text)))
    return {"companies": companies, "tickers": tickers, "metrics": metrics, "years": years}
