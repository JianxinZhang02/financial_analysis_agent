from __future__ import annotations

import re

from agent.state import FinancialAgentState


def _pct_change(new: float, old: float) -> float:
    return (new - old) / old * 100 if old else 0.0


def calculator_node(state: FinancialAgentState) -> dict:
    query = state.get("user_query", "")
    text = "\n".join(card.get("evidence", "") for card in state.get("evidence_cards", []))
    calculations = []

    should_calculate_growth = any(keyword in query for keyword in ["同比", "增长率", "CAGR", "增速"])
    should_calculate_pe = any(keyword in query for keyword in ["市盈率", "PE", "估值", "多少倍"])

    revenue_values = re.findall(r"20(2[2-5])年[^。\n]*?营业收入(?:为)?([0-9.]+)亿元", text)
    if should_calculate_growth and len(revenue_values) >= 2:
        revenue_values = sorted((int("20" + year), float(value)) for year, value in revenue_values)
        old_year, old_value = revenue_values[-2]
        new_year, new_value = revenue_values[-1]
        calculations.append(
            {
                "metric": "营业收入同比",
                "formula": f"({new_value}-{old_value})/{old_value}*100",
                "value": round(_pct_change(new_value, old_value), 2),
                "unit": "%",
                "period": f"{new_year} vs {old_year}",
            }
        )

    market_cap_match = re.search(r"市值(?:为)?([0-9.]+)亿元", text)
    net_profit_match = re.search(r"净利润(?:为)?([0-9.]+)亿元", text)
    if should_calculate_pe and market_cap_match and net_profit_match:
        market_cap = float(market_cap_match.group(1))
        net_profit = float(net_profit_match.group(1))
        calculations.append(
            {
                "metric": "市盈率",
                "formula": f"{market_cap}/{net_profit}",
                "value": round(market_cap / net_profit, 2) if net_profit else None,
                "unit": "x",
            }
        )
    return {"calculations": calculations}
