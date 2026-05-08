from __future__ import annotations

from agent.state import FinancialAgentState
from rag.citation import has_valid_citation


def citation_guard_node(state: FinancialAgentState) -> dict:
    errors = []
    for idx, card in enumerate(state.get("evidence_cards", []), start=1):
        if not has_valid_citation(card):
            errors.append(f"证据卡片 {idx} 缺少强溯源字段。")

    draft = state.get("draft_answer", "")
    if errors:
        draft += "\n\n引用校验未完全通过：" + "；".join(errors)
    return {"citation_errors": errors, "final_answer": draft}
