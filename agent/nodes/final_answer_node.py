from __future__ import annotations

from agent.state import FinancialAgentState


def final_answer_node(state: FinancialAgentState) -> dict:
    critique = state.get("critique_result", {})
    final_answer = state.get("final_answer") or state.get("draft_answer", "")
    if critique and not critique.get("passed", True):
        issues = "；".join(critique.get("issues", []))
        final_answer += f"\n\n审查提示：{issues}"
    return {"final_answer": final_answer.strip()}
