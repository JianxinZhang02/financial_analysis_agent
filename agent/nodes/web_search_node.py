from __future__ import annotations

from agent.state import FinancialAgentState
from utils.logger_handler import log_stage, safe_preview


def web_search_node(state: FinancialAgentState) -> dict:
    query = state.get("user_query", "")
    with log_stage("web_search", query=safe_preview(query), enabled=bool(state.get("needs_web_search"))) as stage:
        note = (
            f"当前未接入实时网页/行情搜索 API；问题“{query}”仅基于本地知识库回答。"
            "如需实时行情或最新公告，可在 config/graph.yaml 开启并接入外部搜索工具。"
        )
        stage.add_done_fields(note_chars=len(note), evidence_cards=len(state.get("evidence_cards", [])))
        return {
            "evidence_cards": state.get("evidence_cards", []),
            "web_search_note": note,
        }
