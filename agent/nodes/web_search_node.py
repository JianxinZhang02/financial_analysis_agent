from __future__ import annotations

from agent.state import FinancialAgentState


def web_search_node(state: FinancialAgentState) -> dict:
    query = state.get("user_query", "")
    return {
        "evidence_cards": state.get("evidence_cards", []),
        "web_search_note": (
            f"当前本地运行未启用实时金融新闻 API，问题“{query}”只能基于本地资料回答；"
            "如需实时行情/新闻，请在 config/graph.yaml 开启外部搜索并配置数据源。"
        ),
    }
