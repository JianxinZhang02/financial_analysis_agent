from __future__ import annotations

from agent.state import FinancialAgentState
from graph_rag.entity_extractor import extract_entities
from utils.logger_handler import log_stage, safe_preview


REALTIME_KEYWORDS = ["今天", "最新", "新闻", "实时", "股价", "公告", "监管"]
CALC_KEYWORDS = ["计算", "多少倍", "同比", "增长率", "CAGR", "市盈率", "敏感性"]
GRAPH_KEYWORDS = ["关系", "上下游", "供应商", "客户", "影响", "产业链"]


def router_node(state: FinancialAgentState) -> dict:
    query = state.get("user_query", "")
    with log_stage("router", query=safe_preview(query), query_chars=len(query)) as stage:
        entities = extract_entities(query)
        if any(keyword in query for keyword in REALTIME_KEYWORDS):
            intent = "realtime_financial_search"
        elif any(keyword in query for keyword in CALC_KEYWORDS):
            intent = "calculation"
        elif any(keyword in query for keyword in GRAPH_KEYWORDS):
            intent = "graph_reasoning"
        else:
            intent = "financial_analysis"

        companies = entities.get("companies", []) if isinstance(entities, dict) else []
        metrics = entities.get("metrics", []) if isinstance(entities, dict) else []
        stage.add_done_fields(
            intent=intent,
            companies=len(companies),
            metrics=len(metrics),
            needs_web_search=intent == "realtime_financial_search",
        )
        return {
            "intent": intent,
            "entities": entities,
            "needs_web_search": intent == "realtime_financial_search",
        }
