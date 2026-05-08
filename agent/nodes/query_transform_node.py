from __future__ import annotations

from agent.llm_utils import compact_json, extract_json_object, invoke_llm
from agent.state import FinancialAgentState


def _metric_expansion(query: str) -> list[str]:
    expansions: list[str] = []
    if "现金流" in query:
        expansions.extend(["经营活动现金流量净额", "自由现金流", "净现比", "应收账款周转"])
    if "盈利" in query or "利润" in query:
        expansions.extend(["营业收入", "归母净利润", "毛利率", "费用率"])
    if "估值" in query or "市盈率" in query or "PE" in query.upper():
        expansions.extend(["市盈率", "盈利预测", "市值", "估值假设"])
    if "风险" in query:
        expansions.extend(["风险提示", "预算波动", "成本变化", "竞争加剧"])
    return list(dict.fromkeys(expansions))


def _fallback_query_plan(query: str, entities: dict) -> dict:
    expansions = _metric_expansion(query)
    entity_text = " ".join(sum(entities.values(), [])) if entities else ""
    rewritten = " ".join(part for part in [query, entity_text, " ".join(expansions)] if part)
    sub_queries = [query]
    if any(keyword in query for keyword in ["和", "影响", "如何", "为什么", "是否"]):
        for metric in expansions[:4]:
            sub_queries.append(f"{entity_text} {metric} {query}".strip())
    hyde = (
        f"假设性金融分析文档：围绕“{query}”，需要检查收入、利润、毛利率、现金流、估值、"
        "管理层表述、研报预测和风险提示，并交叉验证来源页码。"
    )
    sub_queries.append(hyde)
    return {
        "original_query": query,
        "rewritten_query": rewritten,
        "hyde_document": hyde,
        "entities": entities,
        "required_metrics": expansions,
        "sub_queries": list(dict.fromkeys(sub_queries)),
        "llm_used": False,
    }


def _llm_query_plan(query: str, entities: dict) -> dict:
    prompt = f"""
你是金融研报 RAG 系统的查询规划节点。请把用户问题改写成适合混合检索和多跳推理的结构化查询。

要求：
1. 输出严格 JSON，不要 Markdown，不要额外解释。
2. rewritten_query 要包含金融专业术语、时间范围、公司/股票代码、指标。
3. sub_queries 用于多路召回，最多 5 条，必须具体、可检索。
4. hyde_document 是一段假设性金融分析文档，仅用于向量检索，不得当作最终事实。
5. required_metrics 只列与问题直接相关的指标。

用户问题：
{query}

已抽取实体：
{compact_json(entities)}

输出 JSON schema：
{{
  "rewritten_query": "...",
  "sub_queries": ["..."],
  "hyde_document": "...",
  "required_metrics": ["..."],
  "intent_hint": "financial_analysis|calculation|graph_reasoning|realtime_financial_search"
}}
"""
    raw = invoke_llm(prompt)
    data = extract_json_object(raw)
    sub_queries = data.get("sub_queries") or [query]
    if not isinstance(sub_queries, list):
        sub_queries = [str(sub_queries)]
    required_metrics = data.get("required_metrics") or _metric_expansion(query)
    if not isinstance(required_metrics, list):
        required_metrics = [str(required_metrics)]
    hyde = str(data.get("hyde_document") or "")
    if hyde:
        sub_queries.append(hyde)
    return {
        "original_query": query,
        "rewritten_query": str(data.get("rewritten_query") or query),
        "hyde_document": hyde,
        "entities": entities,
        "required_metrics": [str(item) for item in required_metrics],
        "sub_queries": list(dict.fromkeys([str(item) for item in sub_queries if str(item).strip()])),
        "intent_hint": str(data.get("intent_hint") or ""),
        "llm_used": True,
        "llm_raw": raw,
    }


def query_transform_node(state: FinancialAgentState) -> dict:
    query = state.get("user_query", "")
    entities = state.get("entities", {})
    try:
        plan = _llm_query_plan(query, entities)
    except Exception as exc:
        plan = _fallback_query_plan(query, entities)
        plan["llm_error"] = str(exc)

    sub_queries = [query, plan.get("rewritten_query", query), *plan.get("sub_queries", [])]
    return {
        "query_plan": plan,
        "sub_queries": list(dict.fromkeys([item for item in sub_queries if item])),
    }
