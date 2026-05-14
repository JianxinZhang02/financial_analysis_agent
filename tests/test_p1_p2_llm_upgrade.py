"""test_p1_p2_llm_upgrade.py — 验证P1(router LLM)和P2(calculator LLM)的核心逻辑。"""
import pytest
from unittest.mock import patch, MagicMock

# ── P1: router_node LLM意图分类+实体抽取 ──

def test_router_keyword_fallback():
    """关键词fast-path正确分类意图。"""
    from agent.nodes.router_node import _keyword_route_and_extract

    # realtime intent
    result = _keyword_route_and_extract("腾讯最新股价是多少")
    assert result["intent"] == "realtime_financial_search"
    assert result["needs_web_search"] is True

    # calculation intent
    result = _keyword_route_and_extract("腾讯2024营收同比增长率")
    assert result["intent"] == "calculation"

    # graph intent
    result = _keyword_route_and_extract("上游供应商对腾讯的影响")
    assert result["intent"] == "graph_reasoning"

    # default: financial_analysis
    result = _keyword_route_and_extract("腾讯2024年财报分析")
    assert result["intent"] == "financial_analysis"


def test_router_keyword_entities_from_aliases():
    """关键词fast-path能通过knowledge aliases匹配公司。"""
    from agent.nodes.router_node import _keyword_route_and_extract

    result = _keyword_route_and_extract("腾讯2024年营收同比增长率")
    entities = result["entities"]
    companies = entities.get("companies", [])
    assert len(companies) > 0
    assert any("tencent" in c.lower() or "腾讯" in c for c in companies)


def test_router_llm_with_mock():
    """LLM路由：mock invoke_fast_llm，验证JSON解析和字段校验。"""
    from agent.nodes.router_node import _llm_route_and_extract

    mock_response = '{"intent": "calculation", "confidence": 0.92, "entities": {"companies": ["腾讯控股", "Tencent"], "metrics": ["营收", "同比增长率"], "years": ["2024", "2023"], "doc_type": "annual_report"}}'

    with patch("agent.nodes.router_node.invoke_fast_llm", return_value=mock_response):
        result = _llm_route_and_extract("腾讯2024营收同比增长率")

    assert result["intent"] == "calculation"
    assert result["confidence"] == 0.92
    assert "腾讯控股" in result["entities"]["companies"]
    assert "营收" in result["entities"]["metrics"]
    assert result["entities"]["doc_type"] == "annual_report"
    assert result["llm_used"] is True


def test_router_llm_invalid_intent_fallback():
    """LLM返回无效意图时自动降级为financial_analysis。"""
    from agent.nodes.router_node import _llm_route_and_extract

    mock_response = '{"intent": "unknown_intent", "confidence": 0.5, "entities": {"companies": [], "metrics": [], "years": [], "doc_type": "unknown"}}'

    with patch("agent.nodes.router_node.invoke_fast_llm", return_value=mock_response):
        result = _llm_route_and_extract("随便问问")

    assert result["intent"] == "financial_analysis"


def test_router_node_llm_degradation():
    """LLM调用失败时降级到关键词fast-path。"""
    from agent.nodes.router_node import router_node

    state = {"user_query": "腾讯2024营收同比增长率", "intent": "", "entities": {}, "needs_web_search": False}

    with patch("agent.nodes.router_node.using_real_llm", return_value=True), \
         patch("agent.nodes.router_node.invoke_fast_llm", side_effect=Exception("LLM timeout")), \
         patch("agent.nodes.router_node.as_bool", return_value=True):
        result = router_node(state)

    assert result["intent"] == "calculation"


# ── P2: calculator_node LLM数值提取 ──

def test_calculator_regex_fallback_yoy():
    """regex fallback能计算营业收入同比增长率（含逗号数字）。"""
    from agent.nodes.calculator_node import _regex_calculate

    evidence = "2023年营业收入为5600亿元。2024年营业收入为6100亿元。"
    result = _regex_calculate("腾讯2024营收同比增长率", evidence)
    assert len(result["calculations"]) >= 1
    calc = result["calculations"][0]
    assert "营业收入同比" in calc["metric"] or "同比" in calc["metric"]
    assert calc["value"] is not None
    assert abs(calc["value"] - 8.93) < 0.5


def test_calculator_regex_fallback_yoy_with_commas():
    """regex fallback能处理逗号分隔的数字（如5,600）。"""
    from agent.nodes.calculator_node import _regex_calculate

    evidence = "2023年营业收入为5,600亿元。2024年营业收入为6,100亿元。"
    result = _regex_calculate("腾讯2024营收同比增长率", evidence)
    assert len(result["calculations"]) >= 1
    calc = result["calculations"][0]
    assert calc["value"] is not None
    assert abs(calc["value"] - 8.93) < 0.5


def test_calculator_regex_fallback_no_match():
    """regex fallback无匹配时返回空列表。"""
    from agent.nodes.calculator_node import _regex_calculate

    evidence = "这是一段没有数值的文本。"
    result = _regex_calculate("随便问问", evidence)
    assert len(result["calculations"]) == 0


def test_calculator_llm_with_mock():
    """LLM计算：mock invoke_fast_llm，验证JSON解析。"""
    from agent.nodes.calculator_node import _llm_calculate

    mock_response = '{"calculations": [{"metric": "营业收入同比增长率", "formula": "(6100-5600)/5600*100", "value": 8.93, "unit": "%", "period": "2024 vs 2023"}]}'

    with patch("agent.nodes.calculator_node.invoke_fast_llm", return_value=mock_response):
        result = _llm_calculate("腾讯2024营收同比增长率", "2023年营业收入为5600亿元。2024年营业收入为6100亿元。")

    assert len(result["calculations"]) == 1
    calc = result["calculations"][0]
    assert calc["metric"] == "营业收入同比增长率"
    assert calc["value"] == 8.93
    assert calc["unit"] == "%"
    assert result["llm_used"] is True


def test_calculator_llm_bad_response_fallback():
    """LLM调用失败时，calculator_node降级到regex。"""
    from agent.nodes.calculator_node import calculator_node

    state = {
        "user_query": "腾讯2024营收同比增长率",
        "evidence_cards": [{"evidence": "2023年营业收入为5600亿元。2024年营业收入为6100亿元。"}],
        "calculations": [],
    }

    with patch("agent.nodes.calculator_node.using_real_llm", return_value=True), \
         patch("agent.nodes.calculator_node.invoke_fast_llm", side_effect=Exception("bad json")), \
         patch("agent.nodes.calculator_node.as_bool", return_value=True):
        result = calculator_node(state)

    assert len(result["calculations"]) >= 1


def test_query_transform_entity_text_mixed_types():
    """entities dict中既有list又有str值时，query_transform能正确拼接。"""
    from agent.nodes.query_transform_node import _fallback_query_plan

    entities = {
        "companies": ["腾讯"],
        "metrics": ["营收", "净利润"],
        "years": ["2024"],
        "doc_type": "annual_report",
    }
    plan = _fallback_query_plan("腾讯2024财报", entities)
    assert "腾讯" in plan["rewritten_query"]
    assert "营收" in plan["rewritten_query"]