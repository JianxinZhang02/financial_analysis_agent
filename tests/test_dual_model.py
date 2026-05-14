"""test_dual_model.py — 验证主模型+快模型双轨架构。
所有测试使用mock，不触发真实LLM初始化。"""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

# ── 首先mock掉factory的真实模型实例化，避免ChatOpenAI连接超时 ──

# 创建mock模型实例
_mock_main_model = MagicMock()
_mock_main_model.__class__.__name__ = "ChatOpenAI"
_mock_main_model.invoke.return_value = MagicMock(content="main response")

_mock_fast_model = MagicMock()
_mock_fast_model.__class__.__name__ = "ChatOpenAI"
_mock_fast_model.invoke.return_value = MagicMock(content='{"result": "fast"}')

_mock_embed_model = MagicMock()


# ── 1. invoke_fast_llm逻辑验证 ──

def test_invoke_fast_llm_calls_fast_model():
    """invoke_fast_llm应调用fast_model而非chat_model。"""
    with patch("agent.llm_utils.fast_model", _mock_fast_model), \
         patch("agent.llm_utils.using_real_fast_llm", return_value=True):
        from agent.llm_utils import invoke_fast_llm
        result = invoke_fast_llm("test prompt")
        _mock_fast_model.invoke.assert_called_with("test prompt")


def test_invoke_fast_llm_fallback_to_main():
    """快模型不可用时，invoke_fast_llm应自动升级到invoke_llm。"""
    with patch("agent.llm_utils.using_real_fast_llm", return_value=False), \
         patch("agent.llm_utils.invoke_llm", return_value="main model response") as mock_main:
        from agent.llm_utils import invoke_fast_llm
        result = invoke_fast_llm("test prompt")
        mock_main.assert_called_once()


def test_invoke_llm_always_main():
    """invoke_llm应始终调用chat_model（主模型）。"""
    with patch("agent.llm_utils.chat_model", _mock_main_model), \
         patch("agent.llm_utils.using_real_llm", return_value=True):
        from agent.llm_utils import invoke_llm
        result = invoke_llm("test prompt")
        _mock_main_model.invoke.assert_called_once_with("test prompt")


# ── 2. 各节点使用正确的模型层 ──

def test_router_node_uses_fast_model():
    """router_node应调用invoke_fast_llm（快模型）。"""
    mock_response = '{"intent": "calculation", "confidence": 0.92, "entities": {"companies": ["腾讯控股"], "metrics": ["营收"], "years": ["2024"], "doc_type": "annual_report"}}'
    with patch("agent.nodes.router_node.invoke_fast_llm", return_value=mock_response) as mock_fast:
        from agent.nodes.router_node import _llm_route_and_extract
        result = _llm_route_and_extract("腾讯2024营收同比增长率")
        mock_fast.assert_called_once()
    assert result["intent"] == "calculation"


def test_calculator_node_uses_fast_model():
    """calculator_node应调用invoke_fast_llm（快模型）。"""
    mock_response = '{"calculations": [{"metric": "营业收入同比增长率", "formula": "(6100-5600)/5600*100", "value": 8.93, "unit": "%", "period": "2024 vs 2023"}]}'
    with patch("agent.nodes.calculator_node.invoke_fast_llm", return_value=mock_response) as mock_fast:
        from agent.nodes.calculator_node import _llm_calculate
        result = _llm_calculate("腾讯2024营收同比增长率", "evidence text")
        mock_fast.assert_called_once()
    assert len(result["calculations"]) == 1


def test_query_transform_uses_fast_model():
    """query_transform_node应调用invoke_fast_llm（快模型）。"""
    mock_response = '{"rewritten_query": "腾讯控股2024年营业收入同比增速", "sub_queries": ["腾讯2024营收"], "hyde_document": "假设性分析", "required_metrics": ["营收"], "intent_hint": "calculation"}'
    entities = {"companies": ["腾讯控股"], "metrics": ["营收"], "years": ["2024"], "doc_type": "annual_report"}
    with patch("agent.nodes.query_transform_node.invoke_fast_llm", return_value=mock_response) as mock_fast:
        from agent.nodes.query_transform_node import _llm_query_plan
        plan = _llm_query_plan("腾讯2024营收同比增长率", entities)
        mock_fast.assert_called_once()
    assert "腾讯" in plan["rewritten_query"]


def test_critic_node_uses_fast_model():
    """critic_node应调用invoke_fast_llm（快模型）。"""
    mock_response = '{"issues": ["FORMAT:引用格式可改进"], "summary": "整体质量可接受"}'
    state = {
        "user_query": "腾讯2024财报",
        "draft_answer": "根据年报数据，腾讯2024营收6100亿元。",
        "evidence_cards": [{"source_file": "report.md", "chunk_id": "c1", "page_number": 5, "evidence": "营收数据"}],
    }
    with patch("agent.nodes.critic_node.invoke_fast_llm", return_value=mock_response) as mock_fast:
        from agent.nodes.critic_node import _llm_critique
        result = _llm_critique(state)
        mock_fast.assert_called_once()
    assert "issues" in result


def test_reasoning_node_uses_main_model():
    """reasoning_node应调用invoke_llm（主模型），不调用invoke_fast_llm。"""
    with patch("agent.nodes.reasoning_node.invoke_llm", return_value="深度推理结果") as mock_main:
        from agent.nodes.reasoning_node import _llm_reasoning
        state = {
            "user_query": "腾讯2024财报深度分析",
            "evidence_cards": [{"source_file": "r.md", "chunk_id": "c1", "page_number": 5, "evidence": "数据"}],
            "calculations": [{"metric": "营收同比", "value": 8.93}],
            "graph_relations": [],
        }
        result = _llm_reasoning(state)
        mock_main.assert_called_once()


# ── 3. 配置读取验证 ──

def test_config_has_fast_model_section():
    """model.yaml应包含fast_model配置段。"""
    from utils.config_handler import model_cof
    fast_provider = model_cof.get("fast_model_provider")
    fast_name = model_cof.get("fast_model_name")
    assert fast_provider is not None
    assert fast_name is not None


# ── 4. invoke_fast_llm重试逻辑 ──

def test_invoke_fast_llm_retry_on_failure():
    """invoke_fast_llm应有重试机制。"""
    local_mock = MagicMock()
    local_mock.invoke.side_effect = [Exception("timeout"), Exception("timeout"), MagicMock(content="success")]
    with patch("agent.llm_utils.fast_model", local_mock), \
         patch("agent.llm_utils.using_real_fast_llm", return_value=True), \
         patch("agent.llm_utils.time.sleep", MagicMock()):
        from agent.llm_utils import invoke_fast_llm
        result = invoke_fast_llm("test", max_retries=3)
        assert result == "success"
        assert local_mock.invoke.call_count == 3