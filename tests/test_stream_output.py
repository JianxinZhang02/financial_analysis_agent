"""test_stream_output.py — 流式输出全链路验证。"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ── 1. build_reasoning_prompt 可独立调用 ──

def test_build_reasoning_prompt_returns_string():
    """拆出的prompt构建函数能独立工作，不含LLM调用。"""
    from agent.nodes.reasoning_node import build_reasoning_prompt
    state = {
        "user_query": "百度2025年发展趋势如何",
        "evidence_cards": [{"claim": "百度营收增长", "source_file": "baidu_2024.md", "page_number": 3}],
        "graph_relations": [],
        "calculations": [],
        "web_search_note": "",
        "user_profile": {"language_style": "professional", "risk_preference": "neutral"},
    }
    prompt = build_reasoning_prompt(state)
    assert isinstance(prompt, str)
    assert "百度2025年发展趋势如何" in prompt
    assert "专业严谨" in prompt   # style注入
    assert "客观中立" in prompt   # risk注入


# ── 2. reasoning_node skip_llm_reasoning模式 ──

def test_reasoning_node_skip_mode_stores_prompt():
    """skip_llm_reasoning=True时，reasoning_node只构建prompt不调LLM。"""
    from agent.nodes.reasoning_node import reasoning_node
    state = {
        "user_query": "百度2025年营收",
        "evidence_cards": [{"claim": "百度营收100亿", "source_file": "baidu.md", "page_number": 5}],
        "graph_relations": [],
        "calculations": [],
        "skip_llm_reasoning": True,
        "user_profile": {},
        "web_search_note": "",
    }
    result = reasoning_node(state)
    assert result.get("reasoning_prompt"), "should store prompt when skip mode"
    assert result.get("reasoning_llm_skipped") is True
    assert result.get("draft_answer") == ""  # 不调LLM，draft为空


def test_reasoning_node_skip_mode_no_evidence():
    """skip模式+无证据→reasoning_prompt为空（走fallback逻辑被跳过）。"""
    from agent.nodes.reasoning_node import reasoning_node
    state = {
        "user_query": "无证据问题",
        "evidence_cards": [],
        "skip_llm_reasoning": True,
        "user_profile": {},
        "web_search_note": "",
    }
    result = reasoning_node(state)
    assert result.get("reasoning_prompt") == ""  # 无证据，prompt为空
    assert result.get("draft_answer") == ""


def test_reasoning_node_normal_mode_unaffected():
    """skip_llm_reasoning=False时，reasoning_node正常行为不变。"""
    from agent.nodes.reasoning_node import reasoning_node
    # 无证据→fallback
    state_no_evidence = {
        "user_query": "test",
        "evidence_cards": [],
        "user_profile": {},
        "web_search_note": "",
    }
    result = reasoning_node(state_no_evidence)
    assert result.get("draft_answer"), "fallback should still work"
    assert "reasoning_prompt" not in result  # 正常模式不输出prompt字段


# ── 3. invoke_llm_stream generator行为 ──

def test_invoke_llm_stream_yields_tokens():
    """invoke_llm_stream返回generator，逐token yield。"""
    from agent.llm_utils import invoke_llm_stream

    # mock chat_model.stream() 返回几个token chunks
    mock_chunks = [
        MagicMock(content="你"),
        MagicMock(content="好"),
        MagicMock(content="！"),
    ]
    with patch("agent.llm_utils.chat_model") as mock_model, \
         patch("agent.llm_utils.using_real_llm", return_value=True):
        mock_model.stream.return_value = iter(mock_chunks)
        tokens = list(invoke_llm_stream("test prompt"))
        assert tokens == ["你", "好", "！"]
        mock_model.stream.assert_called_once_with("test prompt")


def test_invoke_llm_stream_fallback_when_no_real_llm():
    """主模型不可用时，invoke_llm_stream fallback到invoke_llm一次性输出。"""
    from agent.llm_utils import invoke_llm_stream

    with patch("agent.llm_utils.using_real_llm", return_value=False), \
         patch("agent.llm_utils.invoke_llm", return_value="完整回答"):
        tokens = list(invoke_llm_stream("test prompt"))
        assert tokens == ["完整回答"]  # fallback一次性yield完整文本


def test_invoke_llm_stream_exception_fallback():
    """stream中途异常→已有部分输出时不fallback（保留partial），只在0 token时才fallback。"""
    from agent.llm_utils import invoke_llm_stream

    # 场景1：有partial token → 不fallback，只yield partial
    def broken_stream_partial(prompt):
        yield MagicMock(content="部")
        raise RuntimeError("stream broken")

    with patch("agent.llm_utils.chat_model") as mock_model, \
         patch("agent.llm_utils.using_real_llm", return_value=True):
        mock_model.stream.side_effect = broken_stream_partial
        tokens = list(invoke_llm_stream("test prompt"))
        assert tokens == ["部"]  # 部分token保留，不fallback

    # 场景2：0 token → fallback到invoke_llm
    def broken_stream_no_token(prompt):
        raise RuntimeError("stream broken at start")

    with patch("agent.llm_utils.chat_model") as mock_model, \
         patch("agent.llm_utils.using_real_llm", return_value=True), \
         patch("agent.llm_utils.invoke_llm", return_value="fallback完整回答"):
        mock_model.stream.side_effect = broken_stream_no_token
        tokens = list(invoke_llm_stream("test prompt"))
        assert tokens == ["fallback完整回答"]


# ── 4. invoke_stream 两阶段调用 ──

def test_invoke_stream_returns_state_and_prompt():
    """invoke_stream返回(state, prompt)tuple，state中reasoning_prompt已填充。"""
    from agent.graph import FinancialGraphAgent

    with patch("agent.graph.FinancialGraphAgent.__init__", lambda self, uid: None):
        agent = FinancialGraphAgent.__new__(FinancialGraphAgent)
        agent.user_id = "test_user"
        agent.profiles = MagicMock()
        agent.profiles.get.return_value = {}
        agent.short_memory = MagicMock()
        agent.short_memory.messages = []
        agent.short_memory.snapshot.return_value = {}
        agent.heartbeat = MagicMock()
        agent.heartbeat.should_compact.return_value = False

        # mock compiled_graph.invoke返回带reasoning_prompt的state
        mock_state = {
            "reasoning_prompt": "你是金融分析师...",
            "draft_answer": "",
            "reasoning_llm_used": False,
            "reasoning_llm_skipped": True,
            "evidence_cards": [{"claim": "test"}],
            "entities": {"companies": ["百度"]},
        }
        agent.compiled_graph = MagicMock()
        agent.compiled_graph.invoke.return_value = mock_state

        state, prompt = agent.invoke_stream("百度2025年营收")
        assert prompt == "你是金融分析师..."
        assert state.get("reasoning_prompt")


# ── 5. _stream_finalize 完成记忆和画像更新 ──

def test_stream_finalize_updates_memory_and_profile():
    """_stream_finalize完成流式后更新记忆和画像。"""
    from agent.graph import FinancialGraphAgent

    with patch("agent.graph.FinancialGraphAgent.__init__", lambda self, uid: None):
        agent = FinancialGraphAgent.__new__(FinancialGraphAgent)
        agent.user_id = "test_user"
        agent.profiles = MagicMock()
        agent.short_memory = MagicMock()
        agent.short_memory.messages = []
        agent.heartbeat = MagicMock()
        agent.heartbeat.should_compact.return_value = False

        state = {"entities": {"companies": ["百度"], "metrics": ["营收"]}}
        agent._stream_finalize(state, "百度2025年营收增长20%")

        agent.short_memory.append.assert_called_with("assistant", "百度2025年营收增长20%")
        agent.profiles.remember_focus.assert_called()
        agent.profiles.remember_style.assert_called()
        assert state["final_answer"] == "百度2025年营收增长20%"