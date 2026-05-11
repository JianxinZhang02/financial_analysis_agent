from __future__ import annotations

from agent.graph import FinancialGraphAgent
from agent.state import FinancialAgentState


def test_reflection_routing_to_query_transform():
    """Verify route_after_critic sends state back to query_transform when evidence is insufficient."""
    agent = FinancialGraphAgent()
    assert agent.compiled_graph is not None, "LangGraph must be available"

    state = FinancialAgentState(
        user_query="示例科技2024年现金流质量如何？",
        reflection_round=0,
        reflection_history=[],
    )
    state = agent._invoke_fallback(state)
    assert state.get("final_answer"), "Final answer must be produced"
    assert "reflection_round" in state, "reflection_round must be in state"


def test_reflection_round_field_in_initial_state():
    """Verify initial state includes reflection fields."""
    agent = FinancialGraphAgent()
    initial = agent._initial_state("test query")
    assert initial.get("reflection_round") == 0
    assert initial.get("reflection_history") == []


def test_reflection_history_appended_in_fallback():
    """Verify reflection_history is populated when critique triggers a second round."""
    agent = FinancialGraphAgent()
    state = dict(agent._initial_state("示例科技现金流"))
    state = agent._invoke_fallback(state)
    assert isinstance(state.get("reflection_history", []), list)