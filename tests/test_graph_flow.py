from agent.graph import FinancialGraphAgent


def test_graph_agent_fallback_flow():
    state = FinancialGraphAgent().invoke("示例科技2024年现金流质量如何？")
    assert state["final_answer"]
    assert "来源" in state["final_answer"]
