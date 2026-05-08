from __future__ import annotations

from collections.abc import Iterator

from agent.nodes.calculator_node import calculator_node
from agent.nodes.citation_guard_node import citation_guard_node
from agent.nodes.critic_node import critic_node
from agent.nodes.final_answer_node import final_answer_node
from agent.nodes.graph_rag_node import graph_rag_node
from agent.nodes.query_transform_node import query_transform_node
from agent.nodes.reasoning_node import reasoning_node
from agent.nodes.retrieval_node import retrieval_node
from agent.nodes.router_node import router_node
from agent.nodes.web_search_node import web_search_node
from agent.state import FinancialAgentState
from memory.heartbeat import MemoryHeartbeat
from memory.short_term import ShortTermMemory
from memory.user_profile import UserProfileService
from utils.config_handler import agent_cof, graph_cof, memory_cof


class FinancialGraphAgent:
    """Financial multi-hop analysis agent with optional LangGraph runtime."""

    def __init__(self, user_id: str | None = None):
        self.user_id = user_id or agent_cof.get("default_user_id", "default")
        self.short_memory = ShortTermMemory(window_size=int(memory_cof.get("short_term_window", 8)))
        self.heartbeat = MemoryHeartbeat()
        self.profiles = UserProfileService()
        self.compiled_graph = self._try_build_langgraph()

    def _try_build_langgraph(self):
        try:
            from langgraph.graph import END, START, StateGraph

            workflow = StateGraph(FinancialAgentState)
            workflow.add_node("router", router_node)
            workflow.add_node("query_transform", query_transform_node)
            workflow.add_node("retrieval", retrieval_node)
            workflow.add_node("graph_rag", graph_rag_node)
            workflow.add_node("calculator", calculator_node)
            workflow.add_node("web_search", web_search_node)
            workflow.add_node("reasoning", reasoning_node)
            workflow.add_node("critic", critic_node)
            workflow.add_node("citation_guard", citation_guard_node)
            workflow.add_node("final_answer", final_answer_node)

            workflow.add_edge(START, "router")
            workflow.add_edge("router", "query_transform")
            workflow.add_edge("query_transform", "retrieval")
            workflow.add_edge("retrieval", "graph_rag")
            workflow.add_edge("graph_rag", "calculator")
            workflow.add_edge("calculator", "web_search")
            workflow.add_edge("web_search", "reasoning")
            workflow.add_edge("reasoning", "critic")
            workflow.add_edge("critic", "citation_guard")
            workflow.add_edge("citation_guard", "final_answer")
            workflow.add_edge("final_answer", END)
            return workflow.compile()
        except Exception:
            return None

    def _initial_state(self, query: str) -> FinancialAgentState:
        return {
            "messages": list(self.short_memory.messages) + [{"role": "user", "content": query}],
            "user_id": self.user_id,
            "user_query": query,
            "reflection_round": 0,
            "memory_snapshot": self.short_memory.snapshot(),
        }

    def invoke(self, query: str) -> FinancialAgentState:
        state = self._initial_state(query)
        if self.compiled_graph is not None:
            state = self.compiled_graph.invoke(state)
        else:
            state = self._invoke_fallback(state)

        final_answer = state.get("final_answer", "")
        self.short_memory.append("user", query)
        self.short_memory.append("assistant", final_answer)
        if self.heartbeat.should_compact(list(self.short_memory.messages)):
            self.short_memory.summary = self.heartbeat.summarize(
                list(self.short_memory.messages), self.short_memory.summary
            )
        entities = state.get("entities", {})
        self.profiles.remember_focus(
            self.user_id,
            entities.get("companies", []),
            entities.get("metrics", []),
        )
        return state

    def _invoke_fallback(self, state: FinancialAgentState) -> FinancialAgentState:
        for node in [router_node, query_transform_node, retrieval_node, graph_rag_node, calculator_node]:
            state.update(node(state))
        if state.get("needs_web_search") or graph_cof.get("enable_web_fallback", False):
            state.update(web_search_node(state))
        else:
            state.setdefault("web_search_note", "")
        state.update(reasoning_node(state))
        if graph_cof.get("enable_critic", True):
            state.update(critic_node(state))
        if graph_cof.get("enable_citation_guard", True):
            state.update(citation_guard_node(state))
        state.update(final_answer_node(state))
        return state

    def execute_stream(self, query: str) -> Iterator[str]:
        state = self.invoke(query)
        yield state.get("final_answer", "")


if __name__ == "__main__":
    agent = FinancialGraphAgent()
    result = agent.invoke("示例科技2024年现金流质量如何？")
    print(result["final_answer"])
