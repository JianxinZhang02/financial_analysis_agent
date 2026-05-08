from __future__ import annotations

import sys

from agent.graph import FinancialGraphAgent


def run_cli() -> None:
    agent = FinancialGraphAgent()
    query = " ".join(sys.argv[1:]).strip() or "示例科技2024年现金流质量如何？"
    result = agent.invoke(query)
    print(result["final_answer"])


def run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="金融研报分析 Agent", layout="wide")
    st.title("金融研报分析与多跳推理 Agent")

    if "agent" not in st.session_state:
        st.session_state["agent"] = FinancialGraphAgent()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    left, right = st.columns([0.68, 0.32])
    with left:
        for message in st.session_state["messages"]:
            st.chat_message(message["role"]).write(message["content"])

        query = st.chat_input("输入公司、指标、研报分析或多跳推理问题")
        if query:
            st.chat_message("user").write(query)
            st.session_state["messages"].append({"role": "user", "content": query})
            with st.spinner("正在检索研报、重排证据并进行合规审查..."):
                state = st.session_state["agent"].invoke(query)
            answer = state["final_answer"]
            st.chat_message("assistant").write(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.session_state["last_state"] = state

    with right:
        st.subheader("证据与审查")
        state = st.session_state.get("last_state", {})
        for card in state.get("evidence_cards", [])[:6]:
            with st.expander(card.get("claim", "证据卡片")[:60]):
                st.write(card.get("evidence", ""))
                st.caption(f"{card.get('source_file')} | page={card.get('page_number')} | score={card.get('score'):.3f}")
        critique = state.get("critique_result")
        if critique:
            st.json(critique)


if __name__ == "__main__":
    try:
        import streamlit.runtime.scriptrunner as _st_runner

        if _st_runner.get_script_run_ctx():
            run_streamlit()
        else:
            run_cli()
    except Exception:
        run_cli()
