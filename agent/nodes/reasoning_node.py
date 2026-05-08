from __future__ import annotations

from agent.llm_utils import compact_json, invoke_llm
from agent.state import FinancialAgentState
from rag.citation import EvidenceCard, citation_text
from utils.config_handler import compliance_cof


def _cards(state: FinancialAgentState) -> list[EvidenceCard]:
    return [EvidenceCard.from_dict(card) for card in state.get("evidence_cards", [])]


def _fallback_reasoning(state: FinancialAgentState, error: str | None = None) -> dict:
    query = state.get("user_query", "")
    cards = _cards(state)
    relations = state.get("graph_relations", [])
    calculations = state.get("calculations", [])
    web_note = state.get("web_search_note")

    if not cards:
        return {
            "draft_answer": (
                f"当前资料不足以回答“{query}”。建议补充公司年报、券商研报、电话会议纪要或实时新闻数据后再分析。"
            ),
            "reasoning_llm_used": False,
            "reasoning_llm_error": error,
        }

    lines = [f"问题：{query}", "", "结论摘要："]
    for idx, card in enumerate(cards[:4], start=1):
        lines.append(f"{idx}. {card.claim} {citation_text(card)}")

    if calculations:
        lines.append("")
        lines.append("可复核计算：")
        for item in calculations:
            lines.append(
                f"- {item['metric']} = {item.get('value')} {item.get('unit', '')}，公式：{item.get('formula')}"
            )

    if relations:
        lines.append("")
        lines.append("GraphRAG 实体关系辅助推理：")
        for relation in relations[:5]:
            page = f"第{relation.get('page_number')}页" if relation.get("page_number") else "页码未知"
            lines.append(
                f"- {relation['head']} --{relation['relation']}--> {relation['tail']} "
                f"【来源：{relation.get('source_file')}，{page}】"
            )

    lines.append("")
    lines.append("分析判断：")
    lines.append(
        "综合已检索证据，若多个证据均指向同一趋势，可作为研究判断；若证据来自预测或管理层展望，应视为假设而非既成事实。"
    )
    if web_note:
        lines.append(web_note)
    lines.append(compliance_cof.get("risk_disclaimer", "以上内容不构成投资建议。"))
    return {
        "draft_answer": "\n".join(lines),
        "reasoning_llm_used": False,
        "reasoning_llm_error": error,
    }


def _llm_reasoning(state: FinancialAgentState) -> str:
    query = state.get("user_query", "")
    cards = state.get("evidence_cards", [])
    relations = state.get("graph_relations", [])
    calculations = state.get("calculations", [])
    web_note = state.get("web_search_note", "")
    risk_disclaimer = compliance_cof.get("risk_disclaimer", "以上内容不构成投资建议。")

    prompt = f"""
你是金融研报分析师 Agent。请基于给定 EvidenceCard、GraphRAG 关系和可复核计算回答用户问题。

硬性约束：
1. 所有财务数字、公司结论、预测假设必须来自 EvidenceCard 或计算结果。
2. 每条关键结论必须附引用，格式为：【来源：文件名，第X页】。
3. 区分“事实”“预测/假设”“分析判断”，不要把预测当成事实。
4. 不得输出确定性买卖建议，不得承诺收益。
5. 如果证据不足，明确说“当前资料不足以确认”，不要编造。
6. 用中文输出，结构清晰，最后附风险提示。

用户问题：
{query}

EvidenceCard 列表：
{compact_json(cards)}

GraphRAG 关系：
{compact_json(relations)}

可复核计算：
{compact_json(calculations)}

外部搜索说明：
{web_note}

风险提示固定句：
{risk_disclaimer}
"""
    return invoke_llm(prompt)


def reasoning_node(state: FinancialAgentState) -> dict:
    if not state.get("evidence_cards"):
        return _fallback_reasoning(state)
    try:
        draft = _llm_reasoning(state)
        return {"draft_answer": draft, "reasoning_llm_used": True}
    except Exception as exc:
        return _fallback_reasoning(state, error=str(exc))
