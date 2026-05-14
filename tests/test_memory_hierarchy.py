"""test_memory_hierarchy.py — L1短期记忆持久化 + L2会话摘要消费闭环验证。"""
import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════
# L1: ShortTermMemory 持久化
# ══════════════════════════════════════════════


def test_short_term_memory_persist_writes_session_state():
    """persist() → StorageBackend.save_session_state 被调用，传入summary+slots。"""
    from memory.short_term import ShortTermMemory

    stm = ShortTermMemory(window_size=4)
    stm.summary = "用户之前问了百度营收"
    stm.set_slot("last_company", "百度")
    stm.set_slot("last_metric", "营收")

    mock_backend = MagicMock()
    stm.persist("test_user", backend=mock_backend)

    mock_backend.save_session_state.assert_called_once_with("test_user", {
        "summary": "用户之前问了百度营收",
        "slots": {"last_company": "百度", "last_metric": "营收"},
    })


def test_short_term_memory_restore_reads_session_state():
    """restore() → 从StorageBackend读取summary+slots，恢复到内存。"""
    from memory.short_term import ShortTermMemory

    stm = ShortTermMemory(window_size=4)
    mock_backend = MagicMock()
    mock_backend.get_session_state.return_value = {
        "summary": "关注百度现金流质量",
        "slots": {"last_company": "百度", "last_year": 2024},
    }

    stm.restore("test_user", backend=mock_backend)

    assert stm.summary == "关注百度现金流质量"
    assert stm.slots == {"last_company": "百度", "last_year": 2024}


def test_short_term_memory_persist_restore_roundtrip():
    """persist → restore 完整闭环：数据写入后读取回来一致。"""
    from memory.short_term import ShortTermMemory
    from memory.storage_backend import DiskBackend

    stm = ShortTermMemory(window_size=4)
    stm.summary = "第1轮问了腾讯ROE，第2轮问了百度现金流"
    stm.set_slot("mentioned_companies", ["腾讯", "百度"])

    backend = DiskBackend()
    stm.persist("roundtrip_user", backend=backend)

    # 新建一个STM实例模拟"重启后"
    stm2 = ShortTermMemory(window_size=4)
    stm2.restore("roundtrip_user", backend=backend)

    assert stm2.summary == stm.summary
    assert stm2.slots == stm.slots


def test_disk_backend_session_state_file_path():
    """DiskBackend的session state文件路径格式正确：{user_id}_session.json。"""
    from memory.storage_backend import DiskBackend

    db = DiskBackend()
    f = db._session_state_file("alice")
    assert f.name == "alice_session.json"


def test_disk_backend_session_state_empty_when_no_file():
    """DiskBackend读取不存在用户时返回空模板。"""
    from memory.storage_backend import DiskBackend

    db = DiskBackend()
    state = db.get_session_state("nonexistent_user_xyz")
    assert state == {"summary": "", "slots": {}}


def test_redis_backend_session_state_key_format():
    """RedisBackend的session key格式：fa:session:{user_id}。"""
    from memory.storage_backend import RedisBackend

    assert RedisBackend.SESSION_KEY_PREFIX == "fa:session:"


# ══════════════════════════════════════════════
# L2: 会话摘要消费闭环
# ══════════════════════════════════════════════


def test_reasoning_prompt_includes_session_summary():
    """build_reasoning_prompt注入memory_snapshot.summary → prompt中有"会话历史摘要"段。"""
    from agent.nodes.reasoning_node import build_reasoning_prompt

    state = {
        "user_query": "刚才说的那个公司现金流怎么样",
        "evidence_cards": [{"text": "百度2024现金流100亿", "source": "baidu.md"}],
        "graph_relations": [],
        "calculations": [],
        "web_search_note": "",
        "user_profile": {"language_style": "professional", "risk_preference": "neutral"},
        "memory_snapshot": {
            "summary": "关注实体：百度、腾讯；关注指标：营收、ROE；最近对话摘要：用户询问了百度2024年营收表现",
            "slots": {"last_company": "百度"},
        },
    }

    prompt = build_reasoning_prompt(state)
    assert "会话历史摘要（跨轮上下文）" in prompt
    assert "百度" in prompt
    assert "已确认信息" in prompt
    assert "last_company=百度" in prompt


def test_reasoning_prompt_empty_summary_is_silent():
    """summary为空时不注入摘要段——避免噪音。"""
    from agent.nodes.reasoning_node import build_reasoning_prompt

    state = {
        "user_query": "百度现金流",
        "evidence_cards": [{"text": "百度2024现金流", "source": "baidu.md"}],
        "memory_snapshot": {"summary": "", "slots": {}},
        "user_profile": {},
    }

    prompt = build_reasoning_prompt(state)
    assert "会话历史摘要" not in prompt
    assert "已确认信息" not in prompt


def test_fallback_reasoning_includes_summary_when_no_evidence():
    """_fallback_reasoning无证据时，也注入历史摘要作为context_hint。"""
    from agent.nodes.reasoning_node import _fallback_reasoning

    state = {
        "user_query": "刚才说的那个公司",
        "evidence_cards": [],
        "graph_relations": [],
        "calculations": [],
        "memory_snapshot": {
            "summary": "用户之前问了百度2024年营收",
            "slots": {},
        },
    }

    result = _fallback_reasoning(state)
    assert "历史上下文摘要" in result["draft_answer"]
    assert "百度2024年营收" in result["draft_answer"]


def test_query_transform_llm_plan_accepts_session_summary():
    """_llm_query_plan接受session_summary参数 → prompt中有"会话历史摘要"段。"""
    from agent.nodes.query_transform_node import _llm_query_plan

    with patch("agent.nodes.query_transform_node.invoke_fast_llm") as mock_llm, \
         patch("agent.nodes.query_transform_node.extract_json_object") as mock_extract:
        mock_llm.return_value = "some raw text"
        mock_extract.return_value = {
            "rewritten_query": "百度2024现金流",
            "sub_queries": ["百度现金流质量"],
            "hyde_document": "假设性文档",
            "required_metrics": ["现金流"],
            "intent_hint": "financial_analysis",
        }

        result = _llm_query_plan(
            "刚才说的那个公司现金流",
            {"companies": ["百度"]},
            session_summary="用户之前问了百度营收",
        )
        # 验证invoke_fast_llm被调用（prompt中包含了摘要）
        call_prompt = mock_llm.call_args[0][0]
        assert "会话历史摘要" in call_prompt
        assert "百度营收" in call_prompt


def test_query_transform_fallback_plan_includes_summary():
    """_fallback_query_plan接受session_summary → rewritten query包含摘要关键词。"""
    from agent.nodes.query_transform_node import _fallback_query_plan

    result = _fallback_query_plan(
        "刚才说的那个公司现金流",
        {"companies": ["百度"]},
        session_summary="关注实体：百度；关注指标：营收、ROE",
    )
    # summary[:200]被拼入rewritten
    assert "百度" in result["rewritten_query"]


def test_initial_state_snapshot_includes_restored_summary():
    """FinancialGraphAgent._initial_state → memory_snapshot包含restore后的summary。"""
    from agent.graph import FinancialGraphAgent

    with patch("agent.graph.FinancialGraphAgent._try_build_langgraph", return_value=None), \
         patch.object(FinancialGraphAgent, "__init__", lambda self, user_id: None):
        # 手动初始化
        agent = FinancialGraphAgent.__new__(FinancialGraphAgent)
        from memory.short_term import ShortTermMemory
        from memory.heartbeat import MemoryHeartbeat
        from memory.user_profile import UserProfileService

        agent.user_id = "test_user"
        agent.short_memory = ShortTermMemory(window_size=4)
        agent.short_memory.summary = "恢复的摘要：百度营收"
        agent.short_memory.slots = {"last_company": "百度"}
        agent.heartbeat = MemoryHeartbeat()
        agent.profiles = UserProfileService()
        agent.compiled_graph = None

        state = agent._initial_state("那个公司现金流怎么样")
        snapshot = state["memory_snapshot"]
        assert snapshot["summary"] == "恢复的摘要：百度营收"
        assert snapshot["slots"] == {"last_company": "百度"}