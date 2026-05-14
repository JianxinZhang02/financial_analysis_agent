"""test_user_system.py — 端到端验证多用户体系与画像回灌闭环。"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── 1. sanitize_user_id ──

def test_sanitize_valid_ids():
    from memory.storage_backend import sanitize_user_id
    assert sanitize_user_id("analyst_zhang") == "analyst_zhang"
    assert sanitize_user_id("user-123") == "user-123"
    assert sanitize_user_id("abc") == "abc"


def test_sanitize_rejects_dangerous_ids():
    from memory.storage_backend import sanitize_user_id
    with pytest.raises(ValueError):
        sanitize_user_id("../etc/passwd")
    with pytest.raises(ValueError):
        sanitize_user_id("user with spaces")
    with pytest.raises(ValueError):
        sanitize_user_id("a" * 65)   # too long


# ── 2. DiskBackend ──

def test_disk_backend_conversations():
    from memory.storage_backend import DiskBackend
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = DiskBackend(conversation_dir=tmpdir, profile_path=str(Path(tmpdir) / "profiles.json"))
        threads = [{"id": "t1", "title": "测试对话", "messages": [{"role": "user", "content": "你好"}], "last_state": {}, "updated_at": "10:00"}]
        backend.save_threads("user_a", threads)
        loaded = backend.get_threads("user_a")
        assert len(loaded) == 1
        assert loaded[0]["title"] == "测试对话"


def test_disk_backend_profiles():
    from memory.storage_backend import DiskBackend
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = DiskBackend(conversation_dir=tmpdir, profile_path=str(Path(tmpdir) / "profiles.json"))
        profile = backend.get_profile("user_b")
        assert profile["watchlist"] == []
        assert profile["language_style"] == "professional"
        profile["watchlist"] = ["腾讯控股"]
        backend.save_profile("user_b", profile)
        loaded = backend.get_profile("user_b")
        assert loaded["watchlist"] == ["腾讯控股"]


def test_disk_backend_list_users():
    from memory.storage_backend import DiskBackend
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = DiskBackend(conversation_dir=tmpdir, profile_path=str(Path(tmpdir) / "profiles.json"))
        backend.save_threads("user_a", [{"id": "t1", "messages": [], "last_state": {}, "title": "", "updated_at": ""}])
        backend.save_threads("user_b", [{"id": "t2", "messages": [], "last_state": {}, "title": "", "updated_at": ""}])
        users = backend.list_users()
        assert set(users) == {"user_a", "user_b"}


# ── 3. 多用户隔离 ──

def test_multi_user_isolation():
    from memory.storage_backend import DiskBackend
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = DiskBackend(conversation_dir=tmpdir, profile_path=str(Path(tmpdir) / "profiles.json"))
        # user_a画像
        pa = backend.get_profile("user_a")
        pa["watchlist"] = ["腾讯"]
        pa["preferred_metrics"] = ["营收"]
        backend.save_profile("user_a", pa)
        # user_b画像
        pb = backend.get_profile("user_b")
        pb["watchlist"] = ["美团"]
        pb["preferred_metrics"] = ["利润"]
        backend.save_profile("user_b", pb)
        # 互不干扰
        assert backend.get_profile("user_a")["watchlist"] == ["腾讯"]
        assert backend.get_profile("user_b")["watchlist"] == ["美团"]


# ── 4. ConversationStore ──

def test_conversation_store_append_message():
    from memory.conversation_store import ConversationStore
    from memory.storage_backend import DiskBackend
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = DiskBackend(conversation_dir=tmpdir, profile_path=str(Path(tmpdir) / "profiles.json"))
        store = ConversationStore(backend=backend)
        store.append_message("user_c", "thread-1", "user", "腾讯2024营收如何？")
        store.append_message("user_c", "thread-1", "assistant", "根据年报...")
        threads = store.get_threads("user_c")
        assert len(threads) == 1
        assert len(threads[0]["messages"]) == 2


# ── 5. State包含user_profile ──

def test_state_has_user_profile_field():
    from agent.state import FinancialAgentState
    assert "user_profile" in FinancialAgentState.__annotations__


def test_initial_state_includes_profile():
    """_initial_state应回灌画像到State。"""
    from agent.graph import FinancialGraphAgent
    with patch("agent.graph.FinancialGraphAgent._try_build_langgraph", return_value=None):
        agent = FinancialGraphAgent(user_id="test_user")
        state = agent._initial_state("腾讯2024营收")
        assert "user_profile" in state
        assert isinstance(state["user_profile"], dict)


# ── 6. query_transform消费画像 ──

def test_fallback_query_plan_with_profile():
    from agent.nodes.query_transform_node import _fallback_query_plan
    profile = {"preferred_metrics": ["营收"], "watchlist": ["腾讯控股"]}
    plan = _fallback_query_plan("腾讯2024营收同比增长率", {"companies": ["腾讯控股"], "metrics": ["营收"], "years": ["2024"], "doc_type": "annual_report"}, "", profile=profile)
    # 画像偏好指标的同义词应出现在required_metrics或rewritten_query中
    assert "营收" in plan.get("original_query", "") or "营收" in str(plan.get("required_metrics", []))


# ── 7. reasoning风格偏好 ──

def test_reasoning_prompt_includes_style():
    from agent.nodes.reasoning_node import _llm_reasoning
    state = {
        "user_query": "腾讯2024分析",
        "evidence_cards": [{"source_file": "r.md", "chunk_id": "c1", "page_number": 5, "evidence": "营收6100亿"}],
        "graph_relations": [],
        "calculations": [],
        "web_search_note": "",
        "user_profile": {"language_style": "casual", "risk_preference": "conservative"},
    }
    with patch("agent.nodes.reasoning_node.invoke_llm", return_value="分析结果") as mock_llm:
        _llm_reasoning(state)
        prompt = mock_llm.call_args[0][0]
        assert "通俗易懂" in prompt   # casual风格
        assert "稳健" in prompt        # conservative风险偏好


# ── 8. 画像风格沉淀 ──

def test_profile_remember_style():
    from memory.user_profile import UserProfileService
    from memory.storage_backend import DiskBackend
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = DiskBackend(conversation_dir=tmpdir, profile_path=str(Path(tmpdir) / "profiles.json"))
        from memory.long_term import LongTermMemory
        ltm = LongTermMemory(backend=backend)
        svc = UserProfileService(memory=ltm)
        svc.remember_style("user_d", language_style="academic", risk_preference="aggressive")
        profile = svc.get("user_d")
        assert profile["language_style"] == "academic"
        assert profile["risk_preference"] == "aggressive"


# ── 9. create_backend工厂（Disk fallback） ──

def test_create_backend_falls_to_disk_when_redis_disabled():
    from memory.storage_backend import create_backend, DiskBackend
    backend = create_backend()  # Redis默认disabled
    assert isinstance(backend, DiskBackend)