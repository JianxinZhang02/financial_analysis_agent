# 多用户体系与画像回灌 实现计划

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 让user_id从僵尸字段变为活水——Streamlit登录界面区分用户，各用户拥有独立历史对话和画像，画像数据回灌到Agent State并在核心节点中被消费。

**Architecture:** 三层闭环：(1) UI层——Streamlit登录页+用户切换；(2) 数据层——按user_id隔离的对话历史JSON+画像JSON；(3) Agent层——画像回灌State→query_transform/reasoning参考偏好。当前所有对话历史存于Streamlit session_state（进程内内存），改为持久化到磁盘JSON文件，跨session可恢复。

**Tech Stack:** Streamlit / JSON文件持久化 / LangGraph State / memory模块

---

## 现状诊断

```
graph写入user_id → memory存储画像 → 画像无人消费 → 闭环断裂
app.py无登录 → FinancialGraphAgent()无user_id → 永远"default"
对话历史存session_state内存 → 浏览器关闭即丢失
```

## 目标闭环

```
登录页输入user_id → agent初始化传入user_id → State携带user_id
→ 各节点消费画像偏好 → invoke后沉淀画像 → 画像回灌下次State
对话历史持久化到磁盘 → 换用户加载对应历史 → 关浏览器不丢失
```

---

### Task 1: State新增user_profile字段

**Objective:** 让画像数据在Agent全流程中可见，下游节点可直接从State读取用户偏好。

**Files:**
- Modify: `agent/state.py:6-26`

**Step 1: 在FinancialAgentState中新增字段**

```python
class FinancialAgentState(TypedDict, total=False):
    # ... 现有字段 ...
    user_id: str
    user_profile: dict[str, Any]   # ← 新增：从画像回灌，供下游节点参考
```

**Verification:** `python -c "from agent.state import FinancialAgentState; print('user_profile' in FinancialAgentState.__annotations__)"` → True

---

### Task 2: graph.py画像回灌——_initial_state加载画像

**Objective:** 每次invoke开始时，从LongTermMemory读取该user的画像，注入State的user_profile字段。

**Files:**
- Modify: `agent/graph.py:143-151` (_initial_state方法)

**Step 1: 在_initial_state中注入画像**

当前代码：
```python
def _initial_state(self, query: str) -> FinancialAgentState:
    return {
        "messages": ...,
        "user_id": self.user_id,
        "user_query": query,
        ...
    }
```

改为：
```python
def _initial_state(self, query: str) -> FinancialAgentState:
    profile = self.profiles.get(self.user_id)
    return {
        "messages": list(self.short_memory.messages) + [{"role": "user", "content": query}],
        "user_id": self.user_id,
        "user_profile": profile,       # ← 画像回灌
        "user_query": query,
        "reflection_round": 0,
        "reflection_history": [],
        "memory_snapshot": self.short_memory.snapshot(),
    }
```

**Verification:** 用mock测试_initial_state返回值包含user_profile键且值来自profiles.get()

---

### Task 3: query_transform_node消费画像偏好

**Objective:** 当画像中有preferred_metrics/watchlist时，query改写应参考用户偏好增强搜索语义。

**Files:**
- Modify: `agent/nodes/query_transform_node.py` (在LLM prompt和关键词增强逻辑中加入画像参考)

**Step 1: 读取State中的user_profile，提取偏好**

在query_transform_node函数开头加入：
```python
profile = state.get("user_profile", {})
preferred_metrics = profile.get("preferred_metrics", [])
watchlist = profile.get("watchlist", [])
```

**Step 2: 在LLM prompt中注入偏好提示**

当偏好非空时，在query_transform的prompt中加入偏好上下文段：
```
该用户关注的公司有：{watchlist}，偏好的指标有：{preferred_metrics}。
请优先围绕这些偏好进行改写和扩展。
```

**Step 3: 关键词增强fallback中也参考画像**

当LLM不可用时，关键词增强逻辑也应把preferred_metrics的synonyms加入augmentation。

**Verification:** 测试：传入user_profile含preferred_metrics=["营收","利润"]时，augment_query_for_retrieval的输出中包含"营业收入"等同义词。

---

### Task 4: reasoning_node消费画像风格偏好

**Objective:** reasoning_node的prompt应参考用户的language_style和risk_preference，生成贴合用户偏好的分析文本。

**Files:**
- Modify: `agent/nodes/reasoning_node.py`

**Step 1: 从State读取画像偏好**

```python
profile = state.get("user_profile", {})
style = profile.get("language_style", "professional")
risk = profile.get("risk_preference", "unknown")
```

**Step 2: 在reasoning prompt末尾注入风格指令**

```
回复风格：{style}（professional=专业严谨/casual=通俗易懂/academic=学术规范）
风险偏好视角：{risk}（conservative=侧重稳健/aggressive=侧重增长/neutral=客观中立）
```

**Verification:** 测试mock reasoning_node，确认prompt中包含风格和风险偏好指令。

---

### Task 5: 对话历史持久化——ConversationStore

**Objective:** 将Streamlit内存中的对话历史改为磁盘持久化，按user_id隔离存储，跨session可恢复。

**Files:**
- Create: `memory/conversation_store.py`
- Modify: `config/memory.yaml` (新增conversation_store_path配置)

**Step 1: 创建ConversationStore类**

```python
class ConversationStore:
    """按user_id持久化对话历史到JSON文件。"""
    def __init__(self, path: str | None = None):
        self.path = Path(get_abs_path(path or memory_cof["conversation_store_path"]))
        self.path.mkdir(parents=True, exist_ok=True)

    def _user_file(self, user_id: str) -> Path:
        return self.path / f"{user_id}.json"

    def get_threads(self, user_id: str) -> list[dict]:
        """读取某用户的全部对话线程。"""
        f = self._user_file(user_id)
        if not f.exists():
            return []
        return json.loads(f.read_text(encoding="utf-8"))

    def save_threads(self, user_id: str, threads: list[dict]) -> None:
        """保存某用户的全部对话线程。"""
        f = self._user_file(user_id)
        f.write_text(json.dumps(threads, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_message(self, user_id: str, thread_id: str, role: str, content: str) -> None:
        """追加一条消息到指定线程。"""
        threads = self.get_threads(user_id)
        thread = next((t for t in threads if t["id"] == thread_id), None)
        if thread is None:
            thread = {"id": thread_id, "title": "新对话", "messages": [], "last_state": {}, "updated_at": ""}
            threads.append(thread)
        thread["messages"].append({"role": role, "content": content})
        thread["updated_at"] = datetime.now().strftime("%H:%M")
        self.save_threads(user_id, threads)
```

**Step 2: 在memory.yaml中新增配置**

```yaml
conversation_store_path: data/processed/conversations
```

**Verification:** 测试ConversationStore的读写、append、多用户隔离。

---

### Task 6: app.py登录界面——用户选择/注册页

**Objective:** Streamlit启动时显示登录选择页，用户输入或选择user_id后进入主界面。

**Files:**
- Modify: `app.py` (新增render_login_page函数 + 修改init_session_state和run_streamlit)

**Step 1: 新增render_login_page函数**

登录页UI设计：
- 左侧：品牌logo+欢迎语
- 中间：text_input输入用户名 + 下拉选择已有用户
- 底部：确认按钮

```python
def render_login_page() -> str | None:
    import streamlit as st
    conv_store = ConversationStore()

    # 获取已有用户列表
    existing_users = conv_store.list_users()

    st.markdown("""登录页面HTML/CSS""", unsafe_allow_html=True)

    user_id = st.text_input("输入用户名", placeholder="您的标识，如 analyst_zhang")
    if existing_users:
        selected = st.selectbox("或选择已有用户", existing_users)
        if selected and not user_id:
            user_id = selected

    if st.button("进入工作台") and user_id.strip():
        return user_id.strip()
    return None
```

**Step 2: ConversationStore新增list_users方法**

```python
def list_users(self) -> list[str]:
    """列出所有已注册用户（按最近活跃排序）。"""
    if not self.path.exists():
        return []
    files = sorted(self.path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [f.stem for f in files]
```

**Step 3: init_session_state改造**

```python
def init_session_state() -> None:
    import streamlit as st
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None    # ← 登录前为None
    if "agent" not in st.session_state:
        # agent不再在此初始化，等登录后再创建
        pass
    if "conversation_store" not in st.session_state:
        st.session_state["conversation_store"] = ConversationStore()
```

**Step 4: run_streamlit改造——先判断是否已登录**

```python
def run_streamlit() -> None:
    st.set_page_config(...)
    inject_chat_css()
    init_session_state()

    # ── 未登录 → 显示登录页 ──
    if not st.session_state.get("user_id"):
        user_id = render_login_page()
        if user_id:
            st.session_state["user_id"] = user_id
            st.session_state["agent"] = FinancialGraphAgent(user_id=user_id)
            # 从持久化加载该用户的对话历史到session_state
            _load_user_conversations(user_id)
            st.rerun()
        return

    # ── 已登录 → 主界面（现有逻辑） ──
    sidebar_action, thread_id = render_sidebar()
    # ... 现有代码 ...
```

**Verification:** 手动测试Streamlit登录页，确认输入user_id后能跳转到主界面且agent带有正确user_id。

---

### Task 7: app.py对话历史加载/保存对接

**Objective:** 将Streamlit内存对话与磁盘持久化对接——登录时加载历史，每次交互后保存。

**Files:**
- Modify: `app.py` (新增_load_user_conversations和修改handle_query/init_session_state)

**Step 1: 新增_load_user_conversations**

```python
def _load_user_conversations(user_id: str) -> None:
    import streamlit as st
    store = st.session_state["conversation_store"]
    threads = store.get_threads(user_id)
    if threads:
        st.session_state["threads"] = {t["id"]: t for t in threads}
        st.session_state["thread_order"] = [t["id"] for t in threads]
        st.session_state["active_thread_id"] = threads[0]["id"] if threads else "thread-1"
    else:
        thread_id = f"thread-{st.session_state.get('thread_counter', 1)}"
        st.session_state["threads"] = {thread_id: _make_thread(thread_id)}
        st.session_state["thread_order"] = [thread_id]
        st.session_state["active_thread_id"] = thread_id
```

**Step 2: handle_query中保存对话到磁盘**

```python
def handle_query(query: str) -> None:
    # ... 现有逻辑 ...
    thread["messages"].append({"role": "assistant", "content": answer})
    thread["last_state"] = state

    # ← 新增：持久化到磁盘
    store = st.session_state["conversation_store"]
    store.save_threads(st.session_state["user_id"], list(st.session_state["threads"].values()))
```

**Step 3: sidebar中新增"切换用户"按钮**

在sidebar footer区域新增一个"切换用户"按钮，点击后清除session_state中的user_id，回到登录页。

```python
if st.button("切换用户", key="switch-user"):
    st.session_state["user_id"] = None
    st.rerun()
```

**Verification:** 测试：登录user_A→对话→关闭浏览器→重新打开→登录user_A→对话历史恢复。

---

### Task 8: 画像自动沉淀增强——从reasoning结果提取更多偏好

**Objective:** 当前只沉淀companies和metrics，应扩展到language_style/risk_preference的自动推断。

**Files:**
- Modify: `agent/graph.py:168-173` (invoke后的画像沉淀逻辑)
- Modify: `memory/user_profile.py`

**Step 1: UserProfileService新增remember_style方法**

```python
def remember_style(self, user_id: str, language_style: str, risk_preference: str) -> dict:
    updates = {}
    if language_style:
        updates["language_style"] = language_style
    if risk_preference:
        updates["risk_preference"] = risk_preference
    return self.memory.update_profile(user_id, **updates)
```

**Step 2: graph.py invoke后增加风格沉淀**

在现有remember_focus之后，从State读取推断出的风格：
```python
# 现有
self.profiles.remember_focus(self.user_id, companies, metrics)
# 新增：从critique或最终答案推断风格（简单规则）
draft = state.get("draft_answer", "")
if len(draft) > 200:
    # 长答案→professional风格
    self.profiles.remember_style(self.user_id, "professional", "neutral")
```

注：完整的LLM风格推断可作为后续增强，当前先用简单规则确保闭环存在。

**Verification:** 测试invoke后profile中language_style字段被更新。

---

### Task 9: 测试——多用户体系闭环验证

**Objective:** 端到端验证：不同user_id→不同画像→不同偏好→不同对话历史→持久化恢复。

**Files:**
- Create: `tests/test_user_system.py`

**测试清单：**

1. **State包含user_profile** — 初始化State时user_profile非空
2. **画像回灌闭环** — user_A的preferred_metrics=["营收"]注入State后，query_transform输出的augmented_query包含"营业收入"同义词
3. **多用户隔离** — user_A和user_B的画像独立，互不干扰
4. **对话持久化** — ConversationStore写入后可读取，跨进程不丢失
5. **用户列表** — list_users返回已注册用户，按最近活跃排序
6. **reasoning风格** — 传入language_style="casual"时，reasoning prompt包含通俗风格指令

**Verification:** `pytest tests/test_user_system.py -v` → 6 passed

---

### Task 10: 清理——删除僵尸default_user_id和session_state硬编码

**Objective:** 清理不再需要的硬编码default，确保所有user_id来源统一。

**Files:**
- Modify: `config/agent.yaml` (删除或标注default_user_id为fallback)
- Modify: `app.py` (确保所有地方使用st.session_state["user_id"])

**Step 1:** 检查所有`default_user_id`引用，改为运行时传入优先、配置fallback。

**Verification:** `grep -r "default_user_id" *.py *.yaml` → 只在config fallback中出现，不在业务逻辑中硬编码。

---

## 依赖关系图

```
Task 1 (State字段) → Task 2 (回灌) → Task 3 (query_transform消费) 
                                       → Task 4 (reasoning消费)
                                       → Task 8 (风格沉淀增强)

Task 5 (ConversationStore) → Task 6 (登录页) → Task 7 (加载/保存)
                                                      → Task 9 (端到端测试)

Task 10 (清理) ← 所有Task完成后执行
```

## 关联风险提示

1. **Streamlit session_state与磁盘同步延迟**：每次handle_query后保存磁盘，但Streamlit的rerun机制可能导致中间态丢失。建议用"写后即存"策略，不用异步。
2. **用户名安全**：当前user_id直接作为文件名（`{user_id}.json`），需防路径注入（如`../`）。ConversationStore应对user_id做sanitize（只允许字母数字下划线）。
3. **画像冷启动**：新用户首次登录画像为空模板，query_transform/reasoning应优雅处理空画像（当前已做`profile.get("preferred_metrics", [])`兜底）。
4. **并发写入**：同一用户在多个浏览器tab同时操作可能造成JSON文件写入冲突。生产环境应考虑文件锁或数据库，当前JSON方案适合单用户单tab场景。