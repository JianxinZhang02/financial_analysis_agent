# 金融研报分析 Agent 项目总结

## 项目概述

面向卖方研究分析师与金融投资用户，构建对话式金融研报分析与多跳推理智能体。支持自然语言提问、公司/指标实体识别、混合检索召回、财务指标计算、合规审查与反思修正，输出带来源页码溯源的结构化研报分析。

**核心技术栈**：LangGraph / LangChain / Streamlit / TypedDict & dataclass / Qwen(Dashscope) / Hybrid RAG(Chroma+BM25+Rerank+Compressor) / EvidenceCard溯源体系 / Reflection-Critic

---

## 系统架构总览

```
用户输入
  │
  ▼
router_node ─── 意图识别 + 实体抽取
  │
  ▼
query_transform_node ─── 查询改写/子查询拆解（受 YAML 开关管控）
  │
  ▼
retrieval_node ─── Hybrid RAG: Dense→BM25→Rerank→Compress
  │                        ↑ 结构化元数据过滤（公司+年份）
  ▼
graph_rag_node ─── 知识图谱邻域检索
  │
  ▼
calculator_node ─── 财务指标可复核计算（同比增速/市盈率等）
  │
  ▼
reasoning_node ─── LLM 生成分析草稿（强制溯源约束）
  │
  ▼
critic_node ─── 合规审查（BLOCKING vs FORMAT 分级）
  │
  ├─ needs_more_evidence + BLOCKING issues ──→ 回到 query_transform（反思循环）
  │
  ▼
citation_guard_node ─── 溯源字段二次校验
  │
  ▼
final_answer_node ─── 终答（附反思过程摘要）
```

---

## 区别于普通 RAG 项目的关键创新点

### 创新点 1：Reflection-Critic 闭环 —— BLOCKING vs FORMAT 分级审查

普通 RAG 系统是单次检索→生成→结束。本项目引入 Reflection-Critic 闭环：审查节点区分 **BLOCKING**（证据缺失、合规风险）与 **FORMAT**（溯源格式不完整）两类问题，仅 BLOCKING 问题触发反思重新检索，FORMAT 问题只标记不循环，防止格式瑕疵无限触发反思。配合 `max_reflection_rounds` 硬上限与 `enable_query_rewrite` 配置开关联合控制终止。

```python
# critic_node.py —— BLOCKING 分级逻辑
blocking_issues = [i for i in merged_issues if i.startswith("BLOCKING:")]
result["needs_more_evidence"] = bool(
    result.get("needs_more_evidence", False)
    or (blocking_issues and state.get("reflection_round", 0) < 2)
)

# 确定性规则也分级：证据缺失和投资建议是 BLOCKING，溯源格式只是 FORMAT
if not cards:
    issues.append("BLOCKING:证据不足：没有检索到可引用的证据卡片。")
if any(pattern in draft for pattern in INVESTMENT_ADVICE_PATTERNS):
    issues.append("BLOCKING:合规风险：草稿包含确定性投资建议或收益承诺。")
if missing_citation_lines:
    issues.append("溯源风险：部分含数字表述未直接附来源。")  # FORMAT，不触发反思
```

```python
# graph.py —— 路由决策只看 BLOCKING issues，不看 FORMAT issues
def route_after_critic(state: FinancialAgentState) -> str:
    critique = state.get("critique_result", {})
    blocking_issues = [i for i in critique.get("issues", []) if i.startswith("BLOCKING:")]
    needs_more = bool(
        critique.get("needs_more_evidence", False)
        or (blocking_issues and state.get("reflection_round", 0) < max_reflection_rounds)
    )
    if needs_more and state.get("reflection_round", 0) < max_reflection_rounds:
        return "query_transform"  # 触发反思
    return "citation_guard"       # 正常流转
```

**解决的问题**：金融研报场景中，几乎所有草稿都会有"溯源格式不完整"的 FORMAT 问题，如果这些也触发反思，就会导致无限循环。BLOCKING 分级确保只有真正影响结论可靠性的问题才值得重新检索。

---

### 创新点 2：配置驱动的查询策略对齐 —— YAML 开关与代码行为完全绑定

普通 RAG 项目的配置文件中常见 `enable_hyde`、`enable_query_rewrite` 等开关，但往往是死配置——节点代码永远调用 LLM 改写、永远生成 HyDE、永远拆解子查询。本项目的 `query_transform_node` 严格读取 `rag.yaml` 的三个开关：

- `enable_query_rewrite: false` → 跳过 LLM 改写，走轻量 fallback（0s vs 60s）
- `enable_hyde: false` → 不生成假设性文档
- `enable_sub_queries: false` → 不拆解子查询，仅传原始 query

反射循环也受此管控：`enable_query_rewrite: false` 时，critic 只能路由到 citation_guard/final_answer，**不触发反思循环**。

```python
# query_transform_node.py —— 配置对齐核心逻辑
def query_transform_node(state: FinancialAgentState) -> dict:
    enable_query_rewrite = as_bool(rag_cof.get("enable_query_rewrite"), default=True)

    if not enable_query_rewrite:
        plan = _fallback_query_plan(query, entities, reflection_hint)  # 轻量路径，0s
    else:
        plan = _llm_query_plan(query, entities, reflection_hint)       # LLM 路径，60s

    enable_hyde = as_bool(rag_cof.get("enable_hyde"), default=True)
    enable_sub_queries = as_bool(rag_cof.get("enable_sub_queries"), default=True)
    # 两个开关在 _fallback_query_plan / _llm_query_plan 内部控制
    # hyde=False → 不生成假设文档，sub_queries=False → 只传原始 query
```

```yaml
# config/rag.yaml —— 用户可感知的配置
enable_hyde: false
enable_query_rewrite: false
enable_sub_queries: false
```

**解决的问题**：配置文件声称关闭了改写/HyDE/子查询，但代码完全无视这些开关继续调用 LLM，导致用户困惑（"为什么配置关了还在跑 LLM 改写？"）和性能浪费。对齐后，关闭改写时的请求从 3轮×3次LLM≈7.5分钟 降至 1次reasoning+1次critic≈75秒。

---

### 创新点 3：Hybrid RAG + 结构化元数据过滤 + 查询去噪

普通 RAG 只做语义检索。本项目构建"结构化元数据过滤+Chroma语义召回+BM25词汇召回+加权融合+本地规则重排+上下文压缩"的完整 Hybrid RAG 方案，并在检索前做**查询去噪**：元数据过滤条件中已包含公司名和年份，这些词在向量/词汇搜索时会稀释检索质量，因此被从查询文本中剥离。

```python
# query_filters.py —— 公司别名映射 → 自动推断元数据过滤条件
COMPANY_ALIASES = {
    "3690.HK_meituan": ("美团", "美团-W", "Meituan"),
    "BIDU_baidu": ("百度", "Baidu"),
    ...
}

def infer_metadata_filter(query: str) -> dict[str, str]:
    for company_id, aliases in COMPANY_ALIASES.items():
        if any(alias.lower() in query.lower() for alias in aliases):
            metadata_filter["company_id"] = company_id  # 精确过滤
    year_match = re.search(r"(20\d{2})\s*年?", query)
    if year_match:
        metadata_filter["report_period"] = f"FY{year_match.group(1)}"
    return metadata_filter

def normalize_query_for_metadata_filter(query, metadata_filter):
    # 已通过 metadata_filter 精确匹配的公司名和年份，从搜索文本中移除
    # 避免"美团百度2024年收入占比"中"美团""百度""2024"稀释检索
    normalized = query
    for alias in COMPANY_ALIASES.get(company_id, ()):
        normalized = re.sub(re.escape(alias), " ", normalized, flags=re.I)
    normalized = re.sub(rf"{year}\s*年?", " ", normalized)
    return normalized.strip() or query
```

**解决的问题**：用户问"美团2024年收入占比"，如果直接把"美团"和"2024"送入向量搜索，它们会与核心检索词"收入占比"竞争注意力。去噪后搜索文本变为"收入占比"，同时通过 metadata_filter 精确限定公司+年份范围，检索精度显著提升。

---

### 创新点 4：RetrievalPipeline —— 可插拔的检索流水线

普通 RAG 的检索步骤是硬编码顺序调用。本项目将 Dense→BM25→Rerank→Compress 四步解耦为 `RetrievalStep` ABC 流水线，每步通过 `run(query, candidates, context)` 接口传递中间状态，各步独立可替换、可跳过、可重排。

```python
# hybrid_retriever.py —— Pipeline 模式
class RetrievalStep(ABC):
    def run(self, query, candidates, context) -> list[dict]:
        pass

class DenseSearchStep(RetrievalStep):    # Chroma 语义召回
class BM25SearchStep(RetrievalStep):     # BM25 词汇召回
class RerankStep(RetrievalStep):         # 加权融合 + 规则重排
class CompressStep(RetrievalStep):       # 上下文压缩 → EvidenceCard

class RetrievalPipeline:
    def __init__(self, steps=None):
        self.steps = steps or []
    def add_step(self, step) -> "RetrievalPipeline":
        self.steps.append(step)
        return self
    def execute(self, query, context=None):
        for step in self.steps:
            candidates = step.run(query, candidates, context)
        return candidates
```

**解决的问题**：需要更换重排算法、跳过 BM25 步骤、或插入新的检索步骤时，只需调整 Pipeline 的 step 列表，不需要修改任何调用逻辑。

---

### 创新点 5：财务表格上下文提取 —— 超越句子选择的结构化解读

普通 RAG Compressor 只做句子选择或截断。本项目的 `ContextCompressor` 针对金融研报场景，能从财务表格中提取**结构化解读**：识别指标行→定位年份列→提取对应数值→匹配单位，输出"表格解读：2024年营业收入对应数值为 2,578 亿元，单位：人民币亿元"这样的精确定量摘要。

```python
# context_compressor.py —— 财务表格结构化提取
def _explicit_table_value(self, query, context_lines, metric_line):
    target_year = self._target_year(query)            # 从 query 提取目标年份
    year_line = self._find_year_line(context_lines)   # 定位年份行
    years = self._extract_years(year_line)            # 解析年份序列
    values = self._extract_numeric_values(metric_line) # 解析数值序列
    if target_year in years and len(values) >= len(years):
        value = values[years.index(target_year)]      # 年份与数值对齐
        unit = self._find_unit(context_lines)         # 识别单位（百万元/亿元）
        metric = self._metric_label(metric_line)      # 提取指标名
        return f"表格解读：{target_year}年{metric}对应数值为 {value}，单位：{unit}。"
```

**解决的问题**：金融研报中的关键数据以表格形式呈现，纯句子检索会把表格打碎成无意义的行片段。结构化提取保留指标→年份→数值→单位的完整语义链，使 reasoning 节点能基于精确数据做分析。

---

### 创新点 6：Document Registry + 元数据富化 —— 检索前的数据治理

普通 RAG 直接扫描目录加载数据。本项目引入 CSV-based Document Registry 机制，在索引前对每个文档注入结构化元数据（company_id、ticker、report_period、doc_type、source_priority 等），这些元数据在检索时转为 Chroma 的 metadata filter，实现公司+年份的精确过滤。

```python
# metadata_registry.py —— 注册表驱动的元数据富化
def enrich_documents_from_registry(path, docs):
    row = load_document_registry().get(_normalize_path(path))
    for doc in docs:
        doc.metadata.update({
            "company_id": row.get("company_id"),    # "3690.HK_meituan"
            "report_period": row.get("report_period"), # "FY2024"
            "doc_type": row.get("doc_type"),         # "annual_report"
            "source_priority": row.get("source_priority"),  # "primary"
            ...
        })
```

**解决的问题**：RAG 系统的检索精度很大程度上取决于元数据质量。事后标注（在 prompt 中告诉 LLM 来源）不如事前治理（在索引阶段注入精确元数据），后者让检索层直接按公司/年份/文档类型精确过滤，不依赖 LLM 推理。

---

### 创新点 7：MarkdownHierarchyChunker —— 金融研报的结构感知分块

普通 RAG 用固定字符数分块。本项目的 `MarkdownHierarchyChunker` 保留金融研报的层级结构：沿 Markdown 标题栈维护 section_path（如"经营概况 > 收入分析 > 分部营收"），每个 chunk 附带完整的层级路径。对表格文档采用独立策略：保持表头行与每一组数据行组合，避免表格被截断后丢失列定义。

```python
# markdown_chunker.py —— 层级路径感知分块
def _sections(self, text):
    heading_stack = []
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            heading = match.group(2).strip()
            heading_stack = heading_stack[:level-1]  # 维护标题栈
            heading_stack.append(heading)
            current_path = " > ".join(heading_stack)  # "经营概况 > 收入分析"
            sections.append((current_path, section_text))

# 表格分块：保留表头 + 逐行追加数据行，超长时重带表头
def _split_large_table(self, text):
    header = table_lines[:2]   # 表头+分隔行始终保留
    current = prefix_lines + header
    for row in rows:
        if len("\n".join(current + [row])) <= max_chars:
            current.append(row)
        else:
            chunks.append("\n".join(current))
            current = prefix_lines + header + [row]  # 重新带上表头
```

**解决的问题**：金融研报的表格一旦被固定长度截断，后续 chunk 缺少列定义和单位，无法独立理解。保留表头的分块策略确保每个 chunk 都能被独立解读。section_path 让检索结果具备上下文定位能力。

---

### 创新点 8：增量索引同步 —— ChunkCache + MD5 + Registry指纹

普通 RAG 每次变更数据需要全量重建索引。本项目通过三级缓存签名（源文件 MD5 + 注册表元数据指纹 + 分块参数签名）实现增量同步：只重新处理真正变更的文件，未变更文件直接命中缓存。

```python
# pipeline.py + chunk_cache.py —— 增量构建逻辑
source_hash = file_md5(file)                 # 源文件内容 MD5
metadata_hash = registry_row_fingerprint(file)  # 注册表元数据 MD5
cached_chunks = cache.get(file, source_hash, metadata_hash)
if cached_chunks is not None:
    chunks.extend(cached_chunks)   # 缓存命中，跳过解析分块
    continue

# VectorStoreService.sync_from_chunks —— 增量向量索引同步
stale_ids = existing_ids - target_ids      # 已删除的文档
missing_ids = target_ids - existing_ids    # 新增的文档
changed_ids = [id for id in overlap if _metadata_needs_update(...)]  # 元数据变更的文档
# 只删除 stale + changed，只添加 missing + changed
```

**解决的问题**：金融研报数据持续更新（年报、季报、纪要），全量重建索引耗时且浪费 API 调用（embedding）。增量同步只处理变更部分，大幅降低索引维护成本。

---

### 创新点 9：Citation Guard —— 溯源字段强制校验节点

普通 RAG 的溯源依赖 LLM 在生成时自觉引用来源。本项目在 LLM 生成之后、终答输出之前，设置独立的 `citation_guard_node` 对所有 EvidenceCard 进行确定性校验（source_file + chunk_id + page_number 三字段必须齐全），校验失败直接追加警告到终答。

```python
# citation_guard_node.py
def citation_guard_node(state):
    errors = []
    for idx, card in enumerate(state.get("evidence_cards", [])):
        if not has_valid_citation(card):  # source_file + chunk_id + page_number 三字段齐全
            errors.append(f"证据卡片 {idx} 缺少强溯源字段。")
    if errors:
        draft += "\n\n引用校验未完全通过：" + "；".join(errors)
```

**解决的问题**：金融合规场景要求"每个数字必须有出处"，仅靠 prompt 约束 LLM 并不可靠。确定性校验节点确保溯源合规不被 LLM 的疏忽打破。

---

### 创新点 10：Calculator Node —— 可复核的财务指标计算

普通 RAG 不做数值计算，依赖 LLM 在文本中找数字做推理。本项目设置独立的 `calculator_node`，基于正则从 EvidenceCard 文本中提取年度+数值对，执行同比增速、市盈率等结构化计算，输出 `metric + formula + value + unit + period`，供 reasoning 节点引用。计算结果有明确的公式溯源，可被人工复核。

```python
# calculator_node.py
revenue_values = re.findall(r"20(2[2-5])年[^。\n]*?营业收入(?:为)?([0-9.]+)亿元", text)
if should_calculate_growth and len(revenue_values) >= 2:
    old_year, old_value = revenue_values[-2]
    new_year, new_value = revenue_values[-1]
    calculations.append({
        "metric": "营业收入同比",
        "formula": f"({new_value}-{old_value})/{old_value}*100",
        "value": round(_pct_change(new_value, old_value), 2),
        "unit": "%",
        "period": f"{new_year} vs {old_year}",
    })
```

**解决的问题**：LLM 对数值推理不可靠（可能算错或编造数字）。独立计算节点给出有公式的可复核结果，reasoning 节点直接引用而非自行推算。

---

### 创新点 11：Heartbeat Memory —— 长短期记忆 + 会话压缩

构建三层记忆机制：

- **长期记忆**（JSON文件）：沉淀用户关注公司 watchlist、偏好指标 preferred_metrics、风险倾向等稳定特征，跨会话继承
- **短期记忆**（deque窗口）：维护最近8条对话，保留当前会话约束与上下文
- **Heartbeat压缩**：当消息数≥12或字符数≥16000时，提取公司实体与金融指标关键词生成滚动摘要，避免上下文溢出

```python
# heartbeat.py —— 金融领域感知的摘要提取
def summarize(self, messages, previous_summary=""):
    companies = set(re.findall(r"[一-鿿A-Za-z]+(?:科技|股份|银行|证券|集团|公司)", text))
    metrics = {m for m in ["营收","净利润","毛利率","现金流","ROE","市盈率"] if m in text}
    parts = [previous_summary]
    if companies: parts.append("关注实体：" + "、".join(companies[:8]))
    if metrics:   parts.append("关注指标：" + "、".join(metrics))
    parts.append("最近对话摘要：" + text[-1200:])
```

**解决的问题**：金融分析会话往往持续多轮、跨越多天。纯 messages 列表会超出上下文长度，而通用摘要会丢失金融关键信息。领域感知压缩保留公司实体和指标关键词，确保跨会话偏好继承不丢失专业上下文。

---

### 创新点 12：LLM 调用容错 —— 三次重试 + JSON 自动修复

LLM 输出的 JSON 经常有格式问题（尾逗号、注释、未闭合括号）。本项目在 `invoke_llm` 中实现三次重试+指数退避，在 `extract_json_object` 中实现两轮 JSON 自动修复（移除尾逗号、删除注释、闭合括号），提高生产环境鲁棒性。

```python
# llm_utils.py
def invoke_llm(prompt, max_retries=3, retry_delay=1.0):
    for attempt in range(1, max_retries + 1):
        try:
            response = chat_model.invoke(prompt)
            ...
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(retry_delay)
                retry_delay *= 1.5  # 指数退避

def _repair_json_string(text):
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)   # 移除尾逗号
    cleaned = re.sub(r"//[^\n]*", "", cleaned)       # 移除注释
    open_braces = cleaned.count("{") - cleaned.count("}")
    cleaned += "}" * open_braces                     # 闭合括号
```

---

## 非创新但完整的工程化实践

| 实践 | 说明 |
|------|------|
| **Stage-based 全链路日志** | 每个 node 操作包裹 `log_stage` context manager，记录耗时、输入输出指标，提供完整 pipeline 可观测性 |
| **Streamlit 可视化界面** | 左侧对话+右侧证据卡/审查结果双栏布局，实时展示 EvidenceCard 和 critique_result |
| **GraphRAG 邻域检索** | 基于规则抽取的公司→指标关系，双向邻接表存储，seed-based 邻域检索补充实体关系推理 |
| **向量索引双模式** | Chroma 持久化索引为主，InMemoryVectorStore 为 fallback，自动降级 |
| **本地规则重排** | 无需外部 rerank API，基于词汇重叠+文档权威性+时效性的加权重排 |
| **合规配置化** | risk_disclaimer、forbid_direct_investment_advice 等合规规则通过 YAML 配置，不改代码即可调整 |

---

## 项目文件结构

```
financial_analysis_agent/
├── agent/
│   ├── graph.py                    # LangGraph 状态驱动编排 + 反思闭环
│   ├── state.py                    # FinancialAgentState TypedDict (20字段)
│   ├── llm_utils.py                # LLM 调用容错 + JSON 修复
│   ├── react_agent.py              # Streamlit 入口别名
│   └── nodes/
│   │   ├── router_node.py          # 意图识别 + 实体抽取
│   │   ├── query_transform_node.py # 配置驱动的查询改写
│   │   ├── retrieval_node.py       # 子查询遍历 + 去重截断
│   │   ├── graph_rag_node.py       # 知识图谱邻域检索
│   │   ├── calculator_node.py      # 可复核财务指标计算
│   │   ├── reasoning_node.py       # LLM 分析草稿（含 fallback）
│   │   ├── critic_node.py          # BLOCKING/FORMAT 分级审查
│   │   ├── citation_guard_node.py  # 溯源字段强制校验
│   │   ├── web_search_node.py      # 外部搜索占位节点
│   │   └── final_answer_node.py    # 终答 + 反思过程摘要
│   └── tools/                      # 工具层（ReAct 备用）
├── rag/
│   ├── hybrid_retriever.py         # RetrievalPipeline (Step ABC)
│   ├── vector_store.py             # Chroma + InMemory 双模式 + 增量同步
│   ├── bm25_store.py               # BM25 词汇召回
│   ├── reranker.py                 # 本地规则重排
│   ├── context_compressor.py       # 财务表格结构化提取
│   ├── query_filters.py            # 公司别名映射 + 元数据过滤 + 查询去噪
│   └── citation.py                 # EvidenceCard 数据类 + 溯源校验
├── graph_rag/
│   ├── entity_extractor.py         # 金融实体抽取（公司/指标/年份/代码）
│   ├── relation_extractor.py       # 实体关系抽取
│   ├── graph_store.py              # 双向邻接表
│   └── graph_retriever.py          # seed-based 邻域检索
├── memory/
│   ├── short_term.py               # deque 滑动窗口 + slots
│   ├── long_term.py                # JSON 持久化用户画像
│   ├── heartbeat.py                # 金融领域感知会话压缩
│   └── user_profile.py             # 偏好记忆服务
├── ingestion/
│   ├── pipeline.py                 # 发现→加载→富化→分块→缓存→索引 全流程
│   ├── metadata_registry.py        # CSV 注册表 + 元数据注入
│   ├── chunk_cache.py              # MD5 + Registry 指纹缓存
│   ├── chunkers/
│   │   ├── markdown_chunker.py     # 层级路径感知 + 表格保头分块
│   │   └── semantic_chunker.py     # 语义分块（非金融文档）
│   ├── parsers/                    # PDF/表格/图表解析
│   └── loaders/                    # 多格式加载器
├── model/
│   └── factory.py                  # ChatOpenAI + DashScope 双模式 + SimpleChatModel fallback
├── config/                         # 9个 YAML 配置文件（rag/graph/agent/model/memory/compliance/logging/chroma/prompts）
├── prompts/                        # 7个提示词模板
├── eval/                           # 检索精度 + 溯源完整性评估框架
├── tests/                          # 13个测试覆盖（分块/检索/引用/反思/图谱流程）
├── app.py                          # Streamlit 可视化界面
└── utils/                          # 日志/配置/路径/进度条/中文文本处理
```

---

## 性能优化实践

| 优化 | 效果 |
|------|------|
| 配置关闭改写时跳过 LLM 调用 | 单次请求从 7.5分钟 → 75秒 |
| 增量索引同步（只处理变更文件） | 索引重建从全量 → 仅变更部分 |
| ChunkCache 三级签名缓存 | 未变更文件 0 解析开销 |
| 查询去噪（剥离已被 metadata 过滤的词） | 检索精度提升，减少无关命中 |
| BLOCKING 分级避免无效反思 | 防止 FORMAT 问题触发 3轮 × 3次LLM 的循环 |