# 金融研报分析与多跳推理 Agent 项目对话总结

本文档用于完整记录本轮对话中围绕 `robot_vacuum_agent` 到 `financial_analysis_agent` 的项目需求、架构设计、开发执行、环境调整、已实现能力、当前问题和后续完善路线。

当前项目根目录：

```text
E:\codes\study_LLM\To_work agent\financial_analysis_agent
```

当前依赖运行环境：

```text
F:\anaconda3\envs\llm311\python.exe
```

后续所有命令、验证、启动和代码调试均应默认使用 `llm311` 环境。

---

## 1. 用户原始任务需求

用户最初给出的角色设定是：

> AI 架构师兼全栈工程师，精通 LangGraph 多智能体架构、高级 RAG、金融业务逻辑。

原始工作目录中存在一个名为 `robot_vacuum_agent` 的项目。该项目原本是一个基于基础 LangChain 和简单 RAG 架构的“扫地机器人客服 Agent”，只具备基础线性问答和简单文档检索能力。

用户的核心目标是：

将该项目重构并开发为一个具有工业级深度的：

```text
金融研报分析与多跳推理 Agent
```

用户明确要求：

1. 不能一上来就复制目录和写代码。
2. 必须先输出一份详尽的《金融 Agent 架构设计与开发流程大纲》。
3. 在用户确认大纲后，才可以执行目录复制和代码开发。
4. 用户确认指令为：

```text
大纲确认，开始编码
```

用户要求大纲必须包含以下模块：

1. 数据集来源与异构数据处理
2. 进阶 RAG 检索管线设计
3. 复杂状态记忆系统
4. 业务闭环逻辑与防幻觉机制
5. 硬核进阶功能，例如 GraphRAG 或 Multi-Agent Collaboration

---

## 2. 初始项目结构分析

对 `robot_vacuum_agent` 进行了只读分析。

当时识别到的核心文件包括：

```text
robot_vacuum_agent/
  app.py
  agent/react_agent.py
  agent/tools/agent_tools.py
  agent/tools/middleware.py
  rag/rag_service.py
  rag/vector_store.py
  model/factory.py
  utils/config_handler.py
  utils/file_handler.py
  utils/prompt_loader.py
  config/rag.yaml
  config/chroma.yaml
  config/prompts.yaml
  config/agent.yaml
  prompts/main_prompt.txt
  prompts/rag_summarize.txt
  prompts/report_prompt.txt
  data/
  chroma_db/
```

原项目主要特征如下：

1. 前端入口是 Streamlit。
2. Agent 使用 LangChain `create_agent` 构建 ReAct Agent。
3. RAG 使用 Chroma 作为向量库。
4. 文档处理支持 TXT 和 PDF。
5. 切分方式是 `RecursiveCharacterTextSplitter`，属于固定长度递归字符切分。
6. 检索链路是单路向量检索，没有混合检索、重排序、上下文压缩。
7. 工具层包含一些示例工具，如天气、位置、用户 ID、外部 CSV 数据等。
8. Prompt 内容是扫地机器人客服领域，不适用于金融场景。

原项目的 Chroma 写入方式位于旧项目的：

```text
robot_vacuum_agent\rag\vector_store.py
```

旧逻辑大致为：

```text
读取 data 目录文件
 -> 通过 pdf_loader/txt_loader 加载为 Document
 -> RecursiveCharacterTextSplitter 切分
 -> Chroma.add_documents(split_docs)
 -> 写入 persist_directory
```

用户后续要求新的金融 Agent 也应参考这种方式，将数据真正写入 Chroma，而不是只停留在 JSONL 或内存检索。

---

## 3. 架构设计大纲阶段

在编码之前，先输出了完整的《金融 Agent 架构设计与开发流程大纲》。

大纲设计目标：

```text
从“扫地机器人客服 Agent”升级为“金融研报分析与多跳推理 Agent”
```

设计中的核心升级方向包括：

1. 从线性 ReAct 问答升级为 LangGraph 状态机式多节点 Agent。
2. 从简单向量 RAG 升级为查询改写、HyDE、Sub-query、多路召回、重排序、上下文压缩、强引用溯源。
3. 从单 Agent 升级为分析师 Agent + 合规审查 Agent。
4. 从普通 TXT/PDF 文档处理升级为 PDF、CSV 表格、图像占位、新闻等异构数据处理。
5. 从普通回答升级为带来源文件、页码、证据卡片、风险提示的金融分析输出。

规划的目标目录结构包括：

```text
financial_analysis_agent/
  app.py
  config/
  agent/
    graph.py
    state.py
    nodes/
    tools/
  ingestion/
    loaders/
    parsers/
    chunkers/
    pipeline.py
  rag/
    vector_store.py
    bm25_store.py
    hybrid_retriever.py
    reranker.py
    context_compressor.py
    citation.py
  memory/
  graph_rag/
  prompts/
  data/
  tests/
```

大纲特别强调：

1. 金融指标必须强制带来源和页码。
2. 对资料不足的问题不强行生成，要明确说明资料不足。
3. 多跳问题要拆解为子问题。
4. 实时新闻、行情、政策类问题需要外部 API 或 Web Search。
5. 合规审查 Agent 必须避免确定性投资建议和收益承诺。

用户确认后，开始编码。

---

## 4. 目录状态与路径变化

编码时发现当前目录下已经存在一个 `financial_analysis_agent` 文件夹，该文件夹看起来已经是从 `robot_vacuum_agent` 复制出的副本。

因此实际操作策略为：

1. 不再覆盖原始 `robot_vacuum_agent`。
2. 直接在已有 `financial_analysis_agent` 中继续重构。
3. 保留 `robot_vacuum_agent` 作为原始参考。

后续用户将外层目录名从原来的中文路径调整为：

```text
E:\codes\study_LLM\To_work agent
```

重新确认后的最终项目路径为：

```text
E:\codes\study_LLM\To_work agent\financial_analysis_agent
```

后续所有文件链接、启动命令、建库命令都应以该路径为准。

---

## 5. 运行环境约定

最初使用默认 Python 解释器时，发现默认环境中缺少：

```text
langgraph
langchain
streamlit
```

用户随后说明项目使用的是：

```text
llm311
```

经确认，该环境路径为：

```text
F:\anaconda3\envs\llm311
```

后续验证确认：

```text
langgraph/langchain/streamlit ok
```

因此后续约定为：

所有项目运行、测试、Streamlit 启动、索引构建均使用：

```powershell
& 'F:\anaconda3\envs\llm311\python.exe'
```

例如启动可视化工作台：

```powershell
Set-Location 'E:\codes\study_LLM\To_work agent\financial_analysis_agent'
& 'F:\anaconda3\envs\llm311\python.exe' -m streamlit run app.py
```

重建索引：

```powershell
Set-Location 'E:\codes\study_LLM\To_work agent\financial_analysis_agent'
& 'F:\anaconda3\envs\llm311\python.exe' -m ingestion.pipeline
```

---

## 6. 已实现的核心代码结构

当前项目已经从原来的扫地机器人客服结构，重构为金融分析 Agent 结构。

重要文件包括：

```text
app.py
agent/graph.py
agent/state.py
agent/llm_utils.py
agent/nodes/query_transform_node.py
agent/nodes/retrieval_node.py
agent/nodes/graph_rag_node.py
agent/nodes/calculator_node.py
agent/nodes/web_search_node.py
agent/nodes/reasoning_node.py
agent/nodes/critic_node.py
agent/nodes/citation_guard_node.py
agent/nodes/final_answer_node.py
ingestion/pipeline.py
ingestion/schema.py
ingestion/loaders/text_loader.py
ingestion/loaders/pdf_loader.py
ingestion/loaders/table_loader.py
ingestion/loaders/image_loader.py
ingestion/chunkers/structure_chunker.py
ingestion/chunkers/semantic_chunker.py
rag/vector_store.py
rag/bm25_store.py
rag/hybrid_retriever.py
rag/reranker.py
rag/context_compressor.py
rag/citation.py
memory/short_term.py
memory/long_term.py
memory/heartbeat.py
graph_rag/entity_extractor.py
graph_rag/relation_extractor.py
graph_rag/graph_store.py
graph_rag/graph_retriever.py
```

---

## 7. LangGraph Agent 主流程

核心入口为：

```text
agent/graph.py
```

主类为：

```python
FinancialGraphAgent
```

设计为：

1. 有 LangGraph 时优先编译 `StateGraph`。
2. 如果 LangGraph 不可用，保留本地顺序执行兜底。
3. 在 `llm311` 环境下已确认真实走 `CompiledStateGraph`。

当前节点流大致为：

```text
START
 -> router
 -> query_transform
 -> retrieval
 -> graph_rag
 -> calculator
 -> web_search
 -> reasoning
 -> critic
 -> citation_guard
 -> final_answer
 -> END
```

状态对象定义在：

```text
agent/state.py
```

核心字段包括：

```text
messages
user_query
intent
entities
query_plan
sub_queries
retrieved_docs
evidence_cards
graph_relations
calculations
draft_answer
critique_result
citation_errors
final_answer
needs_web_search
reflection_round
memory_snapshot
```

---

## 8. LLM 调用与 API 接入调整

用户要求：

```text
query_transform_node、reasoning_node、critic_node 改成真实 LLM 调用优先，失败时规则逻辑兜底。
```

已新增：

```text
agent/llm_utils.py
```

用于统一封装：

1. 判断是否真实 LLM。
2. 调用 `chat_model.invoke(prompt)`。
3. 提取 JSON。
4. 捕获异常并抛出 `LLMCallError`。

三个节点的调整：

### 8.1 Query Transform

文件：

```text
agent/nodes/query_transform_node.py
```

现在逻辑为：

```text
优先 LLM 生成：
  rewritten_query
  sub_queries
  hyde_document
  required_metrics
  intent_hint

失败时回退规则逻辑。
```

### 8.2 Reasoning

文件：

```text
agent/nodes/reasoning_node.py
```

现在逻辑为：

```text
优先 LLM 基于 EvidenceCard、GraphRAG 关系、计算结果生成金融分析回答。
失败时回退规则模板。
```

要求 LLM 输出时：

1. 事实、预测、分析判断要区分。
2. 财务指标必须带来源。
3. 不得输出确定性投资建议。
4. 资料不足要说明不足。

### 8.3 Critic

文件：

```text
agent/nodes/critic_node.py
```

现在逻辑为：

```text
优先 LLM 做合规审查。
然后叠加确定性规则审查。
失败时仅用规则审查。
```

审查点包括：

1. 无来源财务数字。
2. 确定性投资建议。
3. 预测伪装成事实。
4. 风险提示缺失。
5. 证据不足。

---

## 9. DashScope 与 compatible-mode 调整

用户在系统中配置了 `DASHSCOPE_API_KEY`。

过程中曾遇到两个问题：

### 9.1 一开始 `dashscope` 包缺失

`llm311` 环境有 LangGraph/LangChain/Streamlit，但最初缺少：

```text
dashscope
```

因此执行安装：

```powershell
& 'F:\anaconda3\envs\llm311\python.exe' -m pip install dashscope --upgrade
```

### 9.2 `qwen3.6-plus` 使用原生 ChatTongyi 报 URL error

当模型名为：

```text
qwen3.6-plus
```

走 `ChatTongyi` 原生 DashScope 接口时出现：

```text
status_code: 400
code: InvalidParameter
message: url error, please check url
```

随后将 Chat 模型切换为 DashScope OpenAI-compatible 模式。

当前配置：

```yaml
chat_provider: dashscope_compatible
chat_model_name: qwen3.6-plus
chat_base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
```

代码位置：

```text
model/factory.py
config/model.yaml
```

当前工厂层会创建：

```text
ChatOpenAI
```

并使用：

```text
base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
api_key = DASHSCOPE_API_KEY
```

---

## 10. 当前数据处理流程

数据入口目录为：

```text
data/raw
```

配置在：

```text
config/rag.yaml
```

核心配置：

```yaml
data_path: data/raw
processed_path: data/processed/chunks.jsonl
index_path: data/indexes
retriever_k: 8
rerank_top_n: 6
chunk_max_chars: 1200
chunk_overlap_chars: 120
```

当前支持的文件类型：

```text
.txt
.md
.pdf
.csv
.png
.jpg
.jpeg
```

数据构建入口：

```text
ingestion/pipeline.py
```

执行流程：

```text
discover_files()
 -> load_documents()
 -> clean_financial_text()
 -> build_chunks()
 -> write_chunks()
 -> VectorStoreService(chunks).build_from_chunks()
```

---

## 11. SourceDocument 与 Chunk

数据先被转成 `SourceDocument`。

定义文件：

```text
ingestion/schema.py
```

`SourceDocument` 字段：

```python
doc_id: str
source_file: str
text: str
doc_type: str
page_number: int | None
metadata: dict
```

随后切分成 `Chunk`。

`Chunk` 字段：

```python
chunk_id: str
doc_id: str
text: str
source_file: str
page_start: int | None
page_end: int | None
section_path: str
doc_type: str
metadata: dict
```

这些字段是后续 Citation、Chroma metadata、EvidenceCard 的基础。

用户特别提醒：

> 后续聊天中涉及代码文件，应以可点击跳转链接形式展示。

因此后续回答中应使用类似：

```markdown
[ingestion/schema.py](<E:\codes\study_LLM\To_work agent\financial_analysis_agent\ingestion\schema.py:1>)
```

这种可点击链接形式。

---

## 12. 当前异构数据加载逻辑

### 12.1 TXT / Markdown

文件：

```text
ingestion/loaders/text_loader.py
```

逻辑：

```text
读取 UTF-8 文本
 -> 根据文件名推断 doc_type
 -> 生成 SourceDocument
```

doc_type 推断包括：

```text
annual_report
research_report
conference_call
financial_news
text
```

### 12.2 PDF

文件：

```text
ingestion/loaders/pdf_loader.py
```

逻辑：

```text
使用 pypdf.PdfReader
 -> 每页提取文本
 -> 每页生成一个 SourceDocument
 -> page_number 保存页码
```

当前 PDF 解析仍较基础：

1. 没有复杂版面识别。
2. 没有表格结构抽取。
3. 没有图表解析。
4. 暂时只是页级文本。

### 12.3 CSV

文件：

```text
ingestion/loaders/table_loader.py
```

逻辑：

```text
csv.DictReader 读取表格
 -> table_to_text()
 -> 转为自然语言表格描述
 -> 生成 SourceDocument
```

原始前 50 行保存在 metadata 中：

```text
raw_rows
```

### 12.4 图片

文件：

```text
ingestion/loaders/image_loader.py
```

当前只是占位：

```text
需要视觉模型生成图表描述后再入库
```

---

## 13. 当前切分逻辑

### 13.1 结构切分

文件：

```text
ingestion/chunkers/structure_chunker.py
```

适用于：

```text
annual_report
research_report
financial_table
pdf
```

逻辑：

```text
识别章节标题
 -> 按 section_path 聚合正文
 -> 章节内按 max_chars 滑窗
 -> overlap_chars 重叠
 -> 生成 Chunk
```

### 13.2 语义切分

文件：

```text
ingestion/chunkers/semantic_chunker.py
```

适用于普通文本、会议纪要、新闻等。

逻辑：

```text
按空行切段
 -> 提取 token set
 -> 用 Jaccard 判断相邻段落相似度
 -> 低相似度或超长时切断
```

当前语义切分是轻量规则版，不是真正基于 embedding 的 Semantic Chunking。

---

## 14. JSONL 中间产物

当前仍会生成：

```text
data/processed/chunks.jsonl
```

用途：

1. 审计。
2. 调试。
3. 快速查看 chunk。
4. 给 BM25 和 GraphRAG 初始化使用。

但用户后续明确要求：

> 不应只停留在 JSONL，数据应真正写入 Chroma 向量库。

因此当前目标是：

```text
JSONL 作为审计产物
Chroma 作为真正检索向量库
```

---

## 15. Chroma 持久化向量库改造

用户要求参考旧项目 `robot_vacuum_agent` 中 Chroma 写入逻辑，把当前金融项目数据真正写入向量库。

已重写：

```text
rag/vector_store.py
```

现在 `VectorStoreService` 会：

1. 初始化 Chroma。
2. 使用 collection name：

```text
financial_agent
```

3. 使用持久化目录：

```text
data/indexes/chroma_db
```

4. 将 `Chunk` 转成 LangChain `Document`。
5. 把 metadata 写入 Chroma。
6. 使用 `chunk_id` 作为 Chroma document id。
7. 使用 `similarity_search_with_score()` 检索。

Chroma 配置文件：

```text
config/chroma.yaml
```

核心配置：

```yaml
collection_name: financial_agent
persist_directory: data/indexes/chroma_db
retriever_k: 8
data_path: data/raw
```

当前 Chroma 写入主方法：

```python
VectorStoreService.build_from_chunks(chunks, force=True)
```

当前入口：

```powershell
& 'F:\anaconda3\envs\llm311\python.exe' -m ingestion.pipeline
```

或者：

```powershell
& 'F:\anaconda3\envs\llm311\python.exe' -m rag.vector_store
```

---

## 16. 当前 Chroma 写入状态

用户发现数据库中的：

```text
embedding_fulltext_search
```

为空。

随后检查 Chroma SQLite 表计数。

当前结果：

```text
collections                 1
segments                    2
embeddings                  0
embedding_metadata          0
embedding_fulltext_search   0
```

说明：

1. Chroma collection 已创建。
2. 底层 segment 已初始化。
3. 但是没有任何向量写入。
4. `embedding_fulltext_search` 为空不是单独问题，本质原因是 `embeddings = 0`。

因此当前状态是：

```text
Chroma 已初始化，但向量数据尚未成功写入。
```

判断 Chroma 是否真正写入，最可靠的是：

```powershell
& 'F:\anaconda3\envs\llm311\python.exe' -c "from rag.vector_store import VectorStoreService; s=VectorStoreService(); print(s.collection_count())"
```

如果返回大于 0，才说明写入成功。

---

## 17. Chroma 写入失败原因

写入 Chroma 的代码已经执行到：

```text
Chroma.add_documents()
```

失败点不在 Chroma 本身，而是在写入前需要调用 Embedding API 生成向量。

过程中出现过两个问题：

### 17.1 DashScope Embedding 原生接口 401

最初 `DashScopeEmbeddings` 返回：

```text
401 InvalidApiKey
```

因此将 Embedding 也切为 OpenAI-compatible 模式。

当前配置：

```yaml
embedding_provider: dashscope_compatible
embedding_model_name: text-embedding-v4
embedding_base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 17.2 OpenAIEmbeddings 下载 tiktoken 文件失败

使用 `OpenAIEmbeddings` 后，它默认会尝试下载：

```text
cl100k_base.tiktoken
```

下载地址：

```text
openaipublic.blob.core.windows.net
```

由于网络/SSL 问题失败。

因此已在 `model/factory.py` 中关闭：

```python
tiktoken_enabled=False
check_embedding_ctx_length=False
```

### 17.3 最后仍出现 Embedding API 401

关闭 tiktoken 后，请求已经进入 DashScope compatible embedding endpoint，但仍返回：

```text
401 invalid_api_key
```

这说明当前 Codex 进程读到的 Key 对 Embedding endpoint 仍无效，或者新的有效 Key 没有被当前进程获取。

用户后续表示已经解决了 API 问题，但目前从检查结果看，当前项目目录中的 Chroma 仍未成功写入向量。

---

## 18. 检索管线当前状态

检索入口：

```text
rag/hybrid_retriever.py
```

当前混合检索逻辑：

```text
read_chunks()
 -> VectorStoreService(chunks)
 -> BM25Store(chunks)
 -> dense search
 -> BM25 search
 -> merge by chunk_id
 -> LocalReranker
 -> ContextCompressor
 -> EvidenceCard
```

其中：

1. 向量检索现在应走 Chroma。
2. BM25 仍基于 JSONL chunk 在内存构建。
3. Reranker 当前是本地词面规则。
4. ContextCompressor 将 chunk 压缩为 EvidenceCard。

EvidenceCard 定义在：

```text
rag/citation.py
```

字段包括：

```text
claim
evidence
source_file
page_number
chunk_id
score
metric
confidence
metadata
```

---

## 19. 当前前端可视化状态

Streamlit 入口：

```text
app.py
```

启动命令：

```powershell
Set-Location 'E:\codes\study_LLM\To_work agent\financial_analysis_agent'
& 'F:\anaconda3\envs\llm311\python.exe' -m streamlit run app.py
```

用户成功运行后，输入示例问题：

```text
计算机行业景气度是否反转？
```

系统输出了较完整的金融分析。

输出内容包括：

1. 核心结论。
2. 细分需求与业务结构。
3. 财务表现与经营质量。
4. 未来展望与核心假设。
5. 风险提示。
6. 分析师提示。
7. 不构成投资建议声明。

该输出说明基础闭环已经跑通：

1. Streamlit 可视化入口正常。
2. 用户问题能进入 Agent。
3. LLM 能生成结构化金融分析。
4. RAG 能召回示例科技相关资料。
5. 引用来源能显示文件名和页码。
6. 模型能避免过度外推，指出资料不足。

因此当前项目可以视为：

```text
v0.1 基础闭环已跑通
```

但数据层和 Chroma 持久化仍需要继续完善。

---

## 20. 当前样例数据

已加入示例金融数据：

```text
data/raw/示例科技2024年年报摘要.txt
data/raw/示例科技深度研报.txt
data/raw/示例科技业绩会纪要.txt
data/raw/示例科技财务指标.csv
```

这些样例支撑了前端测试问题，例如：

```text
示例科技2024年现金流质量如何？
示例科技2025年预测市盈率是多少倍？
计算机行业景气度是否反转？
```

但是这些只是演示语料，不是完整真实金融研报数据集。

---

## 21. 已发现的问题与风险

### 21.1 文件编码显示问题

在 PowerShell 输出中，部分中文文件内容显示为乱码。

这通常是终端编码问题，不一定代表文件本身损坏。

但部分代码文件中确实存在早期由终端/补丁造成的中文乱码字符串，例如一些 fallback 文案、正则关键词。

后续应统一清理为 UTF-8 中文。

### 21.2 Chroma 尚未真正写入向量

当前 Chroma 表状态：

```text
embeddings = 0
```

说明建库未完成。

必须在 API Key 确实可用于 Embedding endpoint 后重新运行：

```powershell
& 'F:\anaconda3\envs\llm311\python.exe' -m ingestion.pipeline
```

### 21.3 PDF 解析能力基础

当前 PDF 解析只是页级文本提取。

还没有：

1. 多栏布局恢复。
2. 表格识别。
3. 图表 OCR/视觉描述。
4. 页眉页脚去噪。
5. PDF 目录结构解析。

### 21.4 表格没有进入 SQL 层

CSV 当前只是转成文本。

还没有：

1. DuckDB/SQLite 表格查询。
2. Text-to-SQL。
3. 财务指标字段标准化。
4. 表格行级 Citation。

### 21.5 Reranker 仍是本地规则

当前 `LocalReranker` 使用词面 overlap、doc_type authority、freshness 简单打分。

还没有接入：

1. BGE-Reranker。
2. Cohere Rerank。
3. DashScope Rerank。
4. CrossEncoder。

### 21.6 外部新闻与行情 API 仍是占位

`web_search_node.py` 和 `agent/tools/web_tools.py` 当前只是占位。

涉及：

```text
今天
最新
新闻
实时
股价
公告
监管
```

这类问题时，后续应接入真实外部数据源。

---

## 22. 后续最优先迭代路线

建议按以下顺序推进。

### 阶段 1：修复并验证 Chroma 入库

目标：

```text
data/raw -> chunks.jsonl -> Chroma embeddings > 0
```

检查命令：

```powershell
Set-Location 'E:\codes\study_LLM\To_work agent\financial_analysis_agent'
& 'F:\anaconda3\envs\llm311\python.exe' -m ingestion.pipeline
& 'F:\anaconda3\envs\llm311\python.exe' -c "from rag.vector_store import VectorStoreService; s=VectorStoreService(); print(s.collection_count())"
```

验收标准：

```text
collection_count() > 0
embeddings 表数量 > 0
```

### 阶段 2：清理 UTF-8 中文乱码

重点文件：

```text
model/factory.py
ingestion/parsers/financial_pdf_parser.py
ingestion/chunkers/structure_chunker.py
rag/context_compressor.py
agent/nodes/*
```

目标：

1. 规则关键词全部恢复为正常中文。
2. fallback 文案全部可读。
3. 正则逻辑不受乱码影响。

### 阶段 3：增强 PDF 解析

目标：

1. 引入 `pdfplumber`。
2. 对表格页进行抽取。
3. 保留 page number。
4. 识别章节标题。
5. 去除页眉页脚、免责声明。
6. 让表格转成 Markdown + JSON + 文本描述。

### 阶段 4：表格进入 SQL 查询

建议引入：

```text
DuckDB 或 SQLite
```

目标：

1. CSV/财报表格结构化入库。
2. 支持指标计算。
3. 支持 Text-to-SQL。
4. 输出表格行级 citation。

### 阶段 5：接入真实 Reranker

可选：

1. BGE-Reranker。
2. DashScope Rerank。
3. Cohere Rerank。

目标：

```text
混合召回后用 cross-encoder/rerank 模型排序
```

### 阶段 6：外部新闻和行情 API

建议接入：

1. 新闻搜索 API。
2. 股票行情 API。
3. 宏观指标 API。
4. 行业景气度数据源。

用于回答：

```text
行业景气度是否反转？
最新公告怎么看？
今天股价为什么变化？
政策变化影响什么？
```

### 阶段 7：构建评测集

评测类型：

1. 单文档事实问答。
2. 多文档交叉验证。
3. 多跳推理。
4. 表格指标计算。
5. 无证据拒答。
6. 引用完整性检查。
7. 预测/事实区分。
8. 合规风险拦截。

---

## 23. 重要命令汇总

### 23.1 启动可视化工作台

```powershell
Set-Location 'E:\codes\study_LLM\To_work agent\financial_analysis_agent'
& 'F:\anaconda3\envs\llm311\python.exe' -m streamlit run app.py
```

如端口冲突：

```powershell
& 'F:\anaconda3\envs\llm311\python.exe' -m streamlit run app.py --server.port 8502
```

### 23.2 CLI 直接问答

```powershell
Set-Location 'E:\codes\study_LLM\To_work agent\financial_analysis_agent'
& 'F:\anaconda3\envs\llm311\python.exe' app.py "示例科技2024年现金流质量如何？"
```

### 23.3 重建 JSONL + Chroma

```powershell
Set-Location 'E:\codes\study_LLM\To_work agent\financial_analysis_agent'
& 'F:\anaconda3\envs\llm311\python.exe' -m ingestion.pipeline
```

### 23.4 单独重建 Chroma

```powershell
Set-Location 'E:\codes\study_LLM\To_work agent\financial_analysis_agent'
& 'F:\anaconda3\envs\llm311\python.exe' -m rag.vector_store
```

### 23.5 检查 Chroma collection 数量

```powershell
Set-Location 'E:\codes\study_LLM\To_work agent\financial_analysis_agent'
& 'F:\anaconda3\envs\llm311\python.exe' -c "from rag.vector_store import VectorStoreService; s=VectorStoreService(); print(s.collection_count())"
```

### 23.6 编译检查

```powershell
Set-Location 'E:\codes\study_LLM\To_work agent\financial_analysis_agent'
& 'F:\anaconda3\envs\llm311\python.exe' -m compileall -q .
```

---

## 24. 当前项目结论

当前项目已经完成从 `robot_vacuum_agent` 到 `financial_analysis_agent` 的初步架构级重构。

已经跑通：

1. Streamlit 可视化入口。
2. LangGraph 主流程。
3. LLM 主路径。
4. 金融分析输出。
5. 基础 RAG 检索。
6. EvidenceCard 引用。
7. Critic 合规审查。
8. GraphRAG 雏形。
9. Memory 雏形。

尚未完全跑通：

1. Chroma 向量数据真实写入。
2. PDF 高质量解析。
3. 表格 SQL 化。
4. 真正 Reranker。
5. 实时金融数据 API。
6. 系统化评测。

项目现在可以定义为：

```text
金融研报分析与多跳推理 Agent v0.1
```

下一阶段最关键的任务是：

```text
优先修复并验证 Chroma 入库，使 data/raw 中的金融语料真正进入持久化向量库。
```

完成该任务后，整个 RAG 链路将从“能演示”进一步升级为“可持续积累和检索真实金融资料”的工程基础。

