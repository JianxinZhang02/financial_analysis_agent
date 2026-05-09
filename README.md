# 金融研报分析与多跳推理 Agent

这是从 `robot_vacuum_agent` 重构而来的金融分析 Agent。核心能力包括：

- 异构金融数据 ingestion：TXT/Markdown/PDF/CSV/图片占位
- 结构切分与语义切分
- Query Rewrite、HyDE 和 Sub-query 拆解
- BM25 + 向量检索 + 本地 Rerank 的混合 RAG
- EvidenceCard 强引用，保留来源文件、页码和 chunk_id
- LangGraph 多节点状态机，缺少依赖时有本地顺序执行降级
- 短期记忆、长期用户画像、Heartbeat 摘要机制
- GraphRAG 雏形：实体关系抽取与邻居检索
- 分析师节点 + Critic 合规审查 + Citation Guard

## 快速运行

当前环境如果未安装 Streamlit/LangGraph，也可以先用 CLI 验证：

```bash
python app.py "示例科技2024年现金流质量如何？"
```

安装完整依赖后启动工作台：

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 重建索引

```bash
python -m ingestion.pipeline
```

原始金融语料放在 `data/raw/`。执行后会生成两类产物：

- `data/processed/chunks.jsonl`：可审计的 chunk 中间产物
- `data/indexes/chroma_db/`：真正用于向量检索的 Chroma 持久化向量库

在 Windows + llm311 环境中推荐使用：

```powershell
& 'F:\anaconda3\envs\llm311\python.exe' -m ingestion.pipeline
```

也可以单独重建 Chroma：

```powershell
& 'F:\anaconda3\envs\llm311\python.exe' -m rag.vector_store
```

## 关键约束

金融指标和关键结论必须来自 EvidenceCard，并携带 `source_file`、`page_number`、`chunk_id`。证据不足时系统会拒绝编造。

## 真实金融数据集

项目已加入 `data/raw/financial_reports/` 数据层，用于采集“中国互联网平台 + AI/云/SaaS 软件”上市公司资料。第一版公司池覆盖腾讯、阿里、百度、美团、网易、拼多多、快手、哔哩哔哩等公司，并用 `company_registry.csv` 和 `document_registry.csv` 管理元数据。

常用命令：

```powershell
& 'F:\anaconda3\envs\llm311\python.exe' scripts\validate_financial_dataset.py
$env:SEC_USER_AGENT='Your Name your_email@example.com'
& 'F:\anaconda3\envs\llm311\python.exe' scripts\collect_sec_filings.py
& 'F:\anaconda3\envs\llm311\python.exe' -m ingestion.pipeline
```

详细说明见 `docs/financial_dataset_v0.md`。
