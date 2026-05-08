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

原始金融语料放在 `data/raw/`，索引输出到 `data/processed/chunks.jsonl`。

## 关键约束

金融指标和关键结论必须来自 EvidenceCard，并携带 `source_file`、`page_number`、`chunk_id`。证据不足时系统会拒绝编造。
