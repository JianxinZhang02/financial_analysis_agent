下面这个版本会更符合一个标准的 GitHub 项目 README 风格，重点做了这些优化：

* 去除所有本地绝对路径（如 `F:\anaconda3\...`）
* 使用跨平台命令
* 增强开源项目的结构感
* 更符合 AI Agent / RAG 项目的技术文档风格
* 补充项目架构感与目录说明
* 命令统一为 `python -m ...`
* 更适合别人 clone 后直接运行

你可以直接替换原 README 对应部分。

---

# Financial Research & Multi-Hop Reasoning Agent

A financial analysis and multi-hop reasoning agent built on top of a modular RAG + Agent architecture.

This project is refactored from the `robot_vacuum_agent` prototype and redesigned for financial research, evidence-grounded analysis, and long-context reasoning workflows.

---

# Features

## Multi-Source Financial Data Ingestion

Supports heterogeneous financial documents:

* TXT
* Markdown
* PDF
* CSV
* Image placeholders (OCR-ready)

---

## Advanced RAG Pipeline

Includes multiple retrieval and reasoning strategies:

* Structural chunking
* Semantic chunking
* Query Rewrite
* HyDE generation
* Sub-query decomposition
* BM25 retrieval
* Dense vector retrieval
* Local reranking
* Hybrid RAG orchestration

---

## Evidence-Grounded Financial Analysis

All financial conclusions are grounded in structured `EvidenceCard` objects.

Each evidence item preserves:

* `source_file`
* `page_number`
* `chunk_id`

The system refuses unsupported financial claims when evidence is insufficient.

---

## Agentic Workflow with LangGraph

Built with a multi-node agent architecture:

* Analyst node
* Critic / compliance reviewer
* Citation guard
* Memory manager
* Retrieval planner

Supports:

* LangGraph state-machine execution
* Graceful local fallback when dependencies are unavailable

---

## Memory System

Includes both:

### Short-Term Memory

Session-level conversational state.

### Long-Term Memory

Persistent user profile and preference storage.

### Heartbeat Summarization

Periodic compression and summarization of historical context.

---

## GraphRAG Prototype

Early GraphRAG support:

* Entity extraction
* Relation extraction
* Neighbor retrieval
* Graph-enhanced reasoning

---

# Quick Start

## CLI Mode

You can run the agent directly from the command line without Streamlit or LangGraph UI dependencies:

```bash
python app.py "How was ExampleTech's cash flow quality in 2024?"
```

---

## Web Workbench

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the Streamlit interface:

```bash
streamlit run app.py
```

---

# Build the Retrieval Index

Run the ingestion pipeline:

```bash
python -m ingestion.pipeline
```

This generates:

```text
data/processed/chunks.jsonl
```

Auditable intermediate chunk artifacts.

and:

```text
data/indexes/chroma_db/
```

Persistent Chroma vector database used for retrieval.

---

## Rebuild Only the Vector Store

```bash
python -m rag.vector_store
```

---

# Financial Dataset Layer

The project includes a real-world financial dataset layer under:

```text
data/raw/financial_reports/
```

Current company coverage includes:

* Tencent
* Alibaba
* Baidu
* Meituan
* NetEase
* Pinduoduo
* Kuaishou
* Bilibili

The dataset focuses on:

* Chinese internet platforms
* AI companies
* Cloud infrastructure
* SaaS businesses

Metadata is managed through:

```text
company_registry.csv
document_registry.csv
```

---

# Dataset Utilities

## Validate Dataset

```bash
python scripts/validate_financial_dataset.py
```

---

## Collect SEC Filings

Set your SEC user agent first:

### Linux / macOS

```bash
export SEC_USER_AGENT="Your Name your_email@example.com"
```

### Windows PowerShell

```powershell
$env:SEC_USER_AGENT="Your Name your_email@example.com"
```

Then run:

```bash
python scripts/collect_sec_filings.py
```

---

# Project Structure

```text
.
├── app.py
├── ingestion/
├── rag/
├── agent/
├── memory/
├── graph/
├── scripts/
├── data/
│   ├── raw/
│   ├── processed/
│   └── indexes/
├── docs/
└── requirements.txt
```

---

# Core Design Principles

## Evidence First

Financial metrics and conclusions must be traceable to source evidence.

---

## Citation Safety

The system includes a citation guard to prevent unsupported hallucinated claims.

---

## Modular Agent Architecture

Each reasoning capability is isolated into composable agent nodes.

---

# Future Roadmap

* OCR pipeline integration
* Full GraphRAG support
* Financial time-series tools
* Multi-agent debate workflows
* Portfolio reasoning
* Earnings-call analysis
* SEC filing auto-sync
* Quantitative factor extraction

---

# Documentation

See:

```text
docs/financial_dataset_v0.md
```

for detailed dataset specifications and collection workflows.

---

# License

MIT License
