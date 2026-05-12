# CLAUDE.md — financial_analysis_agent

## Project Overview

对话式金融研报分析与多跳推理智能体。基于 LangGraph 状态驱动编排，串联路由→改写→检索→图谱→计算→推理→审查→溯源守卫→终答全流程，支持 Reflection-Critic 反思闭环，输出带来源页码溯源的结构化研报分析。

## Tech Stack

- LangGraph 1.1+ / LangChain 0.3+ / Streamlit / Qwen(Dashscope) / Chroma + BM25 + LocalReranker
- Python 3.11, conda env: `F:\anaconda3\envs\llm311`

## Commands

```bash
# Run Streamlit app
F:\anaconda3\envs\llm311\python.exe -m streamlit run app.py

# Run tests
F:\anaconda3\envs\llm311\python.exe -m pytest tests/ -v --tb=short

# Rebuild index (re-chunk + re-embed)
F:\anaconda3\envs\llm311\python.exe -m ingestion.pipeline

# Build Chroma vector store
F:\anaconda3\envs\llm311\python.exe rag\vector_store.py
```

## Architecture

### Graph Flow (agent/graph.py)

`router → query_transform → retrieval → graph_rag → calculator → reasoning → critic → citation_guard → final_answer`

Conditional edges:
- `calculator → web_search` when `needs_web_search=True`
- `critic → query_transform` when BLOCKING issues found and `reflection_round < max_reflection_rounds`
- `critic → citation_guard / final_answer` when no BLOCKING issues or reflection exhausted

### Key Nodes

| Node | File | Role |
|------|------|------|
| router | agent/nodes/router_node.py | Intent classification + entity extraction |
| query_transform | agent/nodes/query_transform_node.py | Config-driven query rewrite/sub-queries/HyDE |
| retrieval | agent/nodes/retrieval_node.py | Iterate sub_queries, deduplicate, cap at 8 cards |
| critic | agent/nodes/critic_node.py | BLOCKING vs FORMAT issue classification |
| calculator | agent/nodes/calculator_node.py | YoY growth, PE ratio from evidence text |
| citation_guard | agent/nodes/citation_guard_node.py | Enforce source_file + chunk_id + page_number |

### Config Alignment

**Critical**: rag.yaml flags (`enable_query_rewrite`, `enable_hyde`, `enable_sub_queries`) must be respected by node code. When disabled, nodes skip LLM calls and use lightweight fallbacks. Reflection loop only activates when `enable_query_rewrite=True`.

### State (agent/state.py)

`FinancialAgentState` — TypedDict with 20 fields, total=False. Key fields:
- `reflection_round: int` — incremented by query_transform_node on re-entry from critic
- `critique_result: dict` — contains `issues` list (prefixed with "BLOCKING:" for critical ones)
- `evidence_cards: list` — capped at 8 by retrieval_node
- `query_plan: dict` — output of query_transform, carries `sub_queries`

## Code Conventions

- Use `as_bool(value, default=False)` from `utils/helpers.py` — never write inline bool coercion
- Use `log_stage(stage_name, **fields)` context manager for every node operation
- Node return dicts must include all state fields they update (LangGraph merges by dict key)
- Never reference undefined variables in f-strings (e.g., `reflection_hint` must be computed before use)
- Chinese text in f-strings: use string concatenation, not nested curly braces inside full-width quotes

## Ingestion Pipeline

- `ingestion/pipeline.py`: discover → load → enrich from registry → chunk → cache → write JSONL → build/sync Chroma
- Document Registry (`data/raw/financial_reports/document_registry.csv`): injects company_id, report_period, doc_type metadata
- `MarkdownHierarchyChunker`: preserves section_path heading hierarchy, keeps table headers with each chunk
- Chunk cache keyed by file MD5 + registry metadata hash + chunking params signature

## RAG Pipeline (rag/)

- `HybridRetriever` → `RetrievalPipeline` with pluggable `RetrievalStep` ABC: Dense → BM25 → Rerank → Compress
- `query_filters.py`: company alias mapping → metadata filter, then remove filtered terms from search query
- `context_compressor.py`: sentence scoring + financial table structured extraction ("表格解读：2024年营业收入 XXX亿元")
- `LocalReranker`: lexical overlap + authority bonus + freshness bonus, weighted by rag.yaml dense/bm25/rerank weights

## Known Issues & Design Notes

- Heartbeat memory compression exists but summary output is not consumed by downstream nodes — not a fully closed loop
- `web_search_node` is a placeholder (no external API wired), returns a note about local-only data
- GraphRAG uses rule-based entity/relation extraction (no LLM), seeded from query entities
- calculator_node uses regex extraction from evidence text, not LLM — limited to patterns it knows

## Testing

13 tests in `tests/`, covering chunking, retrieval, citation, graph flow, markdown ingestion, financial dataset registry, and reflection loop. Run with pytest — each test may take 30-60s due to LLM calls.