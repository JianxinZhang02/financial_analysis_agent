# 中国互联网/AI 软件行业数据集 v0.1

本数据集聚焦中国互联网平台、AI、云与 SaaS 软件公司，不纳入汽车、制造、地产、银行等商业模式差异过大的行业。

## 目录

```text
data/raw/financial_reports/
  company_registry.csv
  document_registry.csv
  <company_id>/
    <year>/
      annual_report/
      interim_report/
      quarterly_results/
      investor_presentation/
      earnings_call/
```

## 公司池

第一版公司池包含 15 家上市公司，分为：

- `large`：腾讯、阿里、百度、美团、京东、网易、拼多多
- `mid`：快手、哔哩哔哩、腾讯音乐、微博、爱奇艺
- `small`：金山云、涂鸦智能、有道

字节跳动不进入核心财务库，因为缺少连续公开年报。它可以作为竞争格局背景资料，但不能用于核心财务指标问答。

## 文档类型

核心库只收这些类型：

- `annual_report`
- `interim_report`
- `quarterly_results`
- `investor_presentation`
- `earnings_call`
- `structured_financials`

以下公告暂不进入核心库：

- 股份奖励计划授出奖励
- 月报表和股份变动月报
- 董事会会议日期
- 日常股份购回公告
- 股东大会通函和委任表格
- ESG 报告，除非问题专门涉及 ESG

## 工作流

1. 检查 registry：

```powershell
& 'F:\anaconda3\envs\llm311\python.exe' scripts\validate_financial_dataset.py
```

2. 自动下载 SEC 可获取的 20-F 年报：

```powershell
$env:SEC_USER_AGENT='Your Name your_email@example.com'
& 'F:\anaconda3\envs\llm311\python.exe' scripts\collect_sec_filings.py
```

3. 对港股 IR 官网文档，按 `document_registry.csv` 的 `source_url` 下载到 `local_path`，并填写真实 `hash`、`publish_date`、`parse_status=downloaded`。

4. 重建 chunks 和 Chroma：

```powershell
& 'F:\anaconda3\envs\llm311\python.exe' -m ingestion.pipeline
```

5. 用 `data/evaluation/financial_agent_questions.csv` 的 30 个问题做回归评测。

## Metadata 约束

所有核心文档必须能通过 `document_registry.csv` 追踪：

```text
doc_id,company_id,ticker,company_name,doc_type,report_period,publish_date,source_url,local_path,file_ext,language,is_core_document,parse_status,hash
```

ingestion 会根据 `local_path` 自动把 registry metadata 写入 `SourceDocument.metadata`，后续 chunk、EvidenceCard 和 citation 可以继承这些字段。
