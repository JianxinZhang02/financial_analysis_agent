from ingestion.chunkers.markdown_chunker import MarkdownHierarchyChunker
from ingestion.parsers.page_cleaner import clean_page_texts
from ingestion.schema import SourceDocument
from rag.query_filters import infer_metadata_filter, normalize_query_for_metadata_filter


def test_clean_page_texts_removes_repeated_report_headers():
    pages = [
        "1 美 团 2024 年度报告\n财务概要\n收入 337,591,576",
        "2 美 团 2024 年度报告\n主席报告\n收入增长",
        "3 美 团 2024 年度报告\n管理层讨论及分析\n毛利增长",
    ]

    cleaned = clean_page_texts(pages)

    assert all("年度报告" not in page.splitlines()[0] for page in cleaned)
    assert "收入 337,591,576" in cleaned[0]


def test_markdown_chunker_keeps_table_block_type():
    doc = SourceDocument(
        doc_id="doc-1",
        source_file="report.pdf",
        text="### Table p1_t1\n\n| 指标 | 2024 |\n| --- | --- |\n| 收入 | 660,257 |",
        doc_type="annual_report",
        page_number=4,
        metadata={"block_type": "table", "table_id": "p1_t1"},
    )

    chunks = MarkdownHierarchyChunker(max_chars=1200, overlap_chars=120).split([doc])

    assert len(chunks) == 1
    assert chunks[0].metadata["block_type"] == "table"
    assert chunks[0].metadata["table_id"] == "p1_t1"
    assert "660,257" in chunks[0].text


def test_infer_metadata_filter_company_and_year():
    metadata_filter = infer_metadata_filter("腾讯2024年收入是多少？")
    assert metadata_filter == {
        "company_id": "0700.HK_tencent",
        "report_period": "FY2024",
    }
    assert normalize_query_for_metadata_filter("腾讯2024年收入是多少？", metadata_filter) == "收入是多少？"
