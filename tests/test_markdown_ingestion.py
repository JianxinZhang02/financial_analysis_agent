from ingestion.chunkers.markdown_chunker import MarkdownHierarchyChunker
from ingestion.parsers.page_cleaner import clean_page_texts
from ingestion.schema import SourceDocument
from rag.query_filters import (
    augment_query_for_retrieval,
    infer_metadata_filter,
    normalize_query_for_metadata_filter,
    matches_metadata_filter,
)


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
    assert metadata_filter["company_id"] == "0700.HK_tencent"
    assert metadata_filter["report_period"] == "FY2024"
    # normalize_query现在会删掉已被filter覆盖的词项
    normalized = normalize_query_for_metadata_filter("腾讯2024年收入是多少？", metadata_filter)
    assert "收入" in normalized
    assert "腾讯" not in normalized


def test_infer_metadata_filter_multi_company():
    """比较型查询：'腾讯和阿里的营收对比' → 两个company_id用$in"""
    metadata_filter = infer_metadata_filter("腾讯和阿里的营收对比")
    assert isinstance(metadata_filter["company_id"], dict)
    assert "$in" in metadata_filter["company_id"]
    company_ids = metadata_filter["company_id"]["$in"]
    assert "0700.HK_tencent" in company_ids
    assert "BABA_alibaba" in company_ids


def test_infer_metadata_filter_industry():
    """行业类查询：'电商行业2024年年报' → 行业推断→company_id列表"""
    metadata_filter = infer_metadata_filter("电商行业2024年年报")
    # 电商 → 电子商务 → BABA_alibaba + PDD_pdd
    assert "company_id" in metadata_filter
    assert isinstance(metadata_filter["company_id"], dict)
    assert "$in" in metadata_filter["company_id"]
    assert "BABA_alibaba" in metadata_filter["company_id"]["$in"]
    assert "PDD_pdd" in metadata_filter["company_id"]["$in"]
    assert metadata_filter["report_period"] == "FY2024"
    assert metadata_filter["doc_type"] == "annual_report"


def test_infer_metadata_filter_near_years():
    """近N年查询：'腾讯近3年营收' → $in时间范围"""
    metadata_filter = infer_metadata_filter("腾讯近3年营收")
    assert metadata_filter["company_id"] == "0700.HK_tencent"
    assert isinstance(metadata_filter["report_period"], dict)
    assert "$in" in metadata_filter["report_period"]
    # 近3年 = [FY2025, FY2024, FY2023]
    periods = metadata_filter["report_period"]["$in"]
    assert len(periods) == 3


def test_infer_metadata_filter_range_years():
    """跨年查询：'腾讯2022到2024年营收' → $in时间范围"""
    metadata_filter = infer_metadata_filter("腾讯2022到2024年营收")
    assert metadata_filter["company_id"] == "0700.HK_tencent"
    assert isinstance(metadata_filter["report_period"], dict)
    periods = metadata_filter["report_period"]["$in"]
    assert "FY2022" in periods
    assert "FY2023" in periods
    assert "FY2024" in periods


def test_augment_query_semantic_enrichment():
    """加法式改写：给Dense通道增强语义信息"""
    metadata_filter = infer_metadata_filter("腾讯营收")
    augmented = augment_query_for_retrieval("腾讯营收", metadata_filter)
    # 应补充公司全称 + 财务指标同义词
    assert "营收" in augmented
    # 应回补同义指标
    assert "营业收入" in augmented or "总收入" in augmented


def test_matches_metadata_filter_multi_value():
    """$in多值过滤匹配"""
    metadata = {"company_id": "0700.HK_tencent", "report_period": "FY2024"}
    filter_with_in = {"company_id": {"$in": ["0700.HK_tencent", "BABA_alibaba"]}}
    assert matches_metadata_filter(metadata, filter_with_in) is True

    metadata_other = {"company_id": "BIDU_baidu", "report_period": "FY2024"}
    assert matches_metadata_filter(metadata_other, filter_with_in) is False


def test_matches_metadata_filter_single_value():
    """单值精确匹配（向后兼容）"""
    metadata = {"company_id": "0700.HK_tencent", "report_period": "FY2024"}
    filter_single = {"company_id": "0700.HK_tencent"}
    assert matches_metadata_filter(metadata, filter_single) is True

    metadata_other = {"company_id": "BIDU_baidu"}
    assert matches_metadata_filter(metadata_other, filter_single) is False