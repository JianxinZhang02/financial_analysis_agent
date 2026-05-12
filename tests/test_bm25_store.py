from rag.bm25_store import tokenize


def test_tokenize_uses_jieba_for_chinese_terms():
    tokens = tokenize("腾讯2024年现金流质量如何？")

    assert "腾讯" in tokens
    assert "2024" in tokens
    assert "现金流" in tokens
    assert not {"腾", "讯", "现", "金", "流"}.issubset(tokens)


def test_tokenize_preserves_english_numbers_and_symbols():
    tokens = tokenize("0700.HK revenue grew 12.5% in 2024")

    assert "0700.hk" in tokens
    assert "revenue" in tokens
    assert "12.5%" in tokens
    assert "2024" in tokens
