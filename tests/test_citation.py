from rag.citation import EvidenceCard, has_valid_citation


def test_citation_guard_fields():
    card = EvidenceCard(
        claim="示例",
        evidence="示例证据",
        source_file="sample.txt",
        page_number=1,
        chunk_id="abc",
    )
    assert has_valid_citation(card)
