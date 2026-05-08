from ingestion.pipeline import build_chunks


def test_build_chunks_has_citations():
    chunks = build_chunks()
    assert chunks
    assert all(chunk.source_file for chunk in chunks)
    assert all(chunk.chunk_id for chunk in chunks)
