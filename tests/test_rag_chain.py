from src.rag_chain import chunk_text, format_citations


def test_chunk_text_splits_long_text():
    text = "word " * 2000
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 1200 for chunk in chunks)


def test_format_citations_wraps_sources():
    citations = [
        {"pdf_name": "alpha.pdf", "page": 3, "text": "This is a key fact."},
        {"pdf_name": "beta.pdf", "page": 7, "text": "Another fact."},
    ]
    result = format_citations(citations)
    assert "[1] alpha.pdf (page 3)" in result
    assert "[2] beta.pdf (page 7)" in result
