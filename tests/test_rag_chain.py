import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_chain import format_citations
from src.chunk_documents import chunk_documents


def test_chunk_documents_splits_long_text():
    docs = [{"text": "word " * 2000, "source": "test.pdf", "page": 1}]
    chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=100)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 1000 for c in chunks)


def test_format_citations_wraps_sources():
    citations = [
        {"source": "alpha.pdf", "page": 3, "score": 0.85, "text": "This is a key fact."},
        {"source": "beta.pdf", "page": 7, "score": 0.82, "text": "Another fact."},
    ]
    result = format_citations(citations)
    assert "alpha.pdf" in result
    assert "beta.pdf" in result
