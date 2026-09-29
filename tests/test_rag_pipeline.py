import sys
import os
import shutil
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_splitter import RecursiveTextSplitter
from src.vector_store import FAISSVectorStore
from src.retriever import Retriever
from src.llm import LLMClient
from src.rag_chain import RAGChain

def test_recursive_text_splitter():
    splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = [{
        "text": "This is a long piece of text designed to test the recursive text splitting capabilities of our custom chunking module.",
        "metadata": {"source": "test.pdf", "page": 1, "total_pages": 1}
    }]
    chunked = splitter.split_documents(docs)
    assert len(chunked) > 1
    assert chunked[0]["metadata"]["source"] == "test.pdf"
    assert "chunk_id" in chunked[0]["metadata"]

def test_faiss_vector_store():
    temp_dir = tempfile.mkdtemp()
    try:
        vs = FAISSVectorStore(embedding_dim=4)
        sample_chunks = [
            {"text": "AI Safety and Alignment", "metadata": {"source": "doc1.pdf", "page": 1, "total_pages": 5, "chunk_id": 1}},
            {"text": "Machine Learning Systems", "metadata": {"source": "doc2.pdf", "page": 3, "total_pages": 10, "chunk_id": 2}}
        ]
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)

        vs.add_documents(sample_chunks, embeddings)
        assert vs.index.ntotal == 2

        vs.save(temp_dir)
        loaded_vs = FAISSVectorStore.load(temp_dir)
        assert loaded_vs.index.ntotal == 2

        query_vec = np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32)
        results = loaded_vs.similarity_search(query_vec, top_k=1)
        assert len(results) == 1
        assert results[0][0]["metadata"]["source"] == "doc1.pdf"
    finally:
        shutil.rmtree(temp_dir)

def test_rag_chain_out_of_scope():
    temp_dir = tempfile.mkdtemp()
    try:
        vs = FAISSVectorStore(embedding_dim=4)
        vs.save(temp_dir)
        loaded_vs = FAISSVectorStore.load(temp_dir)
        
        # Test empty result handling
        results = loaded_vs.similarity_search(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), top_k=2)
        assert len(results) == 0
    finally:
        shutil.rmtree(temp_dir)
