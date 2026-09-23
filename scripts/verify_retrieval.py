import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.embeddings import Embedder
from src.vector_store import FAISSVectorStore

def test_retrieval(query: str, top_k: int = 3):
    print(f"\n--- Searching for: '{query}' (top_k={top_k}) ---")
    
    vector_store = FAISSVectorStore.load(config.VECTOR_STORE_DIR)
    embedder = Embedder(model_name=config.EMBEDDING_MODEL_NAME)
    query_vector = embedder.embed_query(query)
    
    results = vector_store.similarity_search(query_vector, top_k=top_k)
    
    if not results:
        print("No matching results found.")
        return

    for rank, (doc, score) in enumerate(results, start=1):
        meta = doc["metadata"]
        print(f"\nResult #{rank} [Similarity Score: {score:.4f}]")
        print(f" Source Document: {meta['source']}")
        print(f" Page Number: {meta['page']} / {meta['total_pages']}")
        print(f" Chunk ID: {meta['chunk_id']}")
        print(f" Snippet: \"{doc['text'][:250]}...\"")

if __name__ == "__main__":
    sample_queries = [
        "What is AI risk management framework?",
        "How is AI bias measured or mitigated?",
        "What are the ethical concerns of AI governance?"
    ]
    
    for q in sample_queries:
        test_retrieval(q, top_k=3)
