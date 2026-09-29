from typing import List, Dict, Any, Tuple
from src.config import config
from src.embeddings import Embedder
from src.vector_store import FAISSVectorStore

class Retriever:
    def __init__(self, vector_store: FAISSVectorStore = None, embedder: Embedder = None):
        self.vector_store = vector_store or FAISSVectorStore.load(config.VECTOR_STORE_DIR)
        self.embedder = embedder or Embedder(model_name=config.EMBEDDING_MODEL_NAME)

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[Dict[str, Any], float]]:
        k = top_k or config.TOP_K
        query_vector = self.embedder.embed_query(query)
        return self.vector_store.similarity_search(query_vector, top_k=k)
