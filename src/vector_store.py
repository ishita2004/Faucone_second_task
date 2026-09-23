import os
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss

class FAISSVectorStore:
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata_store: List[Dict[str, Any]] = []

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Count of documents must match count of embeddings.")

        if len(embeddings) == 0:
            print("No embeddings to add.")
            return

        normalized_embeddings = self._normalize_vectors(embeddings)
        self.index.add(normalized_embeddings)
        self.metadata_store.extend(documents)
        print(f"Added {len(documents)} chunks to FAISS vector index. Total vectors: {self.index.ntotal}.")

    def save(self, output_dir: str, index_name: str = "index.faiss", metadata_name: str = "metadata.pkl"):
        os.makedirs(output_dir, exist_ok=True)
        index_path = os.path.join(output_dir, index_name)
        metadata_path = os.path.join(output_dir, metadata_name)

        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata_store, f)

        print(f"Vector store successfully saved to '{output_dir}'.")

    @classmethod
    def load(cls, output_dir: str, index_name: str = "index.faiss", metadata_name: str = "metadata.pkl") -> 'FAISSVectorStore':
        index_path = os.path.join(output_dir, index_name)
        metadata_path = os.path.join(output_dir, metadata_name)

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Vector store files not found in '{output_dir}'. Run ingestion first.")

        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            metadata_store = pickle.load(f)

        instance = cls(embedding_dim=index.d)
        instance.index = index
        instance.metadata_store = metadata_store
        print(f"Loaded FAISS vector store from '{output_dir}' with {instance.index.ntotal} vectors.")
        return instance

    def similarity_search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if self.index.ntotal == 0:
            return []

        normalized_query = self._normalize_vectors(query_vector)
        scores, indices = self.index.search(normalized_query, min(top_k, self.index.ntotal))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx < len(self.metadata_store):
                results.append((self.metadata_store[idx], float(score)))

        return results
