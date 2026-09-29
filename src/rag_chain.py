from typing import Dict, Any, List
from src.retriever import Retriever
from src.llm import LLMClient

class RAGChain:
    def __init__(self, retriever: Retriever = None, llm_client: LLMClient = None):
        self.retriever = retriever or Retriever()
        self.llm = llm_client or LLMClient()

    def _build_rag_prompt(self, query: str, retrieved_results: List[Any]) -> str:
        context_blocks = []
        for idx, (doc, score) in enumerate(retrieved_results, start=1):
            meta = doc.get("metadata", {})
            source = meta.get("source", "Unknown")
            page = meta.get("page", "?")
            text = doc.get("text", "")
            context_blocks.append(
                f"--- CONTEXT BLOCK [{idx}] ---\n"
                f"Document: {source}\n"
                f"Page: {page}\n"
                f"Relevance Score: {score:.4f}\n"
                f"Content:\n{text}\n"
            )

        context_str = "\n".join(context_blocks)

        prompt = (
            f"You are an expert Q&A Assistant. Answer the user's question STRICTLY using only the provided context blocks below.\n"
            f"Ground Rules:\n"
            f"1. If the provided context does not contain enough information to answer the question, state clearly: 'I don't know based on the provided documents.'\n"
            f"2. Always cite the exact source document name and page number in your answer using bracket notation like [Document_Name.pdf, Page X].\n"
            f"3. Do not make up facts or use external knowledge outside the context.\n\n"
            f"Retrieved Document Contexts:\n"
            f"{context_str}\n\n"
            f"User Question: {query}\n\n"
            f"Answer with Citations:"
        )
        return prompt

    def query(self, user_question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Executes full RAG pipeline: Query -> Retrieve -> Build Prompt -> Generate LLM Answer + Citations
        """
        retrieved_items = self.retriever.retrieve(user_question, top_k=top_k)
        
        if not retrieved_items:
            return {
                "question": user_question,
                "answer": "I don't know based on the provided documents.",
                "sources": []
            }

        prompt = self._build_rag_prompt(user_question, retrieved_items)

        formatted_contexts = [
            {"document": doc, "score": score} for doc, score in retrieved_items
        ]

        answer = self.llm.generate_answer(prompt, formatted_contexts)

        sources = []
        for doc, score in retrieved_items:
            meta = doc.get("metadata", {})
            sources.append({
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", 1),
                "total_pages": meta.get("total_pages", 1),
                "chunk_id": meta.get("chunk_id", None),
                "similarity_score": score,
                "snippet": doc.get("text", "")
            })

        return {
            "question": user_question,
            "answer": answer,
            "sources": sources
        }
