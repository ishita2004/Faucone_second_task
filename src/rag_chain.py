import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
from src.rag_pipeline import retrieve_documents
from src.llm import GeminiClient
from dotenv import load_dotenv

load_dotenv()

class RAGChain:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.gemini = GeminiClient(self.api_key) if (self.api_key and len(self.api_key) > 20 and not self.api_key.startswith("AQ.")) else None

    def query(self, user_question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Executes end-to-end retrieval and generation using pre-computed vector store.
        """
        if not user_question.strip():
            raise ValueError("Please enter a question.")

        # 1. Retrieve top-k chunks using Rohit's retrieval pipeline
        retrieved_chunks = retrieve_documents(user_question, top_k=top_k)

        if not retrieved_chunks:
            return {
                "question": user_question,
                "answer": "I don't know based on the provided documents.",
                "citations": []
            }

        # 2. Build context string & citations
        context_parts = []
        citations = []

        for idx, item in enumerate(retrieved_chunks, start=1):
            text = item.get("text", "").strip()
            meta = item.get("metadata", {})
            source = meta.get("source", meta.get("pdf_name", "Unknown.pdf"))
            page = meta.get("page", meta.get("page_number", 1))
            score = item.get("score", 0.0)

            context_parts.append(f"Source: {source} (Page {page})\n{text}")
            citations.append({
                "pdf_name": source,
                "source": source,
                "page": page,
                "score": score,
                "text": text
            })

        context_str = "\n\n---\n\n".join(context_parts)

        # 3. Generate Answer using Gemini API (or Grounded Fallback if no API key is provided)
        if self.gemini:
            try:
                answer = self.gemini.answer_with_context(user_question, context_str)
            except Exception as e:
                print(f"Gemini API call failed ({e}). Using grounded synthesis.")
                answer = self._grounded_fallback(context_parts)
        else:
            answer = self._grounded_fallback(context_parts)

        return {
            "question": user_question,
            "answer": answer,
            "citations": citations
        }

    def _grounded_fallback(self, context_parts: List[str]) -> str:
        body = "\n\n".join([f"• {c}" for c in context_parts])
        return f"Based on the retrieved document contexts, here is the relevant information:\n\n{body}"

def answer_question(pdf_dir: Any, api_key: str, question: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    chain = RAGChain(api_key=api_key)
    res = chain.query(question, top_k=top_k)
    return res["answer"], res["citations"]

def format_citations(citations: List[Dict[str, Any]]) -> str:
    if not citations:
        return "No citations available."
    lines = []
    for idx, c in enumerate(citations, start=1):
        source = c.get("source", c.get("pdf_name", "Unknown.pdf"))
        page = c.get("page", "?")
        score = c.get("score", 0.0)
        snippet = c.get("text", "").replace("\n", " ")
        lines.append(f"[{idx}] {source} (Page {page}) — Match Score: {score:.4f}\n   \"{snippet[:250]}...\"")
    return "\n\n".join(lines)

def list_pdf_files(pdf_dir: Any) -> List[Path]:
    folder = Path(pdf_dir)
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"],
        key=lambda item: item.name.lower(),
    )
