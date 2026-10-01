from __future__ import annotations

from typing import Any

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("google-generativeai is required for Gemini-based RAG.") from exc


class GeminiClient:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API key is missing.")
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.chat_model = genai.GenerativeModel(model_name)

    def embed_text(self, text: str) -> list[float]:
        result = genai.embed_content(model="models/embedding-001", content=text)
        values = self._extract_embedding_values(result)
        if not values:
            raise ValueError("Gemini embedding returned no usable vector values.")
        return values

    def answer_with_context(self, question: str, context: str) -> str:
        prompt = (
            "You are a strict document-grounded assistant. "
            "Answer only using the information contained in the provided context. "
            "If the answer is not explicitly supported by the context, say: 'The available PDF chunks do not provide enough evidence to answer this accurately.'\n\n"
            f"Context:\n{context}\n\nQuestion:\n{question}"
        )
        response = self.chat_model.generate_content(prompt)
        return getattr(response, "text", str(response))

    @staticmethod
    def _extract_embedding_values(payload: Any) -> list[float]:
        if isinstance(payload, dict):
            if "embedding" in payload:
                inner = payload["embedding"]
                if isinstance(inner, dict) and "values" in inner:
                    return list(inner["values"])
                if isinstance(inner, list):
                    return [float(x) for x in inner]
            if "values" in payload:
                return [float(x) for x in payload["values"]]
        elif isinstance(payload, list):
            return [float(x) for x in payload]
        return []
