from __future__ import annotations

import os
from typing import Any

try:
    from google import genai
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("google-genai is required for Gemini-based RAG.") from exc


class GeminiClient:
    def __init__(self, api_key: str, model_name: str = "gemini-3.5-flash-lite"):
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API key is missing.")
        self.api_key = api_key.strip()
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)

    def answer_with_context(self, question: str, context: str) -> str:
        prompt = (
            "You are a strict document-grounded assistant. "
            "Answer clearly and concisely using ONLY the information contained in the provided context below. "
            "Cite the specific document and page numbers mentioned in the context. "
            "If the answer is not explicitly supported by the context, respond with: "
            "'I don't know based on the provided documents.'\n\n"
            f"Context:\n{context}\n\nQuestion:\n{question}"
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return getattr(response, "text", str(response))
