import os
from typing import List, Dict, Any
from src.config import config

class LLMClient:
    def __init__(self):
        self.gemini_key = config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.openai_key = config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        self.provider = self._detect_provider()

    def _detect_provider(self) -> str:
        if self.gemini_key:
            return "gemini"
        elif self.openai_key:
            return "openai"
        return "offline_grounded"

    def generate_answer(self, prompt: str, retrieved_contexts: List[Dict[str, Any]]) -> str:
        """
        Generates an answer based on prompt and retrieved contexts.
        """
        if self.provider == "gemini":
            try:
                from google import genai
                client = genai.Client(api_key=self.gemini_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                return response.text
            except Exception as e:
                print(f"Gemini API call failed: {e}. Falling back to grounded synthesis.")

        elif self.provider == "openai":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert Q&A assistant. Answer strictly based on the provided document contexts and cite sources."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API call failed: {e}. Falling back to grounded synthesis.")

        # Offline Grounded Synthesis
        return self._generate_offline_synthesis(retrieved_contexts)

    def _generate_offline_synthesis(self, retrieved_contexts: List[Dict[str, Any]]) -> str:
        if not retrieved_contexts:
            return "I don't know based on the provided documents."

        citations = []
        snippets = []

        for idx, item in enumerate(retrieved_contexts, start=1):
            doc = item.get("document", {})
            meta = doc.get("metadata", {})
            text = doc.get("text", "").strip()
            source = meta.get("source", "Unknown Document")
            page = meta.get("page", "?")

            citations.append(f"[{idx}] {source} (Page {page})")
            snippets.append(f"• From [{source}, Page {page}]: \"{text[:300]}...\"")

        summary_text = "\n\n".join(snippets)
        citation_str = "\n".join(citations)

        return (
            f"Based on the retrieved document contexts, here is the relevant information:\n\n"
            f"{summary_text}\n\n"
            f"### Source Citations:\n{citation_str}"
        )
