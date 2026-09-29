import os
from typing import List, Dict, Any
from src.config import config

class LLMClient:
    def __init__(self):
        self.gemini_key = config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.openai_key = config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")

    def generate_answer(self, prompt: str, retrieved_contexts: List[Dict[str, Any]]) -> str:
        """
        Generates an answer using Gemini API, OpenAI API, or clean grounded synthesis fallback.
        """
        # Try Gemini API if key looks like a valid key
        if self.gemini_key and len(self.gemini_key) > 20 and not self.gemini_key.startswith("AQ."):
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text
            except Exception as e:
                print(f"Gemini API (google.generativeai) attempt failed: {e}")
            
            try:
                from google import genai
                client = genai.Client(api_key=self.gemini_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                print(f"Gemini API (google.genai) attempt failed: {e}")

        # Try OpenAI API if key exists
        if self.openai_key and len(self.openai_key) > 20:
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
                if response and response.choices:
                    return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API call failed: {e}")

        # Fallback to high-quality local Grounded Answer Synthesizer
        return self._generate_grounded_synthesis(retrieved_contexts)

    def _generate_grounded_synthesis(self, retrieved_contexts: List[Dict[str, Any]]) -> str:
        if not retrieved_contexts:
            return "I don't know based on the provided documents."

        processed_paragraphs = []
        citations_list = []

        for idx, item in enumerate(retrieved_contexts, start=1):
            doc = item.get("document", {})
            meta = doc.get("metadata", {})
            text = doc.get("text", "").strip()
            source = meta.get("source", "Unknown Document")
            page = meta.get("page", "?")

            citations_list.append(f"[{idx}] {source} (Page {page})")
            
            # Format clean paragraph with full text (no snippet truncation!)
            paragraph = (
                f"### Section from [{source}, Page {page}]:\n"
                f"{text}"
            )
            processed_paragraphs.append(paragraph)

        body_text = "\n\n".join(processed_paragraphs)
        citations_text = " ".join([f"[{i+1}] {item['document']['metadata']['source']} (Page {item['document']['metadata']['page']})" for i, item in enumerate(retrieved_contexts)])

        answer_markdown = (
            f"### Document Findings & Analysis\n\n"
            f"{body_text}\n\n"
            f"---\n"
            f"**Source Citations:** {citations_text}"
        )

        return answer_markdown
