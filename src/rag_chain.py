from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from src.llm import GeminiClient


def list_pdf_files(pdf_dir: str | Path) -> list[Path]:
    folder = Path(pdf_dir)
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"],
        key=lambda item: item.name.lower(),
    )


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"[Page {page_no}]\n{text.strip()}")
    return "\n\n".join(parts)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 180) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            boundary = max(
                cleaned.rfind(".", start, end),
                cleaned.rfind("\n", start, end),
                cleaned.rfind(" ", start, end),
            )
            if boundary > start + int(chunk_size * 0.6):
                end = boundary + 1
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(start + chunk_size - overlap, end - overlap)
    return chunks or [cleaned]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product / (mag_a * mag_b)


def build_rag_index(pdf_dir: str | Path, api_key: str) -> list[dict[str, Any]]:
    pdf_dir = Path(pdf_dir)
    client = GeminiClient(api_key)
    index: list[dict[str, Any]] = []

    for pdf_path in list_pdf_files(pdf_dir):
        extracted_text = extract_pdf_text(pdf_path)
        for chunk in chunk_text(extracted_text):
            page_match = re.search(r"\[Page\s+(\d+)\]", chunk)
            page_number = int(page_match.group(1)) if page_match else None
            embedding = client.embed_text(chunk)
            index.append({
                "pdf_name": pdf_path.name,
                "page": page_number,
                "chunk": chunk,
                "embedding": embedding,
            })
    return index


def retrieve_relevant_chunks(pdf_dir: str | Path, api_key: str, question: str, top_k: int = 5) -> list[dict[str, Any]]:
    if not question.strip():
        raise ValueError("Please enter a question before asking the RAG system.")

    index = build_rag_index(pdf_dir, api_key)
    if not index:
        raise ValueError("No readable PDF files were found in the selected folder.")

    question_embedding = GeminiClient(api_key).embed_text(question)
    ranked = sorted(
        index,
        key=lambda entry: cosine_similarity(entry["embedding"], question_embedding),
        reverse=True,
    )[:top_k]
    return ranked


def format_citations(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return "No source citations available."

    lines: list[str] = []
    for idx, item in enumerate(citations, start=1):
        page = item.get("page")
        source_name = item.get("pdf_name", "Unknown.pdf")
        text = (item.get("text") or "").strip().replace("\n", " ")
        if page is not None:
            lines.append(f"[{idx}] {source_name} (page {page}) — {text}")
        else:
            lines.append(f"[{idx}] {source_name} — {text}")
    return "\n".join(lines)


def answer_question(pdf_dir: str | Path, api_key: str, question: str, top_k: int = 5) -> tuple[str, list[dict[str, Any]]]:
    relevant_chunks = retrieve_relevant_chunks(pdf_dir, api_key, question, top_k=top_k)
    if not relevant_chunks:
        raise ValueError("The PDF folder does not contain enough indexed content to answer this question.")

    context = "\n\n---\n\n".join(
        f"Source: {chunk['pdf_name']}\n{chunk['chunk']}" for chunk in relevant_chunks
    )

    client = GeminiClient(api_key)
    answer = client.answer_with_context(question, context)

    citations: list[dict[str, Any]] = []
    for item in relevant_chunks:
        clean_chunk = re.sub(r"\[Page\s+\d+\]\s*", "", item["chunk"]).strip()
        citations.append({
            "pdf_name": item["pdf_name"],
            "page": item.get("page"),
            "text": clean_chunk,
        })

    return answer, citations
