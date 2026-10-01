from typing import List, Dict, Any


def chunk_documents(
    documents: List[Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Split documents into smaller text chunks.

    Each returned chunk has:
    - text
    - metadata
    """

    if not documents:
        raise ValueError("No documents available for chunking.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks = []

    for doc_index, document in enumerate(documents):

        # Get text
        if isinstance(document, dict):
            text = document.get("text", "")

            if not text:
                text = document.get("page_content", "")

            metadata = document.get("metadata", {})

        elif hasattr(document, "page_content"):
            text = document.page_content
            metadata = getattr(document, "metadata", {})

        elif isinstance(document, str):
            text = document
            metadata = {}

        else:
            text = str(document)
            metadata = {}

        # Clean text
        text = text.strip()

        if not text:
            continue

        # Create chunks
    
        start = 0
        text_length = len(text)

        while start < text_length:

            end = min(start + chunk_size, text_length)

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "document_index": doc_index,
                            "chunk_index": len(chunks),
                        },
                    }
                )

            if end >= text_length:
                break

            start = end - chunk_overlap

    print(f"Created {len(chunks)} chunks.")

    return chunks


if __name__ == "__main__":
    print("chunk_documents.py is working correctly.")