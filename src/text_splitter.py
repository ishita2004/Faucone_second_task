from typing import List, Dict, Any

class RecursiveTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separator = self.separators[-1]
        for s in self.separators:
            if s in text:
                separator = s
                break

        splits = text.split(separator) if separator else list(text)
        chunks = []
        current_chunk = ""

        for split in splits:
            piece = split + (separator if separator != "" else "")
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + piece

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunked_docs = []
        chunk_counter = 0

        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]
            raw_chunks = self._split_text(text)

            for idx, chunk_text in enumerate(raw_chunks):
                chunk_counter += 1
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = chunk_counter
                chunk_metadata["chunk_index_in_page"] = idx

                chunked_docs.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })

        print(f"Created {len(chunked_docs)} chunks from {len(documents)} pages (Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}).")
        return chunked_docs
