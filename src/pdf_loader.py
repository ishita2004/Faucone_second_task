import os
import re
from typing import List, Dict, Any
from pypdf import PdfReader

class PDFLoader:
    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir

    def _clean_text(self, text: str) -> str:
        """Fixes hyphenated line breaks and normalizes whitespace."""
        # Replace word-hyphen-newline-space (e.g., "organi- zational" -> "organizational")
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        # Replace word-hyphen-space (e.g., "mag- nitude" -> "magnitude")
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        # Replace multiple spaces/newlines with single spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def load_documents(self) -> List[Dict[str, Any]]:
        documents = []
        if not os.path.exists(self.pdf_dir):
            raise FileNotFoundError(f"PDF directory not found: {self.pdf_dir}")

        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF file(s) in '{self.pdf_dir}'.")

        for pdf_file in pdf_files:
            file_path = os.path.join(self.pdf_dir, pdf_file)
            try:
                reader = PdfReader(file_path)
                num_pages = len(reader.pages)
                for page_num, page in enumerate(reader.pages, start=1):
                    extracted_text = page.extract_text() or ""
                    cleaned_text = self._clean_text(extracted_text)
                    if cleaned_text:
                        documents.append({
                            "text": cleaned_text,
                            "metadata": {
                                "source": pdf_file,
                                "page": page_num,
                                "total_pages": num_pages
                            }
                        })
            except Exception as e:
                print(f"Warning: Failed to parse '{pdf_file}': {e}")

        print(f"Extracted {len(documents)} page documents in total.")
        return documents
