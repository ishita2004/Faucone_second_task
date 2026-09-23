import os
from typing import List, Dict, Any
from pypdf import PdfReader

class PDFLoader:
    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir

    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Reads all PDFs from pdf_dir and extracts page-by-page text along with metadata.
        """
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
                    extracted_text = extracted_text.strip()
                    if extracted_text:
                        documents.append({
                            "text": extracted_text,
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
