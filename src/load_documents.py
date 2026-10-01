from pathlib import Path
from pypdf import PdfReader


def load_documents():
    pdf_folder = Path(__file__).parent.parent / "pdfs"

    documents = []
    total_pages = 0

    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in pdfs folder.")
        return documents

    for file in pdf_files:
        try:
            reader = PdfReader(str(file))

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text()

                if text and text.strip():
                    documents.append({
                        "text": text.strip(),
                        "source": file.name,
                        "page": page_number
                    })

            total_pages += len(reader.pages)

        except Exception as e:
            print(f"Error reading {file.name}: {e}")

    print(f"PDF files found: {len(pdf_files)}")
    print(f"Total pages: {total_pages}")
    print(f"Pages with text loaded: {len(documents)}")

    return documents


if __name__ == "__main__":
    docs = load_documents()

    if docs:
        print("\nFirst document:")
        print("Source:", docs[0]["source"])
        print("Page:", docs[0]["page"])
        print("Text preview:", docs[0]["text"][:300])