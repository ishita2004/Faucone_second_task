import time
from src.config import config
from src.pdf_loader import PDFLoader
from src.text_splitter import RecursiveTextSplitter
from src.embeddings import Embedder
from src.vector_store import FAISSVectorStore

def run_ingestion_pipeline():
    start_time = time.time()
    print("=== Starting RAG Document Ingestion Pipeline ===")
    print(f"PDF Directory: {config.RAW_PDFS_DIR}")
    print(f"Output Directory: {config.VECTOR_STORE_DIR}")

    loader = PDFLoader(pdf_dir=config.RAW_PDFS_DIR)
    page_docs = loader.load_documents()

    if not page_docs:
        print("No documents found to ingest. Pipeline aborted.")
        return

    splitter = RecursiveTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunked_docs = splitter.split_documents(page_docs)

    embedder = Embedder(model_name=config.EMBEDDING_MODEL_NAME)
    texts = [doc["text"] for doc in chunked_docs]
    embeddings = embedder.embed_texts(texts)

    vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
    vector_store.add_documents(chunked_docs, embeddings)
    vector_store.save(
        output_dir=config.VECTOR_STORE_DIR,
        index_name=config.FAISS_INDEX_NAME,
        metadata_name=config.METADATA_NAME
    )

    elapsed = time.time() - start_time
    print(f"\n[OK] Ingestion complete in {elapsed:.2f} seconds!")
    print(f"Vector Store saved to: {config.VECTOR_STORE_DIR}")

if __name__ == "__main__":
    run_ingestion_pipeline()
