import sys
from pathlib import Path

# Project root ko Python path mein add karo
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project modules
from src.load_documents import load_documents
from src.chunk_documents import chunk_documents
from src.embeddings import create_embeddings
from src.vector_store import save_embeddings


def run_pipeline():
    print("=" * 60)
    print("              DOCUMENT RAG PIPELINE")
    print("=" * 60)

    # STEP 1: Load Documents
    print("\n[1/4] Loading documents...")

    try:
        documents = load_documents()

        if not documents:
            print("No documents found.")
            print("Please check the 'pdfs' folder.")
            return

        print(f"Documents loaded successfully: {len(documents)}")

    except Exception as e:
        print(f"Error while loading documents: {e}")
        return

    # STEP 2: Chunk Documents
    print("\n[2/4] Creating document chunks...")

    try:
        chunks = chunk_documents(documents)

        if not chunks:
            print("No chunks were created.")
            return

        print(f"Chunks created successfully: {len(chunks)}")

    except Exception as e:
        print(f"Error while creating chunks: {e}")
        return

    # STEP 3: Create Embeddings
    print("\n[3/4] Creating embeddings...")
    print("This may take some time depending on the number of chunks.")

    try:
        embeddings = create_embeddings(chunks)

        if embeddings is None:
            print("Embedding generation failed.")
            return

        print("Embeddings created successfully.")
        print(f"Embedding shape: {embeddings.shape}")

    except Exception as e:
        print(f"Error while creating embeddings: {e}")
        return

    # STEP 4: Save Embeddings + Chunks
    print("\n[4/4] Saving embeddings and chunks...")

    try:
        save_embeddings(chunks, embeddings)

        print("\nFiles saved successfully.")

    except Exception as e:
        print(f"Error while saving embeddings: {e}")
        return

    # FINAL SUMMARY
    print("\n" + "=" * 60)
    print("             PIPELINE COMPLETED")
    print("=" * 60)

    print(f"Documents : {len(documents)}")
    print(f"Chunks    : {len(chunks)}")
    print(f"Embeddings: {embeddings.shape}")

    output_dir = PROJECT_ROOT / "output"

    print(f"\nOutput folder: {output_dir}")
    print("Generated files:")

    print("  - embeddings.npy")
    print("  - chunks.pkl")

    print("\nRAG data preparation completed successfully!")


if __name__ == "__main__":
    run_pipeline()