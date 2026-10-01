from sentence_transformers import SentenceTransformer


# Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"


def create_embeddings(chunks):
    """
    Convert document chunks into numerical embeddings.
    """

    if not chunks:
        raise ValueError("No chunks available for embedding.")

    # Load the embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Extract text from chunks
    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


if __name__ == "__main__":

    # Import project modules
    from .load_documents import load_documents
    from .chunk_documents import chunk_documents
    from .vector_store import save_embeddings

    print("Loading documents...")

    # Load PDF documents
    documents = load_documents()

    print("Creating document chunks...")

    # Split documents into chunks
    chunks = chunk_documents(documents)

    print("Creating embeddings...")

    # Generate embeddings
    embeddings = create_embeddings(chunks)

    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")

    # Save embeddings and chunks
    save_embeddings(chunks, embeddings)

    print("Embedding pipeline completed successfully!")