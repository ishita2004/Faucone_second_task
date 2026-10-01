import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npy"
CHUNKS_FILE = OUTPUT_DIR / "chunks.pkl"

# Model

MODEL_NAME = "all-MiniLM-L6-v2"

def load_rag_data():
    """
    Load saved embeddings and document chunks.
    """

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_FILE}"
        )

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    embeddings = np.load(EMBEDDINGS_FILE)

    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    return embeddings, chunks

# Create query embedding

def create_query_embedding(query, model):
    """
    Convert user query into an embedding.
    """

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return query_embedding

# Retrieve relevant chunks

def retrieve_documents(query, top_k=5):
    """
    Retrieve the most relevant document chunks
    for the given query.
    """

    # Load saved data
    embeddings, chunks = load_rag_data()

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Create query embedding
    query_embedding = create_query_embedding(query, model)

    # Calculate cosine similarity

    scores = np.dot(embeddings, query_embedding)

    # Get top K results
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        chunk = chunks[index]

        result = {
            "text": chunk.get("text", ""),
            "score": float(scores[index]),
            "metadata": {
                key: value
                for key, value in chunk.items()
                if key != "text"
            }
        }

        results.append(result)

    return results

# Display search results

def print_results(query, results):
    """
    Print retrieved documents in a readable format.
    """

    print("\n" + "=" * 70)
    print("RAG SEARCH RESULTS")
    print("=" * 70)

    print(f"\nQuery: {query}")

    if not results:
        print("\nNo relevant documents found.")
        return

    for i, result in enumerate(results, start=1):

        print("\n" + "-" * 70)
        print(f"Result {i}")
        print(f"Similarity Score: {result['score']:.4f}")

        if result["metadata"]:
            print(f"Metadata: {result['metadata']}")

        print("\nText:")
        print(result["text"])

# Interactive search

def run_rag_search():
    """
    Run interactive RAG retrieval.
    """

    print("=" * 70)
    print("                 RAG DOCUMENT SEARCH")
    print("=" * 70)

    print("\nLoading RAG data...")

    try:
        embeddings, chunks = load_rag_data()

        print(f"Embeddings loaded : {embeddings.shape}")
        print(f"Chunks loaded     : {len(chunks)}")

    except Exception as e:
        print(f"\nError loading RAG data: {e}")
        return

    print("\nLoading embedding model...")

    try:
        # Load once for initial validation
        SentenceTransformer(MODEL_NAME)
        print("Embedding model loaded successfully.")

    except Exception as e:
        print(f"\nError loading embedding model: {e}")
        return

    print("\nRAG search is ready.")
    print("Type your question.")
    print("Type 'exit' to stop.\n")

    while True:

        query = input("Enter your question: ").strip()

        if query.lower() == "exit":
            print("\nRAG search stopped.")
            break

        if not query:
            print("Please enter a question.")
            continue

        try:

            results = retrieve_documents(
                query=query,
                top_k=5
            )

            print_results(query, results)

        except Exception as e:

            print(f"\nError during retrieval: {e}")


# Main

if __name__ == "__main__":
    run_rag_search()