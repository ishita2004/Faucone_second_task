import numpy as np
import pickle
from pathlib import Path


def get_output_dir():
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def save_embeddings(chunks, embeddings):
    output_dir = get_output_dir()

    np.save(
        output_dir / "embeddings.npy",
        embeddings
    )

    with open(output_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Embeddings saved successfully!")
    print("Chunks saved successfully!")


def load_embeddings():
    output_dir = get_output_dir()

    embeddings_file = output_dir / "embeddings.npy"
    chunks_file = output_dir / "chunks.pkl"

    if not embeddings_file.exists():
        raise FileNotFoundError(
            "embeddings.npy not found. Run run_pipeline.py first."
        )

    if not chunks_file.exists():
        raise FileNotFoundError(
            "chunks.pkl not found. Run run_pipeline.py first."
        )

    embeddings = np.load(embeddings_file)

    with open(chunks_file, "rb") as f:
        chunks = pickle.load(f)

    return chunks, embeddings