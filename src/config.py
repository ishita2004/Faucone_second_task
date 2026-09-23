import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_PDFS_DIR: str = os.getenv("RAW_PDFS_DIR", os.path.join(PROJECT_ROOT, "pdfs"))
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", os.path.join(PROJECT_ROOT, "data", "vector_store"))
    
    # Chunking options
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Embedding options
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    
    FAISS_INDEX_NAME: str = "index.faiss"
    METADATA_NAME: str = "metadata.pkl"

config = Config()
