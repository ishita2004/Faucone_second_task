# Document Q&A Assistant (RAG) - Task 2 (Days 1–3 Baseline)

A retrieval-augmented generation (RAG) system built to ingest PDF document sets, perform page-aware text chunking, compute dense embeddings, and store them in a persistent FAISS vector index with source citations.

---

## 📁 Repository Structure

```
.
├── pdfs/                   # 18 NIST & ArXiv PDF documents
├── src/
│   ├── __init__.py
│   ├── config.py           # Project settings
│   ├── pdf_loader.py       # Page-level PDF text extraction
│   ├── text_splitter.py    # Recursive chunker preserving metadata
│   ├── embeddings.py       # SentenceTransformers embedding generator
│   ├── vector_store.py     # FAISS vector store manager
│   └── ingest.py           # Ingestion pipeline entry point
├── scripts/
│   └── verify_retrieval.py # Similarity search & citation testing
├── tests/
│   └── test_ingestion.py   # Unit tests
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Ingestion Pipeline
```bash
python -m src.ingest
```

### 3. Verify Retrieval & Citations
```bash
python scripts/verify_retrieval.py
```

### 4. Run Unit Tests
```bash
pytest
```
