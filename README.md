# Document Q&A Assistant (RAG System)

A retrieval-augmented generation (RAG) system built to answer questions from a set of PDF documents with source citations (document name and page number).

---

## 🛠️ RAG System Architecture

```
PDF Documents (pdfs/)
       │
       ▼
[1. PDF Loader (pypdf)] ──> Extracts page text + metadata (source, page_num)
       │
       ▼
[2. Recursive Text Splitter] ──> Chunks text (500 chars, 50 overlap) with metadata
       │
       ▼
[3. Embedder (all-MiniLM-L6-v2)] ──> Computes 384-d dense embeddings
       │
       ▼
[4. FAISS Vector DB] ──> Stores vectors & metadata payload (data/vector_store/)
       │
       ▼
[5. Top-K Retriever] ──> Fetches top-k relevant chunks for user query
       │
       ▼
[6. LLM Generator & Chain] ──> Generates grounded answers with [Doc, Page X] citations
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ingest PDFs & Build FAISS Vector Database
```bash
python -m src.ingest
```

### 3. Run Interactive Streamlit Web Application
```bash
streamlit run app.py
```

### 4. Run CLI Interface
```bash
python main.py
```

### 5. Run Pytest Suite
```bash
pytest
```
