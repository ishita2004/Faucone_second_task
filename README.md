# 📚 Document Q&A Assistant (RAG) with Quantitative Evaluation Benchmark

An end-to-end Retrieval-Augmented Generation (RAG) system built to answer questions across a collection of **18 academic and technical PDF documents (618 pages)** with grounded reasoning, precise source citations `[Document, Page]`, and an empirical **30-question evaluation benchmark**.

---

## 👥 Team Structure & Roles

This project was developed by **Team 2** as part of AI Engineer Intern Task 2:
* **Rohit**: Ingestion & Vector Retrieval Pipeline (`src/load_documents.py`, `src/chunk_documents.py`, `src/embeddings.py`, `src/vector_store.py`).
* **Lakshan**: LLM Answer Generation & Web Interface (`src/llm.py`, `src/rag_chain.py`, `app.py`).
* **Ishita**: Benchmark Evaluation, Experimentation & Reporting (`data/evaluation_set.json`, `scripts/evaluate.py`, `scripts/run_experiments.py`, `EVALUATION_REPORT.md`).

---

## 🚀 Key Features

* **Sub-second Dense Vector Search**: Pre-computed 384-dimensional dense vectors using `SentenceTransformer("all-MiniLM-L6-v2")` cached as a singleton for ~250ms query responses.
* **Page-Level Grounded Citations**: Every answer provides verifiable citations with source document name, page number, match score, and chunk excerpt.
* **Anti-Hallucination Safe Guards**: Out-of-scope questions that cannot be answered from the PDFs are safely rejected with *"I don't know based on the provided documents"*.
* **Interactive Web Interface**: Streamlit UI with real-time latency tracking, PDF library overview, and expandable citation inspectors.
* **30-Question Hand-Crafted Benchmark**: Realistic evaluation set covering factual lookups, complex multi-page synthesis, and negative controls.
* **Systematic Experimentation**: Automated comparison of Top-$K$ retrieval depths ($K = 2, 4, 8$) with recorded Hit Rate, MRR, and Latency metrics.

---

## 📊 Benchmark & Experiment Results

### Evaluation Benchmark Summary (`scripts/evaluate.py`):
```text
======================================================================
 [RESULTS] FINAL EVALUATION RESULTS
======================================================================
 Total In-Scope Queries    : 20
 Successful Retrieval Hits : 18
 Retrieval Hit Rate        : 90.00%
 Out-of-Scope Queries      : 10
 Out-of-Scope Protection   : 100.00% (Safely rejected)
======================================================================
```

### Top-K Experiment Comparison Table (`scripts/run_experiments.py`):
| Top-$K$ | Hit Rate (%) | Mean Reciprocal Rank (MRR) | Avg Latency (ms) |
| :---: | :---: | :---: | :---: |
| **$K = 2$** | 75.00% | 0.7250 | 631.95 ms |
| **$K = 4$** | **90.00%** | **0.7708** | **334.76 ms** |
| **$K = 8$** | **90.00%** | **0.7708** | **257.71 ms** |

*See full analysis in [EVALUATION_REPORT.md](EVALUATION_REPORT.md).*

---

## 📁 Repository Structure

```text
├── pdfs/                     # 18 Technical & Academic PDF documents (618 pages)
├── output/                   # Pre-computed vector store (embeddings.npy, chunks.pkl)
├── data/
│   └── evaluation_set.json   # 30 Hand-crafted Q&A evaluation benchmark pairs
├── src/
│   ├── load_documents.py     # PDF text extraction and whitespace normalization
│   ├── chunk_documents.py    # Recursive chunking preserving source & page metadata
│   ├── embeddings.py         # SentenceTransformer all-MiniLM-L6-v2 embedding model
│   ├── vector_store.py       # Vector saving & loading utilities
│   ├── rag_pipeline.py       # High-speed cached cosine similarity retriever
│   ├── llm.py                # Google Gemini API client
│   └── rag_chain.py          # Unified RAG chain orchestrator with latency tracking
├── scripts/
│   ├── evaluate.py           # Automated 30-question benchmark evaluation script
│   └── run_experiments.py    # Top-K comparison experiment runner
├── tests/
│   └── test_rag_chain.py     # Pytest unit test suite
├── app.py                    # Streamlit Web User Interface
├── EVALUATION_REPORT.md      # 2-Page technical evaluation report & failure analysis
├── PRESENTATION.md           # 10-Minute presentation slides outline
└── requirements.txt          # Project dependencies
```

---

## ⚡ Quick Start Guide

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/ishita2004/Faucone_second_task.git
cd Faucone_second_task

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Run Ingestion Pipeline
*The vector store in `output/` is already pre-computed. To rebuild it from raw PDFs:*
```bash
python src/run_pipeline.py
```

### 3. Run the Streamlit Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🧪 Running Benchmarks & Tests

### Run 30-Question Evaluation Benchmark
```bash
python scripts/evaluate.py
```

### Run Top-K Experiments
```bash
python scripts/run_experiments.py
```

### Run Unit Tests
```bash
python -m pytest tests/
```

---

## 🌟 Stretch Goals Implemented

- ✅ **Out-of-Scope Negative Handling**: The system detects unanswerable queries and responds with *"I don't know based on the provided documents"*.
- ✅ **Per-Query Latency Tracking**: Real-time display of retrieval and generation latency.
- ✅ **Conversation History**: Retains multi-turn questions and citations throughout user sessions.
- ✅ **Grounded Fallback Engine**: Enables seamless offline usage even without an active API key.
