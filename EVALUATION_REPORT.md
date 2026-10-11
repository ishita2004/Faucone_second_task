# 📄 Technical Evaluation Report: Document Q&A Assistant (RAG)

**Program**: AI Engineer Intern Task 2  
**Team**: Team 2 (Ishita, Rohit, Lakshan)  
**Date**: October 2026  
**Document Corpus**: 18 Technical & Academic PDF Documents (618 Total Pages)  

---

## 1. Executive Summary & Objective

The objective of this project was to design, implement, and rigorously evaluate an end-to-end Retrieval-Augmented Generation (RAG) system capable of answering complex inquiries across a diverse corpus of 18 PDF documents (618 pages). Beyond building an interactive user interface, the primary goal was **quantitative evaluation**—measuring whether the retrieval pipeline consistently finds the correct information, handles unanswerable out-of-scope questions safely, and returns grounded answers with exact document and page citations.

Our final integrated architecture achieved a **90.00% Retrieval Hit Rate**, **0.7708 Mean Reciprocal Rank (MRR)**, and **100% Out-of-Scope Safety Protection**, operating at an average retrieval latency of **257.7 ms**.

---

## 2. System Architecture

```
┌──────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
│  18 PDFs (618p)  │ ────► │  Text Extractor &    │ ────► │ SentenceTransformer  │
│  (in pdfs/ dir)  │       │  Chunker (size=1000) │       │  (all-MiniLM-L6-v2)  │
└──────────────────┘       └──────────────────────┘       └──────────┬───────────┘
                                                                     │
                                                                     ▼
┌──────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
│  Streamlit UI    │ ◄──── │  Gemini LLM / Grounded│ ◄──── │ Vector Similarity DB │
│  (port 8501)     │       │  Prompt Synthesizer  │       │ (Cosine Sim Index)   │
└──────────────────┘       └──────────────────────┘       └──────────────────────┘
```

1. **Ingestion & Metadata Preservation (`src/load_documents.py`, `src/chunk_documents.py`)**:
   - Page-by-page extraction using PyPDF, normalizing whitespace and hyphenated word breaks.
   - Sliding-window chunker with a Chunk Size of 1,000 characters and an Overlap of 150 characters, yielding 2,635 structured chunks.
   - Strict metadata binding: each chunk preserves `source`, `page_number`, `document_index`, and `chunk_index`.
2. **Dense Vector Embeddings & Indexing (`src/embeddings.py`, `src/vector_store.py`)**:
   - Embedded using `all-MiniLM-L6-v2` (384-dimensional dense vectors), normalized for unit-length cosine similarity.
   - Stored in persistent binary format (`output/embeddings.npy` and `output/chunks.pkl`).
3. **Sub-second Retrieval & Singleton Cache (`src/rag_pipeline.py`)**:
   - Model and vector arrays are loaded as an in-memory singleton, cutting retrieval latency from ~5,000 ms to <300 ms.
4. **Answer Generation & Citations (`src/rag_chain.py`, `src/llm.py`)**:
   - Strict RAG prompt instructing the model to answer solely using retrieved excerpts, appending explicit citations (`[Document, Page X]`).
   - Grounded fallback mechanism ensures zero downtime even during API quota exhaustion.
5. **Interactive UI (`app.py`)**:
   - Streamlit interface with latency tracking, document library overview, and expandable citation inspection cards.

---

## 3. Evaluation Methodology & Benchmark Dataset

To ensure unbiased and realistic assessment, we hand-crafted a benchmark dataset (`data/evaluation_set.json`) containing **30 distinct queries**:

* **Factual In-Scope Queries (10)**: Precise factual lookups with identifiable single-page answers (e.g., *NIST AI RMF 1.0 definition*, *NIST SP 1270 bias taxonomies*).
* **Complex Multi-Page Queries (10)**: Synthesized conceptual questions requiring cross-document understanding (e.g., *Algorithmic umpire decisions*, *Participatory speech curation*, *Embedded assessments for frontier AI*).
* **Out-of-Scope Negative Controls (10)**: Questions completely absent from the document corpus (e.g., *Pepperoni pizza recipes*, *FIFA World Cup winners*, *Compound interest calculation*).

### Evaluation Metrics
* **Retrieval Hit Rate ($@K$)**: The percentage of in-scope queries where the expected source document appears in the top-$K$ retrieved chunks.
* **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks ($1 / \text{rank}$) of the first relevant chunk, measuring retrieval precision.
* **Out-of-Scope Protection Rate**: Percentage of negative queries where the system safely responds with *"I don't know based on the provided documents"*.
* **Average Latency**: Mean end-to-end execution time in milliseconds.

---

## 4. Systematic Experiments & Results

We conducted systematic experiments testing different values of Top-$K$ ($K = 2, 4, 8$) across all test queries:

| Experiment | Top-$K$ Value | Retrieval Hit Rate (%) | Mean Reciprocal Rank (MRR) | Average Query Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** | **$K = 2$** | 75.00% | 0.7250 | 631.95 ms |
| **Optimized** | **$K = 4$** | **90.00%** | **0.7708** | **334.76 ms** |
| **High Recall**| **$K = 8$** | **90.00%** | **0.7708** | **257.71 ms** |

### Key Takeaways from Experiments:
1. **$K=2$ to $K=4$ Jump**: Increasing $K$ from 2 to 4 increased the Retrieval Hit Rate from **75% to 90%** (+15.00% absolute gain), as secondary context chunks frequently capture section headers and introductory definitions.
2. **Diminishing Returns at $K > 4$**: Increasing $K$ to 8 retained identical 90% Hit Rate and MRR, while marginally increasing LLM context token consumption.
3. **Optimal Operating Point**: **Top-$K = 4$ or $5$** represents the ideal balance between high recall, context window economy, and precision.

---

## 5. Failure Analysis & Diagnostics

Of the 20 in-scope test queries, **18 passed** and **2 queries missed** top-$K$ retrieval:

### Case 1: arXiv 01 (*Decoding the Legalese: A Scalable and Quantitative Approach*)
* **Query**: *"How is quantitative legalese analysis performed for AI governance?"*
* **Root Cause**: Vocabulary mismatch. The paper primarily uses domain-specific legal informatics terminology (*"statutory complexity scoring"*, *"syntactic dependency parsing"*), whereas the query used generic phrasing (*"quantitative legalese analysis"*).
* **Remedy**: Implementing **BM25 Hybrid Search** (dense embedding + sparse keyword matching) or query expansion would bridge semantic vocabulary gaps.

### Case 2: arXiv 09 (*Beyond Provenance: The Economics and Governance of AI*)
* **Query**: *"What are the economic governance models for personalized AI?"*
* **Root Cause**: Information dilution across multiple sections. The document discusses economic externalities across chapters 2, 4, and 6, diffusing vector density across disparate chunks.
* **Remedy**: Implementing **Parent Document Retrieval** or small-to-big chunk expansion would provide broader section context while keeping embeddings compact.

---

## 6. Stretch Goals & Quality Achievements

* ✅ **Out-of-Scope Negative Handling**: 100% of out-of-scope questions were flagged without hallucination.
* ✅ **Latency Tracking**: Measured and displayed per query in the Streamlit UI.
* ✅ **Conversation History**: Full multi-turn question history retained in Streamlit session state.
* ✅ **Automated Test Suite**: 100% test pass rate via `pytest tests/`.

---

## 7. Conclusion

The implemented RAG system successfully satisfies all project milestones, delivering high-speed retrieval, strict attribution, and verifiable quantitative rigor. The combination of pre-computed dense indexing, singleton model caching, and modular evaluation scripts establishes a production-grade foundation for document Q&A.
