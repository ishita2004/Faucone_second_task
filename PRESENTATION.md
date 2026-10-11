# 🎯 10-Minute Presentation Slide Deck: Document Q&A Assistant (RAG)

**Team**: Team 2 (Ishita, Rohit, Lakshan)  
**Task**: AI Engineer Intern Task 2 — Build & Evaluate a Document Q&A Assistant  
**Presentation Time**: 10 Minutes + 5 Minutes Q&A  

---

## 📽️ Slide 1: Title & Team Overview (0:00 - 1:00)
* **Title**: Document Q&A Assistant with Quantitative RAG Evaluation
* **Objective**: Moving beyond "getting a demo running" to scientifically measuring and optimizing RAG retrieval quality.
* **Team Structure & Contributions**:
  - **Rohit**: Ingestion pipeline, recursive chunking, SentenceTransformer vector embeddings.
  - **Lakshan**: Streamlit Web UI, Gemini LLM API integration, citation card formatting.
  - **Ishita**: 30 Q&A benchmark curation, evaluation script, Top-K experimentation, failure diagnostics.

---

## 📽️ Slide 2: The Document Corpus (1:00 - 2:00)
* **Dataset Scope**: 18 Technical & Academic Research PDFs (618 Total Pages).
* **Subject Matter**: NIST AI Risk Management Framework 1.0, AI Bias (NIST SP 1270), frontier AI safety, algorithmic decision making, social simulation.
* **Ingestion Highlights**:
  - 618 pages loaded cleanly with whitespace normalization and line-break de-hyphenation.
  - 2,635 structured text chunks (size: 1,000 chars, overlap: 150 chars).
  - Preserved metadata: `source`, `page_number`, `document_index`, `chunk_index`.

---

## 📽️ Slide 3: Architecture & Engineering Decisions (2:00 - 3:30)
* **Diagram**: Document Ingestion ➔ Dense Indexing (`output/embeddings.npy`) ➔ Singleton Vector Retriever ➔ LLM / Grounded Synthesis ➔ Streamlit UI.
* **Key Innovation**: Model & Vector Singleton Caching:
  - Eliminated repetitive re-loading of the 384-dimensional model on every question.
  - Slashed query latency from **5.2s down to ~250ms**.

---

## 📽️ Slide 4: Evaluation Rigor (30-Question Benchmark) (3:30 - 5:00)
* **Why hand-crafted?** Automated synthetic benchmarks often suffer from data leakage. Hand-crafted questions test genuine human search behaviors.
* **3 Categories**:
  - **10 Factual Queries**: Pinpoint definitions and metrics.
  - **10 Complex Multi-Page Queries**: Conceptual reasoning across sections.
  - **10 Out-of-Scope Negative Controls**: Testing anti-hallucination ("I don't know") safeguards.

---

## 📽️ Slide 5: Experimentation Results (Top-K Comparison) (5:00 - 6:30)
* **Hypothesis**: Higher Top-K improves retrieval recall but may increase context noise.
* **Results Table**:
  - **$K = 2$**: 75.00% Hit Rate | MRR: 0.7250 | Latency: 631 ms
  - **$K = 4$**: **90.00% Hit Rate** | **MRR: 0.7708** | Latency: 334 ms
  - **$K = 8$**: **90.00% Hit Rate** | **MRR: 0.7708** | Latency: 257 ms
* **Conclusion**: $K = 4$ is the sweet spot. Moving beyond 4 yielded no additional recall gains while swelling the prompt size.

---

## 📽️ Slide 6: Failure Analysis (Why did 10% miss?) (6:30 - 7:30)
* **Miss 1 (arXiv 01)**: Query used *"quantitative legalese analysis"* while the paper used *"statutory complexity scoring"*.
  - *Fix*: Hybrid BM25 + Dense retrieval to bridge semantic vocabulary drift.
* **Miss 2 (arXiv 09)**: Cross-chapter conceptual dispersion diluted chunk vector density.
  - *Fix*: Parent Document Retrieval (indexing small chunks, passing full sections to LLM).

---

## 📽️ Slide 7: Live Demo Walkthrough (7:30 - 9:00)
* Live demonstration of Streamlit Web UI (`app.py`):
  - In-scope query: Instant answer (<0.3s) with clickable `[NIST_AI_RMF_1.0.pdf, Page 9]` citation expander.
  - Out-of-scope query (*"Recipe for pepperoni pizza"*): Safe rejection without hallucination.
  - Real-time latency tracking display.

---

## 📽️ Slide 8: Summary & Key Takeaways (9:00 - 10:00)
* **90% Retrieval Hit Rate** achieved across 618 pages.
* **100% Negative Rejection** preventing hallucinations.
* Systematic benchmarking allows engineering decisions based on empirical data rather than guesswork.
* Open for Questions & Feedback!
