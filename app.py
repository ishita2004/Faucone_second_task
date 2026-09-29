import sys
import os
import streamlit as st

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import config
from src.rag_chain import RAGChain

st.set_page_config(
    page_title="Document Q&A Assistant (RAG)",
    page_icon="📖",
    layout="wide"
)

st.title("📖 Document Q&A Assistant (RAG System)")
st.markdown("Ask questions grounded strictly in your ingested PDF document set with precise page-level citations.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
top_k = st.sidebar.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=config.TOP_K)
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Document Collection")

if os.path.exists(config.RAW_PDFS_DIR):
    pdfs = [f for f in os.listdir(config.RAW_PDFS_DIR) if f.endswith('.pdf')]
    st.sidebar.write(f"**Loaded Documents:** {len(pdfs)} PDFs")
    with st.sidebar.expander("View PDF List"):
        for pdf in pdfs:
            st.write(f"- {pdf}")
else:
    st.sidebar.error("PDF directory not found.")

@st.cache_resource
def get_rag_chain():
    return RAGChain()

try:
    rag = get_rag_chain()
    vector_store_ready = True
except Exception as e:
    st.error(f"Vector Store not found. Please run ingestion first (`python -m src.ingest`). Details: {e}")
    vector_store_ready = False

# Example Questions
st.subheader("💡 Example Questions")
example_cols = st.columns(3)
selected_example = None

with example_cols[0]:
    if st.button("What is the NIST AI Risk Management Framework?"):
        selected_example = "What is the NIST AI Risk Management Framework?"

with example_cols[1]:
    if st.button("How is AI bias defined and mitigated in NIST SP 1270?"):
        selected_example = "How is AI bias defined and mitigated in NIST SP 1270?"

with example_cols[2]:
    if st.button("What are the key ethical concerns of AI governance?"):
        selected_example = "What are the key ethical concerns of AI governance?"

query_input = st.text_input(
    "Enter your question:",
    value=selected_example if selected_example else "",
    placeholder="e.g. What are the core functions of NIST AI RMF?"
)

if st.button("Get Answer", type="primary") or selected_example:
    if not query_input.strip():
        st.warning("Please enter a question.")
    elif not vector_store_ready:
        st.error("Vector store is not initialized. Run `python -m src.ingest` first.")
    else:
        with st.spinner("Searching vector index and generating answer with citations..."):
            result = rag.query(query_input, top_k=top_k)

        st.markdown("### 💬 Answer")
        st.info(result["answer"])

        st.markdown("### 📚 Source Citations & Contexts")
        sources = result.get("sources", [])

        if not sources:
            st.write("No matching context chunks retrieved.")
        else:
            for idx, src in enumerate(sources, start=1):
                with st.expander(f"Citation [{idx}]: {src['source']} (Page {src['page']}) | Similarity: {src['similarity_score']:.4f}"):
                    st.write(f"**Document:** {src['source']}")
                    st.write(f"**Page:** {src['page']} of {src['total_pages']}")
                    st.write(f"**Chunk ID:** {src['chunk_id']}")
                    st.write(f"**Similarity Score:** `{src['similarity_score']:.4f}`")
                    st.markdown("**Excerpt:**")
                    st.code(src['snippet'], language=None)
