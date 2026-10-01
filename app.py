from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.rag_chain import answer_question, format_citations, list_pdf_files

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "pdfs"


st.set_page_config(page_title="PDF RAG Assistant", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #f5f7ff 0%, #eef3ff 100%);
    }
    .block-container {
        padding-top: 2rem;
    }
    div[data-testid="stSidebar"] {
        background: #f7f9ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 PDF RAG Assistant")
st.caption("Ask questions about the PDFs in the local folder. Answers are grounded strictly in the retrieved PDF chunks.")

if "history" not in st.session_state:
    st.session_state.history = []

pdf_files = list_pdf_files(PDF_DIR)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        value=st.session_state.get("gemini_api_key", ""),
        help="Paste your Gemini API key to query the PDF collection.",
    )
    st.session_state.gemini_api_key = api_key

    st.subheader("PDF library")
    st.write(f"Loaded PDFs: {len(pdf_files)}")
    for pdf_path in pdf_files[:10]:
        st.caption(pdf_path.name)
    if len(pdf_files) > 10:
        st.caption(f"... and {len(pdf_files) - 10} more")

    if st.button("Refresh files"):
        st.rerun()

if not pdf_files:
    st.warning("No PDF files were found in the pdfs folder.")
    st.stop()

col1, col2 = st.columns(2)
col1.metric("Total PDFs", len(pdf_files))
col2.metric("Retrieval mode", "Chunk-based")

st.subheader("Ask a question")
question = st.text_area(
    "Question",
    placeholder="Example: What are the main AI governance concerns identified in these PDFs?",
    height=140,
)

if st.button("Ask question", use_container_width=True):
    if not api_key:
        st.warning("Please enter your Gemini API key in the sidebar first.")
    elif not question.strip():
        st.warning("Please enter a question before asking the RAG system.")
    else:
        try:
            with st.spinner("Searching the PDF chunks and generating a grounded answer..."):
                answer, citations = answer_question(PDF_DIR, api_key, question, top_k=5)
            st.session_state.history.append({
                "question": question,
                "answer": answer,
                "citations": citations,
            })
            st.success("Answer generated from the PDF library.")
        except Exception as exc:  # pragma: no cover - surfaced to the user
            st.error(f"RAG failed: {exc}")

for item in reversed(st.session_state.history):
    st.markdown("---")
    st.markdown(f"**Question:** {item['question']}")
    st.markdown(f"**Answer:**\n{item['answer']}")
    with st.expander("Citations"):
        st.text(format_citations(item["citations"]))
