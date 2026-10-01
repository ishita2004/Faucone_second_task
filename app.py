import streamlit as st

from rag_pipeline import search_documents


st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


st.title("📚 PDF RAG Assistant")
st.write(
    "Ask questions about the information contained in the uploaded PDFs."
)


query = st.text_input(
    "Enter your question:",
    placeholder="What is the main topic of the document?"
)


top_k = st.slider(
    "Number of relevant chunks:",
    min_value=1,
    max_value=10,
    value=5
)


if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents..."):

            try:
                results = search_documents(
                    query,
                    top_k=top_k
                )

                st.subheader("Relevant Information")

                for i, result in enumerate(results, start=1):

                    st.markdown(
                        f"### Result {i}"
                    )

                    st.write(
                        f"**Source:** {result['source']}"
                    )

                    st.write(
                        f"**Page:** {result['page']}"
                    )

                    st.write(
                        f"**Similarity:** "
                        f"{result['score']:.4f}"
                    )

                    st.write(result["text"])

                    st.divider()

            except FileNotFoundError as e:
                st.error(str(e))

            except Exception as e:
                st.error(
                    f"Something went wrong: {e}"
                )