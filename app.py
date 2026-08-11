import streamlit as st
from dotenv import load_dotenv

from src.config import PAPERS, TOP_K
from src.generate import generate_answer, generate_baseline

load_dotenv()

st.set_page_config(page_title="RAG over Agent & RAG Papers", page_icon="📚")

st.title("📚 RAG over Agent & RAG-Eval Papers")
st.caption(
    f"Grounded Q&A over {len(PAPERS)} arXiv papers on LLM agents, retrieval-augmented "
    "generation, and RAG evaluation methodology."
)

with st.sidebar:
    st.header("Settings")
    k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=10, value=TOP_K)
    rerank = st.checkbox("Use cross-encoder reranking", value=True)
    show_baseline = st.checkbox("Also show no-retrieval baseline", value=False)
    st.markdown("---")
    st.caption(
        "Baseline = same model, same question, no retrieved context. "
        "Useful for seeing whether grounding actually changes the answer."
    )

query = st.text_input(
    "Ask a question about the corpus",
    placeholder="e.g. How does ReAct combine reasoning and acting?",
)
ask = st.button("Ask", type="primary")

if ask and query:
    with st.spinner("Retrieving and generating..."):
        result = generate_answer(query, k=k, rerank=rerank)

    if not result["grounded"]:
        st.warning(result["answer"])
    else:
        st.markdown(result["answer"])

        st.subheader("Sources")
        for source, chunk in zip(result["sources"], result["chunks"]):
            with st.expander(f"[{source['index']}] {source['title']}"):
                st.write(chunk["text"])
                st.caption(source["url"])

    if show_baseline:
        st.divider()
        st.subheader("Baseline (no retrieval)")
        with st.spinner("Generating baseline answer..."):
            baseline_answer = generate_baseline(query)
        st.markdown(baseline_answer)
elif ask:
    st.info("Enter a question first.")
