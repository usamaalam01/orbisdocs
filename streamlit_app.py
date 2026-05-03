"""Streamlit Cloud entrypoint: chat UI over the Iris Accelerator docs."""
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from app.config import settings
from app.llm import build_llm
from app.retriever import build_embeddings, load_vectorstore
from app.rag_chain import build_rag_chain, is_in_scope, REFUSAL
from app.ui import render_header, render_sidebar, render_sources, password_gate

VERSION = "0.1.0"

st.set_page_config(
    page_title="Iris Accelerator Q&A",
    page_icon=settings.LOGO_PATH if settings.LOGO_PATH else None,
    layout="wide",
)


@st.cache_resource(show_spinner="Loading embeddings, FAISS index, and LLM...")
def bootstrap():
    embeddings = build_embeddings()
    vectorstore = load_vectorstore(embeddings)
    llm = build_llm()
    chain = build_rag_chain(llm, vectorstore)
    return vectorstore, chain


def to_lc_history(messages):
    history = []
    for m in messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))
    return history


def main():
    render_header()
    render_sidebar(VERSION)

    if not password_gate():
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        vectorstore, chain = bootstrap()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                render_sources(m["sources"])

    user_input = st.chat_input("Ask a question about the Iris Accelerator…")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    in_scope, score = is_in_scope(vectorstore, user_input)

    with st.chat_message("assistant"):
        if not in_scope:
            st.markdown(REFUSAL)
            st.session_state.messages.append(
                {"role": "assistant", "content": REFUSAL, "sources": []}
            )
            return

        history = to_lc_history(st.session_state.messages[:-1])
        full_text = ""
        sources = []
        placeholder = st.empty()

        for chunk in chain.stream({"input": user_input, "chat_history": history}):
            if "answer" in chunk and chunk["answer"]:
                full_text += chunk["answer"]
                placeholder.markdown(full_text)
            if "context" in chunk and chunk["context"]:
                sources = chunk["context"]

        render_sources(sources)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_text, "sources": sources}
        )


main()
