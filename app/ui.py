"""Streamlit UI components: header, sidebar, sources panel, optional password gate."""
import streamlit as st
from app.config import settings


def render_header():
    col1, col2 = st.columns([1, 6])
    with col1:
        try:
            st.image(settings.LOGO_PATH, width=140)
        except Exception:
            pass
    with col2:
        st.title("Iris Accelerator — Documentation Q&A (MVP)")
        st.markdown(
            f"<hr style='border:0;border-top:3px solid {settings.BRAND_COLOR};margin:0 0 8px 0;'>",
            unsafe_allow_html=True,
        )


def render_sidebar(version: str):
    st.sidebar.markdown("### Configuration")
    st.sidebar.markdown(f"**Provider:** `{settings.LLM_PROVIDER}`")
    st.sidebar.markdown(f"**Model:** `{settings.LLM_MODEL}`")
    st.sidebar.markdown(f"**Top-K:** {settings.TOP_K}")
    st.sidebar.markdown(f"**Threshold:** {settings.SIMILARITY_THRESHOLD}")
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown("[Source on GitHub](https://github.com/usamaalam01/orbisdocs)")
    st.sidebar.caption(f"v{version}")


def render_sources(docs):
    if not docs:
        return
    seen = {}
    for d in docs:
        url = d.metadata.get("source_url", "#")
        title = d.metadata.get("page_title", "Source")
        heading = d.metadata.get("section_heading", "")
        if url not in seen:
            seen[url] = (title, heading)
    with st.expander(f"Sources ({len(seen)})"):
        for url, (title, heading) in seen.items():
            label = f"{title} — {heading}" if heading else title
            st.markdown(f"- [{label}]({url})")


def password_gate() -> bool:
    if not settings.APP_PASSWORD:
        return True
    if st.session_state.get("authed"):
        return True
    pw = st.text_input("Enter password to continue", type="password")
    if pw and pw == settings.APP_PASSWORD:
        st.session_state.authed = True
        st.rerun()
    elif pw:
        st.error("Incorrect password.")
    return False
