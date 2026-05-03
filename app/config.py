"""Loads settings from .env locally and falls back to st.secrets on Streamlit Cloud."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _get(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


class Settings:
    LLM_PROVIDER = _get("LLM_PROVIDER", "groq")
    LLM_MODEL = _get("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_API_KEY = _get("LLM_API_KEY", "")
    EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    TOP_K = int(_get("TOP_K", "4") or "4")
    SIMILARITY_THRESHOLD = float(_get("SIMILARITY_THRESHOLD", "0.35") or "0.35")
    CHUNK_SIZE = int(_get("CHUNK_SIZE", "800") or "800")
    CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "120") or "120")
    BRAND_COLOR = _get("BRAND_COLOR", "#1F7A5C")
    LOGO_PATH = _get("LOGO_PATH", "images/logo.jpg")
    APP_PASSWORD = _get("APP_PASSWORD", "")


settings = Settings()
