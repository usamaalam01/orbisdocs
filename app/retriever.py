"""Loads the persisted FAISS index built by build_index.py."""
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import settings

VECTORSTORE_DIR = Path(__file__).resolve().parent.parent / "vectorstore"


def build_embeddings() -> HuggingFaceEmbeddings:
    # normalize_embeddings=True must match build_index.py so that the FAISS
    # MAX_INNER_PRODUCT scores are real cosine similarities in [-1, 1].
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectorstore(embeddings: HuggingFaceEmbeddings) -> FAISS:
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {VECTORSTORE_DIR}. "
            f"Run `python build_index.py` first."
        )
    # allow_dangerous_deserialization is required: LangChain pickles docstore
    # metadata. Safe here because the file is built and committed by us.
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
