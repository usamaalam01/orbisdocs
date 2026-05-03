"""Offline scraper + FAISS indexer for the Iris Accelerator docs.

Run locally before each commit when docs change:
    python build_index.py

Reads URLs from urls.txt, fetches each page, converts HTML to markdown,
splits into chunks, embeds with the model in .env, persists FAISS to ./vectorstore/.

The embedding model and the persisted index are coupled — if you change
EMBEDDING_MODEL in .env you MUST rerun this script and recommit vectorstore/.
"""
import sys
import time
import warnings
from pathlib import Path

import requests
from urllib3.exceptions import InsecureRequestWarning
from bs4 import BeautifulSoup
from markdownify import markdownify
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from app.config import settings

ROOT = Path(__file__).resolve().parent
URLS_FILE = ROOT / "urls.txt"
VECTORSTORE_DIR = ROOT / "vectorstore"

# The lab portal uses a self-signed TLS cert; treating it as trusted is
# acceptable here because the host is internal-IT-managed and the docs
# themselves are non-sensitive.
warnings.simplefilter("ignore", InsecureRequestWarning)


def fetch(url: str) -> bytes:
    """Return raw response bytes. BeautifulSoup will detect the charset from
    the page's <meta> tag, which is more reliable than requests' header sniffing."""
    try:
        r = requests.get(url, timeout=30, allow_redirects=True, verify=True)
        r.raise_for_status()
        return r.content
    except requests.exceptions.SSLError:
        r = requests.get(url, timeout=30, allow_redirects=True, verify=False)
        r.raise_for_status()
        return r.content


def extract_main(html: bytes) -> tuple[str, str]:
    """Return (page_title, main_html). Try semantic containers, fall back to body."""
    soup = BeautifulSoup(html, "html.parser")
    title = "Untitled"
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    node = None
    for selector in [("main", None), ("article", None), ("div", "document"), ("div", "body")]:
        tag, cls = selector
        node = soup.find(tag, class_=cls) if cls else soup.find(tag)
        if node:
            break
    if node is None:
        node = soup.body or soup

    for bad in node.find_all(["nav", "footer", "script", "style", "header"]):
        bad.decompose()

    return title, str(node)


def split_chunks(md: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    return splitter.split_text(md)


def nearest_heading(chunk: str, full_md: str) -> str:
    if not chunk:
        return ""
    probe = chunk[:60]
    idx = full_md.find(probe)
    if idx < 0:
        return ""
    last_h = ""
    for line in full_md[:idx].splitlines():
        s = line.strip()
        if s.startswith("## ") or s.startswith("### "):
            last_h = s.lstrip("# ").strip()
    return last_h


def main():
    start = time.time()

    urls = [
        u.strip()
        for u in URLS_FILE.read_text(encoding="utf-8").splitlines()
        if u.strip() and not u.strip().startswith("#")
    ]
    print(f"[build_index] Reading {len(urls)} URLs from {URLS_FILE.name}")

    docs: list[Document] = []
    for url in urls:
        print(f"[build_index]   fetching {url}")
        try:
            html = fetch(url)
        except Exception as e:
            print(f"[build_index]   FAILED: {e}", file=sys.stderr)
            raise

        title, main_html = extract_main(html)
        md = markdownify(main_html, heading_style="ATX")
        chunks = split_chunks(md)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source_url": url,
                    "page_title": title,
                    "section_heading": nearest_heading(chunk, md),
                    "chunk_index": i,
                },
            ))
        print(f"[build_index]     -> {len(chunks)} chunks  (title: {title!r})")

    print(f"[build_index] Total chunks: {len(docs)}")
    print(f"[build_index] Embedding with {settings.EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    # MAX_INNER_PRODUCT on already-normalized vectors = cosine similarity in [-1, 1].
    db = FAISS.from_documents(
        docs,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    db.save_local(str(VECTORSTORE_DIR))
    print(f"[build_index] Saved FAISS index to {VECTORSTORE_DIR.name}/  (index.faiss + index.pkl)")
    print(f"[build_index] Done — pages: {len(urls)}, chunks: {len(docs)}, "
          f"elapsed: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
