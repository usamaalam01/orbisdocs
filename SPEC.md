# Orbis Iris Accelerator — RAG Demo MVP

A Streamlit application that lets employees ask plain-English questions against the Iris Accelerator lab-machine documentation, demonstrating Retrieval-Augmented Generation (RAG) for Orbis management.

## 1. Goal

Show management that internal documentation can be queried in natural language with cited answers, using the Iris Accelerator docs (9 pages) as a concrete example. The MVP must be presentable in a live demo, deployable to Streamlit Community Cloud from an existing GitHub repo, and configurable enough that the LLM provider/model/key can be swapped via a config file.

## 2. Scope

### In scope
- Ingest the 9 listed Iris Accelerator pages from `docs.lab.orbisholding.com`.
- Build a local FAISS vector index offline; commit it to the repo.
- Streamlit chat UI with multi-turn history, streaming answers, and inline citations linking back to the original doc URLs.
- Polite refusal for questions not covered by the docs.
- Orbis-branded header (logo + accent color).
- Deployable to Streamlit Community Cloud with a single `requirements.txt`.

### Out of scope (MVP)
- Authentication / SSO beyond an optional shared-secret password gate.
- Crawling beyond the 9 listed URLs.
- Automatic re-indexing on a schedule (manual rebuild only).
- Evaluation harness, automated tests, feedback collection.
- Reranker, hybrid search, query rewriting.
- Image/diagram extraction from the docs.
- Per-user usage limits or token accounting.

## 3. Document sources

```
http://docs.lab.orbisholding.com/lab-machines/iris-accelerator/index.html
http://docs.lab.orbisholding.com/lab-machines/iris-accelerator/architecture.html
http://docs.lab.orbisholding.com/lab-machines/iris-accelerator/infrastructure.html
http://docs.lab.orbisholding.com/lab-machines/iris-accelerator/database.html
http://docs.lab.orbisholding.com/lab-machines/iris-accelerator/services.html
http://docs.lab.orbisholding.com/lab-machines/iris-accelerator/device-manager.html
http://docs.lab.orbisholding.com/lab-machines/iris-accelerator/imssdk.html
http://docs.lab.orbisholding.com/lab-machines/iris-accelerator/developer-guide.html
http://docs.lab.orbisholding.com/lab-machines/iris-accelerator/troubleshooting.html
```

The list lives in `urls.txt` at the repo root so it can be edited without code changes.

### TLS note
A test fetch was auto-upgraded to HTTPS and returned `self signed certificate`. The scraper must:
1. Honor the `http://` scheme (do not auto-upgrade), and
2. If the server redirects to HTTPS, retry with `requests.get(url, verify=False)` and suppress the `InsecureRequestWarning`.

This is acceptable for a public-but-internal-IT-managed lab portal; document the rationale in a code comment in `build_index.py`.

## 4. Technology stack

| Layer | Choice |
|---|---|
| Language | Python 3.13 |
| Packaging | `requirements.txt` (pip) |
| UI | Streamlit ≥ 1.32 (chat elements + streaming) |
| Orchestration | LangChain (`langchain`, `langchain-community`, `langchain-groq`, `langchain-huggingface`) |
| LLM (default) | Groq `llama-3.3-70b-versatile` via `langchain-groq` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` (CPU) |
| Vector store | FAISS (`faiss-cpu`) — local, two-file persistence (`index.faiss` + `index.pkl`) committed under `vectorstore/` |
| Scraper | `requests` + `beautifulsoup4` |
| Config | `python-dotenv`, single `.env` file |
| Branding | Orbis logo asset + Streamlit `theme` overrides |
| Hosting | Streamlit Community Cloud, deployed from existing GitHub repo |

`requirements.txt` (target):
```
streamlit>=1.32
langchain>=0.2
langchain-community>=0.2
langchain-groq>=0.1
langchain-huggingface>=0.0.3
faiss-cpu>=1.8
sentence-transformers>=2.7
beautifulsoup4>=4.12
markdownify>=0.12
requests>=2.31
python-dotenv>=1.0
```

## 5. Configuration

Single `.env` file at repo root, loaded via `python-dotenv`. On Streamlit Community Cloud, the same keys are mirrored into the secrets UI.

```dotenv
# LLM provider — swap by changing these three values
LLM_PROVIDER=groq                  # groq | openai | anthropic | azure | ollama
LLM_MODEL=llama-3.3-70b-versatile
LLM_API_KEY=

# Embeddings (local — no key needed for the default)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Retrieval — cosine similarity in [0, 1], higher = better
TOP_K=4
SIMILARITY_THRESHOLD=0.35          # if best chunk's cosine score < this, refuse as off-topic

# Chunking (used by build_index.py)
CHUNK_SIZE=800
CHUNK_OVERLAP=120

# Branding
BRAND_COLOR=#1F7A5C                # deep teal-green, eye-picked from images/logo.jpg
LOGO_PATH=images/logo.jpg

# Optional access gate (leave empty to disable)
APP_PASSWORD=
```

`.env.example` is committed; `.env` is gitignored.

The LangChain LLM is constructed in a small factory (`app/llm.py`) that switches on `LLM_PROVIDER` and returns a streaming-capable chat model. Adding a new provider = adding one branch.

## 6. Repository layout

```
DocsRAG/
├── .env.example
├── .gitignore
├── README.md
├── SPEC.md                       # this file
├── requirements.txt
├── urls.txt                      # the 9 doc URLs, one per line
├── build_index.py                # offline scraper + indexer
├── streamlit_app.py              # Streamlit Cloud entrypoint
├── app/
│   ├── __init__.py
│   ├── config.py                 # loads .env, exposes typed settings
│   ├── llm.py                    # LangChain LLM factory (provider switch)
│   ├── retriever.py              # loads persisted FAISS, returns retriever
│   ├── rag_chain.py              # builds the LangChain RAG pipeline
│   └── ui.py                     # Streamlit components (header, chat, citations)
├── images/
│   └── logo.jpg                  # Orbis logo, provided
└── vectorstore/                  # committed, generated by build_index.py
    ├── index.faiss
    └── index.pkl
```

## 7. Indexing pipeline (`build_index.py`)

Run locally before each commit when docs change:

1. Read URLs from `urls.txt`.
2. For each URL, fetch HTML with `requests` (HTTP-first, fall back to `verify=False` HTTPS).
3. Parse with BeautifulSoup; extract the main content container (try `<main>`, `<article>`, `div.document`, `div.body` in order; fall back to `<body>`). Strip nav, footer, script, style.
4. Convert the cleaned HTML to markdown with `markdownify` so headings become `## ` / `### ` lines (this is what the splitter's heading-aware separators rely on).
5. Split with LangChain `RecursiveCharacterTextSplitter`, `chunk_size=CHUNK_SIZE`, `chunk_overlap=CHUNK_OVERLAP` (read from `.env`), separators `["\n## ", "\n### ", "\n\n", "\n", " "]` (heading-aware).
6. Attach metadata to each chunk: `source_url`, `page_title` (from `<title>`), `section_heading` (nearest preceding `## ` / `### ` line in the chunk), `chunk_index`.
7. Embed with `HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)` (read from `.env` — same model the runtime app will load).
8. Build the index via `FAISS.from_documents(chunks, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT, normalize_L2=True)` (gives cosine similarity in [0, 1]) and persist with `db.save_local("vectorstore")` — this writes `index.faiss` and `index.pkl`.
9. Print a summary: pages fetched, chunks created, elapsed time.

`build_index.py` loads `.env` at the top via `python-dotenv` so it reads `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP` from the same source as the runtime app. **The embedding model and the persisted index are coupled — if you change `EMBEDDING_MODEL` you MUST rerun `build_index.py` and recommit, otherwise loading will fail with a dimension mismatch.**

`markdownify` is added to `requirements.txt`.

Idempotent: rerunning overwrites both files in `vectorstore/`.

## 8. Runtime app (`streamlit_app.py`)

### Layout
- **Header**: Orbis logo (left) + title "Iris Accelerator — Documentation Q&A" + subtle `BRAND_COLOR` divider.
- **Sidebar**: model name, provider, top_k, "Clear chat" button, link to source repo, app version.
- **Main**: chat area (Streamlit `st.chat_message` + `st.chat_input`).
- **Optional gate**: if `APP_PASSWORD` is set, render a single password input before the chat is shown.

### Behavior
1. On first load (cached via `@st.cache_resource`):
   - Load `.env`.
   - Load embeddings model.
   - Load the persisted FAISS index from `./vectorstore/` via `FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)` (the flag is required because LangChain pickles docstore metadata; safe here because we built and committed the file ourselves).
   - Construct LLM via `app/llm.py` factory.
   - Build the LangChain RAG chain (history-aware retriever + answer generator).
2. On each user turn:
   - Run history-aware retrieval (rewrite query using prior turns, then retrieve `TOP_K` chunks).
   - If the top similarity score is below `SIMILARITY_THRESHOLD`, return the canned refusal:
     > "I can only answer questions about the Iris Accelerator documentation. Try asking about its architecture, services, infrastructure, device manager, IMS SDK, or troubleshooting."
   - Otherwise, stream the LLM answer token-by-token using `st.write_stream`.
   - After streaming completes, render a "Sources" section listing each unique `source_url` with the `page_title` and `section_heading` as a clickable link.
3. Multi-turn: full message history is kept in `st.session_state.messages` and passed into the history-aware retriever.

### System prompt (default, lives in `app/rag_chain.py`)
```
You are Orbis Iris Accelerator's documentation assistant. Answer the user's
question using ONLY the provided context excerpts. If the context does not
contain the answer, say you don't know — do not invent details.

Cite your sources inline using bracketed numbers like [1], [2] that map to the
context excerpts in the order given. Keep answers concise and structured;
prefer bullet lists for steps and short paragraphs for explanations.
```

## 9. Branding

- Logo file: `images/logo.jpg` — provided.
- Accent color: `BRAND_COLOR=#1F7A5C` (deep teal-green of the "rbis" wordmark). Eye-picked from the JPG, so refine in `.env` if it reads off on the deployed page. The orange-red star inside the globe is a secondary accent (~`#D14F2A`) but is **not** used in the Streamlit theme — only one `primaryColor` is set.
- Streamlit theme set in `.streamlit/config.toml` with the **hex hardcoded** (Streamlit does not interpolate env vars in its config files):
  ```toml
  [theme]
  primaryColor = "#1F7A5C"
  font = "sans serif"
  ```
  `BRAND_COLOR` in `.env` remains the source of truth for any in-app uses (custom dividers, markdown). If you change the brand color, update **both** files.
- Logo rendered in the header via `st.image(LOGO_PATH, width=160)`.

## 10. Deployment

1. Provide brand color hex in `.env` (logo is already at `images/logo.jpg`).
2. Build the index locally:
   ```bash
   python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   python build_index.py
   git add vectorstore/ && git commit -m "rebuild index"
   ```
3. Push to the existing GitHub repo at https://github.com/usamaalam01/orbisdocs.
4. On Streamlit Community Cloud:
   - Connect the repo, point to `streamlit_app.py`, Python 3.13.
   - Open the secrets UI and add each `.env` key in **TOML** format (Streamlit secrets are TOML, not dotenv — the values must be quoted strings). Example:
     ```toml
     LLM_PROVIDER = "groq"
     LLM_MODEL = "llama-3.3-70b-versatile"
     LLM_API_KEY = "gsk_..."
     EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
     TOP_K = "4"
     SIMILARITY_THRESHOLD = "0.35"
     CHUNK_SIZE = "800"
     CHUNK_OVERLAP = "120"
     BRAND_COLOR = "#1F7A5C"
     LOGO_PATH = "images/logo.jpg"
     APP_PASSWORD = ""
     ```
     `app/config.py` reads them via `os.getenv` after `dotenv` locally and via `st.secrets` on Streamlit Cloud — `config.py` falls back from one to the other.
   - Deploy.

## 11. Verification (end-to-end test plan)

After deploy, manually verify:

1. **Cold start succeeds** within ~30 s; chat input is reachable.
2. **In-scope question** ("What services run on the Iris Accelerator?") returns a streamed answer with at least one citation linking back to `services.html` or `architecture.html`.
3. **Multi-turn follow-up** ("And how do they communicate?") uses prior context and returns a coherent answer with citations.
4. **Off-topic question** ("What's the weather in London?") returns the polite refusal text and no citations.
5. **Citation links open** the correct upstream `docs.lab.orbisholding.com` page.
6. **Provider swap** works: change `LLM_PROVIDER`/`LLM_MODEL`/`LLM_API_KEY` in Streamlit secrets, redeploy, ask a question — answer comes from the new provider.
7. **Index rebuild** works: change a URL in `urls.txt`, rerun `build_index.py`, commit, redeploy — new content is retrievable.
8. **Optional password gate**: if `APP_PASSWORD` is set, app refuses access until correct password is entered.

## 12. Open items requiring user input before build

- [x] Orbis logo asset — `images/logo.jpg`
- [x] Brand accent color hex — `#1F7A5C` (eye-picked from the logo; refine if needed)
- [x] GitHub repo URL — https://github.com/usamaalam01/orbisdocs
- [ ] Groq API key (user will add to `.env` locally and to Streamlit Cloud secrets at deploy time)

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Docs portal HTTPS cert breaks scraper | HTTP-first fetch with `verify=False` HTTPS fallback, documented in `build_index.py` |
| Public Streamlit URL leaks Groq tokens | Optional `APP_PASSWORD` gate; Groq free tier limits exposure |
| Streamlit Cloud cold-start RAM | MiniLM is small; cache resources with `@st.cache_resource` |
| Off-topic answers from LLM general knowledge | Strict similarity threshold + system-prompt instruction to refuse when context is empty |
| Stale committed index | Manual rebuild flow documented; small page count makes rebuilds fast |
| FAISS pickle deserialization warning | `allow_dangerous_deserialization=True` is required by LangChain when loading FAISS; safe here because we author and commit the pickle ourselves (no external/untrusted source). Documented at the call site. |
| `sentence-transformers` pulls PyTorch (~700 MB) and may pressure Streamlit Cloud's 1 GB cap | If install fails or app OOMs, swap to `fastembed` (ONNX, ~150 MB) — it has a drop-in `FastEmbedEmbeddings` wrapper in `langchain-community`. Embedding model name and index would need to be regenerated. |
| Embedding model and persisted FAISS index drift apart | Treat them as coupled: any change to `EMBEDDING_MODEL` requires rerunning `build_index.py` and recommitting `vectorstore/`. Documented in §7. |
| Cited URL is unreachable to demo audience on the projector laptop | Confirm projector laptop is on Orbis network before demo (the docs portal is public per user, but worth verifying live before the meeting) |
