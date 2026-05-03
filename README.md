# Orbis Iris Accelerator — Docs RAG MVP

Streamlit chat app that answers plain-English questions about the Iris Accelerator
documentation, with inline citations back to the source pages.

See [SPEC.md](SPEC.md) for the full design or open [SPEC.html](SPEC.html) for a
branded presentation view.

## Quick start

```bash
# 1. Create venv and install
python -m venv .venv
# Windows:  .venv\Scripts\activate
# *nix:     source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# edit .env and set LLM_API_KEY=<your Groq key>

# 3. Build the index (scrapes the 9 doc URLs)
python build_index.py

# 4. Run the app
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. Make sure `vectorstore/` is committed (run `python build_index.py` and commit).
2. In the Streamlit Cloud UI, connect this repo and point to `streamlit_app.py`.
3. Paste your config into the **secrets** UI in TOML format (quoted strings):
   ```toml
   LLM_PROVIDER = "groq"
   LLM_MODEL = "llama-3.3-70b-versatile"
   LLM_API_KEY = "gsk_..."
   ```
   See [SPEC.md §10](SPEC.md#10-deployment) for the full key list.
4. Deploy.

## Layout

```
build_index.py        # offline scraper + FAISS indexer
streamlit_app.py      # Streamlit Cloud entrypoint
app/
  config.py           # .env + st.secrets loader
  llm.py              # LangChain LLM factory (provider switch)
  retriever.py        # loads the persisted FAISS index
  rag_chain.py        # history-aware retriever + answer chain
  ui.py               # Streamlit components
urls.txt              # the 9 doc URLs to ingest
vectorstore/          # generated FAISS index (committed)
images/logo.jpg       # Orbis logo
```

## Swapping the LLM

Edit `.env` (or Streamlit secrets) — no code change needed:

```dotenv
LLM_PROVIDER=openai           # or anthropic, ollama, groq
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...
```

`app/llm.py` handles the provider switch. New providers = one new branch.
