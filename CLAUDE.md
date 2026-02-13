# CLAUDE.md - Project Ascent AI

## Project Overview

Project Ascent is a **Streamlit-based AI research agent** for tax, legal, and audit solutions. It implements a RAG (Retrieval-Augmented Generation) pipeline that lets users upload documents, build an in-memory knowledge base, and query it alongside live web search results using Google Gemini as the LLM.

**Stage**: Early development / prototype. Single-developer project with no tests, no CI/CD, and no persistent storage.

## File Structure

```
project-ascent-app/
├── app.py                 # Main application (sidebar-based UI, uses gemini-pro)
├── update app.py          # Enhanced version (tabbed UI, uses gemini-1.5-flash-latest, configurable temperature/prompt)
├── requirements.txt       # Python dependencies (unpinned)
└── CLAUDE.md              # This file
```

- `app.py` — Original version with sidebar layout for document upload and web search toggle.
- `update app.py` — Newer version with three-tab UI (Knowledge Base, Agent Configuration, Chat), configurable LLM temperature, and editable system prompt. Uses the updated `gemini-1.5-flash-latest` model.

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Streamlit |
| LLM | Google Gemini (via `langchain-google-genai`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (via `sentence-transformers`) |
| Vector store | ChromaDB (in-memory, no persistence) |
| RAG orchestration | LangChain (chains, prompts, output parsers) |
| Document parsing | PyMuPDF (`fitz`) for PDFs, native decode for TXT |
| Web search | DuckDuckGo (via `langchain-community` tools) |

## Commands

### Setup
```bash
pip install -r requirements.txt
```

### Run the app
```bash
# Main version
streamlit run app.py

# Enhanced tabbed version
streamlit run "update app.py"
```

### No test, lint, or build commands exist
There is no test suite, linter configuration, or build pipeline in this project.

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Google Gemini API authentication |

Set via OS environment or Streamlit secrets. The app displays an error if this key is missing.

## Architecture & Data Flow

1. **Document Upload** — User uploads PDF/TXT files via Streamlit UI
2. **Text Extraction** — `get_document_text()` extracts text using PyMuPDF or UTF-8 decode
3. **Chunking** — `get_text_chunks()` splits text (chunk_size=1000, overlap=200) via `RecursiveCharacterTextSplitter`
4. **Embedding** — HuggingFace `all-MiniLM-L6-v2` model generates embeddings
5. **Vector Storage** — Chunks stored in ChromaDB in-memory vector store
6. **Query** — On user question:
   - Retrieve top-3 similar chunks from vector store
   - Optionally fetch DuckDuckGo web search results
   - Combine context and pass to Gemini LLM via LangChain chain
   - Display response in chat UI

## Code Conventions

- **Single-file architecture** — All logic lives in one Python file (`app.py` or `update app.py`)
- **Streamlit session state** — Used for all persistent state (`messages`, `vector_store`, `enable_web_search`, `llm_temperature`, `prompt_template`)
- **`@st.cache_resource`** — Applied to expensive model loading functions (`load_embedding_model`, `load_llm`)
- **Docstrings** — Every function has a one-line docstring
- **Error handling** — Try/except blocks with `st.error()` for user-facing errors; fallback strings for non-critical failures
- **LangChain LCEL chains** — Uses pipe (`|`) operator for composing prompt -> LLM -> output parser chains
- **Section comments** — Code sections delimited with `# --- Section Name ---` comments
- **No type hints** — Functions do not use type annotations
- **No logging** — Uses `st.error`/`st.warning`/`st.success` instead of Python logging

## Key Session State Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `messages` | list[dict] | Initial greeting | Chat history (role + content dicts) |
| `vector_store` | Chroma or None | None | In-memory vector database |
| `enable_web_search` | bool | True | Toggle for DuckDuckGo search |
| `llm_temperature` | float | 0.7 | LLM creativity parameter (0.0-1.0) |
| `prompt_template` | str | DEFAULT_PROMPT_TEMPLATE | Editable system prompt |

## Important Notes for AI Assistants

- **Two app versions exist**: `app.py` is the original; `update app.py` is newer with more features. Clarify with the user which file to modify.
- **The filename `update app.py` contains a space** — quote it in shell commands.
- **Dependencies are unpinned** in `requirements.txt` — be cautious about breaking changes between library versions.
- **No tests exist** — any new functionality should ideally include tests, but this is not yet an established pattern in this project.
- **In-memory only** — The vector store is not persisted. Data is lost on app restart.
- **Security**: Documents are processed in-memory only. The API key is read from environment, never hardcoded.
- **Git branch**: Main branch is `master`.
