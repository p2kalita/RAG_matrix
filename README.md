# RAG Comparison App

A Streamlit application that compares **Traditional RAG**, **Hybrid RAG**, and **Agentic RAG** side-by-side. Upload any PDF document and ask questions — all three pipelines respond simultaneously so you can see exactly how retrieval strategy affects answer quality.

## Features

- 📄 **PDF ingestion** — chunked, embedded, and stored in ChromaDB
- 🔵 **Traditional RAG** — dense semantic search → single LLM call
- 🟣 **Hybrid RAG** — BM25 keyword search + dense vector search fused with Reciprocal Rank Fusion (RRF)
- 🟢 **Agentic RAG** — ReAct-style multi-step loop with three specialised agent roles (Analyzer → Researcher → Writer)
- 🤖 **Unified LLM selector** — switch between Groq (Llama 3.3 70B, Llama 3.1 8B, Qwen QWQ 32B) and Gemini 2.5 Flash from the sidebar
- 🧲 **Local HuggingFace embeddings** — `all-MiniLM-L6-v2`, no OpenAI key needed
- 📝 **Citation tracking** — every response shows source page numbers and chunk scores
- 🛡️ **Pure-Python guardrails** — citation presence, hedging detection, length checks
- 📊 **Optional Opik tracking** — logs locally if `OPIK_API_KEY` is not set

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Praneeth16/rag-comparison-app.git
cd rag-comparison-app
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```
> ⚠️ First install downloads the `all-MiniLM-L6-v2` model (~90 MB) and PyTorch. Allow ~5 minutes.

4. **Configure API keys** in `.env`:
```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
OPIK_API_KEY=your_opik_api_key   # optional
```

---

## Usage

```bash
streamlit run app.py
```

1. Select an LLM model from the sidebar dropdown
2. Upload a PDF document
3. Type a question in the chat input
4. Compare responses from all three RAG pipelines side-by-side

---

## Project Structure

```
rag-comparison-app/
├── app.py                          # Main Streamlit application (3-column UI)
├── requirements.txt                # Project dependencies
├── .env                            # API keys (not committed)
├── components/
│   ├── __init__.py
│   ├── config.py                   # Unified LLM config (LiteLLM) + model registry
│   ├── pdf_processor.py            # PDF parsing and chunking
│   ├── vector_store.py             # ChromaDB singleton with HF embeddings
│   ├── rag_handler.py              # Traditional RAG implementation
│   ├── hybrid_rag_handler.py       # Hybrid RAG (BM25 + Dense + RRF)
│   ├── agentic_rag_handler.py      # Agentic RAG (ReAct loop, no crewai)
│   ├── tracking.py                 # Interaction tracking (Opik or local)
│   └── guardrails.py               # Pure-Python response validation
└── vector_store_db/                # ChromaDB persistent storage (auto-created)
```

---

## Pipelines in Detail

### 🔵 Traditional RAG
- Dense similarity search via ChromaDB
- Optional query expansion using conversation history
- Single LLM call with retrieved context
- Page-level citations

### 🟣 Hybrid RAG
- **BM25** keyword retrieval over all stored chunks (`rank_bm25`)
- **Dense** vector retrieval via ChromaDB
- **Reciprocal Rank Fusion** (RRF, k=60) merges both ranked lists
- Scores shown per source (BM25 + RRF)

### 🟢 Agentic RAG
- **Context Analyzer** agent: decomposes query into focused sub-queries (JSON output)
- **Researcher** agent: iterative retrieval loop (up to 3 rounds) with sufficiency assessment
- **Technical Writer** agent: synthesizes final answer strictly from research notes
- All agents run via LiteLLM — no crewai dependency

---

## Available Models

| Display Name | Provider | Model ID |
|---|---|---|
| Llama 3.3 70B | Groq | `llama-3.3-70b-versatile` |
| Llama 3.1 8B | Groq | `llama-3.1-8b-instant` |
| Qwen QWQ 32B | Groq | `qwen-qwq-32b` |
| Gemini 2.5 Flash | Google | `gemini-2.5-flash` |

---

## Tech Stack

| Component | Library |
|---|---|
| UI | Streamlit |
| LLM Gateway | LiteLLM |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | ChromaDB |
| Keyword Search | rank-bm25 |
| PDF Parsing | pypdf |
| Text Splitting | langchain-text-splitters |
| Tracking | Opik (optional) |
