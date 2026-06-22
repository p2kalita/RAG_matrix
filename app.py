"""
app.py — RAG Comparison App: Traditional vs Hybrid vs Agentic

Streamlit entry-point. Orchestrates all three RAG pipelines side-by-side.
"""

import streamlit as st
import time
from dotenv import load_dotenv

from components.config import AVAILABLE_MODELS, DEFAULT_MODEL
from components.pdf_processor import process_pdf
from components.vector_store import VectorStoreManager
from components.rag_handler import TraditionalRAG
from components.hybrid_rag_handler import HybridRAG
from components.agentic_rag_handler import AgenticRAG
from components.tracking import track_interaction, start_query_trace, track_pipeline, end_query_trace
from components.guardrails import apply_guardrails

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Comparison App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Light premium theme ── */
:root {
    --bg-primary:   #f5f7fa;
    --bg-card:      #ffffff;
    --bg-card2:     #eef2f7;
    --bg-sidebar:   #ffffff;
    --accent-trad:  #2563eb;
    --accent-hyb:   #7c3aed;
    --accent-agent: #059669;
    --text-primary: #1e293b;
    --text-muted:   #64748b;
    --border:       rgba(0,0,0,0.09);
    --shadow:       0 2px 12px rgba(0,0,0,0.07);
}

/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* App background */
.stApp { background: var(--bg-primary) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border);
    box-shadow: 2px 0 8px rgba(0,0,0,0.04);
}

/* Column pipeline headers */
.rag-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 16px;
    font-size: 1.0rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    box-shadow: var(--shadow);
}
.rag-header.trad {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border-left: 4px solid var(--accent-trad);
    color: #1d4ed8;
}
.rag-header.hyb {
    background: linear-gradient(135deg, #f5f3ff, #ede9fe);
    border-left: 4px solid var(--accent-hyb);
    color: #6d28d9;
}
.rag-header.agent {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border-left: 4px solid var(--accent-agent);
    color: #047857;
}

/* Badge chips */
.chip {
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 999px;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.chip-trad  { background: #dbeafe; color: #1d4ed8; }
.chip-hyb   { background: #ede9fe; color: #6d28d9; }
.chip-agent { background: #d1fae5; color: #047857; }

/* Citation box */
.citation-box {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.76rem;
    color: var(--text-muted);
    margin-top: 6px;
    line-height: 1.6;
}

/* Divider */
.msg-divider { border-top: 1px solid var(--border); margin: 6px 0; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #ffffff, #eef2f7);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: var(--shadow);
}
.hero-icon  { font-size: 3rem; }
.hero-title { font-size: 1.65rem; font-weight: 700; color: var(--text-primary); margin: 0; }
.hero-sub   { color: var(--text-muted); font-size: 0.9rem; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages":       [],
        "document_store": None,
        "pdf_name":       None,
        "vector_store":   None,
        "selected_model": DEFAULT_MODEL,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Lazy-init VectorStoreManager (triggers HF model download once)
    if st.session_state.vector_store is None:
        with st.spinner("⏳ Loading embedding model (first run ~90s)…"):
            st.session_state.vector_store = VectorStoreManager()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar():
    with st.sidebar:
        st.markdown("### 🧠 RAG Comparison")
        st.markdown("---")

        # Model selector
        st.markdown("**🤖 LLM Model**")
        model_names = list(AVAILABLE_MODELS.keys())
        selected = st.selectbox(
            label="Select model",
            options=model_names,
            index=model_names.index(st.session_state.selected_model),
            key="model_selector",
            label_visibility="collapsed",
        )
        st.session_state.selected_model = selected
        st.caption(f"`{AVAILABLE_MODELS[selected]}`")

        # Rate-limit tip for Groq models
        if "Groq" in selected:
            if "8B" in selected:
                st.caption("30,000 TPM — good for 3-pipeline comparisons")
            elif "70B" in selected:
                st.caption("12,000 TPM — may hit limits; retry auto-handled")
            elif "Qwen" in selected:
                st.caption("6,000 TPM — lowest limit; use for single queries")
        else:
            st.caption("No strict TPM limit on free tier")

        st.markdown("---")

        # PDF upload
        st.markdown("**📄 Document**")
        uploaded_file = st.file_uploader(
            "Upload PDF", type="pdf", label_visibility="collapsed"
        )
        if uploaded_file:
            if uploaded_file.name != st.session_state.pdf_name:
                with st.spinner("🔄 Processing PDF…"):
                    st.session_state.pdf_name       = uploaded_file.name
                    st.session_state.document_store = process_pdf(uploaded_file)
                    st.session_state.messages       = []  # reset chat on new doc
                st.success("✅ PDF indexed successfully!")

        if st.session_state.pdf_name:
            st.info(f"📎 **{st.session_state.pdf_name}**")

        st.markdown("---")

        # Legend
        st.markdown("**📖 Pipeline Legend**")
        st.markdown(
            '<span style="color:#2563eb">■</span> **Traditional** — Dense retrieval → single LLM call  \n'
            '<span style="color:#7c3aed">■</span> **Hybrid** — BM25 + Dense → RRF fusion → LLM  \n'
            '<span style="color:#059669">■</span> **Agentic** — ReAct loop with 3 specialised agents',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Clear Chat
        st.markdown("**🗑️ Session**")
        if st.button(
            "Clear Chat History",
            use_container_width=True,
            type="secondary",
            key="clear_chat_btn",
        ):
            st.session_state.messages = []
            st.success("Chat cleared!")
            st.rerun()


# ── Message Renderers ─────────────────────────────────────────────────────────
def _render_column(rag_type: str, color: str, messages: list[dict]):
    for msg in messages:
        if msg.get("rag_type") != rag_type:
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                st.markdown(
                    f'<div class="citation-box">{msg["citations"]}</div>',
                    unsafe_allow_html=True,
                )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    _init_state()
    _render_sidebar()

    # Hero header
    st.markdown(
        """
        <div class="hero">
          <div class="hero-icon">🧠</div>
          <div>
            <p class="hero-title">RAG Strategy Comparison</p>
            <p class="hero-sub">
              Compare <strong>Traditional</strong>, <strong>Hybrid</strong>, and <strong>Agentic</strong>
              RAG pipelines side-by-side on the same document and question.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.document_store is None:
        st.info("📂 Upload a PDF in the sidebar to get started.")
        st.stop()

    # Column headers
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="rag-header trad">'
            '🔵 Traditional RAG '
            '<span class="chip chip-trad">Dense Only</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="rag-header hyb">'
            '🟣 Hybrid RAG '
            '<span class="chip chip-hyb">BM25 + Dense + RRF</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="rag-header agent">'
            '🟢 Agentic RAG '
            '<span class="chip chip-agent">ReAct Loop</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Render existing messages
    col1, col2, col3 = st.columns(3)
    with col1:
        _render_column("traditional", "#4f8ef7", st.session_state.messages)
    with col2:
        _render_column("hybrid", "#a855f7", st.session_state.messages)
    with col3:
        _render_column("agentic", "#22d3a8", st.session_state.messages)

    # ── Chat Input ────────────────────────────────────────────────────────────
    prompt = st.chat_input("Ask a question about your document…")
    if not prompt:
        return

    model = st.session_state.selected_model
    file_name = st.session_state.pdf_name

    # Store user messages for all 3 pipelines
    for rag_type in ("traditional", "hybrid", "agentic"):
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "rag_type": rag_type}
        )

    # ── Run pipelines sequentially with gap to spread token usage ────────────
    col1, col2, col3 = st.columns(3)
    vs = st.session_state.vector_store

    from litellm.exceptions import RateLimitError

    # Open one parent Opik Trace for this query
    trace_id = start_query_trace(query=prompt, model=model)
    pipeline_results: dict[str, dict] = {}

    def _run_pipeline(label, rag_type, rag_cls, col, gap_before=0.0):
        """Run one pipeline, record Opik span with timing, return (resp, cit) or (None,None)."""
        if gap_before:
            time.sleep(gap_before)
        with col:
            with st.spinner(f"{label} thinking…"):
                t_start = time.time()
                try:
                    history   = vs.get_relevant_history(prompt, rag_type)
                    rag       = rag_cls(st.session_state.document_store, file_name)
                    resp, cit = rag.get_response(prompt, history, model=model)
                    resp_g    = apply_guardrails(resp)
                    latency   = time.time() - t_start

                    has_warnings = resp_g.startswith("⚠️")

                    # Record span in Opik
                    track_pipeline(
                        trace_id=trace_id,
                        rag_type=rag_type,
                        query=prompt,
                        response=resp_g,
                        citations=cit,
                        latency_s=latency,
                        model=model,
                        has_guardrail_warnings=has_warnings,
                    )

                    # Accumulate for trace summary
                    pipeline_results[rag_type] = {
                        "latency_s":  round(latency, 3),
                        "word_count": len(resp_g.split()),
                    }

                    vs.add_conversation(prompt, resp_g, rag_type)
                    return resp_g, cit

                except RateLimitError:
                    latency = time.time() - t_start
                    st.error(
                        f"Rate limit reached for **{label}**. "
                        "Switch to *Llama 3.1 8B* (30k TPM) or wait and retry."
                    )
                    pipeline_results[rag_type] = {"latency_s": latency, "word_count": 0}
                    return None, None

                except Exception as exc:
                    latency = time.time() - t_start
                    st.error(f"Error in {label}: {exc}")
                    pipeline_results[rag_type] = {"latency_s": latency, "word_count": 0}
                    return None, None

    # Run Traditional → Hybrid (+1s) → Agentic (+1s)
    t_resp_g, t_cit = _run_pipeline("Traditional RAG", "traditional", TraditionalRAG, col1)
    h_resp_g, h_cit = _run_pipeline("Hybrid RAG",      "hybrid",      HybridRAG,      col2, gap_before=1.0)
    a_resp_g, a_cit = _run_pipeline("Agentic RAG",     "agentic",     AgenticRAG,     col3, gap_before=1.0)

    # Close parent Opik Trace with comparison summary
    end_query_trace(trace_id, pipeline_results)

    # Store successful responses in session
    for rag_type, resp, cit in [
        ("traditional", t_resp_g, t_cit),
        ("hybrid",      h_resp_g, h_cit),
        ("agentic",     a_resp_g, a_cit),
    ]:
        if resp is not None:
            st.session_state.messages.append(
                {
                    "role":      "assistant",
                    "content":   resp,
                    "citations": cit,
                    "rag_type":  rag_type,
                }
            )

    st.rerun()


if __name__ == "__main__":
    main()