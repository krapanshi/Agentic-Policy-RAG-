"""
app.py  —  Policy & Process Copilot
-------------------------------------
This is the main entry point.  Run it with:
    streamlit run app.py
"""

import os
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ─── Load .env file FIRST (sets OPENAI_API_KEY before anything else) ────────
load_dotenv()

# ─── Page configuration (must be the very first Streamlit call) ─────────────
st.set_page_config(
    page_title="Policy & Process Copilot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Logging setup  ──────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename=f"logs/app_{datetime.now().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ─── Inline CSS for a polished look ─────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Main header */
        .cop-header {
            font-size: 2rem;
            font-weight: 700;
            color: #1a3c6e;
            margin-bottom: 0;
        }
        .cop-subtitle {
            color: #555;
            font-size: 0.95rem;
            margin-top: 0;
        }

        /* Action badge colours */
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.05em;
        }
        .badge-retrieve      { background:#dbeafe; color:#1d4ed8; }
        .badge-clarify       { background:#fef3c7; color:#92400e; }
        .badge-direct_answer { background:#dcfce7; color:#166534; }

        /* Citation card */
        .citation-card {
            background: #f0f7ff;
            border-left: 4px solid #3b82f6;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.85rem;
            margin: 4px 0;
            color: #1e3a5f;
        }

        /* Agent trace box */
        .trace-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
            font-size: 0.82rem;
            line-height: 1.6;
            color: #334155;
        }

        /* Sidebar stats */
        .stat-box {
            background: #f1f5f9;
            border-radius: 8px;
            padding: 10px 14px;
            text-align: center;
        }
        .stat-num  { font-size: 1.6rem; font-weight: 700; color: #1a3c6e; }
        .stat-label{ font-size: 0.75rem; color: #64748b; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════
# HELPER — get or create API key
# ════════════════════════════════════════════════════════════════════════════

def get_api_key() -> str:
    """
    Looks for the OpenAI API key in three places (in priority order):
      1. Entered by the user in the sidebar text input this session
      2. OPENAI_API_KEY environment variable (set via .env file)
    Returns the key string, or empty string if none found.
    """
    return (
        st.session_state.get("sidebar_api_key", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )


# ════════════════════════════════════════════════════════════════════════════
# HELPER — initialize models (cached in session_state)
# ════════════════════════════════════════════════════════════════════════════

def initialize_system(api_key: str):
    """
    Creates the VectorStore and AgenticRAGWorkflow once per session and caches
    them in st.session_state so we don't reload them on every Streamlit rerun.
    """
    if "vector_store" not in st.session_state:
        from rag.vector_store import VectorStore
        from rag.agents import AgenticRAGWorkflow

        with st.spinner("⚙️  Loading AI models (first run may take ~10 s)…"):
            vs = VectorStore(api_key=api_key)
            wf = AgenticRAGWorkflow(api_key=api_key, vector_store=vs)
            st.session_state.vector_store  = vs
            st.session_state.workflow      = wf
            st.session_state.chat_history  = []   # list of {role, content, meta}
            st.session_state.ingested      = not vs.is_empty()


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️  Configuration")
    st.divider()

    # ── API Key ──────────────────────────────────────────────────────────
    st.markdown("**OpenAI API Key**")
    env_key = os.getenv("OPENAI_API_KEY", "")
    if env_key:
        st.success("✅ API key loaded from .env file")
    else:
        st.session_state["sidebar_api_key"] = st.text_input(
            "Paste your OpenAI API key",
            type="password",
            placeholder="sk-...",
            help="Your key is never stored — it lives only in memory for this session.",
        )

    st.divider()

    # ── Document ingestion ───────────────────────────────────────────────
    st.markdown("**📂 Knowledge Base**")
    docs_folder = st.text_input(
        "Documents folder path",
        value="documents",
        help="Relative or absolute path to the folder containing your policy files.",
    )

    col_ingest, col_clear = st.columns(2)

    with col_ingest:
        ingest_btn = st.button("➕ Ingest Docs", use_container_width=True)

    with col_clear:
        clear_btn = st.button("🗑️ Clear DB", use_container_width=True, type="secondary")

    # ── Stats ────────────────────────────────────────────────────────────
    if "vector_store" in st.session_state:
        chunk_count = st.session_state.vector_store.count()
        st.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-num">{chunk_count}</div>'
            f'<div class="stat-label">chunks in vector DB</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**🛠️  Model Settings**")
    model_choice = st.selectbox(
        "LLM model",
        ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        index=0,
        help="gpt-4o gives the most accurate answers. gpt-4o-mini is faster and cheaper.",
    )
    st.session_state["model_choice"] = model_choice

    st.divider()
    st.markdown(
        "<small>📖 See README.md for setup instructions</small>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="cop-header">📚 Policy &amp; Process Copilot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="cop-subtitle">Agentic RAG assistant — ask anything about company policies</p>',
    unsafe_allow_html=True,
)
st.divider()


# ════════════════════════════════════════════════════════════════════════════
# BOOT-UP CHECKS
# ════════════════════════════════════════════════════════════════════════════

api_key = get_api_key()

if not api_key:
    st.warning(
        "🔑 **No API key found.**  "
        "Either create a `.env` file with `OPENAI_API_KEY=sk-...`  "
        "or paste your key in the sidebar."
    )
    st.stop()

# Initialize the AI system (only runs once per session)
initialize_system(api_key)

# Reinitialise if the user changed the model mid-session
current_model = st.session_state.get("model_choice", "gpt-4o")
if st.session_state.get("_last_model") != current_model:
    from rag.agents import AgenticRAGWorkflow
    st.session_state.workflow = AgenticRAGWorkflow(
        api_key=api_key,
        vector_store=st.session_state.vector_store,
        model=current_model,
    )
    st.session_state["_last_model"] = current_model


# ════════════════════════════════════════════════════════════════════════════
# DOCUMENT INGESTION LOGIC  (triggered by sidebar buttons)
# ════════════════════════════════════════════════════════════════════════════

if ingest_btn:
    from rag.document_loader import DocumentLoader

    folder_path = Path(docs_folder)
    if not folder_path.exists():
        st.sidebar.error(f"❌ Folder not found: {docs_folder}")
    else:
        with st.sidebar:
            with st.spinner("📥 Ingesting documents…"):
                loader = DocumentLoader(str(folder_path))
                chunks = loader.load_all()

                if not chunks:
                    st.warning("⚠️  No supported files (TXT / PDF / MD) found.")
                else:
                    st.session_state.vector_store.add_documents(chunks)
                    st.session_state.ingested = True
                    st.success(f"✅ Ingested {len(chunks)} chunks from {docs_folder}")
                    logging.info(f"Ingested {len(chunks)} chunks from '{docs_folder}'")
            st.rerun()

if clear_btn:
    if "vector_store" in st.session_state:
        st.session_state.vector_store.clear()
        st.session_state.ingested = False
        st.sidebar.success("🗑️  Vector DB cleared.")
        logging.info("Vector DB cleared by user.")
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# INGESTION REMINDER
# ════════════════════════════════════════════════════════════════════════════

if not st.session_state.get("ingested", False):
    st.info(
        "📂 **No documents ingested yet.**  "
        "Click **➕ Ingest Docs** in the sidebar to load the policy files."
    )


# ════════════════════════════════════════════════════════════════════════════
# CHAT HISTORY
# ════════════════════════════════════════════════════════════════════════════

for msg in st.session_state.get("chat_history", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show agent trace & citations for assistant messages
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]

            # Action badge
            action = meta.get("action", "").upper()
            if action:
                badge_cls = f"badge-{action.lower().replace(' ', '_')}"
                st.markdown(
                    f'<span class="badge {badge_cls}">{action}</span>',
                    unsafe_allow_html=True,
                )

            # Citations
            citations = meta.get("citations", [])
            if citations:
                st.markdown("**Sources used:**")
                for cite in citations:
                    st.markdown(
                        f'<div class="citation-card">📄 {cite}</div>',
                        unsafe_allow_html=True,
                    )

            # Agent reasoning trace (collapsible)
            agent_log = meta.get("agent_log", [])
            if agent_log:
                with st.expander("🔍 Agent reasoning trace", expanded=False):
                    trace_html = "<br>".join(agent_log)
                    st.markdown(
                        f'<div class="trace-box">{trace_html}</div>',
                        unsafe_allow_html=True,
                    )


# ════════════════════════════════════════════════════════════════════════════
# CHAT INPUT  —  this is the main interaction loop
# ════════════════════════════════════════════════════════════════════════════

user_input = st.chat_input("Ask a question about company policies…")

if user_input:
    # 1. Display the user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # 2. Run the agentic RAG workflow
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking…"):
            result = st.session_state.workflow.run(user_input)

        answer    = result.get("final_answer", "I could not generate an answer.")
        citations = result.get("citations", [])
        action    = result.get("action", "").upper()
        agent_log = result.get("agent_log", [])

        # 3. Render the answer
        st.markdown(answer)

        # 4. Action badge
        if action:
            badge_cls = f"badge-{action.lower().replace(' ', '_')}"
            st.markdown(
                f'<span class="badge {badge_cls}">{action}</span>',
                unsafe_allow_html=True,
            )

        # 5. Citations
        if citations:
            st.markdown("**Sources used:**")
            for cite in citations:
                st.markdown(
                    f'<div class="citation-card">📄 {cite}</div>',
                    unsafe_allow_html=True,
                )

        # 6. Agent reasoning trace
        if agent_log:
            with st.expander("🔍 Agent reasoning trace", expanded=False):
                trace_html = "<br>".join(agent_log)
                st.markdown(
                    f'<div class="trace-box">{trace_html}</div>',
                    unsafe_allow_html=True,
                )

    # 7. Save to chat history (with meta for re-render)
    st.session_state.chat_history.append(
        {
            "role":    "assistant",
            "content": answer,
            "meta": {
                "action":    action,
                "citations": citations,
                "agent_log": agent_log,
            },
        }
    )


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    "<small>Policy &amp; Process Copilot — Powered by LangChain · LangGraph · "
    "ChromaDB · OpenAI · Streamlit</small>",
    unsafe_allow_html=True,
)
