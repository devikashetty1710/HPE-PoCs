"""
app.py — Streamlit Frontend for AI Document Retrieval Agent

Run with:
    streamlit run app.py

Features:
  - Chat-style query interface with message history
  - Real-time tool trace (Thought / Action / Observation)
  - Source file attribution per answer
  - Upload documents from the UI (auto-indexed into FAISS)
"""

import os
import subprocess
import streamlit as st
from pathlib import Path

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Document Retrieval Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0F1117; }

.user-bubble {
    background: #1E3A5F;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 8px 0 8px auto;
    color: #FFFFFF;
    max-width: 80%;
    font-size: 15px;
}
.agent-bubble {
    background: #1A1F2E;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    margin: 8px auto 8px 0;
    color: #E8EAF6;
    max-width: 85%;
    border-left: 3px solid #2E86AB;
    font-size: 15px;
}
.source-badge {
    background: #0D2137;
    border: 1px solid #2E86AB;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #2E86AB;
    display: inline-block;
    margin: 6px 4px 0 0;
}
.tool-badge {
    background: #1A0D2E;
    border: 1px solid #7B2D8B;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #CE93D8;
    display: inline-block;
    margin: 6px 4px 0 0;
}
.trace-box {
    background: #0D1117;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    color: #8B949E;
    margin-top: 8px;
    white-space: pre-wrap;
}
.metric-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    margin-bottom: 10px;
}
.metric-val { font-size: 28px; font-weight: bold; color: #2E86AB; }
.metric-lbl { font-size: 12px; color: #8B949E; margin-top: 4px; }

section[data-testid="stSidebar"] {
    background-color: #0D1117;
    border-right: 1px solid #21262D;
}
</style>
""", unsafe_allow_html=True)


# ── FAISS helpers ─────────────────────────────────────────────
FAISS_DIR = "faiss_db"


@st.cache_resource(show_spinner=False)
def _load_faiss_store():
    """Load FAISS vectorstore once and cache it."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    if not os.path.exists(FAISS_DIR):
        return None

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_indexed_info():
    """Return (list of source filenames, total chunk count)."""
    vs = _load_faiss_store()
    if vs is None:
        return [], 0

    try:
        docs = vs.similarity_search("the", k=200)
        sources = sorted(set(
            d.metadata.get("source_file", "unknown") for d in docs
        ))
        total = len(vs.index_to_docstore_id)
        return sources, total
    except Exception:
        return [], 0


# ── Lazy init — build agent once and cache ───────────────────
@st.cache_resource(show_spinner="🚀 Starting agent...")
def get_agent():
    from agent.agent_core import build_agent
    return build_agent()


# ── Session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤖 AI Doc Agent")
    st.markdown("**LangChain · Groq · FAISS**")
    st.divider()

    # Stats
    indexed_sources, total_chunks = get_indexed_info()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{total_chunks}</div>
            <div class="metric-lbl">Chunks</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{len(indexed_sources)}</div>
            <div class="metric-lbl">Documents</div>
        </div>""", unsafe_allow_html=True)

    # Indexed files list
    st.markdown("### 📁 Indexed Documents")
    if indexed_sources:
        for src in indexed_sources:
            fname = Path(src).name
            ext = Path(src).suffix.lower().lstrip(".")
            icon = {
                "pdf": "📕",
                "txt": "📄",
                "json": "📋",
                "csv": "📊",
                "md": "📝"
            }.get(ext, "📄")
            st.markdown(f"{icon} `{fname}`")
    else:
        st.caption("No documents indexed yet. Run `python ingest.py` first.")

    st.divider()

    # Upload documents
    st.markdown("### ⬆️ Upload Documents")
    st.caption("PDF, TXT, JSON, CSV, MD")

    uploaded_files = st.file_uploader(
        "Drop files here",
        accept_multiple_files=True,
        type=["pdf", "txt", "json", "csv", "md"],
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("📥 Index Files", use_container_width=True, type="primary"):
            docs_dir = Path("./sample_docs")
            docs_dir.mkdir(exist_ok=True)

            saved = []
            for uf in uploaded_files:
                dest = docs_dir / uf.name
                with open(dest, "wb") as f:
                    f.write(uf.getbuffer())
                saved.append(uf.name)

            with st.spinner(f"Indexing {len(saved)} file(s)..."):
                try:
                    result = subprocess.run(
                        ["python", "ingest.py"],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    if result.returncode == 0:
                        _load_faiss_store.clear()
                        get_agent.clear()
                        st.success(f"✅ {len(saved)} file(s) indexed successfully.")
                        st.rerun()
                    else:
                        st.error(f"Indexing failed:\n{result.stderr[:300]}")
                except subprocess.TimeoutExpired:
                    st.error("Indexing timed out. Try fewer or smaller files.")
                except Exception as e:
                    st.error(f"Error during indexing: {str(e)}")

    st.divider()

    # Settings
    st.markdown("### ⚙️ Settings")
    show_trace = st.toggle("Show tool trace", value=True)
    show_sources = st.toggle("Show source files", value=True)

    st.divider()

    if os.path.exists(FAISS_DIR):
        st.success("✅ FAISS index loaded")
    else:
        st.error("⚠️ No FAISS index found. Run `python ingest.py`")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Devika Shetty · HPE POC-2 · 2026")


# ══════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════
st.markdown("## 🤖 AI Document Retrieval Agent")
st.markdown(
    "Ask anything. I search **local documents first** — "
    "falling back to Wikipedia only when the answer isn't found locally."
)

# Tool legend
c1, c2, c3, c4 = st.columns(4)
c1.markdown("🔵 **LocalDocumentSearch**")
c2.markdown("🟣 **PDFReader / FileReader**")
c3.markdown("🟠 **WikipediaSearch**")
c4.markdown("🔴 **WebURLLoader**")

st.divider()

# ── Chat history ──────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 40px; color: #8B949E;">
        <div style="font-size: 48px;">💬</div>
        <div style="font-size: 18px; margin-top: 12px;">
            Ask a question to get started
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">🧑 {msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="agent-bubble">🤖 {msg["content"]}</div>',
            unsafe_allow_html=True
        )

        badges = ""
        if show_sources and msg.get("sources"):
            for src in msg["sources"]:
                badges += f'<span class="source-badge">📄 {src}</span>'
        if msg.get("tools"):
            badges += f'<span class="tool-badge">🔧 {" → ".join(msg["tools"])}</span>'
        if badges:
            st.markdown(badges, unsafe_allow_html=True)

        if show_trace and msg.get("trace"):
            with st.expander("🔍 View reasoning trace", expanded=False):
                st.markdown(
                    f'<div class="trace-box">{msg["trace"]}</div>',
                    unsafe_allow_html=True
                )

st.divider()

# ── Input form ────────────────────────────────────────────────
with st.form(key="chat_form", clear_on_submit=True):
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        user_input = st.text_input(
            "question",
            placeholder="Type your question here...",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button(
            "Send ➤",
            use_container_width=True,
            type="primary"
        )


# ── Helper: parse agent result ────────────────────────────────
def parse_result(result):
    steps = result.get("intermediate_steps", [])
    trace_lines = []
    sources = set()
    tools_used = []

    for action, observation in steps:
        tools_used.append(action.tool)
        obs_str = str(observation)
        if len(obs_str) > 600:
            obs_str = obs_str[:600] + "... [truncated]"

        trace_lines.append(f"ACTION : {action.tool}")
        trace_lines.append(f"INPUT  : {action.tool_input}")
        trace_lines.append(f"RESULT : {obs_str}")
        trace_lines.append("")

        for line in obs_str.split("\n"):
            if "Source:" in line:
                src = line.split("Source:")[-1].strip().rstrip("]").rstrip(")")
                src = src.split("|")[0].strip()
                if src and len(src) < 120:
                    sources.add(
                        Path(src).name if ("/" in src or "\\" in src) else src
                    )

    return "\n".join(trace_lines), list(sources), tools_used


# ── Process query ─────────────────────────────────────────────
if submitted and user_input and user_input.strip():
    query_to_run = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": query_to_run})

    if not os.path.exists(FAISS_DIR):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ No FAISS index found. Please run `python ingest.py` first, then refresh.",
            "sources": [],
            "tools": [],
            "trace": "",
        })
        st.rerun()

    with st.spinner("🔍 Searching..."):
        try:
            agent = get_agent()
            result = agent.invoke({"input": query_to_run})
            answer = result.get("output", "No answer returned.")
            trace_str, sources, tools_used = parse_result(result)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "tools": tools_used,
                "trace": trace_str,
            })

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate_limit" in err.lower():
                msg = "⚠️ Groq rate limit hit. Please wait 30 seconds and try again."
            elif "API_KEY" in err or "authentication" in err.lower():
                msg = "⚠️ API key error. Check your .env file."
            elif "FAISS" in err or "faiss" in err:
                msg = "⚠️ FAISS index error. Run `python ingest.py` and refresh."
            else:
                msg = f"⚠️ Error: {err[:300]}"

            st.session_state.messages.append({
                "role": "assistant",
                "content": msg,
                "sources": [],
                "tools": [],
                "trace": "",
            })

    st.rerun()