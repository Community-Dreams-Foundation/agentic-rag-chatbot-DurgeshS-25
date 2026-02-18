"""
streamlit_app.py — Web UI for the Local Agentic RAG Chatbot.

Run:
    streamlit run streamlit_app.py
"""

import os
import re
import streamlit as st
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────

UPLOADS_DIR      = "uploads"
FAISS_INDEX_PATH = os.path.join("artifacts", "faiss.index")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# ── cleanup on session end ─────────────────────────────────────────────────────

def _cleanup_uploads():
    supported = {".pdf", ".txt", ".md"}
    uploads = Path(UPLOADS_DIR)
    if uploads.exists():
        for f in uploads.rglob("*"):
            if f.is_file() and f.suffix.lower() in supported:
                try: f.unlink()
                except Exception: pass
    for artifact in ["artifacts/faiss.index", "artifacts/chunks.jsonl", "artifacts/embed_meta.json"]:
        try:
            if Path(artifact).exists(): Path(artifact).unlink()
        except Exception: pass

if "cleanup_registered" not in st.session_state:
    st.session_state["cleanup_registered"] = True
    import atexit
    atexit.register(_cleanup_uploads)

# ── security helpers ───────────────────────────────────────────────────────────

_INJECTION_RE = re.compile(
    r"(?i)("
    r"ignore prior instructions|reveal secrets?|show system prompt"
    r"|dump memory|expose internal|bypass rules?|give me hidden"
    r"|print hidden|confidential data|api[_\s\-]?key|secret[_\s\-]?key"
    r")"
)
_CLASSIFIED_RE = re.compile(
    r"(?i)\b(phone\s+number|email\s+address|cro|api\s?key|token|password|secret)\b"
)
_MEMORY_ONLY_RE = re.compile(
    r"(?i)\b("
    r"i\s+prefer|i\s+like|i\s+love|i\s+enjoy|i'?m\s+into|i\s+am\s+into"
    r"|my\s+name\s+is|call\s+me|i'?m\s+an?|i\s+am\s+an?"
    r"|my\s+role\s+is|i\s+work\s+as|i\s+am\s+working\s+as|send\s+me"
    r"|don'?t\s+explain|don'?t\s+summarize|do\s+not\s+explain"
    r"|do\s+not\s+summarize|no\s+summary|no\s+explanation|no\s+briefing|no\s+brief"
    r"|our\s+team\s+(uses?|prefers?|follows?|relies?|always|never)"
    r"|we\s+use\b|the\s+bottleneck\s+is|everyone\s+uses?"
    r")"
)
_MEMORY_QUESTION_RE = re.compile(
    r"(?i)\b("
    r"what\s+(do|did|don'?t)\s+i\s+(like|love|enjoy|prefer|hate|want|need)"
    r"|what\s+is\s+my\s+(name|role|job|preference|hobby)"
    r"|who\s+am\s+i|what\s+are\s+my\s+(preferences?|interests?|hobbies|goals?)"
    r"|do\s+you\s+(know|remember)\s+(me|my)"
    r")"
)

def _is_malicious(text):       return bool(_INJECTION_RE.search(text))
def _is_classified(text):      return bool(_CLASSIFIED_RE.search(text))
def _is_memory_question(text): return bool(_MEMORY_QUESTION_RE.search(text))

def _is_memory_only(text):
    frags = [f.strip() for f in re.split(r"\band\b", text, flags=re.IGNORECASE) if f.strip()]
    return any(bool(_MEMORY_ONLY_RE.search(f)) for f in frags)

# ── index management ───────────────────────────────────────────────────────────

def _has_uploads() -> bool:
    supported = {".pdf", ".txt", ".md"}
    uploads = Path(UPLOADS_DIR)
    return uploads.exists() and any(
        f.suffix.lower() in supported for f in uploads.rglob("*") if f.is_file()
    )

def _build_index():
    from app.ingest import ingest
    from app.chunk  import chunk
    from app.embed  import build_index
    from collections import Counter

    if not _has_uploads():
        st.warning("No documents in uploads/ — please upload files first.")
        return None, []

    docs = ingest(UPLOADS_DIR)
    chunks = chunk(docs, chunk_size=500, overlap=100)

    if not chunks:
        st.error("No text could be extracted from uploaded files.")
        return None, []

    # terminal summary
    print(f"\n{'='*60}")
    print(f"  DOCUMENT INDEXING SUMMARY")
    print(f"{'='*60}")
    for fname, count in Counter(c['filename'] for c in chunks).items():
        print(f"  📄 {fname} → {count} chunks")
    print(f"\n  Total chunks indexed: {len(chunks)}")
    print(f"{'─'*60}")
    print(f"  CHUNK PREVIEW (first 5)")
    print(f"{'─'*60}")
    for i, c in enumerate(chunks[:5]):
        print(f"\n  [{i+1}] {c['filename']}  |  page {c['page']}  |  {len(c['text'])} chars")
        print(f"  ID    : {c['chunk_id']}")
        print(f"  Text  : {c['text'][:120].replace(chr(10), ' ').strip()}...")
    if len(chunks) > 5:
        print(f"\n  ... and {len(chunks) - 5} more chunks")
    print(f"\n{'='*60}\n")

    index, meta = build_index(chunks)
    return index, meta

@st.cache_resource(show_spinner="Loading index …")
def _load_cached():
    from app.embed import load_index
    return load_index()

def _force_rebuild():
    st.cache_resource.clear()
    for f in ["artifacts/faiss.index", "artifacts/chunks.jsonl", "artifacts/embed_meta.json"]:
        if Path(f).exists(): Path(f).unlink()
    return _build_index()

# ── memory helpers ─────────────────────────────────────────────────────────────

def _answer_from_memory() -> str:
    from app.memory import load_memory, USER_MEMORY_PATH
    contents = load_memory(USER_MEMORY_PATH).strip()
    if not contents:
        return "I don't have anything stored in your memory yet."
    facts = [l.lstrip("- ").strip() for l in contents.splitlines() if l.strip().startswith("-")]
    if not facts:
        return "I don't have anything stored in your memory yet."
    return "Based on what I know about you:\n" + "\n".join(f"• {f}" for f in facts)

def _is_refusal(text: str) -> bool:
    return any(phrase in text for phrase in [
        "I don't have enough information",
        "I cannot assist",
        "I can't share",
        "I couldn't find",
        "No documents indexed",
        "Got it",
        "I'll remember",
    ])

# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    h1 { font-size: 2rem !important; font-weight: 700 !important; }
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2d3047;
    }
    .stButton > button {
        border-radius: 8px !important;
        border: 1px solid #2d3047 !important;
        background-color: #1e2130 !important;
        color: #e0e0e0 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #2d3047 !important;
        border-color: #4a5080 !important;
    }
    .streamlit-expanderHeader {
        background-color: #1a1d27 !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
    }
    .stAlert { border-radius: 8px !important; font-size: 0.85rem !important; }
    .stChatInput { border-radius: 12px !important; border: 1px solid #2d3047 !important; }
    .stChatInput textarea {
        border-radius: 12px !important;
        background-color: #1a1d27 !important;
        border: none !important;
    }
    .status-banner {
        background: linear-gradient(135deg, #1a1d27, #1e2535);
        border: 1px solid #2d3047;
        border-left: 3px solid #4a90d9;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 16px;
        font-size: 0.85rem;
        color: #a0b4cc;
    }
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 80px 20px;
        text-align: center;
    }
    .empty-state .icon { font-size: 3rem; margin-bottom: 16px; }
    .empty-state h3 { font-size: 1.3rem; color: #6b7db3; margin-bottom: 8px; font-weight: 600; }
    .empty-state p { font-size: 0.9rem; color: #4a5080; margin: 4px 0; line-height: 1.6; }
    .empty-state .hints {
        margin-top: 24px;
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        justify-content: center;
    }
    .empty-state .hint-chip {
        background: #1a1d27;
        border: 1px solid #2d3047;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.8rem;
        color: #6b7db3;
    }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model = st.selectbox("🤖 Ollama model", ["mistral", "llama3", "phi3", "gemma"], index=0)
    top_k = st.slider("🔍 Retrieval chunks (Top-K)", min_value=1, max_value=10, value=7)

    st.divider()
    st.markdown("**🗂️ Index Controls**")

    if st.button("🔄 Reindex", use_container_width=True):
        with st.spinner("Rebuilding index …"):
            idx, cks = _force_rebuild()
            if idx:
                st.session_state["index"]  = idx
                st.session_state["chunks"] = cks
                st.session_state["rag_cache"] = {}
                st.success(f"✅ Index rebuilt — {idx.ntotal} vectors")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state["messages"]  = []
        st.session_state["rag_cache"] = {}
        st.rerun()

    if st.button("🧹 Clear memory", use_container_width=True):
        from app.memory import USER_MEMORY_PATH, COMPANY_MEMORY_PATH
        for path in [USER_MEMORY_PATH, COMPANY_MEMORY_PATH]:
            if Path(path).exists():
                Path(path).write_text("", encoding="utf-8")
        st.success("✅ Memory cleared!")
        st.rerun()

    st.divider()
    st.markdown("**📁 Upload Documents**")
    uploaded_files = st.file_uploader(
        "Drag & drop or browse",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        saved = []
        for f in uploaded_files:
            dest = Path(UPLOADS_DIR) / f.name
            dest.write_bytes(f.read())
            saved.append(f.name)
        if saved:
            st.success(f"✅ Saved: {', '.join(saved)}")
            with st.spinner("Indexing …"):
                idx, cks = _force_rebuild()
                if idx:
                    st.session_state["index"]  = idx
                    st.session_state["chunks"] = cks
                    st.session_state["rag_cache"] = {}
                    st.rerun()

    st.divider()
    st.caption("🔒 Fully local · No cloud APIs · Data stays on your machine")

# ── main area ──────────────────────────────────────────────────────────────────

st.title("🤖 Local Agentic RAG Chatbot")
st.caption("Fully offline · Citation-grounded · Persistent memory · Powered by Ollama + FAISS")

# load index on first run — only if it already exists
if "index" not in st.session_state:
    if Path(FAISS_INDEX_PATH).exists():
        with st.spinner("Loading index …"):
            idx, cks = _load_cached()
            st.session_state["index"]  = idx
            st.session_state["chunks"] = cks
    else:
        st.session_state["index"]  = None
        st.session_state["chunks"] = []

if "messages"  not in st.session_state: st.session_state["messages"]  = []
if "rag_cache" not in st.session_state: st.session_state["rag_cache"] = {}

# status banner
if not _has_uploads() or st.session_state.get("index") is None:
    st.markdown("""
    <div class="status-banner">
        👋 &nbsp;<strong>Welcome!</strong> &nbsp;·&nbsp;
        Upload your documents using the sidebar to get started.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-banner">
        📤 &nbsp;<strong>Answering from your uploaded documents</strong> &nbsp;·&nbsp;
        <code>uploads/</code>
    </div>
    """, unsafe_allow_html=True)

# chat history
if not st.session_state["messages"]:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🤖</div>
        <h3>Local Agentic RAG Chatbot</h3>
        <p>Upload your documents using the sidebar, then ask questions.</p>
        <p>All answers are grounded in your documents with citations.</p>
        <div class="hints">
            <span class="hint-chip">📄 Upload PDF or TXT</span>
            <span class="hint-chip">🔍 Ask questions</span>
            <span class="hint-chip">🧠 Tell me about yourself</span>
            <span class="hint-chip">🔒 Fully offline</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations") and not _is_refusal(msg["content"]):
            with st.expander("📄 Sources"):
                for c in msg["citations"]:
                    st.markdown(f"- **{c['filename']}** · page {c['page']} · `{c['chunk_id']}`")
        if msg.get("memory_note"):
            st.info(f"🧠 {msg['memory_note']}")

# ── chat input ─────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask a question or tell me about yourself …")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    bot_response = ""
    citations    = []
    memory_note  = ""

    if _is_malicious(user_input):
        bot_response = "I cannot assist with that request."

    elif _is_classified(user_input):
        bot_response = "I can't share confidential or classified details."

    elif _is_memory_question(user_input):
        bot_response = _answer_from_memory()

    elif _is_memory_only(user_input):
        from app.memory import maybe_write_memory
        frags = [f.strip() for f in re.split(r"\band\b", user_input, flags=re.IGNORECASE) if f.strip()]
        notes = []
        for frag in frags:
            r = maybe_write_memory(frag, "")
            if r.get("written"):
                notes.append(f"Stored ({r['target']}): {r['summary']}")
            elif r.get("should_write"):
                notes.append(f"Already noted: {r['summary']}")
        bot_response = "Got it — I'll remember that."
        if notes:
            memory_note = " · ".join(notes)

    else:
        index  = st.session_state.get("index")
        chunks = st.session_state.get("chunks", [])
        cache  = st.session_state["rag_cache"]
        cache_key = f"{' '.join(user_input.lower().split())}|{model}|{top_k}"

        if index is None or index.ntotal == 0:
            bot_response = "No documents indexed yet — please upload files using the sidebar."

        elif cache_key in cache:
            cached       = cache[cache_key]
            bot_response = cached["bot_response"]
            citations    = cached["citations"]
            st.toast("⚡ Served from cache", icon="💾")

        else:
            from app.retrieve import retrieve
            from app.rag      import answer
            from app.memory   import maybe_write_memory
            import re as _re

            with st.spinner("Thinking …"):
                hits = retrieve(user_input, index, chunks, top_k=top_k)
                if not hits:
                    bot_response = "I couldn't find relevant content in the documents."
                else:
                    out          = answer(user_input, hits, model=model)
                    bot_response = _re.sub(r"\[source:[^\]]+\]", "", out["answer"]).strip()
                    citations    = out.get("citations", [])

                    _cit_re  = _re.compile(r"\[source:([^#\]]+)#([^\s\]]+)\s+p=(\d+)\]")
                    seen_ids = {c["chunk_id"] for c in citations}
                    for _m in _cit_re.finditer(out["answer"]):
                        if _m.group(2) not in seen_ids:
                            citations.append({
                                "filename": _m.group(1),
                                "chunk_id": _m.group(2),
                                "page":     int(_m.group(3)),
                            })
                            seen_ids.add(_m.group(2))

                    cache[cache_key] = {"bot_response": bot_response, "citations": citations}

                    mem = maybe_write_memory(user_input, "")
                    if mem.get("written"):
                        memory_note = f"Memory updated ({mem['target']}): {mem['summary']}"

    with st.chat_message("assistant"):
        st.markdown(bot_response)
        if citations and not _is_refusal(bot_response):
            with st.expander("📄 Sources"):
                for c in citations:
                    st.markdown(f"- **{c['filename']}** · page {c['page']} · `{c['chunk_id']}`")
        if memory_note:
            st.info(f"🧠 {memory_note}")

    st.session_state["messages"].append({
        "role":        "assistant",
        "content":     bot_response,
        "citations":   citations,
        "memory_note": memory_note,
    })