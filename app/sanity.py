"""
streamlit_app.py — Web UI for the Local Agentic RAG Chatbot.

Run:
    streamlit run streamlit_app.py
"""

import os
import re
import shutil
import streamlit as st
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────

UPLOADS_DIR      = "uploads"
SAMPLE_DOCS_DIR  = "sample_docs"
FAISS_INDEX_PATH = os.path.join("artifacts", "faiss.index")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# ── security helpers (mirrors cli.py) ─────────────────────────────────────────

_INJECTION_RE = re.compile(
    r"(?i)("
    r"ignore prior instructions|reveal secrets?|show system prompt"
    r"|dump memory|expose internal|bypass rules?|give me hidden"
    r"|print hidden|confidential data|api[_\s\-]?key|secret[_\s\-]?key"
    r")"
)
_CLASSIFIED_RE = re.compile(
    r"(?i)\b(phone|email|cro|api\s?key|token|password|secret|contact|reach|call\s+me)\b"
)
_MEMORY_ONLY_RE = re.compile(
    r"(?i)\b("
    r"i\s+prefer|i\s+like|i\s+love|i\s+enjoy|i'?m\s+into|i\s+am\s+into"
    r"|my\s+name\s+is|call\s+me|i'?m\s+an?|i\s+am\s+an?"
    r"|my\s+role\s+is|i\s+work\s+as|i\s+am\s+working\s+as|send\s+me"
    r"|don'?t\s+explain|don'?t\s+summarize|do\s+not\s+explain"
    r"|do\s+not\s+summarize|no\s+summary|no\s+explanation|no\s+briefing|no\s+brief"
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

def _is_malicious(text):      return bool(_INJECTION_RE.search(text))
def _is_classified(text):     return bool(_CLASSIFIED_RE.search(text))
def _is_memory_only(text):
    frags = [f.strip() for f in re.split(r"\band\b", text, flags=re.IGNORECASE) if f.strip()]
    return any(bool(_MEMORY_ONLY_RE.search(f)) for f in frags)
def _is_memory_question(text): return bool(_MEMORY_QUESTION_RE.search(text))


# ── index management ───────────────────────────────────────────────────────────

def _all_source_dirs():
    """
    Priority: uploads/ if it has files, otherwise fall back to sample_docs/.
    Never mix both — uploaded docs are the user's intent.
    """
    uploads = Path(UPLOADS_DIR)
    supported = {".pdf", ".txt", ".md"}
    has_uploads = uploads.exists() and any(
        f.suffix.lower() in supported for f in uploads.rglob("*") if f.is_file()
    )
    if has_uploads:
        return [UPLOADS_DIR]
    return [SAMPLE_DOCS_DIR]


def _build_index():
    from app.ingest import ingest
    from app.chunk  import chunk
    from app.embed  import build_index

    all_docs, all_chunks = [], []
    for d in _all_source_dirs():
        docs = ingest(d)
        all_docs.extend(docs)
        all_chunks.extend(chunk(docs))

    if not all_chunks:
        st.error("No documents found in sample_docs/ or uploads/.")
        return None, []

    index, meta = build_index(all_chunks)
    return index, meta


@st.cache_resource(show_spinner="Loading index …")
def _load_or_build():
    if not Path(FAISS_INDEX_PATH).exists():
        return _build_index()
    from app.embed import load_index
    return load_index()


def _force_rebuild():
    """Clear cache and rebuild."""
    st.cache_resource.clear()
    return _build_index()


# ── memory helpers ─────────────────────────────────────────────────────────────

def _read_user_memory():
    from app.memory import load_memory, USER_MEMORY_PATH
    raw = load_memory(USER_MEMORY_PATH).strip()
    facts = [l.lstrip("- ").strip() for l in raw.splitlines() if l.strip().startswith("-")]
    return facts


def _answer_from_memory():
    facts = _read_user_memory()
    if not facts:
        return "I don't have anything stored in your memory yet."
    return "Based on what I know about you:\n" + "\n".join(f"• {f}" for f in facts)


# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    model  = st.selectbox("Ollama model", ["mistral", "llama3", "phi3", "gemma"], index=0)
    top_k  = st.slider("Top-K retrieval chunks", min_value=1, max_value=10, value=5)

    st.divider()

    if st.button("🔄 Reindex", use_container_width=True):
        with st.spinner("Rebuilding index …"):
            idx, cks = _force_rebuild()
            if idx:
                st.session_state["index"]  = idx
                st.session_state["chunks"] = cks
                st.session_state["rag_cache"] = {}  # invalidate cache
                st.success(f"Index rebuilt — {idx.ntotal} vectors")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state["messages"]  = []
        st.session_state["rag_cache"] = {}
        st.rerun()

    st.divider()
    st.caption("📁 Upload documents")
    uploaded_files = st.file_uploader(
        "Add .pdf, .txt, or .md files",
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
            st.success(f"Saved: {', '.join(saved)}")
            with st.spinner("Reindexing with new files …"):
                idx, cks = _force_rebuild()
                if idx:
                    st.session_state["index"]  = idx
                    st.session_state["chunks"] = cks
                    st.session_state["rag_cache"] = {}  # invalidate — new docs = stale answers

    st.divider()
    with st.expander("🧠 My Memory"):
        facts = _read_user_memory()
        if facts:
            for f in facts:
                st.markdown(f"- {f}")
        else:
            st.caption("Nothing stored yet.")

    st.divider()
    with st.expander("⚡ Cache"):
        cache = st.session_state.get("rag_cache", {})
        st.caption(f"{len(cache)} cached response(s)")
        for k in list(cache.keys()):
            st.markdown(f"- `{k[:60]}...`" if len(k) > 60 else f"- `{k}`")

# ── main area ──────────────────────────────────────────────────────────────────

st.title("🤖 Local Agentic RAG Chatbot")
st.caption("Fully offline · Citation-grounded · Persistent memory · Powered by Ollama + FAISS")

# show which source is active
_active_dirs = _all_source_dirs()
if UPLOADS_DIR in _active_dirs:
    st.info("📂 Answering from your **uploaded documents** (`uploads/`)", icon="📤")
else:
    st.info("📂 Answering from **sample documents** (`sample_docs/`) — upload files to use your own.", icon="📁")

# load index into session state on first run
if "index" not in st.session_state or "chunks" not in st.session_state:
    with st.spinner("Loading index …"):
        idx, cks = _load_or_build()
        st.session_state["index"]  = idx
        st.session_state["chunks"] = cks

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "rag_cache" not in st.session_state:
    st.session_state["rag_cache"] = {}  # query -> {bot_response, citations}

# render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📄 Sources"):
                for c in msg["citations"]:
                    st.markdown(f"- **{c['filename']}** · page {c['page']} · `{c['chunk_id']}`")
        if msg.get("memory_note"):
            st.info(f"🧠 {msg['memory_note']}")

# ── chat input ─────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask a question or tell me about yourself …")

if user_input:
    # display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    bot_response  = ""
    citations     = []
    memory_note   = ""

    # ── prompt injection guard ─────────────────────────────────────────────────
    if _is_malicious(user_input):
        bot_response = "I cannot assist with that request."

    # ── classified field guard ─────────────────────────────────────────────────
    elif _is_classified(user_input):
        bot_response = "I can't share confidential or classified details."

    # ── memory question ────────────────────────────────────────────────────────
    elif _is_memory_question(user_input):
        bot_response = _answer_from_memory()

    # ── memory-only input ──────────────────────────────────────────────────────
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

    # ── normal RAG query ───────────────────────────────────────────────────────
    else:
        index  = st.session_state.get("index")
        chunks = st.session_state.get("chunks", [])

        if index is None or index.ntotal == 0:
            bot_response = "No documents indexed yet. Please upload files or check sample_docs/."
        else:
            from app.retrieve import retrieve
            from app.rag      import answer
            from app.memory   import maybe_write_memory

            with st.spinner("Thinking …"):
                hits = retrieve(user_input, index, chunks, top_k=top_k)
                if not hits:
                    bot_response = "I couldn't find relevant content in the documents."
                else:
                    out = answer(user_input, hits, model=model)
                    # strip inline [source:...] markers from displayed answer
                    import re as _re
                    bot_response = _re.sub(
                        r"\[source:[^\]]+\]", "", out["answer"]
                    ).strip()
                    citations = out.get("citations", [])
                    # also collect any citations embedded in answer but missing from list
                    _cit_re = _re.compile(r"\[source:([^#\]]+)#([^\s\]]+)\s+p=(\d+)\]")
                    seen_ids = {c["chunk_id"] for c in citations}
                    for _m in _cit_re.finditer(out["answer"]):
                        if _m.group(2) not in seen_ids:
                            citations.append({
                                "filename": _m.group(1),
                                "chunk_id": _m.group(2),
                                "page":     int(_m.group(3)),
                            })
                            seen_ids.add(_m.group(2))

                    # memory on user input only
                    mem = maybe_write_memory(user_input, "")
                    if mem.get("written"):
                        memory_note = f"Memory updated ({mem['target']}): {mem['summary']}"

    # render bot response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
        if citations:
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