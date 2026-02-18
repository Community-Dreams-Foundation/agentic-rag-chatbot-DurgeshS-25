"""
cli.py — Entry-point CLI for the local Agentic RAG chatbot.

Commands:
    python -m app.cli ingest  [--source-dir sample_docs]
    python -m app.cli chat    [--source-dir sample_docs] [--model mistral]
                              [--top-k 5] [--rebuild]
    python -m app.cli sanity
"""

import argparse
import os
import re
import sys
from pathlib import Path

FAISS_INDEX_PATH = os.path.join("artifacts", "faiss.index")

# ── security gate ──────────────────────────────────────────────────────────────

_INJECTION_RE = re.compile(
    r"(?i)("
    r"ignore prior instructions"
    r"|reveal secrets?"
    r"|show system prompt"
    r"|dump memory"
    r"|expose internal"
    r"|bypass rules?"
    r"|give me hidden"
    r"|print hidden"
    r"|confidential data"
    r"|api[_\s\-]?key"
    r"|secret[_\s\-]?key"
    r")"
)

def _is_malicious_input(text: str) -> bool:
    """Return True if the input contains prompt injection or secret extraction patterns."""
    return bool(_INJECTION_RE.search(text))


# ── memory-only detection ──────────────────────────────────────────────────────

_MEMORY_ONLY_RE = re.compile(
    r"(?i)\b("
    r"i\s+prefer"
    r"|i\s+like"
    r"|i\s+love"
    r"|i\s+enjoy"
    r"|i'?m\s+into"
    r"|i\s+am\s+into"
    r"|my\s+name\s+is"
    r"|call\s+me"
    r"|i'?m\s+a[n]?"
    r"|i\s+am\s+a[n]?"
    r"|my\s+role\s+is"
    r"|i\s+work\s+as"
    r"|i\s+am\s+working\s+as"
    r"|send\s+me"
    r"|don'?t\s+explain"
    r"|don'?t\s+summarize"
    r"|do\s+not\s+explain"
    r"|do\s+not\s+summarize"
    r"|no\s+summary"
    r"|no\s+explanation"
    r"|no\s+briefing"
    r"|no\s+brief"
    r")"
)

def _is_memory_only_input(text: str) -> bool:
    """Return True if ANY fragment of the input is a user preference or identity statement."""
    fragments = [f.strip() for f in re.split(r"\band\b", text, flags=re.IGNORECASE) if f.strip()]
    return any(bool(_MEMORY_ONLY_RE.search(frag)) for frag in fragments)


# ── memory question detection ──────────────────────────────────────────────────

_MEMORY_QUESTION_RE = re.compile(
    r"(?i)\b("
    r"what\s+(do|did|don't|doesn't)\s+i\s+(like|love|enjoy|prefer|hate|want|need)"
    r"|what\s+is\s+my\s+(name|role|job|preference|hobby|interest)"
    r"|who\s+am\s+i"
    r"|what\s+are\s+my\s+(preferences?|interests?|hobbies|goals?)"
    r"|do\s+you\s+(know|remember)\s+(me|my)"
    r")"
)

def _is_memory_question(text: str) -> bool:
    """Return True if the user is asking about something stored in their memory."""
    return bool(_MEMORY_QUESTION_RE.search(text))


def _answer_from_memory(query: str) -> str:
    """Build a plain-text answer from USER_MEMORY.md contents."""
    from app.memory import load_memory, USER_MEMORY_PATH
    contents = load_memory(USER_MEMORY_PATH).strip()
    if not contents:
        return "I don't have anything stored in your memory yet."
    facts = [
        line.lstrip("- ").strip()
        for line in contents.splitlines()
        if line.strip().startswith("-")
    ]
    if not facts:
        return "I don't have anything stored in your memory yet."
    return "Based on what I know about you:\n" + "\n".join(f"  • {f}" for f in facts)


# ── help text ──────────────────────────────────────────────────────────────────

CHAT_HELP = """
┌─────────────────────────────────────────────────────┐
│         Local Agentic RAG Chatbot  🤖                │
│                                                     │
│  Slash commands:                                    │
│    /exit   or  /quit  — end the session             │
│    /memory            — show persistent memory      │
│    /reindex           — rebuild index from docs     │
│    /help              — show this message           │
└─────────────────────────────────────────────────────┘
"""

# ── index helpers ──────────────────────────────────────────────────────────────

def _build(source_dir: str):
    """Ingest → chunk → embed. Returns (index, chunks)."""
    from app.ingest import ingest
    from app.chunk  import chunk
    from app.embed  import build_index

    print(f"[cli] ingesting documents from '{source_dir}' …")
    docs = ingest(source_dir)
    if not docs:
        print("[cli] no documents found — add files to the source directory and retry.")
        sys.exit(1)
    chunks = chunk(docs)
    index, meta = build_index(chunks)
    return index, meta


def _load():
    """Load existing index from artifacts/."""
    from app.retrieve import load_retriever_assets
    return load_retriever_assets()


# ── commands ───────────────────────────────────────────────────────────────────

def cmd_ingest(args):
    _build(args.source_dir)
    print("[cli] index built successfully.")


def cmd_chat(args):
    from app.retrieve import retrieve
    from app.rag      import answer
    from app.memory   import load_memory, maybe_write_memory, USER_MEMORY_PATH, COMPANY_MEMORY_PATH

    # ── load or build index ────────────────────────────────────────────────────
    needs_build = args.rebuild or not Path(FAISS_INDEX_PATH).exists()
    if needs_build:
        index, chunks = _build(args.source_dir)
    else:
        print("[cli] loading existing index …")
        index, chunks = _load()

    print(CHAT_HELP)

    # ── chat loop ──────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[cli] session ended.")
            break

        if not user_input:
            continue

        # ── slash commands ─────────────────────────────────────────────────────
        if user_input.lower() in ("/exit", "/quit"):
            print("[cli] goodbye!")
            break

        if user_input.lower() == "/help":
            print(CHAT_HELP)
            continue

        if user_input.lower() == "/memory":
            user_mem    = load_memory(USER_MEMORY_PATH)
            company_mem = load_memory(COMPANY_MEMORY_PATH)
            print("\n── USER MEMORY ──────────────────────────────")
            print(user_mem if user_mem.strip() else "(empty)")
            print("── COMPANY MEMORY ───────────────────────────")
            print(company_mem if company_mem.strip() else "(empty)")
            print("─────────────────────────────────────────────\n")
            continue

        if user_input.lower() == "/reindex":
            print("[cli] rebuilding index …")
            try:
                index, chunks = _build(args.source_dir)
                print("[cli] reindex complete.")
            except Exception as e:
                print(f"[cli] reindex failed: {e}")
            continue

        # ── security gate ──────────────────────────────────────────────────────
        if _is_malicious_input(user_input):
            print("bot> I cannot assist with that request.\n")
            continue

        # ── memory question (what do I like / what is my name etc.) ───────────
        if _is_memory_question(user_input):
            print(f"\nbot> {_answer_from_memory(user_input)}\n")
            continue

        # ── memory-only shortcut (preference / identity statements) ────────────
        if _is_memory_only_input(user_input):
            fragments = [f.strip() for f in re.split(r"\band\b", user_input, flags=re.IGNORECASE) if f.strip()]
            any_written = False
            any_already_known = False
            for fragment in fragments:
                mem_result = maybe_write_memory(fragment, "")
                if mem_result.get("written"):
                    print(f"     🧠 memory updated ({mem_result['target']}): {mem_result['summary']}\n")
                    any_written = True
                elif mem_result.get("should_write"):
                    print(f"     💡 already noted: {mem_result['summary']}\n")
                    any_already_known = True

            if any_already_known and not any_written:
                print("bot> I already have that noted in your memory from a previous session.\n")
            else:
                print("bot> Got it — I'll remember that.\n")
            continue

        # ── normal query ───────────────────────────────────────────────────────
        try:
            hits = retrieve(user_input, index, chunks, top_k=args.top_k)

            if not hits:
                print("bot> I couldn't find any relevant content in the documents.\n")
                continue

            out = answer(user_input, hits, model=args.model)
            print(f"\nbot> {out['answer']}\n")

            # citations summary
            if out["citations"]:
                srcs = ", ".join(
                    f"{c['filename']} p{c['page']}" for c in out["citations"]
                )
                print(f"     📄 sources: {srcs}\n")

            # memory — user input only, never the answer (avoids citation pattern matches)
            mem_result = maybe_write_memory(user_input, "")
            if mem_result.get("written"):
                print(f"     🧠 memory updated ({mem_result['target']}): {mem_result['summary']}\n")

        except Exception as e:
            print(f"[cli] error during query: {e}\n")
            continue


def cmd_sanity():
    from app.sanity import run_sanity
    run_sanity()


# ── argument parser ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="ragbot",
        description="Local Agentic RAG Chatbot — fully offline, citation-grounded.",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # ingest
    p_ingest = sub.add_parser("ingest", help="Build or rebuild the vector index")
    p_ingest.add_argument("--source-dir", default="sample_docs",
                          help="Directory containing documents (default: sample_docs)")

    # chat
    p_chat = sub.add_parser("chat", help="Start an interactive chat session")
    p_chat.add_argument("--source-dir", default="sample_docs",
                        help="Directory containing documents (default: sample_docs)")
    p_chat.add_argument("--model",   default="mistral",
                        help="Ollama model name (default: mistral)")
    p_chat.add_argument("--top-k",   default=5, type=int,
                        help="Number of chunks to retrieve (default: 5)")
    p_chat.add_argument("--rebuild", action="store_true",
                        help="Force rebuild the index before chatting")

    # sanity
    sub.add_parser("sanity", help="Run sanity checks and write artifacts/sanity_output.json")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "sanity":
        cmd_sanity()


if __name__ == "__main__":
    main()