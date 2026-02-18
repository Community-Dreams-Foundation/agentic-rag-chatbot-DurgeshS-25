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

REFUSAL_INJECTION    = "I cannot assist with that request."
REFUSAL_CONFIDENTIAL = "I can't share confidential or classified details."

# ── security: prompt injection ─────────────────────────────────────────────────

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

# ── security: classified field requests ───────────────────────────────────────

_CLASSIFIED_RE = re.compile(
    r"(?i)\b(phone\s+number|email\s+address|cro|api\s?key|token|password|secret)\b"
)

def _is_malicious_input(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))

def _is_classified_request(text: str) -> bool:
    return bool(_CLASSIFIED_RE.search(text))


# ── memory-only detection ──────────────────────────────────────────────────────

_MEMORY_ONLY_RE = re.compile(
    r"(?i)\b("
    r"i\s+prefer|i\s+like|i\s+love|i\s+enjoy|i'?m\s+into|i\s+am\s+into"
    r"|my\s+name\s+is|call\s+me|i'?m\s+an?|i\s+am\s+an?"
    r"|my\s+role\s+is|i\s+work\s+as|i\s+am\s+working\s+as|send\s+me"
    r"|don'?t\s+explain|don'?t\s+summarize|do\s+not\s+explain"
    r"|do\s+not\s+summarize|no\s+summary|no\s+explanation|no\s+briefing|no\s+brief"
    r"|our\s+team\s+(uses?|prefers?|follows?|relies?|always|never)"
    r"|we\s+use\b|the\s+bottleneck\s+is"
    r"|everyone\s+uses?"
    r")"
)

def _is_memory_only_input(text: str) -> bool:
    """True if ANY fragment (split on 'and') is a preference/identity statement."""
    fragments = [f.strip() for f in re.split(r"\band\b", text, flags=re.IGNORECASE) if f.strip()]
    return any(bool(_MEMORY_ONLY_RE.search(frag)) for frag in fragments)


# ── memory question detection ──────────────────────────────────────────────────

_MEMORY_QUESTION_RE = re.compile(
    r"(?i)\b("
    r"what\s+(do|did|don'?t|doesn'?t)\s+i\s+(like|love|enjoy|prefer|hate|want|need)"
    r"|what\s+is\s+my\s+(name|role|job|preference|hobby|interest)"
    r"|who\s+am\s+i"
    r"|what\s+are\s+my\s+(preferences?|interests?|hobbies|goals?)"
    r"|do\s+you\s+(know|remember)\s+(me|my)"
    r")"
)

def _is_memory_question(text: str) -> bool:
    return bool(_MEMORY_QUESTION_RE.search(text))

def _answer_from_memory(query: str) -> str:
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


# ── help banner ────────────────────────────────────────────────────────────────

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

def _build(source_dir: str, verbose: bool = False):
    from app.ingest import ingest
    from app.chunk  import chunk
    from app.embed  import build_index
    print(f"[cli] ingesting documents from '{source_dir}' …")
    docs = ingest(source_dir)
    if not docs:
        print("[cli] no documents found — add files and retry.")
        sys.exit(1)
    chunks = chunk(docs, chunk_size=500, overlap=100)

    if verbose:
        print(f"\n[verbose] ── chunk preview ──────────────────────────")
        for i, c in enumerate(chunks):
            print(f"\n  Chunk {i+1}/{len(chunks)} | {c['filename']} | p{c['page']} | {len(c['text'])} chars")
            print(f"  ID: {c['chunk_id']}")
            print(f"  Text: {c['text'][:150].replace(chr(10), ' ')}...")
        print(f"\n[verbose] total chunks: {len(chunks)}")
        print(f"[verbose] ────────────────────────────────────────────\n")

    index, meta = build_index(chunks)
    return index, meta

def _load():
    from app.retrieve import load_retriever_assets
    return load_retriever_assets()


# ── commands ───────────────────────────────────────────────────────────────────

def cmd_ingest(args):
    from app.ingest import ingest
    from app.chunk  import chunk
    from app.embed  import build_index
    print(f"[cli] ingesting documents from '{args.source_dir}' …")
    docs = ingest(args.source_dir)
    if not docs:
        print("[cli] no documents found — add files and retry.")
        sys.exit(1)
    chunks = chunk(docs, chunk_size=500, overlap=100)

    if args.verbose:
        print(f"\n[verbose] ── chunk preview ──────────────────────────")
        for i, c in enumerate(chunks):
            print(f"\n  Chunk {i+1}/{len(chunks)} | {c['filename']} | p{c['page']} | {len(c['text'])} chars")
            print(f"  ID: {c['chunk_id']}")
            print(f"  Text: {c['text'][:150].replace(chr(10), ' ')}...")
        print(f"\n[verbose] total chunks: {len(chunks)}")
        print(f"[verbose] ────────────────────────────────────────────\n")

    build_index(chunks)
    print("[cli] index built successfully.")


def cmd_chat(args):
    from app.retrieve import retrieve
    from app.rag      import answer
    from app.memory   import load_memory, maybe_write_memory, USER_MEMORY_PATH, COMPANY_MEMORY_PATH

    needs_build = args.rebuild or not Path(FAISS_INDEX_PATH).exists()
    if needs_build:
        index, chunks = _build(args.source_dir)
    else:
        print("[cli] loading existing index …")
        index, chunks = _load()

    print(CHAT_HELP)

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
            try:
                index, chunks = _build(args.source_dir)
                print("[cli] reindex complete.")
            except Exception as e:
                print(f"[cli] reindex failed: {e}")
            continue

        # ── prompt injection guard ─────────────────────────────────────────────
        if _is_malicious_input(user_input):
            print(f"bot> {REFUSAL_INJECTION}\n")
            continue

        # ── classified / confidential field guard ──────────────────────────────
        if _is_classified_request(user_input):
            print(f"bot> {REFUSAL_CONFIDENTIAL}\n")
            continue

        # ── memory question ────────────────────────────────────────────────────
        if _is_memory_question(user_input):
            print(f"\nbot> {_answer_from_memory(user_input)}\n")
            continue

        # ── memory-only input (preferences / identity) ─────────────────────────
        if _is_memory_only_input(user_input):
            fragments = [f.strip() for f in re.split(r"\band\b", user_input, flags=re.IGNORECASE) if f.strip()]
            any_written      = False
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

        # ── normal RAG query ───────────────────────────────────────────────────
        try:
            hits = retrieve(user_input, index, chunks, top_k=args.top_k)
            if not hits:
                print("bot> I couldn't find any relevant content in the documents.\n")
                continue

            out = answer(user_input, hits, model=args.model)
            print(f"\nbot> {out['answer']}\n")

            if out["citations"]:
                srcs = ", ".join(f"{c['filename']} p{c['page']}" for c in out["citations"])
                print(f"     📄 sources: {srcs}\n")

            # memory scan on user input ONLY — never pass assistant answer
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

    p_ingest = sub.add_parser("ingest", help="Build or rebuild the vector index")
    p_ingest.add_argument("--source-dir", default="sample_docs")
    p_ingest.add_argument("--verbose", "-v", action="store_true",
                          help="Print chunk details during indexing")

    p_chat = sub.add_parser("chat", help="Start an interactive chat session")
    p_chat.add_argument("--source-dir", default="sample_docs")
    p_chat.add_argument("--model",   default="mistral")
    p_chat.add_argument("--top-k",   default=5, type=int)
    p_chat.add_argument("--rebuild", action="store_true")

    sub.add_parser("sanity", help="Run sanity checks")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "sanity":
        cmd_sanity()


if __name__ == "__main__":
    main()