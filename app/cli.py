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
import sys
from pathlib import Path

FAISS_INDEX_PATH = os.path.join("artifacts", "faiss.index")

# ── help text shown at chat start ──────────────────────────────────────────────

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
    docs   = ingest(source_dir)
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

            # memory
            mem_result = maybe_write_memory(user_input, out["answer"])
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