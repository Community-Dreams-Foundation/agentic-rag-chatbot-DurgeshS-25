"""Entry-point CLI.  Usage: python -m app.cli [command] [args]"""
import argparse

def main():
    parser = argparse.ArgumentParser(prog="ragbot", description="Local Agentic RAG chatbot")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("ingest",  help="Ingest documents into the vector store")
    sub.add_parser("chat",    help="Start an interactive chat session")
    sub.add_parser("sanity",  help="Run sanity checks and emit artifacts/sanity_output.json")

    args = parser.parse_args()

    if args.command == "sanity":
        from app.sanity import run_sanity
        run_sanity()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
