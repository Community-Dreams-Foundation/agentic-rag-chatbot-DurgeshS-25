# Local Agentic RAG Chatbot

A free, local-first agentic RAG chatbot built for hackathons.
No cloud API keys required — runs entirely on your machine with Ollama + FAISS.

---

## Quickstart

### 1. Clone & enter the project
```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the sanity check
```bash
python -m app.sanity
```

This writes **`artifacts/sanity_output.json`** confirming the skeleton is intact.

### 5. Explore the CLI
```bash
python -m app.cli --help
```

---

## Project Layout
```
.
+-- app/
¦   +-- ingest.py     # PDF / TXT / MD ingestion
¦   +-- chunk.py      # Sliding-window chunking
¦   +-- embed.py      # FAISS dense index (sentence-transformers)
¦   +-- retrieve.py   # Hybrid BM25 + FAISS retrieval
¦   +-- rag.py        # Prompt assembly + LLM call (Ollama)
¦   +-- memory.py     # Persistent markdown memory
¦   +-- cli.py        # CLI entry-point
¦   +-- sanity.py     # Smoke-test runner
+-- artifacts/        # Generated outputs
+-- USER_MEMORY.md
+-- COMPANY_MEMORY.md
+-- requirements.txt
+-- Makefile
+-- README.md
```
