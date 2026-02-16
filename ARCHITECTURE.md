# Architecture Overview

## Goal
Provide a brief, readable overview of how your chatbot works:
- ingestion
- indexing
- retrieval + grounding with citations
- memory writing
- optional safe tool execution

Keep this short (1–2 pages).

---

## High-Level Flow

### 1) Ingestion (Upload → Parse → Chunk)
- Supported inputs:
- Parsing approach:
- Chunking strategy:
- Metadata captured per chunk (recommended):
  - source filename
  - page/section (if available)
  - chunk_id

### 2) Indexing / Storage
- Vector store choice (FAISS/Chroma/pgvector/etc):
- Persistence:
- Optional lexical index (BM25):

### 3) Retrieval + Grounded Answering
- Retrieval method (top-k, filters, reranking):
- How citations are built:
  - citation includes: source, locator (page/section), snippet
- Failure behavior:
  - what happens when retrieval is empty/low confidence

### 4) Memory System (Selective)
- What counts as “high-signal” memory:
- What you explicitly do NOT store (PII/secrets/raw transcript):
- How you decide when to write:
- Format written to:
  - `USER_MEMORY.md`
  - `COMPANY_MEMORY.md`

### 5) Optional: Safe Tooling (Open-Meteo)
- Tool interface shape:
- Safety boundaries:
  - timeouts
  - restricted imports / sandbox isolation
  - network access rules (if applicable)

---

## Tradeoffs & Next Steps
- Why this design?
- What you would improve with more time: