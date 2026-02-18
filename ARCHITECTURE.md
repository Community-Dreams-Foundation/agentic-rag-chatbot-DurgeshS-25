# Architecture Overview

**Project:** Local Agentic RAG Chatbot  
**Author:** Durgesh Sakhardande  
**Stack:** Python 3.13 · FAISS · sentence-transformers · Ollama (Mistral 7B) · Streamlit

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                        │
│          streamlit_app.py — browser at localhost:8501       │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────────┐
         │               │                   │
    Security Gate   Memory Gate         RAG Pipeline
    (pre-LLM)       (pre-LLM)          (retrieval + generation)
         │               │                   │
    cli.py /        memory.py          ingest → chunk
    streamlit        USER_MEMORY.md    embed → retrieve
                    COMPANY_MEMORY.md  rag → answer
                                            │
                                       Ollama (Mistral 7B)
```

---

## Module Breakdown

### `app/ingest.py` — Document Ingestion
Recursively loads `.pdf`, `.txt`, and `.md` files from a source directory.

- **PDF extraction:** Uses `pdfminer.six` as primary extractor with `pypdf` as fallback
- **Text cleaning:** Normalizes line endings, fixes hyphenated word breaks, removes decorative separator lines (`====`, `----`) while preserving paragraph structure
- **Full-document approach:** Entire PDF is concatenated into one document before chunking to preserve cross-page context
- **Output:** List of `{doc_id, filename, pages: [{page, text}]}`

**Tradeoff:** Full-document concatenation improves semantic coherence but loses page-level granularity for very long documents.

---

### `app/chunk.py` — Paragraph-Aware Chunking
Splits documents into overlapping text chunks for embedding.

- **Strategy:** Paragraph-aware sliding window — splits on blank lines first, then accumulates paragraphs up to `chunk_size=500` characters
- **Overlap:** Last `overlap=100` characters carried into next chunk to prevent context loss at boundaries
- **Fallback:** Long paragraphs split by sentences; single long sentences split by characters
- **Output:** Flat list of `{chunk_id, doc_id, filename, page, text}`

**Why paragraph-aware over fixed-character:** Fixed-character chunking cuts mid-sentence, degrading embedding quality. Paragraph boundaries are natural semantic units that produce more coherent, retrievable chunks.

---

### `app/embed.py` — Vector Embedding + FAISS Index
Encodes chunks into dense vectors and builds a searchable index.

- **Model:** `multi-qa-MiniLM-L6-cos-v1` (sentence-transformers) — 384-dimensional embeddings, specifically trained for question-answer retrieval
- **Index type:** `IndexFlatIP` (inner product) with L2-normalized vectors = cosine similarity
- **Batch encoding:** Chunks encoded in batches of 64 to manage memory
- **Persistence:** Index saved to `artifacts/faiss.index`, metadata to `artifacts/chunks.jsonl`
- **Determinism:** Chunks sorted by `chunk_id` before embedding — same input always produces same index

**Why `multi-qa-MiniLM-L6-cos-v1` over `all-MiniLM-L6-v2`:** The `multi-qa` variant is fine-tuned on question-answer pairs, making it significantly better at matching natural language questions to document passages.

---

### `app/retrieve.py` — Hybrid Retrieval
Finds the most relevant chunks using both dense and sparse retrieval fused with RRF.

- **Dense retrieval:** FAISS inner product search on normalized vectors (cosine similarity)
- **Sparse retrieval:** BM25 keyword search using `rank-bm25` — good for exact terms, acronyms, proper nouns
- **Fusion:** Reciprocal Rank Fusion (RRF) — combines both ranked lists using `score = Σ 1/(k+rank)`
- **Candidate pool:** Each retriever fetches `top_k * 3` candidates before fusion for better coverage
- **Output:** Top-K chunks sorted by descending RRF score

**Why RRF over weighted sum:** RRF is rank-based not score-based — immune to scale differences between BM25 and cosine similarity scores. Items ranking highly in both lists get naturally boosted.

**Example:**
```
Dense:  chunk_A=#1, chunk_B=#2, chunk_C=#3
BM25:   chunk_C=#1, chunk_A=#2, chunk_D=#3
RRF:    chunk_A wins (top 2 in both lists)
```

---

### `app/rag.py` — Grounded Answer Generation
Assembles context, calls Ollama, extracts citations.

**Prompt Design:**
```
STRICT RULES:
1. Use ONLY information from provided sources
2. Every claim MUST cite: [source:<filename>#<chunk_id> p=<page>]
3. Page must be single integer — no ranges, commas, or hyphens
4. If answer not in sources: "I don't have enough information..."
```

**Citation pipeline:**
1. Build prompt with SOURCE headers containing exact citation format
2. Call `ollama run mistral` via subprocess with prompt on stdin
3. Post-process: repair malformed citations (`p=1-3` → `p=1`)
4. Extract valid citations via strict regex
5. If no citations found: retry once with formatting reminder
6. If still none: use top retrieved chunk as fallback citation
7. Final safety check: if answer contains email/phone → replace with refusal

**Security layer:**
- `REFUSAL_CONTACT`: blocks answers containing emails or phone numbers
- `_repair_common_citation_mistakes()`: fixes LLM citation formatting errors deterministically

---

### `app/memory.py` — Deterministic Persistent Memory
Stores high-signal facts without any LLM involvement.

**Decision structure:**
```python
{
  "should_write": bool,
  "target":       "USER" | "COMPANY" | "NONE",
  "summary":      str,
  "confidence":   float  # 0.9 for USER, 0.85 for COMPANY
}
```

**USER memory rules** (regex-based, match on combined user+assistant text):
- Name: `"my name is X"`, `"call me X"`
- Role: `"I'm a data analyst"`, `"I work as a PM"`
- Preferences: concise, bullet points, step-by-step, direct answers
- Hobbies: `"I love reading"`, `"I enjoy hiking"`
- Goals: `"preparing for interviews"`
- Weekly cadence: `"I prefer weekly summaries on Mondays"`

**COMPANY memory rules** (regex-based, match on user text ONLY):
- Tech stack: FAISS, Ollama, Mistral mentions
- Workflow bottlenecks: `"the bottleneck is X"`
- Team tools: `"our team uses Slack"`
- Org practices: `"we use Jira for tracking"`

**Deduplication:** Case-insensitive substring check — same fact never stored twice.

**Security:** Rejects inputs containing emails, phone numbers, API keys, tokens, or long random strings before any rule matching.

**Tradeoff:** Deterministic regex rules are fast, free, and predictable but miss nuanced phrasing. An LLM-based memory classifier would have higher recall but adds latency and cost — inappropriate for a local-first tool.

---

### `app/cli.py` — Security Gates + Chat Orchestration
Coordinates the full pipeline with two pre-LLM security layers.

**Request flow:**
```
User input
    │
    ├─► Prompt injection check (_INJECTION_RE)
    │       "ignore prior instructions" → BLOCK
    │
    ├─► Classified field check (_CLASSIFIED_RE)  
    │       "phone number", "email address" → BLOCK
    │
    ├─► Memory question check
    │       "what do I prefer?" → answer from USER_MEMORY.md
    │
    ├─► Memory-only check
    │       "I'm a data analyst" → store → acknowledge
    │
    └─► RAG pipeline
            retrieve → answer → extract citations → write memory
```

**Why two security layers:**
- Layer 1 (query-time): Catches intent before any document retrieval — zero LLM exposure to malicious inputs
- Layer 2 (answer-time): Catches sensitive data the LLM might have extracted from documents and included in its response

---

### `app/sanity.py` — Pipeline Validation
End-to-end smoke test that validates all features and writes judge-readable output.

Runs: ingest → chunk → embed → retrieve → answer → memory write → security check

Outputs `artifacts/sanity_output.json` in format required by `scripts/verify_output.py`:
```json
{
  "implemented_features": ["A", "B"],
  "qa": [{"question": "...", "answer": "...", "citations": [...]}],
  "demo": {"memory_writes": [{"target": "USER", "summary": "..."}]},
  "meta": {"status": "ok", "errors": []}
}
```

---

## Data Flow Diagram

```
User uploads file
        │
        ▼
   ingest.py ──► clean text ──► pages[]
        │
        ▼
   chunk.py ──► paragraph split ──► chunks[]
        │
        ▼
   embed.py ──► encode (multi-qa-MiniLM) ──► FAISS index
        │
        ▼
   artifacts/faiss.index + chunks.jsonl
        │
   User asks question
        │
        ▼
   Security gates (inject / classified / memory)
        │
        ▼
   retrieve.py ──► embed query ──► FAISS search ──► top-K chunks
        │
        ▼
   rag.py ──► build prompt ──► ollama run mistral ──► raw answer
        │
        ▼
   repair citations ──► extract citations ──► safety filter
        │
        ▼
   Streamlit UI ──► display answer + Sources expander
        │
        ▼
   memory.py ──► decide_memory_write ──► USER/COMPANY_MEMORY.md
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model | `multi-qa-MiniLM-L6-cos-v1` | Trained on QA pairs — better retrieval than general-purpose models |
| Retrieval | FAISS + BM25 hybrid (RRF) | Dense handles semantics, BM25 handles keywords — RRF combines both |
| LLM interface | Ollama subprocess | Zero dependencies, works fully offline, model-agnostic |
| Memory | Regex rules | Deterministic, fast, free — no LLM latency for simple fact extraction |
| Chunking | Paragraph-aware | Preserves semantic units, avoids mid-sentence cuts |
| Security | Pre-LLM gates | LLM never sees malicious input — safer than post-processing |
| Citation format | `[source:file#id p=N]` | Parseable by regex, human-readable, traceable to exact chunk |

---

## What I Would Improve Next

1. **Hybrid retrieval (BM25 + dense):** Add BM25 sparse retrieval with Reciprocal Rank Fusion — better recall for keyword-heavy queries like acronyms and proper nouns

2. **Streaming responses:** Stream Mistral output token-by-token via Ollama's REST API for better UX on long answers

3. **Conversation history:** Pass last N turns as context to Ollama for multi-turn coherence

4. **LLM-assisted memory:** Use a lightweight classifier for company memory to catch nuanced org-level insights that regex misses

5. **Re-ranking:** Add a cross-encoder reranker after FAISS retrieval to improve precision on ambiguous queries

6. **Feature C — Open-Meteo sandbox:** Implement safe Python execution with restricted builtins to allow weather time series analysis via the Open-Meteo API