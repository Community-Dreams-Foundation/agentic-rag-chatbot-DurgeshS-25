"""
rag.py — Prompt assembly, Ollama LLM call, citation extraction.

Pipeline:
    context_chunks → build_prompt() → _ollama_generate() → answer()

Usage:
    from app.rag import answer
    out = answer("what is X?", hits, model="mistral")
    print(out["answer"])
    print(out["citations"])

CLI demo:
    python -m app.rag
"""

import re
import subprocess
from typing import Optional

REFUSAL_CONTACT = "I can't share that information because it is confidential."

# keywords that trigger immediate refusal before calling Ollama
_CONTACT_KEYWORDS = re.compile(
    r"(?i)\b(phone|email|contact|reach|call|mail|address)\b"
)

# sensitive data patterns in generated answers
_SENSITIVE_PATTERNS = [
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),   # email
    re.compile(r"""(?x)
        (?:\+?1[\s\-.]?)?          # optional country code
        (?:\(?\d{3}\)?[\s\-.]?)    # area code
        \d{3}[\s\-.]               # first 3 digits
        \d{4}                      # last 4 digits
    """),                                                               # US phone
    re.compile(r"\+\d{1,3}[\s\-.]?\d{4,14}"),                         # international
]

def _contains_sensitive(text: str) -> bool:
    """Return True if text contains an email or phone number."""
    return any(p.search(text) for p in _SENSITIVE_PATTERNS)

MAX_CHUNKS      = 5
MAX_CHUNK_CHARS = 1200
REFUSAL_MSG     = (
    "I don't have enough information in the uploaded documents to answer that."
)

# strict regex: [source:<filename>#<chunk_id> p=<single-integer>]
# filename : no #, ], or whitespace
# chunk_id : no whitespace or ]
# page     : exactly one integer — no commas, ranges, or spaces
_CITATION_RE = re.compile(
    r"\[source:([^#\]]+)#([^\s\]]+)\s+p=(\d+)\]"
)

# repair patterns — only the p=... portion inside [source:...] blocks
_REPAIR_PATTERNS = [
    # p=1 - p=19  →  p=1
    (re.compile(r"(p=\d+)\s*-\s*p=\d+"),   r"\1"),
    # p=1 - 19    →  p=1
    (re.compile(r"(p=\d+)\s*-\s*\d+"),      r"\1"),
    # p=1, 3      →  p=1
    (re.compile(r"(p=\d+),\s*\d+"),         r"\1"),
]

# matches the full [source:...] block so repairs stay scoped inside it
_SOURCE_BLOCK_RE = re.compile(r"\[source:[^\]]*\]")

_RETRY_SUFFIX = (
    "\n\nYour previous answer had invalid citation formatting. "
    "Output citations ONLY in the exact format "
    "[source:<filename>#<chunk_id> p=<page>] "
    "with a single integer page. "
    "Any citation not matching this exact format is invalid."
)


# ── prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Assemble a grounded RAG prompt from the query and retrieved chunks.

    Uses at most MAX_CHUNKS chunks, each truncated to MAX_CHUNK_CHARS characters.
    """
    chunks = context_chunks[:MAX_CHUNKS]

    sources_block = ""
    for c in chunks:
        header   = f"SOURCE [source:{c['filename']}#{c['chunk_id']} p={c['page']}]"
        body     = c["text"][:MAX_CHUNK_CHARS]
        sources_block += f"{header}\n{body}\n\n"

    prompt = f"""You are a precise document assistant. Answer the user's question using ONLY the sources below.

STRICT RULES:
1. Use ONLY information from the provided sources. Do not use outside knowledge.
2. Every factual claim MUST include an inline citation in this EXACT format:
   [source:<filename>#<chunk_id> p=<page>]
   Where <page> is a SINGLE integer (e.g. p=3). No commas, no ranges, no spaces.
   Example of a VALID citation:   [source:report.pdf#report_abc_p3_0 p=3]
   Example of an INVALID citation: [source:report.pdf#chunk p=1, 3]
   Any citation not matching the exact format is invalid and will be rejected.
3. The page number is fixed per SOURCE header. You MUST copy it exactly as shown.
4. Never output page ranges, multiple pages, commas, hyphens, or a second 'p='.
   Each citation must contain exactly one 'p=' token followed by one integer.
5. If the answer cannot be found in the sources, respond with exactly:
   "{REFUSAL_MSG}"
6. Do not guess, infer, or speculate beyond what the sources state.

─────────────────────────────────────────────
{sources_block.strip()}
─────────────────────────────────────────────

Question: {query.strip()}

Answer:"""

    return prompt


# ── ollama caller ──────────────────────────────────────────────────────────────

def _ollama_generate(prompt: str, model: str = "mistral") -> str:
    """
    Call `ollama run <model>` via subprocess, passing the prompt through stdin.

    Raises:
        RuntimeError if ollama is not installed or the model is unavailable.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,                # 2-minute safety timeout
        )
    except FileNotFoundError:
        raise RuntimeError(
            "[rag] 'ollama' command not found. "
            "Install Ollama from https://ollama.com and run: ollama pull mistral"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("[rag] Ollama timed out after 120 seconds.")

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"[rag] Ollama exited with code {result.returncode}.\n"
            f"Stderr: {stderr}\n"
            f"Tip: make sure the model is pulled — run: ollama pull {model}"
        )

    output = result.stdout.strip()
    if not output:
        raise RuntimeError(
            f"[rag] Ollama returned an empty response. "
            f"Try: ollama run {model} in a separate terminal to verify it works."
        )

    return output


# ── citation extractor ─────────────────────────────────────────────────────────

def _repair_common_citation_mistakes(text: str) -> str:
    """
    Conservatively fix malformed page tokens inside [source:...] blocks only.

    Handles:
        p=1 - p=19   →  p=1
        p=1 - 19     →  p=1
        p=1, 3       →  p=1

    Text outside citation blocks is never touched.
    """
    def _fix_block(m: re.Match) -> str:
        block = m.group(0)
        for pattern, replacement in _REPAIR_PATTERNS:
            block = pattern.sub(replacement, block)
        return block

    return _SOURCE_BLOCK_RE.sub(_fix_block, text)


def _extract_citations(text: str) -> list[dict]:
    """
    Parse all strictly-formatted [source:<filename>#<chunk_id> p=<page>] markers.
    Only matches citations where page is a single integer.
    Returns deduplicated list preserving first-seen order.
    """
    seen   = set()
    result = []
    for m in _CITATION_RE.finditer(text):
        filename, chunk_id, page = m.group(1), m.group(2), m.group(3)
        key = (filename, chunk_id, page)
        if key not in seen:
            seen.add(key)
            result.append({
                "filename": filename,
                "chunk_id": chunk_id,
                "page":     int(page),
            })
    return result


# ── main public function ───────────────────────────────────────────────────────

def answer(
    query: str,
    context_chunks: list[dict],
    model: str = "mistral",
) -> dict:
    """
    Generate a grounded answer with citations.

    Args:
        query:          The user's question.
        context_chunks: Retrieved chunks from retrieve(), each with score.
        model:          Ollama model name (must be pulled locally).

    Returns:
        {
            "answer":    str,
            "citations": [{"filename": str, "page": int, "chunk_id": str}, ...]
        }
    """
    if not query or not query.strip():
        raise ValueError("[rag] query must be a non-empty string")

    # ── immediate refusal for contact-seeking queries ──────────────────────────
    if _CONTACT_KEYWORDS.search(query):
        print("[rag] contact keyword detected — refusing without LLM call")
        return {"answer": REFUSAL_CONTACT, "citations": []}

    if not context_chunks:
        return {"answer": REFUSAL_MSG, "citations": []}

    # build prompt and call the model
    prompt     = build_prompt(query, context_chunks)
    raw_answer = _ollama_generate(prompt, model=model)
    raw_answer = _repair_common_citation_mistakes(raw_answer)   # fix before extracting
    citations  = _extract_citations(raw_answer)

    # ── retry once if citations are missing ────────────────────────────────────
    if not citations and context_chunks:
        print("[rag] no valid citations found — retrying with strict formatting reminder …")
        retry_prompt = prompt + _RETRY_SUFFIX
        raw_answer   = _ollama_generate(retry_prompt, model=model)
        citations    = _extract_citations(raw_answer)

    # if still no citations after retry, return refusal
    if not citations:
        print("[rag] warning: still no valid citations after retry — returning refusal")
        return {"answer": REFUSAL_MSG, "citations": []}

    print(f"[rag] answer generated — {len(citations)} unique citation(s)")

    # ── post-generation sensitive data filter ──────────────────────────────────
    if _contains_sensitive(raw_answer):
        print("[rag] sensitive data detected in answer — refusing")
        return {"answer": REFUSAL_CONTACT, "citations": []}

    return {"answer": raw_answer, "citations": citations}


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from app.retrieve import load_retriever_assets, retrieve

    index, chunks = load_retriever_assets()

    query = "what is this document about?"
    hits  = retrieve(query, index, chunks, top_k=3)

    print(f"\n[rag] running query: '{query}'\n" + "─" * 60)
    out = answer(query, hits, model="mistral")

    print("\nANSWER:")
    print(out["answer"])
    print("\nCITATIONS:")
    for c in out["citations"]:
        print(f"  - {c['filename']} | p{c['page']} | {c['chunk_id']}")