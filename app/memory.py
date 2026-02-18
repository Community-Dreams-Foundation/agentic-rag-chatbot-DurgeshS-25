"""
memory.py — Deterministic, LLM-free persistent memory for the RAG chatbot.

Backed by USER_MEMORY.md and COMPANY_MEMORY.md at the repo root.

Usage:
    from app.memory import load_memory, update_memory, maybe_write_memory

CLI demo:
    python -m app.memory
"""

import re
from datetime import datetime, timezone
from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────────

USER_MEMORY_PATH    = "USER_MEMORY.md"
COMPANY_MEMORY_PATH = "COMPANY_MEMORY.md"

# ── low-level I/O ──────────────────────────────────────────────────────────────

def load_memory(path: str) -> str:
    """Return full UTF-8 contents of a memory file, or '' if missing."""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def update_memory(path: str, new_fact: str) -> None:
    """
    Append a timestamped bullet to the memory file.

    - Creates the file if it does not exist.
    - Skips silently if new_fact is already present (case-insensitive).
    - Guarantees the file ends with a newline.
    """
    new_fact = new_fact.strip()
    if not new_fact:
        return

    existing = load_memory(path)

    # dedup: skip if fact already appears anywhere in the file
    if new_fact.lower() in existing.lower():
        return

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bullet   = f"- [{date_str}] {new_fact}\n"

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # ensure a blank line between existing content and new bullet
    if existing and not existing.endswith("\n"):
        existing += "\n"

    p.write_text(existing + bullet, encoding="utf-8")


# ── secret detection ───────────────────────────────────────────────────────────

_SECRET_PATTERNS = [
    re.compile(r"[A-Za-z0-9+/]{20,}={0,2}"),          # base64 / long random string
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),  # email
    re.compile(r"\b\d[\d\s\-().]{7,}\d\b"),            # phone number
    re.compile(r"(?i)(token|secret|password|api[_-]?key|bearer)\s*[=:]\s*\S+"),
]

def _looks_like_secret(text: str) -> bool:
    return any(p.search(text) for p in _SECRET_PATTERNS)


# ── USER memory rules ──────────────────────────────────────────────────────────

_USER_RULES = [
    # name
    (re.compile(r"(?i)\b(my name is|call me)\s+([A-Z][a-z]+)"),
     lambda m: f"User's name is {m.group(2)}"),
    # preference: concise
    (re.compile(r"(?i)\bprefer\b.{0,40}\b(concise|brief|short)\b"),
     lambda m: "User prefers concise answers"),
    # preference: bullet points
    (re.compile(r"(?i)\bprefer\b.{0,40}\bbullet[s\s]"),
     lambda m: "User prefers bullet point answers"),
    # preference: step-by-step
    (re.compile(r"(?i)\bprefer\b.{0,40}\bstep.by.step\b"),
     lambda m: "User prefers step-by-step explanations"),
    # long-term goal
    (re.compile(r"(?i)\b(preparing for|studying for|practicing for)\s+(.{4,60})"),
     lambda m: f"User is preparing for: {m.group(2).strip()}"),
]

# ── COMPANY memory rules ───────────────────────────────────────────────────────

_COMPANY_RULES = [
    (re.compile(r"(?i)\b(faiss|sentence.transformers|ollama|mistral|bm25|rank.bm25)\b"),
     lambda m: f"Project uses {m.group(1)} in its stack"),
    (re.compile(r"\[source:[^\]#]+#[^\]]+\s+p=\d+\]"),
     lambda _: "Project uses citation format [source:<filename>#<chunk_id> p=<page>]"),
    (re.compile(r"(?i)artifacts[\\/](sanity_output\.json|faiss\.index|chunks\.jsonl)"),
     lambda m: f"Project artifact: artifacts/{m.group(1)}"),
    (re.compile(r"(?i)\bartifacts[\\/]\b"),
     lambda _: "Project stores outputs in the artifacts/ directory"),
]


# ── decision engine ────────────────────────────────────────────────────────────

def decide_memory_write(user_text: str, assistant_text: str) -> dict:
    """
    Deterministically decide whether to write to memory (no LLM required).

    Returns:
        {
            "should_write": bool,
            "target":       "USER" | "COMPANY" | "NONE",
            "summary":      str,
            "confidence":   float,
        }
    """
    combined = f"{user_text} {assistant_text}"

    _none = {"should_write": False, "target": "NONE", "summary": "", "confidence": 0.0}

    # ── reject secrets immediately ─────────────────────────────────────────────
    if _looks_like_secret(combined):
        return _none

    # ── check USER rules ───────────────────────────────────────────────────────
    for pattern, summarise in _USER_RULES:
        m = pattern.search(combined)
        if m:
            return {
                "should_write": True,
                "target":       "USER",
                "summary":      summarise(m),
                "confidence":   0.9,
            }

    # ── check COMPANY rules ────────────────────────────────────────────────────
    for pattern, summarise in _COMPANY_RULES:
        m = pattern.search(combined)
        if m:
            return {
                "should_write": True,
                "target":       "COMPANY",
                "summary":      summarise(m),
                "confidence":   0.85,
            }

    return _none


def maybe_write_memory(user_text: str, assistant_text: str) -> dict:
    """
    Decide and, if appropriate, persist a memory fact.

    Returns the decision dict with an added "written" bool field.
    """
    decision = decide_memory_write(user_text, assistant_text)
    decision["written"] = False

    if decision["should_write"] and decision["confidence"] >= 0.8:
        target = decision["target"]
        if target == "USER":
            update_memory(USER_MEMORY_PATH, decision["summary"])
            decision["written"] = True
        elif target == "COMPANY":
            update_memory(COMPANY_MEMORY_PATH, decision["summary"])
            decision["written"] = True

    return decision


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("My name is Durgesh and I prefer concise answers", "Okay"),
        ("This chatbot uses FAISS + Ollama mistral with strict citations", "Okay"),
        ("My API key is sk-abc123XYZsecret999", "Got it"),           # should NOT write
        ("I prefer step-by-step explanations", "Sure"),
        ("We store outputs in artifacts/faiss.index", "Confirmed"),
    ]

    print("─" * 60)
    for u, a in tests:
        result = maybe_write_memory(u, a)
        status = "WRITTEN" if result["written"] else "skipped"
        print(f"[{status}] target={result['target']} conf={result['confidence']:.2f} | {result['summary'] or '—'}")
        print(f"  input: \"{u[:70]}\"")
        print()

    print("─" * 60)
    print("\nUSER_MEMORY.md:\n", load_memory(USER_MEMORY_PATH))
    print("COMPANY_MEMORY.md:\n", load_memory(COMPANY_MEMORY_PATH))