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

    if new_fact.lower() in existing.lower():
        return

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bullet   = f"- [{date_str}] {new_fact}\n"

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if existing and not existing.endswith("\n"):
        existing += "\n"

    p.write_text(existing + bullet, encoding="utf-8")


# ── secret detection ───────────────────────────────────────────────────────────

_SECRET_PATTERNS = [
    re.compile(r"[A-Za-z0-9+/]{20,}={0,2}"),
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    re.compile(r"\b\d[\d\s\-().]{7,}\d\b"),
    re.compile(r"(?i)(token|secret|password|api[_-]?key|bearer)\s*[=:]\s*\S+"),
]

def _looks_like_secret(text: str) -> bool:
    return any(p.search(text) for p in _SECRET_PATTERNS)


# ── USER memory rules ──────────────────────────────────────────────────────────

_USER_RULES = [
    # name
    (
        re.compile(r"(?i)\b(my name is|call me|i go by|you can call me)\s+([A-Z][a-z]+)"),
        lambda m: f"User's name is {m.group(2)}",
    ),
    # hobbies / interests
    (
        re.compile(r"(?i)\b(i\s+love|i\s+enjoy|i'?m\s+into|i\s+am\s+into)\s+(?P<hobby>[\w\s]{3,50})"),
        lambda m: f"User enjoys {m.group('hobby').strip().rstrip('.')}",
    ),
    # preference: concise
    (
        re.compile(r"(?i)\b(prefer|like|want)\b.{0,40}\b(concise|brief|short)\b"),
        lambda m: "User prefers concise answers",
    ),
    # preference: bullet points
    (
        re.compile(r"(?i)\b(prefer|like|want)\b.{0,40}\bbullet[s\s]"),
        lambda m: "User prefers bullet point answers",
    ),
    # preference: step-by-step
    (
        re.compile(r"(?i)\b(prefer|like|want)\b.{0,40}\bstep.by.step\b"),
        lambda m: "User prefers step-by-step explanations",
    ),
    # preference: direct / exact answers
    (
        re.compile(r"(?i)\b(exact|direct)\s+answers?\b"),
        lambda m: "User prefers direct answers without extra explanation",
    ),
    # preference: to the point
    (
        re.compile(r"(?i)\bto\s+the\s+point\b"),
        lambda m: "User prefers direct answers without extra explanation",
    ),
    # preference: no briefing / no brief
    (
        re.compile(r"(?i)\bno\s+brie(f|fing)\b"),
        lambda m: "User prefers direct answers without extra explanation",
    ),
    # preference: no summary / don't summarize / do not summarize
    (
        re.compile(r"(?i)\b(no\s+summar(y|ies)|(don'?t|do\s+not)\s+summarize)\b"),
        lambda m: "User prefers direct answers without extra explanation",
    ),
    # preference: no explanation / don't explain / do not explain
    (
        re.compile(r"(?i)\b(no\s+explanation|(don'?t|do\s+not)\s+explain)\b"),
        lambda m: "User prefers direct answers without extra explanation",
    ),
    # preference: weekly summaries/news/updates with optional day
    (
        re.compile(
            r"(?i)\b(prefer|like|send me|want)\b.{0,30}"
            r"\bweekly\s+(summar(y|ies)|news|updates?|reports?)\b"
            r"(.{0,30}\bon\s+(?P<day>\w+days?)\b)?"
        ),
        lambda m: (
            f"User prefers weekly {m.group(2).lower()}"
            + (f" on {m.group('day').capitalize()}" if m.group('day') else "")
        ),
    ),
    # role / identity — flexible phrasing
    (
        re.compile(
            r"(?i)\b(i'?m\s+an?|i\s+am\s+an?|my\s+role\s+is"
            r"|i\s+work\s+as\s+an?|i\s+work\s+as"
            r"|i\s+am\s+working\s+as\s+an?|i\s+serve\s+as\s+an?)"
            r"\s+(?P<role>[A-Za-z][A-Za-z\s]{2,50})"
        ),
        lambda m: f"User's role is {m.group('role').strip().title()}",
    ),
    # long-term goal
    (
        re.compile(
            r"(?i)\b(preparing for|studying for|practicing for"
            r"|getting ready for|training for)\s+(.{4,60})"
        ),
        lambda m: f"User is preparing for: {m.group(2).strip()}",
    ),
]

# ── COMPANY memory rules ───────────────────────────────────────────────────────

_COMPANY_RULES = [
    # tech stack (existing)
    (
        re.compile(r"(?i)\b(faiss|sentence.transformers|ollama|mistral|bm25|rank.bm25)\b"),
        lambda m: f"Project uses {m.group(1)} in its stack",
    ),
    # citation format (existing)
    (
        re.compile(r"\[source:[^\]#]+#[^\]]+\s+p=\d+\]"),
        lambda _: "Project uses citation format [source:<filename>#<chunk_id> p=<page>]",
    ),
    # artifacts (existing)
    (
        re.compile(r"(?i)artifacts[\\/](sanity_output\.json|faiss\.index|chunks\.jsonl)"),
        lambda m: f"Project artifact: artifacts/{m.group(1)}",
    ),
    # workflow bottleneck
    (
        re.compile(r"(?i)\b(bottleneck|blocker|recurring issue|pain point)\s+(is|are|was|has been)\s+(?P<issue>.{5,80})"),
        lambda m: f"Recurring workflow bottleneck: {m.group('issue').strip().rstrip('.')}",
    ),
    # team / department interface
    (
        re.compile(r"(?i)\b(?P<teamA>[A-Z][a-z]+(\s[A-Z][a-z]+)?)\s+(interfaces?|works?\s+with|collaborates?\s+with|reports?\s+to)\s+(?P<teamB>[A-Z][a-z]+(\s[A-Z][a-z]+)?)"),
        lambda m: f"{m.group('teamA')} interfaces with {m.group('teamB')}",
    ),
    # team preference / process
    (
        re.compile(r"(?i)\b(our team|the team|we)\s+(prefer|use|follow|rely on|always|never)\s+(?P<practice>.{5,80})"),
        lambda m: f"Team practice: {m.group('practice').strip().rstrip('.')}",
    ),
    # recurring meeting / reporting cadence
    (
        re.compile(r"(?i)\b(weekly|daily|monthly|quarterly)\s+(meeting|standup|report|review|sync|update)\s+(is|are|happens?|occurs?|on)\s+(?P<detail>.{3,40})"),
        lambda m: f"Team has {m.group(1).lower()} {m.group(2).lower()} on {m.group('detail').strip()}",
    ),
    # org-wide tool or process adoption
    (
        re.compile(r"(?i)\b(we\s+use|our\s+company\s+uses?|org\s+uses?|everyone\s+uses?)\s+(?P<tool>[A-Za-z][\w\s]{2,40})\s+(for|to)\s+(?P<purpose>.{5,60})"),
        lambda m: f"Org uses {m.group('tool').strip()} for {m.group('purpose').strip().rstrip('.')}",
    ),
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

    if _looks_like_secret(combined):
        return _none

    for pattern, summarise in _USER_RULES:
        m = pattern.search(combined)
        if m:
            return {
                "should_write": True,
                "target":       "USER",
                "summary":      summarise(m),
                "confidence":   0.9,
            }

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
        ("My API key is sk-abc123XYZsecret999", "Got it"),
        ("I prefer step-by-step explanations", "Sure"),
        ("We store outputs in artifacts/faiss.index", "Confirmed"),
        ("I want exact answers and no briefing", "Understood"),
        ("Give direct answers", "Sure"),
        ("Don't explain, be to the point", "Got it"),
        ("No summary please", "Okay"),
        ("Do not summarize your responses", "Understood"),
        ("I am a data analyst", "Got it"),
        ("I love reading books", "Nice!"),
        ("I like weekly news on Mondays", "Sure"),
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