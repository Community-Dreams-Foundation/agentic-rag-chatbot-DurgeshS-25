"""
memory.py — Deterministic, LLM-free persistent memory for the RAG chatbot.

Backed by USER_MEMORY.md and COMPANY_MEMORY.md at the repo root.

Usage:
    from app.memory import load_memory, update_memory, maybe_write_memory
"""

import re
from datetime import datetime, timezone
from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────────

USER_MEMORY_PATH    = "USER_MEMORY.md"
COMPANY_MEMORY_PATH = "COMPANY_MEMORY.md"

# ── low-level I/O ──────────────────────────────────────────────────────────────

def load_memory(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def update_memory(path: str, new_fact: str) -> None:
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
    # preference: concise / brief / short
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
    # preference: exact / direct answers
    (
        re.compile(r"(?i)\b(exact|direct)\s+answers?\b"),
        lambda m: "User prefers direct answers without extra explanation",
    ),
    # preference: to the point
    (
        re.compile(r"(?i)\bto\s+the\s+point\b"),
        lambda m: "User prefers direct answers without extra explanation",
    ),
    # preference: no brief / no briefing
    (
        re.compile(r"(?i)\bno\s+brie(f|fing)\b"),
        lambda m: "User prefers direct answers without extra explanation",
    ),
    # preference: no summary / don't summarize
    (
        re.compile(r"(?i)\b(no\s+summar(y|ies)|(don'?t|do\s+not)\s+summarize)\b"),
        lambda m: "User prefers direct answers without extra explanation",
    ),
    # preference: no explanation / don't explain
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
            + (f" on {m.group('day').capitalize()}" if m.group("day") else "")
        ),
    ),
    # role / identity
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
            r"|getting ready for|training for)\s+(?P<goal>.{4,60})"
        ),
        lambda m: f"User is preparing for: {m.group('goal').strip()}",
    ),
]

# ── COMPANY memory rules (match on user input ONLY — never assistant text) ─────

_COMPANY_RULES = [
    # tech stack
    (
        re.compile(r"(?i)\b(faiss|sentence.transformers|ollama|mistral|bm25|rank.bm25)\b"),
        lambda m: f"Project uses {m.group(1)} in its stack",
    ),
    # workflow bottleneck
    (
        re.compile(r"(?i)\b(bottleneck|blocker|recurring issue|pain point)\s+(is|are|was|has been)\s+(?P<issue>.{5,80})"),
        lambda m: f"Recurring workflow bottleneck: {m.group('issue').strip().rstrip('.')}",
    ),
    # team interface
    (
        re.compile(r"(?i)\b(?P<teamA>[A-Z][a-z]+(\s[A-Z][a-z]+)?)\s+(interfaces?|works?\s+with|collaborates?\s+with)\s+(?P<teamB>[A-Z][a-z]+(\s[A-Z][a-z]+)?)"),
        lambda m: f"{m.group('teamA')} interfaces with {m.group('teamB')}",
    ),
    # team practice
    (
        re.compile(r"(?i)\b(our team|the team|we)\s+(prefer|use|follow|rely on|always|never)\s+(?P<practice>.{5,80})"),
        lambda m: f"Team practice: {m.group('practice').strip().rstrip('.')}",
    ),
    # meeting cadence
    (
        re.compile(r"(?i)\b(weekly|daily|monthly|quarterly)\s+(meeting|standup|report|review|sync|update)\s+(is|are|happens?|occurs?|on)\s+(?P<detail>.{3,40})"),
        lambda m: f"Team has {m.group(1).lower()} {m.group(2).lower()} on {m.group('detail').strip()}",
    ),
    # org tool usage
    (
        re.compile(r"(?i)\b(we\s+use|our\s+company\s+uses?|everyone\s+uses?)\s+(?P<tool>[A-Za-z][\w\s]{2,40})\s+(for|to)\s+(?P<purpose>.{5,60})"),
        lambda m: f"Org uses {m.group('tool').strip()} for {m.group('purpose').strip().rstrip('.')}",
    ),
]


# ── decision engine ────────────────────────────────────────────────────────────

def decide_memory_write(user_text: str, assistant_text: str) -> dict:
    """
    Decide whether to write to memory deterministically (no LLM).
    COMPANY rules evaluated on user_text ONLY to prevent citation bleed.
    """
    _none = {"should_write": False, "target": "NONE", "summary": "", "confidence": 0.0}

    if _looks_like_secret(user_text):
        return _none

    # USER rules
    combined = f"{user_text} {assistant_text}"
    for pattern, summarise in _USER_RULES:
        m = pattern.search(combined)
        if m:
            return {
                "should_write": True,
                "target":       "USER",
                "summary":      summarise(m),
                "confidence":   0.9,
            }

    # COMPANY rules — user_text only, never assistant answer
    for pattern, summarise in _COMPANY_RULES:
        m = pattern.search(user_text)
        if m:
            return {
                "should_write": True,
                "target":       "COMPANY",
                "summary":      summarise(m),
                "confidence":   0.85,
            }

    return _none


def maybe_write_memory(user_text: str, assistant_text: str) -> dict:
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