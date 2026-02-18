"""
memory.py — Persistent memory for the RAG chatbot.

- USER_MEMORY.md  : user-specific facts (name, role, preferences, hobbies)
- COMPANY_MEMORY.md : org-wide learnings (tools, bottlenecks, practices)

Detection strategy:
1. Fast regex rules for common patterns
2. Ollama LLM fallback for natural phrases regex misses
"""

import re
import subprocess
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
    (
        re.compile(r"(?i)\b(my name is|call me|i go by|you can call me)\s+([A-Z][a-z]+)"),
        lambda m: f"User's name is {m.group(2)}",
    ),
    (
        re.compile(r"(?i)\b(i\s+love|i\s+enjoy|i[''\u2019]?m\s+into|i\s+am\s+into)\s+(?P<hobby>[\w\s]{3,50})"),
        lambda m: f"User enjoys {m.group('hobby').strip().rstrip('.')}",
    ),
    (
        re.compile(r"(?i)\b(prefer|like|want)\b.{0,40}\b(concise|brief|short)\b"),
        lambda m: "User prefers concise answers",
    ),
    (
        re.compile(r"(?i)\b(prefer|like|want)\b.{0,40}\bbullet[s\s]"),
        lambda m: "User prefers bullet point answers",
    ),
    (
        re.compile(r"(?i)\b(prefer|like|want)\b.{0,40}\bstep.by.step\b"),
        lambda m: "User prefers step-by-step explanations",
    ),
    (
        re.compile(r"(?i)\b(exact|direct)\s+answers?\b"),
        lambda m: "User prefers direct answers without extra explanation",
    ),
    (
        re.compile(r"(?i)\bto\s+the\s+point\b"),
        lambda m: "User prefers direct answers without extra explanation",
    ),
    (
        re.compile(r"(?i)\bno\s+brie(f|fing)\b"),
        lambda m: "User prefers direct answers without extra explanation",
    ),
    (
        re.compile(r"(?i)\b(no\s+summar(y|ies)|(don[''\u2019]?t|do\s+not)\s+summarize)\b"),
        lambda m: "User prefers direct answers without extra explanation",
    ),
    (
        re.compile(r"(?i)\b(no\s+explanation|(don[''\u2019]?t|do\s+not)\s+explain)\b"),
        lambda m: "User prefers direct answers without extra explanation",
    ),
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
    (
        re.compile(
            r"(?i)\b(i[''\u2019]?m\s+an?|i\s+am\s+an?|my\s+role\s+is"
            r"|i\s+work\s+as\s+an?|i\s+work\s+as"
            r"|i\s+am\s+working\s+as\s+an?|i\s+serve\s+as\s+an?)"
            r"\s+(?P<role>[A-Za-z][A-Za-z\s]{2,50})"
        ),
        lambda m: f"User's role is {m.group('role').strip().title()}",
    ),
    (
        re.compile(
            r"(?i)\b(preparing for|studying for|practicing for"
            r"|getting ready for|training for)\s+(?P<goal>.{4,60})"
        ),
        lambda m: f"User is preparing for: {m.group('goal').strip()}",
    ),
]

# ── COMPANY memory rules ───────────────────────────────────────────────────────

_COMPANY_RULES = [
    (
        re.compile(r"(?i)\b(faiss|sentence.transformers|ollama|mistral|bm25|rank.bm25)\b"),
        lambda m: f"Project uses {m.group(1)} in its stack",
    ),
    (
        re.compile(r"(?i)\b(bottleneck|blocker|recurring issue|pain point)\s+(is|are|was|has been)\s+(?P<issue>.{5,80})"),
        lambda m: f"Recurring workflow bottleneck: {m.group('issue').strip().rstrip('.')}",
    ),
    (
        re.compile(r"(?i)\b(?P<teamA>[A-Z][a-z]+(\s[A-Z][a-z]+)?)\s+(interfaces?|works?\s+with|collaborates?\s+with)\s+(?P<teamB>[A-Z][a-z]+(\s[A-Z][a-z]+)?)"),
        lambda m: f"{m.group('teamA')} interfaces with {m.group('teamB')}",
    ),
    (
        re.compile(r"(?i)\b(our team|the team|we)\s+(prefer|use|follow|rely on|always|never)\s+(?P<practice>.{5,80})"),
        lambda m: f"Team practice: {m.group('practice').strip().rstrip('.')}",
    ),
    (
        re.compile(r"(?i)\b(weekly|daily|monthly|quarterly)\s+(meeting|standup|report|review|sync|update)\s+(is|are|happens?|occurs?|on)\s+(?P<detail>.{3,40})"),
        lambda m: f"Team has {m.group(1).lower()} {m.group(2).lower()} on {m.group('detail').strip()}",
    ),
    (
        re.compile(r"(?i)\b(we\s+use|our\s+company\s+uses?|everyone\s+uses?)\s+(?P<tool>[A-Za-z][\w\s]{2,40})\s+(for|to)\s+(?P<purpose>.{5,60})"),
        lambda m: f"Org uses {m.group('tool').strip()} for {m.group('purpose').strip().rstrip('.')}",
    ),
]


# ── LLM fallbacks ──────────────────────────────────────────────────────────────

def _llm_extract_user_fact(text: str) -> str | None:
    """Use Ollama to extract a personal fact when regex misses it."""
    prompt = (
        "You are a memory extraction assistant.\n"
        "Given this user statement, extract ONE personal fact about the user "
        "(name, role, job title, preference, hobby, goal).\n"
        "If no personal fact exists, reply with exactly: NONE\n\n"
        "Rules:\n"
        "- Reply with ONE short sentence starting with 'User' \n"
        "- Keep it under 10 words\n"
        "- Examples: \"User's role is Financial Analyst\", \"User enjoys hiking\"\n"
        "- If nothing personal, reply: NONE\n\n"
        f"Statement: {text}\n\nReply:"
    )
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        output = result.stdout.strip().split("\n")[0].strip()
        if not output or output.upper() == "NONE" or len(output) < 5:
            return None
        return output
    except Exception:
        return None


def _llm_extract_company_fact(text: str) -> str | None:
    """Use Ollama to extract a company/org fact when regex misses it."""
    prompt = (
        "You are a company memory extraction assistant.\n"
        "Given this statement, extract ONE org-wide or company-level fact "
        "(tools used, workflows, bottlenecks, team practices, hiring, processes).\n"
        "If no company fact exists, reply with exactly: NONE\n\n"
        "Rules:\n"
        "- Reply with ONE short sentence under 12 words\n"
        "- Start with 'Org', 'Team', 'Company' or 'Project'\n"
        "- Examples: \"Org uses Slack for communications\", \"Team is hiring engineers\"\n"
        "- If nothing company-related, reply: NONE\n\n"
        f"Statement: {text}\n\nReply:"
    )
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        output = result.stdout.strip().split("\n")[0].strip()
        if not output or output.upper() == "NONE" or len(output) < 5:
            return None
        return output
    except Exception:
        return None


# ── decision engine ────────────────────────────────────────────────────────────

def decide_memory_write(user_text: str, assistant_text: str) -> dict:
    """
    Decide whether to write to memory.
    1. Fast regex rules first
    2. Ollama LLM fallback for both USER and COMPANY statements regex misses
    """
    _none = {"should_write": False, "target": "NONE", "summary": "", "confidence": 0.0}

    if _looks_like_secret(user_text):
        return _none

    # USER rules — fast regex
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

    # COMPANY rules — user_text only
    for pattern, summarise in _COMPANY_RULES:
        m = pattern.search(user_text)
        if m:
            return {
                "should_write": True,
                "target":       "COMPANY",
                "summary":      summarise(m),
                "confidence":   0.85,
            }

    text     = user_text.strip()
    is_short = len(text) < 120

    # USER LLM fallback
    is_personal = any(w in text.lower() for w in [
        "i am", "i'm", "i work", "i love", "i enjoy", "i prefer",
        "my name", "my role", "my job", "i study", "i do", "i have been"
    ])
    if is_short and is_personal:
        summary = _llm_extract_user_fact(text)
        if summary:
            return {
                "should_write": True,
                "target":       "USER",
                "summary":      summary,
                "confidence":   0.8,
            }

    # COMPANY LLM fallback
    is_company = any(w in text.lower() for w in [
        "our company", "our team", "our org", "we use", "we are", "we have",
        "the company", "the team", "company is", "team is", "hiring",
        "our workflow", "our process", "our stack", "our tool",
        "we're", "we've", "management", "department"
    ])
    if is_short and is_company:
        summary = _llm_extract_company_fact(text)
        if summary:
            return {
                "should_write": True,
                "target":       "COMPANY",
                "summary":      summary,
                "confidence":   0.8,
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