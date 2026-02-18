"""
ingest.py — Recursively load .pdf, .txt, and .md files from a directory.

Usage:
    from app.ingest import ingest
    docs = ingest("sample_docs/")
"""

import os
import hashlib
from pathlib import Path
from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _clean_text(text: str) -> str:
    """
    Normalize line endings and remove decorative separator lines.
    Preserves ALL newlines — does not join or collapse lines.
    """
    import re
    # normalize Windows and Mac line endings to Unix
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # fix hyphenated word breaks: "stan-\ndards" -> "standards"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # replace decorative lines (====, ----) with a blank line
    text = re.sub(r"^[=\-]{4,}\s*$", "", text, flags=re.MULTILINE)
    # collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _doc_id(filepath: str) -> str:
    """Stable doc ID: stem + first 8 chars of SHA1 of absolute path."""
    abs_path = str(Path(filepath).resolve())
    return Path(filepath).stem + "_" + hashlib.sha1(abs_path.encode()).hexdigest()[:8]


def _extract_pdf_text(filepath: str) -> str:
    """Extract full text from PDF using pdfminer, falling back to pypdf."""
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(filepath)
        if text and len(text.strip()) > 100:
            return text
    except Exception:
        pass
    try:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        print(f"  [error] PDF read failed for '{filepath}': {e}")
        return ""


def _read_pdf(filepath: str) -> list[dict]:
    """Read entire PDF as one document (preserves cross-page context)."""
    try:
        raw = _extract_pdf_text(filepath)
        text = _clean_text(raw)
        return [{"page": 1, "text": text}] if text else []
    except Exception as e:
        print(f"  [error] failed to read PDF '{filepath}': {e}")
        return []


def _read_text(filepath: str) -> list[dict]:
    """Read .txt / .md file with explicit binary read to preserve newlines."""
    try:
        # read as binary then decode — avoids Python's newline translation
        raw_bytes = Path(filepath).read_bytes()
        text = raw_bytes.decode("utf-8", errors="replace")
        text = _clean_text(text)
        return [{"page": 1, "text": text}] if text else []
    except Exception as e:
        print(f"  [error] failed to read text file '{filepath}': {e}")
        return []


def ingest(source_dir: str) -> list[dict]:
    """
    Recursively ingest all supported files in source_dir.
    Ingests ALL files — PDF, TXT, and MD — without deduplication.
    Returns list of {doc_id, filename, pages: [{page, text}]}
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: '{source_dir}'")

    all_files = sorted(
        p for p in source_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not all_files:
        print(f"[ingest] no supported files found in '{source_dir}'")
        return []

    docs = []
    print(f"[ingest] found {len(all_files)} file(s) in '{source_dir}'")

    for filepath in tqdm(all_files, desc="Ingesting", unit="file"):
        ext = filepath.suffix.lower()
        pages = _read_pdf(str(filepath)) if ext == ".pdf" else _read_text(str(filepath))

        if not pages or all(not p.get("text", "").strip() for p in pages):
            print(f"  [skip] no text extracted from '{filepath.name}'")
            continue

        docs.append({
            "doc_id":   _doc_id(str(filepath)),
            "filename": filepath.name,
            "pages":    pages,
        })

    print(f"[ingest] successfully ingested {len(docs)} document(s)")
    return docs