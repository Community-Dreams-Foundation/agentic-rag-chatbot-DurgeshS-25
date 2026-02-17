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


def _doc_id(filepath: str) -> str:
    """Stable, filename-based doc ID (SHA1 of the absolute path)."""
    abs_path = str(Path(filepath).resolve())
    return Path(filepath).stem + "_" + hashlib.sha1(abs_path.encode()).hexdigest()[:8]


def _read_pdf(filepath: str) -> list[dict]:
    """Return list of {page, text} dicts from a PDF."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                print(f"  [warn] could not extract page {i} of '{filepath}': {e}")
                text = ""
            pages.append({"page": i + 1, "text": text.strip()})
        return pages
    except Exception as e:
        print(f"  [error] failed to read PDF '{filepath}': {e}")
        return []


def _read_text(filepath: str) -> list[dict]:
    """Return a single-page list for .txt / .md files."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return [{"page": 1, "text": text.strip()}]
    except Exception as e:
        print(f"  [error] failed to read text file '{filepath}': {e}")
        return []


def ingest(source_dir: str) -> list[dict]:
    """
    Recursively ingest all supported files in source_dir.

    Returns:
        List of dicts:
        {
            "doc_id":   str,
            "filename": str,
            "pages":    [{"page": int, "text": str}, ...]
        }
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: '{source_dir}'")

    # Collect all supported files (sorted for determinism)
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
        str_path = str(filepath)

        if ext == ".pdf":
            pages = _read_pdf(str_path)
        else:
            pages = _read_text(str_path)

        # Skip files that produced no usable text
        if not pages or all(p["text"] == "" for p in pages):
            print(f"  [skip] no text extracted from '{filepath.name}'")
            continue

        docs.append({
            "doc_id":   _doc_id(str_path),
            "filename": filepath.name,
            "pages":    pages,
        })

    print(f"[ingest] successfully ingested {len(docs)} document(s)")
    return docs