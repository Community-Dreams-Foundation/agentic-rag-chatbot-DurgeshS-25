"""
chunk.py — Split ingested document pages into overlapping character-based chunks.

Usage:
    from app.chunk import chunk
    chunks = chunk(docs, chunk_size=800, overlap=150)
"""
import re
from tqdm import tqdm


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Paragraph-aware sliding window chunker.

    1. Split text into paragraphs on blank lines.
    2. Greedily accumulate paragraphs until chunk_size is reached.
    3. Overlap by carrying the last paragraph(s) into the next chunk.
    Never cuts mid-sentence or mid-word.
    """
    if not text:
        return []

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    # split into paragraphs, filter empty
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if not paragraphs:
        return []

    chunks   = []
    current  = []
    curr_len = 0

    for para in paragraphs:
        para_len = len(para)

        # if adding this paragraph exceeds chunk_size, flush current chunk
        if curr_len + para_len > chunk_size and current:
            chunks.append("\n\n".join(current))
            # carry overlap: keep last paragraph(s) up to overlap chars
            overlap_paras = []
            overlap_len   = 0
            for p in reversed(current):
                if overlap_len + len(p) <= overlap:
                    overlap_paras.insert(0, p)
                    overlap_len += len(p)
                else:
                    break
            current  = overlap_paras
            curr_len = overlap_len

        current.append(para)
        curr_len += para_len

    # flush remaining
    if current:
        chunks.append("\n\n".join(current))

    return chunks


def chunk(
    docs: list[dict],
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[dict]:
    """
    Split each page of each document into overlapping text chunks.

    Args:
        docs:       Output of ingest() — list of {doc_id, filename, pages}.
        chunk_size: Maximum characters per chunk.
        overlap:    Characters of overlap between consecutive chunks.

    Returns:
        Flat list of dicts:
        {
            "chunk_id": str,   # "{doc_id}_p{page}_{index}"
            "doc_id":   str,
            "filename": str,
            "page":     int,
            "text":     str,
        }
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    all_chunks = []

    for doc in tqdm(docs, desc="Chunking", unit="doc"):
        doc_id   = doc.get("doc_id", "unknown")
        filename = doc.get("filename", "unknown")
        pages    = doc.get("pages", [])

        for page_obj in pages:
            page_num = page_obj.get("page", 0)
            text     = page_obj.get("text", "")

            if not text:
                continue

            splits = _split_text(text, chunk_size, overlap)

            for idx, split_text in enumerate(splits):
                all_chunks.append({
                    "chunk_id": f"{doc_id}_p{page_num}_{idx}",
                    "doc_id":   doc_id,
                    "filename": filename,
                    "page":     page_num,
                    "text":     split_text,
                })

    print(f"[chunk] produced {len(all_chunks)} chunk(s) from {len(docs)} doc(s)")
    return all_chunks