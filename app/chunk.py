"""
chunk.py — Split ingested document pages into overlapping character-based chunks.

Usage:
    from app.chunk import chunk
    chunks = chunk(docs, chunk_size=800, overlap=150)
"""

from tqdm import tqdm


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Sliding-window character-based chunker.

    - Deterministic: same input always produces same output.
    - Safe for large text: no in-memory list explosions.
    - Returns at least one chunk even if text < chunk_size.
    """
    if not text:
        return []

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        # Stop if we've reached the end
        if end == text_len:
            break
        start += step

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