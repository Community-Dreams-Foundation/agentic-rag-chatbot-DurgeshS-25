"""
retrieve.py — Dense retrieval using FAISS + cosine similarity.

Usage:
    from app.retrieve import retrieve, load_retriever_assets

    index, chunks = load_retriever_assets()
    hits = retrieve("what is this about?", index, chunks, top_k=5)

    # CLI demo:
    python -m app.retrieve
"""

from app.embed import load_index, embed_query

DEFAULT_MODEL = "multi-qa-MiniLM-L6-cos-v1"


# ── public API ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    index,
    chunks: list[dict],
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query using FAISS dense search.

    Args:
        query:      Natural language search query.
        index:      FAISS index built by embed.build_index().
        chunks:     Chunk metadata list aligned to the FAISS index rows.
        top_k:      Number of results to return.
        model_name: Must match the model used during build_index().

    Returns:
        List of chunk dicts (copies) sorted by descending cosine similarity score,
        each with an added "score" key.
    """
    # ── validation ─────────────────────────────────────────────────────────────
    if not query or not query.strip():
        raise ValueError("[retrieve] query must be a non-empty string")
    if index.ntotal == 0:
        raise ValueError("[retrieve] FAISS index is empty — run embed.build_index() first")
    if len(chunks) != index.ntotal:
        raise ValueError(
            f"[retrieve] mismatch: index has {index.ntotal} vectors "
            f"but chunks list has {len(chunks)} entries — rebuild the index"
        )

    # ── clamp top_k to available vectors ───────────────────────────────────────
    k = min(top_k, index.ntotal)

    # ── embed the query ────────────────────────────────────────────────────────
    query_vec = embed_query(query, model_name=model_name)   # shape (1, dim)

    # ── FAISS search ───────────────────────────────────────────────────────────
    scores, ids = index.search(query_vec, k)   # both shape (1, k)

    # ── assemble results ───────────────────────────────────────────────────────
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:          # FAISS returns -1 for empty slots
            continue
        hit = dict(chunks[idx])   # copy — never mutate the original list
        hit["score"] = float(score)
        results.append(hit)

    # already sorted descending by FAISS, but make it explicit
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"[retrieve] query='{query[:60]}{'...' if len(query) > 60 else ''}' → {len(results)} hit(s)")
    return results


def load_retriever_assets(path_prefix: str = "artifacts") -> tuple:
    """
    Convenience loader — returns (faiss_index, chunks_metadata).
    Call this once at startup, then pass the results into retrieve().
    """
    index, chunks = load_index(path_prefix=path_prefix)
    print(f"[retrieve] assets ready — {index.ntotal} vector(s), {len(chunks)} chunk(s)")
    return index, chunks


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    index, chunks = load_retriever_assets()

    query = "what is this document about?"
    hits  = retrieve(query, index, chunks, top_k=3)

    print(f"\nTop {len(hits)} results for: \"{query}\"\n" + "─" * 60)
    for i, hit in enumerate(hits, 1):
        preview = hit["text"][:80].replace("\n", " ")
        print(f"{i}. score={hit['score']:.4f} | {hit['filename']} | p{hit['page']} | {hit['chunk_id']}")
        print(f"   {preview}…\n")