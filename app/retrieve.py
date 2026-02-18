"""
retrieve.py — Hybrid retrieval: dense (FAISS) + sparse (BM25) with Reciprocal Rank Fusion.

Usage:
    from app.retrieve import retrieve, load_retriever_assets

    index, chunks = load_retriever_assets()
    hits = retrieve("what is this about?", index, chunks, top_k=5)

CLI demo:
    python -m app.retrieve
"""

from app.embed import load_index, embed_query

DEFAULT_MODEL = "multi-qa-MiniLM-L6-cos-v1"
RRF_K         = 60   # RRF constant — higher = smoother fusion, lower = more aggressive


# ── BM25 sparse retrieval ──────────────────────────────────────────────────────

def _bm25_search(query: str, chunks: list[dict], top_k: int) -> list[tuple[int, float]]:
    """
    Run BM25 keyword search over chunk texts.
    Returns list of (chunk_index, bm25_score) sorted by descending score.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("rank-bm25 is not installed. Run: pip install rank-bm25")

    # tokenize: lowercase + split on whitespace
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    tokenized_query  = query.lower().split()

    bm25   = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    # return top_k (index, score) pairs sorted descending
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def _rrf_fuse(
    dense_ids:  list[int],
    sparse_ids: list[int],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Combine dense and sparse ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i)

    Higher score = more relevant. Rewards items that rank highly in BOTH lists.
    """
    scores: dict[int, float] = {}

    for rank, idx in enumerate(dense_ids, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

    for rank, idx in enumerate(sparse_ids, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── public API ─────────────────────────────────────────────────────────────────

def retrieve(
    query:      str,
    index,
    chunks:     list[dict],
    top_k:      int = 5,
    model_name: str = DEFAULT_MODEL,
) -> list[dict]:
    """
    Hybrid retrieval: FAISS dense + BM25 sparse, fused with RRF.

    Args:
        query:      Natural language search query.
        index:      FAISS index built by embed.build_index().
        chunks:     Chunk metadata list aligned to the FAISS index rows.
        top_k:      Number of final results to return.
        model_name: Embedding model — must match the one used at index build time.

    Returns:
        List of chunk dicts sorted by descending RRF score, each with added "score" key.
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

    k = min(top_k, index.ntotal)
    # fetch more candidates from each retriever for better fusion
    candidate_k = min(k * 3, index.ntotal)

    # ── dense retrieval (FAISS) ────────────────────────────────────────────────
    query_vec = embed_query(query, model_name=model_name)
    scores, ids = index.search(query_vec, candidate_k)
    dense_ids = [int(i) for i in ids[0] if i != -1]

    # ── sparse retrieval (BM25) ────────────────────────────────────────────────
    bm25_results = _bm25_search(query, chunks, candidate_k)
    sparse_ids   = [idx for idx, _ in bm25_results]

    # ── RRF fusion ─────────────────────────────────────────────────────────────
    fused = _rrf_fuse(dense_ids, sparse_ids)

    # ── assemble final results ─────────────────────────────────────────────────
    results = []
    for idx, rrf_score in fused[:k]:
        hit = dict(chunks[idx])
        hit["score"] = round(rrf_score, 6)
        results.append(hit)

    print(
        f"[retrieve] query='{query[:60]}{'...' if len(query) > 60 else ''}' "
        f"→ {len(results)} hit(s) (hybrid BM25+FAISS)"
    )
    return results


def load_retriever_assets(path_prefix: str = "artifacts") -> tuple:
    """Load FAISS index and chunk metadata from disk."""
    index, chunks = load_index(path_prefix=path_prefix)
    print(f"[retrieve] assets ready — {index.ntotal} vector(s), {len(chunks)} chunk(s)")
    return index, chunks


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    index, chunks = load_retriever_assets()

    query = "what is the purpose and scope?"
    hits  = retrieve(query, index, chunks, top_k=3)

    print(f"\nTop {len(hits)} results for: \"{query}\"\n" + "─" * 60)
    for i, hit in enumerate(hits, 1):
        preview = hit["text"][:80].replace("\n", " ")
        print(f"{i}. rrf_score={hit['score']} | {hit['filename']} | p{hit['page']}")
        print(f"   {preview}…\n")