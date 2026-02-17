"""Hybrid retrieval: dense (FAISS) + sparse (BM25), with reciprocal-rank fusion."""

def retrieve(query: str, index, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Return top-k chunks most relevant to query. TODO: implement."""
    raise NotImplementedError
