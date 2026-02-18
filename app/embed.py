"""
embed.py — Embed chunks with sentence-transformers and build a FAISS index.

Workflow:
    from app.embed import build_index, load_index, embed_query

    index, meta = build_index(chunks)          # first run — builds & saves
    index, meta = load_index()                 # subsequent runs — loads from disk
    q_vec       = embed_query("my question")   # single query vector for retrieval
"""

import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

ARTIFACTS_DIR   = "artifacts"
FAISS_PATH      = os.path.join(ARTIFACTS_DIR, "faiss.index")
CHUNKS_PATH     = os.path.join(ARTIFACTS_DIR, "chunks.jsonl")
META_PATH       = os.path.join(ARTIFACTS_DIR, "embed_meta.json")

DEFAULT_MODEL = "multi-qa-MiniLM-L6-cos-v1"
BATCH_SIZE      = 64          # chunks per encoding batch — safe for low-RAM machines


# ── helpers ────────────────────────────────────────────────────────────────────

def _normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize each row so dot-product == cosine similarity."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid div-by-zero for zero vectors
    return (vecs / norms).astype(np.float32)


def _load_model(model_name: str):
    """Load (and cache) a SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[embed] loading model '{model_name}' …")
        return SentenceTransformer(model_name)
    except ImportError:
        raise ImportError("sentence-transformers is not installed. Run: pip install sentence-transformers")


def _faiss():
    """Import faiss with a friendly error."""
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError("faiss-cpu is not installed. Run: pip install faiss-cpu")


# ── public API ─────────────────────────────────────────────────────────────────

def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """
    Embed a single query string and return a normalized (1, dim) float32 array.
    Used at retrieval time — keeps the model interface consistent with build_index.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    model = _load_model(model_name)
    vec = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
    return _normalize(vec)   # shape (1, dim)


def build_index(
    chunks: list[dict],
    model_name: str = DEFAULT_MODEL,
) -> tuple:
    """
    Encode all chunks, build a FAISS IndexFlatIP index, and persist to artifacts/.

    Args:
        chunks:     Flat list of chunk dicts from chunk().
        model_name: Any sentence-transformers model name.

    Returns:
        (faiss_index, chunks_metadata)
        chunks_metadata is sorted by chunk_id — same order as the FAISS row indices.
    """
    if not chunks:
        raise ValueError("[embed] chunk list is empty — run ingest + chunk first")

    faiss = _faiss()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── 1. Sort for determinism ────────────────────────────────────────────────
    sorted_chunks = sorted(chunks, key=lambda c: c["chunk_id"])
    texts = [c["text"] for c in sorted_chunks]
    print(f"[embed] embedding {len(texts)} chunk(s) with model '{model_name}' …")

    # ── 2. Encode in batches ───────────────────────────────────────────────────
    model = _load_model(model_name)
    all_vecs = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding", unit="batch"):
        batch = texts[start : start + BATCH_SIZE]
        vecs  = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_vecs.append(vecs)

    embeddings = np.vstack(all_vecs).astype(np.float32)   # (N, dim)
    embeddings = _normalize(embeddings)
    dim        = embeddings.shape[1]
    print(f"[embed] encoded {len(texts)} chunk(s), dim={dim}")

    # ── 3. Build FAISS index (inner product on normalized vecs = cosine sim) ───
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[embed] FAISS index built — {index.ntotal} vector(s)")

    # ── 4. Persist ─────────────────────────────────────────────────────────────
    faiss.write_index(index, FAISS_PATH)
    print(f"[embed] index saved → {FAISS_PATH}")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in sorted_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[embed] chunks saved → {CHUNKS_PATH}")

    meta = {"model_name": model_name, "dim": dim, "num_chunks": len(sorted_chunks)}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[embed] metadata saved → {META_PATH}")

    return index, sorted_chunks


def load_index(path_prefix: str = ARTIFACTS_DIR) -> tuple:
    """
    Load a previously built FAISS index and its chunk metadata from disk.

    Args:
        path_prefix: Directory containing faiss.index and chunks.jsonl.

    Returns:
        (faiss_index, chunks_metadata)
    """
    faiss      = _faiss()
    idx_path   = os.path.join(path_prefix, "faiss.index")
    chunks_path = os.path.join(path_prefix, "chunks.jsonl")
    meta_path  = os.path.join(path_prefix, "embed_meta.json")

    for p in [idx_path, chunks_path]:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"[embed] '{p}' not found — run build_index() first"
            )

    index = faiss.read_index(idx_path)
    print(f"[embed] loaded FAISS index ({index.ntotal} vectors) ← {idx_path}")

    chunks_meta = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks_meta.append(json.loads(line))
    print(f"[embed] loaded {len(chunks_meta)} chunk(s) ← {chunks_path}")

    if Path(meta_path).exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print(f"[embed] meta — model: {meta.get('model_name')}, dim: {meta.get('dim')}")

    if index.ntotal != len(chunks_meta):
        raise ValueError(
            f"[embed] mismatch: FAISS has {index.ntotal} vectors "
            f"but chunks.jsonl has {len(chunks_meta)} entries — rebuild the index"
        )

    return index, chunks_meta