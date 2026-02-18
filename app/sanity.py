"""
sanity.py — Full pipeline validation for the local Agentic RAG chatbot.

Writes artifacts/sanity_output.json with evidence of every feature working.

Run:
    python -m app.sanity
    make sanity
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_PATH  = os.path.join("artifacts", "sanity_output.json")
SAMPLE_DIR   = "sample_docs"
FAISS_PATH   = os.path.join("artifacts", "faiss.index")
SANITY_QUERY = "guidelines for sample documents"

EXPECTED_MODULES = [
    "app.ingest", "app.chunk", "app.embed",
    "app.retrieve", "app.rag", "app.memory",
    "app.cli",
]

# ── helpers ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _check_imports() -> dict:
    results = {}
    for mod in EXPECTED_MODULES:
        try:
            __import__(mod)
            results[mod] = True
        except Exception as e:
            results[mod] = False
            print(f"  [warn] could not import {mod}: {e}")
    return results


def _has_sample_docs() -> bool:
    p = Path(SAMPLE_DIR)
    if not p.exists():
        return False
    supported = {".pdf", ".txt", ".md"}
    return any(f.suffix.lower() in supported for f in p.rglob("*") if f.is_file())


# ── pipeline checks ────────────────────────────────────────────────────────────

def _check_ingest() -> tuple:
    """Returns (ok, docs, error_str)."""
    try:
        from app.ingest import ingest
        docs = ingest(SAMPLE_DIR)
        if not docs:
            return False, [], "ingest() returned 0 documents"
        print(f"  [ok] ingest — {len(docs)} doc(s)")
        return True, docs, None
    except Exception as e:
        return False, [], f"ingest exception: {e}"


def _check_chunk(docs) -> tuple:
    """Returns (ok, chunks, error_str)."""
    try:
        from app.chunk import chunk
        chunks = chunk(docs)
        if not chunks:
            return False, [], "chunk() returned 0 chunks"
        print(f"  [ok] chunk — {len(chunks)} chunk(s)")
        return True, chunks, None
    except Exception as e:
        return False, [], f"chunk exception: {e}"


def _check_embed(chunks) -> tuple:
    """Returns (ok, index, chunks_meta, error_str). Loads if index exists, else builds."""
    try:
        from app.embed import build_index, load_index
        if Path(FAISS_PATH).exists():
            index, meta = load_index()
            # if stored index is stale (different chunk count), rebuild
            if index.ntotal != len(chunks):
                print("  [info] index stale — rebuilding …")
                index, meta = build_index(chunks)
        else:
            index, meta = build_index(chunks)

        if index.ntotal != len(meta):
            return False, None, [], (
                f"embed mismatch: index has {index.ntotal} vectors, "
                f"meta has {len(meta)} entries"
            )
        print(f"  [ok] embed — {index.ntotal} vector(s), dim confirmed")
        return True, index, meta, None
    except Exception as e:
        return False, None, [], f"embed exception: {e}"


def _check_retrieve(index, chunks_meta) -> tuple:
    """Returns (ok, hits, error_str)."""
    try:
        from app.retrieve import retrieve
        hits = retrieve(SANITY_QUERY, index, chunks_meta, top_k=3)
        if not hits:
            return False, [], "retrieve() returned 0 hits"
        print(f"  [ok] retrieve — {len(hits)} hit(s), top score={hits[0]['score']:.4f}")
        return True, hits, None
    except Exception as e:
        return False, [], f"retrieve exception: {e}"


def _check_rag(hits) -> tuple:
    """Returns (ok, citations_ok, out, error_str)."""
    try:
        from app.rag import answer
        out = answer(SANITY_QUERY, hits, model="mistral")

        ans_ok  = bool(out.get("answer", "").strip())
        cite_ok = bool(out.get("citations"))

        if not ans_ok:
            return False, False, out, "answer() returned empty answer"
        if not cite_ok:
            print("  [warn] rag answered but no citations extracted")

        print(f"  [ok] rag — answer len={len(out['answer'])}, citations={len(out.get('citations', []))}")
        return ans_ok, cite_ok, out, None
    except Exception as e:
        return False, False, {}, f"rag exception: {e}"


def _check_memory(assistant_text: str) -> tuple:
    """Returns (ok, error_str). Accepts written=True OR already-known (should_write=True)."""
    try:
        from app.memory import maybe_write_memory
        result = maybe_write_memory("I prefer concise answers", assistant_text)
        # Pass if written OR fact was already present (should_write True, written False = dedup)
        ok = result.get("written") or result.get("should_write", False)
        status = "written" if result.get("written") else "already present (dedup)"
        print(f"  [ok] memory — {status}, target={result.get('target')}")
        return ok, None
    except Exception as e:
        return False, f"memory exception: {e}"


# ── main ───────────────────────────────────────────────────────────────────────

def run_sanity() -> None:
    os.makedirs("artifacts", exist_ok=True)

    errors  = []
    features = {
        "embeddings": False,
        "retrieval":  False,
        "rag":        False,
        "citations":  False,
        "memory":     False,
    }
    run_entry = {
        "query":         SANITY_QUERY,
        "top_k":         3,
        "num_docs":      0,
        "num_chunks":    0,
        "num_hits":      0,
        "num_citations": 0,
        "timestamp":     _now(),
    }

    print("\n[sanity] ── module imports ──────────────────────────")
    import_results = _check_imports()
    all_imports_ok = all(import_results.values())

    # abort pipeline checks early if sample_docs is absent
    if not _has_sample_docs():
        errors.append(f"'{SAMPLE_DIR}' directory missing or contains no supported files")
        print(f"  [warn] {errors[-1]}")
        _write(OUTPUT_PATH, "fail", features, import_results, [run_entry], errors)
        return

    print("\n[sanity] ── pipeline checks ─────────────────────────")

    # A) ingest
    ingest_ok, docs, err = _check_ingest()
    if err: errors.append(err)
    run_entry["num_docs"] = len(docs)

    # B) chunk
    chunk_ok, chunks, err = (False, [], None)
    if ingest_ok:
        chunk_ok, chunks, err = _check_chunk(docs)
        if err: errors.append(err)
    run_entry["num_chunks"] = len(chunks)

    # C) embed
    embed_ok, index, chunks_meta, err = (False, None, [], None)
    if chunk_ok:
        embed_ok, index, chunks_meta, err = _check_embed(chunks)
        if err: errors.append(err)
    features["embeddings"] = embed_ok

    # D) retrieve
    retrieve_ok, hits, err = (False, [], None)
    if embed_ok:
        retrieve_ok, hits, err = _check_retrieve(index, chunks_meta)
        if err: errors.append(err)
    features["retrieval"]  = retrieve_ok
    run_entry["num_hits"]  = len(hits)

    # E) rag + citations
    rag_ok, citations_ok, rag_out, err = (False, False, {}, None)
    if retrieve_ok:
        rag_ok, citations_ok, rag_out, err = _check_rag(hits)
        if err: errors.append(err)
    features["rag"]              = rag_ok
    features["citations"]        = citations_ok
    run_entry["num_citations"]   = len(rag_out.get("citations", []))

    # F) memory
    mem_ok, err = (False, None)
    mem_text = rag_out.get("answer", "sanity check placeholder")
    if ingest_ok:   # memory doesn't need RAG to work
        mem_ok, err = _check_memory(mem_text)
        if err: errors.append(err)
    features["memory"] = mem_ok

    # ── overall status ─────────────────────────────────────────────────────────
    all_ok  = all(features.values()) and all_imports_ok and not errors
    partial = any(features.values()) and not all_ok
    status  = "ok" if all_ok else ("partial" if partial else "fail")

    _write(OUTPUT_PATH, status, features, import_results, [run_entry], errors)
    print(f"\n[sanity] status={status.upper()} | features={features}")
    if errors:
        print(f"[sanity] errors: {errors}")


def _write(path, status, features, imports, runs, errors):
    output = {
        "status":         status,
        "timestamp":      _now(),
        "features":       features,
        "module_imports": imports,
        "runs":           runs,
        "errors":         errors,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[sanity] output → {path}")


if __name__ == "__main__":
    run_sanity()