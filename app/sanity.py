"""
Sanity check: verifies the project skeleton is intact and writes
artifacts/sanity_output.json.

Run:  python -m app.sanity
      make sanity
"""
import json
import os
from datetime import datetime, timezone

OUTPUT_PATH = os.path.join("artifacts", "sanity_output.json")

EXPECTED_MODULES = [
    "app.ingest", "app.chunk", "app.embed",
    "app.retrieve", "app.rag", "app.memory",
    "app.cli",
]

def _check_imports() -> dict:
    results = {}
    for mod in EXPECTED_MODULES:
        try:
            __import__(mod)
            results[mod] = True
        except Exception:
            results[mod] = False
    return results

def run_sanity() -> None:
    os.makedirs("artifacts", exist_ok=True)

    import_results = _check_imports()
    all_ok = all(import_results.values())

    output = {
        "status": "skeleton_ok" if all_ok else "missing_modules",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": {
            "rag": False,
            "citations": False,
            "memory": False,
        },
        "module_imports": import_results,
        "runs": [],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    status = "OK all modules importable" if all_ok else "WARN some modules missing"
    print(f"[sanity] {status}")
    print(f"[sanity] output -> {OUTPUT_PATH}")

if __name__ == "__main__":
    run_sanity()
