"""Lightweight persistent memory backed by USER_MEMORY.md / COMPANY_MEMORY.md."""

USER_MEMORY_PATH = "USER_MEMORY.md"
COMPANY_MEMORY_PATH = "COMPANY_MEMORY.md"

def load_memory(path: str) -> str:
    """Read memory file and return contents. TODO: implement."""
    raise NotImplementedError

def update_memory(path: str, new_fact: str) -> None:
    """Append a new fact to memory file. TODO: implement."""
    raise NotImplementedError
