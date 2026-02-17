"""RAG pipeline: retrieval -> prompt assembly -> LLM call -> cited answer."""

def answer(query: str, context_chunks: list[dict], model: str = "ollama/mistral") -> dict:
    """
    Return {answer: str, citations: list[dict]}.
    TODO: call local LLM (Ollama) with assembled context prompt.
    """
    raise NotImplementedError
