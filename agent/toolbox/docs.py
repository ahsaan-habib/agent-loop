from __future__ import annotations

from functools import lru_cache

from ..tools import tool


@lru_cache(maxsize=1)
def _retrieval():
    # rag-grounded's hybrid retriever + reranker, without its generation step
    from rag_grounded.retrieval.hybrid import HybridRetriever
    from rag_grounded.retrieval.rerank import Reranker

    return HybridRetriever(), Reranker()


@tool
def search(query: str) -> list[dict]:
    """Search the documentation.

    Args:
        query: what to search for
    """
    hybrid, reranker = _retrieval()
    ranked = reranker.rank(query, hybrid.vectors.get(hybrid.retrieve(query)))[:6]
    return [{"source": c.source, "heading": c.heading, "text": c.text[:1200]} for c, _ in ranked]
