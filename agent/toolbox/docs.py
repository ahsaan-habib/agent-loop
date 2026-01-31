from __future__ import annotations

from functools import lru_cache
from typing import Literal

from ..tools import tool


@lru_cache(maxsize=1)
def _retrieval():
    # rag-grounded's hybrid retriever + reranker, without its generation step
    from rag_grounded.retrieval.hybrid import HybridRetriever
    from rag_grounded.retrieval.rerank import Reranker

    return HybridRetriever(), Reranker()


SECTIONS = {"laravel": "laravel/", "eloquent": "laravel/eloquent", "filament": "filament/",
            "livewire": "livewire/"}


@tool
def search_docs(query: str, section: Literal["laravel", "eloquent", "filament", "livewire"] | None = None) -> list[dict]:
    """Search the Laravel, Filament and Livewire documentation.
    Returns up to 6 passages, each with its source file and heading.
    Covers official framework documentation ONLY — not Stack Overflow,
    not changelogs, not application source code, not order or billing data.
    If the first call returns nothing relevant, do not rephrase and retry:
    the answer is very likely not in this corpus. Say so instead.

    Args:
        query: A natural-language question. Full sentences beat keywords.
        section: Optional filter, one of "laravel" | "eloquent" | "filament" | "livewire".
    """
    hybrid, reranker = _retrieval()
    candidates = hybrid.vectors.get(hybrid.retrieve(query))
    if section:
        candidates = [c for c in candidates if c.source.startswith(SECTIONS[section])]
    # below this rerank score nothing is about the question; returning it
    # anyway just invites the model to keep searching
    ranked = [(c, s) for c, s in reranker.rank(query, candidates)[:6] if s > -2.0]
    return [{"source": c.source, "heading": c.heading, "text": c.text[:1200]} for c, _ in ranked]
