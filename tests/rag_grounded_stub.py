"""Just enough of rag-grounded's retriever shape for search_docs."""
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    source: str
    heading: str
    text: str


CHUNKS = [Chunk("1", "laravel/eloquent.md", "Global Scopes", "withoutGlobalScope ..."),
          Chunk("2", "laravel/queues.md", "Jobs", "dispatch ..."),
          Chunk("3", "laravel/eloquent-relationships.md", "Eager", "with ...")]


class Hybrid:
    class vectors:
        @staticmethod
        def get(ids):
            return [c for c in CHUNKS if c.id in ids]

    def retrieve(self, q):
        return ["1", "2", "3"]


class Rerank:
    def rank(self, q, chunks):
        scores = {"1": 5.0, "2": 1.0, "3": -4.0}
        return sorted(((c, scores[c.id]) for c in chunks), key=lambda x: -x[1])
