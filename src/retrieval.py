"""Embedding-based retrieval over the KB + resolved-case corpus.

The corpus is embedded once per process (see ``get_index``) and cached, so
repeated questions in the same run don't re-embed the whole knowledge base.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .corpus import Chunk, load_all_chunks
from .models import cosine, get_embedder


@dataclass
class IndexedChunk:
    chunk: Chunk
    embedding: list[float]


@lru_cache(maxsize=1)
def get_index() -> list[IndexedChunk]:
    embedder = get_embedder()
    chunks = load_all_chunks()
    return [IndexedChunk(c, embedder.encode(c.text)) for c in chunks]


def search(query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
    embedder = get_embedder()
    q_vec = embedder.encode(query)
    index = get_index()
    scored = [(ic.chunk, cosine(q_vec, ic.embedding)) for ic in index]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
