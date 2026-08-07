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

    keywords = {
        w.lower().strip(".,?!:;")
        for w in query.split()
        if len(w) > 2
    }

    scored = []

    for ic in get_index():
        semantic = cosine(q_vec, ic.embedding)

        text = ic.chunk.text.lower()

        lexical = 0.0
        for word in keywords:
            if word in text:
                lexical += 0.04

        final_score = semantic + lexical
        scored.append((ic.chunk, final_score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Diversify by source
    selected = []
    seen_sources = set()

    for chunk, score in scored:
        if chunk.source_id not in seen_sources:
            selected.append((chunk, score))
            seen_sources.add(chunk.source_id)

        if len(selected) >= top_k:
            break

    return selected