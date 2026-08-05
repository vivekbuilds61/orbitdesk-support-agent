"""Retrieval node: fetches the top-k passages for an answerable/escalation question."""

from __future__ import annotations

from ..retrieval import search
from ..state import AgentState

TOP_K = 5


def retrieve(state: AgentState) -> dict:
    trace = state.get("trace", []) + ["retrieve"]
    matches = search(state["question"], top_k=TOP_K)
    retrieved = [
        {"source_id": chunk.source_id, "passage": chunk.text, "score": round(score, 4)}
        for chunk, score in matches
    ]
    return {
        "retrieved": retrieved,
        "retrieval_max_score": retrieved[0]["score"] if retrieved else 0.0,
        "trace": trace,
    }
