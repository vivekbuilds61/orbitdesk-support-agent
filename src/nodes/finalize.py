"""Finalize node.

Every path in the graph ends here. This is where the final JSON object
(matching data/output_schema.json exactly) gets assembled - deterministically,
from whatever the earlier nodes populated in state. No node before this one
returns end-user-facing JSON; they only populate state fields.
"""

from __future__ import annotations

from ..state import AgentState
from . import templates


def _confidence_from_grounding(state: AgentState) -> float:
    retrieved = state.get("retrieved", [])
    if not retrieved:
        return 0.0
    top = retrieved[0]["score"]
    # squash into [0,1]-ish confidence; cosine sims for MiniLM rarely exceed ~0.8
    return round(min(1.0, max(0.0, top / 0.8)), 2)


def finalize_answerable(state: AgentState) -> dict:
    trace = state.get("trace", []) + ["finalize"]
    sources = [{"source_id": r["source_id"], "passage": r["passage"][:280]} for r in state.get("retrieved", [])[:3]]
    warnings = []
    superseded = [s["source_id"] for s in sources if _is_superseded(s["source_id"])]
    if superseded:
        warnings.append(f"Includes historical context from superseded source(s): {', '.join(superseded)}.")
    return {
        "classification": "answerable",
        "answer": state["draft_answer"],
        "sources": sources,
        "confidence": _confidence_from_grounding(state),
        "requires_human": False,
        "reason": state.get("triage_reason", ""),
        "clarification_question": None,
        "warnings": warnings,
        "trace": trace,
    }


def finalize_escalation(state: AgentState) -> dict:
    trace = state.get("trace", []) + ["finalize"]
    sources = [{"source_id": r["source_id"], "passage": r["passage"][:280]} for r in state.get("retrieved", [])[:3]]
    return {
        "classification": "requires_escalation",
        "answer": templates.ESCALATION_ACK_TEMPLATE,
        "sources": sources,
        "confidence": 0.9,
        "requires_human": True,
        "reason": state.get("triage_reason", ""),
        "clarification_question": None,
        "warnings": [],
        "trace": trace,
    }


def finalize_clarification(state: AgentState) -> dict:
    trace = state.get("trace", []) + ["finalize"]
    return {
        "classification": "requires_clarification",
        "answer": templates.CLARIFICATION_MESSAGE,
        "sources": [],
        "confidence": 0.0,
        "requires_human": False,
        "reason": state.get("triage_reason", ""),
        "clarification_question": templates.CLARIFICATION_MESSAGE,
        "warnings": [],
        "trace": trace,
    }


def finalize_out_of_scope(state: AgentState) -> dict:
    trace = state.get("trace", []) + ["finalize"]
    return {
        "classification": "out_of_scope",
        "answer": templates.OUT_OF_SCOPE_MESSAGE,
        "sources": [],
        "confidence": 1.0,
        "requires_human": False,
        "reason": state.get("triage_reason", ""),
        "clarification_question": None,
        "warnings": [],
        "trace": trace,
    }


def finalize_safe_failure(state: AgentState) -> dict:
    trace = state.get("trace", []) + ["finalize"]
    failed = state.get("verification", {}).get("failed_checks", [])
    return {
        "classification": "safe_failure",
        "answer": templates.SAFE_FAILURE_MESSAGE,
        "sources": [{"source_id": r["source_id"], "passage": r["passage"][:280]} for r in state.get("retrieved", [])[:3]],
        "confidence": 0.0,
        "requires_human": True,
        "reason": f"Verification failed after retry: {', '.join(failed)}",
        "clarification_question": None,
        "warnings": ["Automated answer failed verification and was withheld."],
        "trace": trace,
    }


def _is_superseded(source_id: str) -> bool:
    from ..corpus import load_all_chunks

    for chunk in load_all_chunks():
        if chunk.source_id == source_id:
            return chunk.status == "superseded"
    return False
