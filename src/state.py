"""Shared typed state passed between every node in the LangGraph workflow.

Keeping this in one place is what the assignment calls "shared typed state" -
every node reads and writes to the same TypedDict, and LangGraph merges
partial updates returned by each node into this structure.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

Classification = Literal[
    "answerable",
    "requires_clarification",
    "requires_escalation",
    "out_of_scope",
    "safe_failure",
]


class SourcePassage(TypedDict):
    source_id: str
    passage: str
    score: float


class VerificationResult(TypedDict):
    passed: bool
    failed_checks: list[str]


class AgentState(TypedDict, total=False):
    # --- input ---
    question: str

    # --- triage node output ---
    classification: Classification
    triage_reason: str

    # --- retrieval node output ---
    retrieved: list[SourcePassage]
    retrieval_max_score: float

    # --- generation node output ---
    draft_answer: str
    generation_attempts: int

    # --- verification node output ---
    verification: VerificationResult

    # --- final structured output (matches data/output_schema.json) ---
    answer: str
    sources: list[dict[str, str]]
    confidence: float
    requires_human: bool
    reason: str
    clarification_question: Optional[str]
    warnings: list[str]

    # --- observability ---
    trace: list[str]  # ordered list of node names that executed
    timings_ms: dict[str, float]
