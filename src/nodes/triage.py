
"""Triage node.

Classification uses deterministic safety rules and a local embedding model.
Out-of-scope requests are rejected before retrieval-based intent scoring.
Other questions are classified as escalation, clarification, or answerable
using intent prototypes and retrieval confidence.
"""

from __future__ import annotations

import re

from ..models import cosine, get_embedder
from ..retrieval import search
from ..state import AgentState


# --- deterministic safety net -------------------------------------------- #

OUT_OF_SCOPE_PATTERNS = [
    r"\brefund\b",
    r"\bcancel (my |the )?subscription\b",
    r"\bchargeback\b",
    r"\blegal advice\b",
    r"\bsue\b|\blawsuit\b",
    r"\bmedical advice\b",
    r"\bfinancial advice\b",
    r"\binvest(ment)? advice\b",
    r"\bdelete my account\b",
    r"\bignore (the |your |all )?(previous |above )?instructions\b",
    r"\bignore .*(knowledge base|documentation|policy)\b",
    r"\bact as\b.*\b(admin|developer|unrestricted)\b",
    r"\breveal (the |your )?(system prompt|credential secret)\b",
    r"\bcontact (them|the (customer|user|provider)) "
    r"(for|on behalf of) me\b",
]

_OUT_OF_SCOPE_RE = re.compile(
    "|".join(OUT_OF_SCOPE_PATTERNS),
    re.IGNORECASE,
)


# --- embedding-based intent scoring -------------------------------------- #

ESCALATION_PROTOTYPES = [
    "I already tried the documented troubleshooting steps and it still fails",
    "Two consecutive runs failed with the same error code, what should we do next",
    "What information should I collect before escalating this to support",
    "The suggested fix did not work, this needs to be escalated",
]

CLARIFICATION_PROTOTYPES = [
    "Something is broken, please fix it",
    "It is not working, can you help",
    "Sync is not working",
    "My thing is broken can you fix it now",
]

RETRIEVAL_LOW_CONFIDENCE = 0.30
ESCALATION_THRESHOLD = 0.55
CLARIFICATION_THRESHOLD = 0.55


# --- deterministic clarification and escalation cues --------------------- #

_VAGUE_SYNC_RE = re.compile(
    r"\b(?:our\s+)?(?:data\s+)?sync\s+"
    r"(?:is not working|isn't working|is broken)\b",
    re.IGNORECASE,
)

_PRIOR_ATTEMPT_RE = re.compile(
    r"\balready\s+(?:tried|checked|attempted)\b|"
    r"\btwo\s+(?:consecutive|repeated|failed)\b|"
    r"\bstill\s+(?:fails|failing|doesn't work|does not work)\b|"
    r"\bescalat",
    re.IGNORECASE,
)


def _prototype_score(
    embedder,
    query_vec: list[float],
    prototypes: list[str],
) -> float:
    """Return the maximum cosine similarity to the intent prototypes."""
    scores = [
        cosine(query_vec, embedder.encode(prototype))
        for prototype in prototypes
    ]
    return max(scores)


def triage(state: AgentState) -> dict:
    """Classify a support question before answer generation."""
    question = state["question"]
    trace = state.get("trace", []) + ["triage"]

    # 1. Safety-critical requests are rejected first.
    if _OUT_OF_SCOPE_RE.search(question):
        return {
            "classification": "out_of_scope",
            "triage_reason": (
                "Matched a deterministic out-of-scope / unsupported-action "
                "pattern (KB-010)."
            ),
            "trace": trace,
        }

    # 2. A vague sync problem needs more details before troubleshooting.
    # Do not apply this shortcut when the user mentions prior attempts or
    # repeated failures; those questions should reach escalation scoring.
    if (
        _VAGUE_SYNC_RE.search(question)
        and not _PRIOR_ATTEMPT_RE.search(question)
    ):
        return {
            "classification": "requires_clarification",
            "triage_reason": (
                "The sync problem is described too vaguely to select "
                "a reliable troubleshooting path. Ask for the affected "
                "system, observed symptom, and any error message."
            ),
            "trace": trace,
        }

    # 3. Score intent using the existing local embedding model.
    embedder = get_embedder()
    q_vec = embedder.encode(question)

    top_matches = search(question, top_k=3)
    retrieval_max_score = (
        top_matches[0][1] if top_matches else 0.0
    )

    escalation_score = _prototype_score(
        embedder, q_vec, ESCALATION_PROTOTYPES
    )
    clarification_score = _prototype_score(
        embedder, q_vec, CLARIFICATION_PROTOTYPES
    )
    has_prior_attempt_language = bool(
        _PRIOR_ATTEMPT_RE.search(question)
    )

    # 4. Escalate when both intent similarity and prior-attempt evidence
    # satisfy the existing escalation criteria.
    if (
        escalation_score > ESCALATION_THRESHOLD
        and has_prior_attempt_language
    ):
        classification = "requires_escalation"
        reason = (
            f"Escalation-intent similarity {escalation_score:.2f} "
            "with prior-attempt language present "
            "(KB-008 escalation conditions)."
        )

    # 5. Ask for clarification when retrieval is weak and the question
    # resembles an underspecified support request.
    elif (
        retrieval_max_score < RETRIEVAL_LOW_CONFIDENCE
        and clarification_score > CLARIFICATION_THRESHOLD
    ):
        classification = "requires_clarification"
        reason = (
            f"Low retrieval confidence ({retrieval_max_score:.2f}) "
            "and the question lacks the object/symptom/error detail "
            "needed to pick a documented path (KB-010)."
        )

    # 6. Otherwise, continue to the answer-generation workflow.
    else:
        classification = "answerable"
        reason = (
            f"Best KB match score {retrieval_max_score:.2f}; "
            "proceeding to retrieval."
        )

    return {
        "classification": classification,
        "triage_reason": reason,
        "retrieval_max_score": retrieval_max_score,
        "trace": trace,
    }
