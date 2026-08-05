"""Triage node.

Engineering decision (documented in the README too): classification is a
*hybrid* of deterministic rules and a local embedding model, not a single
opaque model call.

- Out-of-scope / unsupported-action requests are caught by deterministic
  keyword rules first. These are safety-critical per KB-010 ("Unsupported
  Actions", "Out-of-Scope Requests") and must not depend on a model's
  judgement call - a refund request must never slip through because an
  embedding similarity score landed just under a threshold.
- Escalation vs. clarification vs. answerable is then scored using the same
  local embedding model already loaded for retrieval (no second model to
  load), by comparing the question against a small set of prototype
  utterances for each intent, combined with a retrieval-confidence signal
  (a question with no good KB match is more likely to need clarification).

This keeps the deterministic code and the model reasoning clearly separated,
as asked for in the assignment brief.
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
    r"\bcontact (them|the (customer|user|provider)) (for|on behalf of) me\b",
]
_OUT_OF_SCOPE_RE = re.compile("|".join(OUT_OF_SCOPE_PATTERNS), re.IGNORECASE)

# --- embedding-based intent scoring --------------------------------------- #

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

# secondary deterministic cue: escalation language almost always references
# a prior attempt or a repeated failure count
_PRIOR_ATTEMPT_RE = re.compile(
    r"\balready (tried|checked|attempted)\b|\btwo (consecutive|repeated|failed)\b|"
    r"\bstill (fails|failing|doesn't work|does not work)\b|\bescalat",
    re.IGNORECASE,
)


def _prototype_score(embedder, query_vec: list[float], prototypes: list[str]) -> float:
    scores = [cosine(query_vec, embedder.encode(p)) for p in prototypes]
    return max(scores)


def triage(state: AgentState) -> dict:
    question = state["question"]
    trace = state.get("trace", []) + ["triage"]

    if _OUT_OF_SCOPE_RE.search(question):
        return {
            "classification": "out_of_scope",
            "triage_reason": "Matched a deterministic out-of-scope / unsupported-action pattern (KB-010).",
            "trace": trace,
        }

    embedder = get_embedder()
    q_vec = embedder.encode(question)

    top_matches = search(question, top_k=3)
    retrieval_max_score = top_matches[0][1] if top_matches else 0.0

    escalation_score = _prototype_score(embedder, q_vec, ESCALATION_PROTOTYPES)
    clarification_score = _prototype_score(embedder, q_vec, CLARIFICATION_PROTOTYPES)
    has_prior_attempt_language = bool(_PRIOR_ATTEMPT_RE.search(question))

    if escalation_score > ESCALATION_THRESHOLD and has_prior_attempt_language:
        classification = "requires_escalation"
        reason = (
            f"Escalation-intent similarity {escalation_score:.2f} with prior-attempt "
            "language present (KB-008 escalation conditions)."
        )
    elif retrieval_max_score < RETRIEVAL_LOW_CONFIDENCE and clarification_score > CLARIFICATION_THRESHOLD:
        classification = "requires_clarification"
        reason = (
            f"Low retrieval confidence ({retrieval_max_score:.2f}) and the question lacks "
            "the object/symptom/error detail needed to pick a documented path (KB-010)."
        )
    else:
        classification = "answerable"
        reason = f"Best KB match score {retrieval_max_score:.2f}; proceeding to retrieval."

    return {
        "classification": classification,
        "triage_reason": reason,
        "retrieval_max_score": retrieval_max_score,
        "trace": trace,
    }
