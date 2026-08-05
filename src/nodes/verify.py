"""Verification node.

Runs three independent, deterministic checks against the model's draft:

1. Non-empty / not a template artifact.
2. Grounded - the draft must be semantically close to the retrieved
   passages (embedding cosine similarity), so the model can't wander off
   into unsupported claims.
3. Safe - the draft must not claim to perform an action KB-010 says support
   cannot do (refunds, credential creation, settings changes, ...), and
   must not present a `superseded` case as current guidance.

If any check fails and a retry budget remains, the graph loops back to
`generate` with the failure reason attached. If it fails again, the graph
routes to a deterministic safe-failure response instead of returning an
ungrounded answer.
"""

from __future__ import annotations

import re

from ..models import cosine, get_embedder
from ..state import AgentState

GROUNDING_THRESHOLD = 0.35

BANNED_ACTION_PATTERNS = [
    r"\bI(?:'ve| have)? (?:issued|processed) (?:a |the )?refund\b",
    r"\bI(?:'ve| have)? (?:created|generated|revealed) (?:a |the |your )?(?:credential|api key|secret)\b",
    r"\bI(?:'ve| have)? changed (?:your |the )?(?:role|permission|setting)\b",
    r"\bI(?:'ve| have)? (?:run|executed|triggered) (?:the |an? )?(?:export|refresh)\b",
    r"\bI(?:'ve| have)? contacted\b",
    r"\bI(?:'ve| have)? cancell?ed (?:your |the )?subscription\b",
]
_BANNED_RE = re.compile("|".join(BANNED_ACTION_PATTERNS), re.IGNORECASE)

_EMPTY_OR_PLACEHOLDER_RE = re.compile(r"^\s*$|\{|\}|<insert|TODO", re.IGNORECASE)


def _presents_superseded_as_current(draft: str, retrieved: list[dict], corpus_status: dict[str, str]) -> bool:
    """Heuristic: if the only supporting source for a claim is `superseded`
    and the draft doesn't say so, flag it. We approximate this by checking
    whether a superseded source id is cited without the word 'historical'/
    'previously'/'no longer' nearby."""
    superseded_ids = [r["source_id"] for r in retrieved if corpus_status.get(r["source_id"]) == "superseded"]
    if not superseded_ids:
        return False
    mentions_caveat = re.search(r"\b(historical|previously|no longer|superseded|outdated|used to)\b", draft, re.IGNORECASE)
    cites_superseded = any(sid in draft for sid in superseded_ids)
    return cites_superseded and not mentions_caveat


def verify(state: AgentState) -> dict:
    trace = state.get("trace", []) + ["verify"]
    draft = state.get("draft_answer", "")
    retrieved = state.get("retrieved", [])
    failed_checks: list[str] = []

    if _EMPTY_OR_PLACEHOLDER_RE.match(draft) or len(draft.strip()) < 15:
        failed_checks.append("empty_or_placeholder_answer")

    if retrieved:
        embedder = get_embedder()
        draft_vec = embedder.encode(draft)
        passage_text = " ".join(r["passage"] for r in retrieved)
        passage_vec = embedder.encode(passage_text)
        grounding_score = cosine(draft_vec, passage_vec)
        if grounding_score < GROUNDING_THRESHOLD:
            failed_checks.append(f"insufficient_grounding(score={grounding_score:.2f})")
    else:
        grounding_score = 0.0
        failed_checks.append("no_retrieved_evidence")

    if _BANNED_RE.search(draft):
        failed_checks.append("claims_unsupported_action")

    corpus_status = {r["source_id"]: _lookup_status(r["source_id"]) for r in retrieved}
    if _presents_superseded_as_current(draft, retrieved, corpus_status):
        failed_checks.append("presents_superseded_case_as_current")

    passed = len(failed_checks) == 0

    return {
        "verification": {"passed": passed, "failed_checks": failed_checks},
        "trace": trace,
    }


def _lookup_status(source_id: str) -> str:
    # Lightweight lookup avoiding a second import cycle; reads the same
    # corpus module used by retrieval so status stays authoritative.
    from ..corpus import load_all_chunks

    for chunk in load_all_chunks():
        if chunk.source_id == source_id:
            return chunk.status
    return "unknown"
