"""Generation node.

The model is only asked to write prose. Everything structural - which
sources get cited, the confidence score, the final JSON shape - is computed
deterministically in verify.py / graph.py, not left to the model. This is
the "clear separation between deterministic code and model reasoning" the
brief asks for.
"""

from __future__ import annotations

from .. import models
from ..state import AgentState

SYSTEM_INSTRUCTIONS = """You are the OrbitDesk support assistant. Answer ONLY using the passages \
below. Do not use outside knowledge. If a passage is marked (status: superseded), you may only \
mention it as historical context, never as current guidance. Never claim to perform an action \
support cannot do (change settings, issue refunds, create credentials, contact people). If the \
passages do not contain enough information, say so plainly instead of guessing. Cite the source \
ids you relied on in parentheses, e.g. (KB-004)."""

PROMPT_TEMPLATE = """{system}

PASSAGES:
{passages}

QUESTION:
{question}

Write a concise, accurate answer grounded only in the passages above."""

REVISION_SUFFIX = """

Your previous answer failed this check: {failure_reason}
Revise the answer so it fixes that problem, staying grounded only in the passages above."""


def _format_passages(retrieved: list[dict]) -> str:
    """
    Keep the prompt short enough for FLAN-T5.
    """
    lines = []

    for r in retrieved[:3]:
        passage = r["passage"].replace("\n", " ")

        if len(passage) > 250:
            passage = passage[:250] + "..."

        lines.append(f"[{r['source_id']}] {passage}")

    return "\n\n".join(lines)


def generate(state: AgentState) -> dict:
    trace = state.get("trace", []) + ["generate"]
    generator = models.get_generator()

    prompt = PROMPT_TEMPLATE.format(
        system=SYSTEM_INSTRUCTIONS,
        passages=_format_passages(state.get("retrieved", [])),
        question=state["question"],
    )

    attempts = state.get("generation_attempts", 0)
    if attempts > 0 and state.get("verification"):
        failed = ", ".join(state["verification"].get("failed_checks", [])) or "grounding check"
        prompt += REVISION_SUFFIX.format(failure_reason=failed)

    draft = generator.generate(prompt)

    return {
        "draft_answer": draft,
        "generation_attempts": attempts + 1,
        "trace": trace,
    }
