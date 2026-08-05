"""Deterministic (non-model) response templates.

Per KB-010, out-of-scope refusals and clarification requests must not
"guess an error code or claim to have inspected an account" - so these are
plain Python strings, not model output. This is intentional, not a
shortcut: it removes an entire class of hallucination risk from the two
highest-stakes response paths.
"""

from __future__ import annotations

OUT_OF_SCOPE_MESSAGE = (
    "That request is outside the OrbitDesk support knowledge base available to me "
    "(for example: refunds, subscription cancellation, legal/medical/financial advice, "
    "or changing account settings on your behalf). I can't act on it or advise on it here. "
    "For account or billing actions, please contact OrbitDesk support directly through your "
    "workspace's billing page."
)

CLARIFICATION_MESSAGE = (
    "I need a bit more detail before I can point you to the right fix. Could you share: "
    "(1) which feature or object is affected (e.g. a specific schedule, connection, or "
    "dashboard), (2) the exact error code or message you're seeing, if any, and (3) whether "
    "this started after a recent change (like a timezone update)?"
)

ESCALATION_ACK_TEMPLATE = (
    "This looks like it meets the escalation criteria in KB-008. Based on what you've "
    "described, please gather: the workspace ID, the affected object ID (schedule/dashboard/"
    "connection ID), the exact error code, timestamps with timezone, and the troubleshooting "
    "steps already attempted. Do not include passwords, API secrets, tokens, or full exported "
    "customer data. I'll summarize this for the appropriate human team; I can't resolve it "
    "directly myself."
)

SAFE_FAILURE_MESSAGE = (
    "I wasn't able to produce an answer I'm confident is fully supported by the OrbitDesk "
    "documentation and resolved cases for this question. Rather than guess, I'm flagging this "
    "for a human to review. Please include the workspace ID and any specific error codes you "
    "have when this is picked up."
)
