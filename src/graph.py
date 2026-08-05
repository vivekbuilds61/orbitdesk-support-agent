"""Builds the LangGraph StateGraph for the OrbitDesk support agent.

Routing summary (see diagram/graph_diagram.png):

    triage --(answerable | requires_escalation)--> retrieve --> generate --> verify
        --(pass)--> finalize --> END
        --(fail, attempts < MAX_GENERATION_ATTEMPTS)--> generate   [retry loop, capped]
        --(fail, attempts exhausted)--> finalize_safe_failure --> END
    triage --(requires_clarification)--> finalize_clarification --> END
    triage --(out_of_scope)--> finalize_out_of_scope --> END

The retry loop is capped by MAX_GENERATION_ATTEMPTS so a persistently
ungrounded model response can never cause infinite recursion; LangGraph's
own recursion_limit is also set defensively in run_graph() as a second
safety net.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes.finalize import (
    finalize_answerable,
    finalize_clarification,
    finalize_escalation,
    finalize_out_of_scope,
    finalize_safe_failure,
)
from .nodes.generate import generate
from .nodes.retrieve import retrieve
from .nodes.triage import triage
from .nodes.verify import verify
from .state import AgentState

MAX_GENERATION_ATTEMPTS = 2  # 1 initial attempt + 1 retry


def _route_after_triage(state: AgentState) -> str:
    classification = state["classification"]
    if classification in ("answerable", "requires_escalation"):
        return "retrieve"
    if classification == "requires_clarification":
        return "finalize_clarification"
    return "finalize_out_of_scope"


def _route_after_verify(state: AgentState) -> str:
    """Single dispatch point out of `verify`.

    - failed + retry budget remaining -> loop back to `generate`
    - failed + budget exhausted        -> deterministic safe-failure path
    - passed                           -> the right finalize node for this
                                           question's classification
    """
    verification = state["verification"]
    if not verification["passed"]:
        if state.get("generation_attempts", 0) < MAX_GENERATION_ATTEMPTS:
            return "retry"
        return "finalize_safe_failure"
    if state["classification"] == "requires_escalation":
        return "finalize_escalation"
    return "finalize_answerable"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_node("finalize_answerable", finalize_answerable)
    graph.add_node("finalize_escalation", finalize_escalation)
    graph.add_node("finalize_clarification", finalize_clarification)
    graph.add_node("finalize_out_of_scope", finalize_out_of_scope)
    graph.add_node("finalize_safe_failure", finalize_safe_failure)

    graph.set_entry_point("triage")

    graph.add_conditional_edges(
        "triage",
        _route_after_triage,
        {
            "retrieve": "retrieve",
            "finalize_clarification": "finalize_clarification",
            "finalize_out_of_scope": "finalize_out_of_scope",
        },
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        _route_after_verify,
        {
            "retry": "generate",
            "finalize_safe_failure": "finalize_safe_failure",
            "finalize_answerable": "finalize_answerable",
            "finalize_escalation": "finalize_escalation",
        },
    )

    graph.add_edge("finalize_answerable", END)
    graph.add_edge("finalize_escalation", END)
    graph.add_edge("finalize_clarification", END)
    graph.add_edge("finalize_out_of_scope", END)
    graph.add_edge("finalize_safe_failure", END)

    return graph.compile()


def run_graph(question: str) -> AgentState:
    app = build_graph()
    initial_state: AgentState = {"question": question, "trace": [], "generation_attempts": 0}
    # recursion_limit is a second, independent safety net on top of
    # MAX_GENERATION_ATTEMPTS above.
    return app.invoke(initial_state, config={"recursion_limit": 25})
