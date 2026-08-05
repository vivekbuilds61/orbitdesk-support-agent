"""Renders diagram/graph_diagram.png from the actual node/edge names used in
src/graph.py, so the diagram can't drift out of sync with the real graph."""

import graphviz

dot = graphviz.Digraph("orbitdesk_agent", format="png")
dot.attr(rankdir="TB", fontname="Helvetica", bgcolor="white")
dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="11")

dot.node("start", "User question", shape="ellipse", fillcolor="#eeeeee")
dot.node("triage", "triage\n(deterministic safety rules\n+ embedding intent scoring)", fillcolor="#d9c9f0")
dot.node("retrieve", "retrieve\n(embedding top-k search)", fillcolor="#bfe3e0")
dot.node("generate", "generate\n(local LLM, grounded prompt)", fillcolor="#bfe3e0")
dot.node("verify", "verify\n(schema + grounding\n+ banned-action checks)", fillcolor="#f5d38a")
dot.node("finalize_answerable", "finalize_answerable", fillcolor="#bfe3ba")
dot.node("finalize_escalation", "finalize_escalation", fillcolor="#bfe3ba")
dot.node("finalize_clarification", "finalize_clarification", fillcolor="#f0a898")
dot.node("finalize_out_of_scope", "finalize_out_of_scope", fillcolor="#f0a898")
dot.node("finalize_safe_failure", "finalize_safe_failure\n(after retry exhausted)", fillcolor="#dddddd")
dot.node("end", "END\n(schema-conformant JSON)", shape="ellipse", fillcolor="#eeeeee")

dot.edge("start", "triage")
dot.edge("triage", "retrieve", label="answerable /\nrequires_escalation")
dot.edge("triage", "finalize_clarification", label="requires_clarification")
dot.edge("triage", "finalize_out_of_scope", label="out_of_scope")
dot.edge("retrieve", "generate")
dot.edge("generate", "verify")
dot.edge("verify", "generate", label="failed, retry budget left\n(max 1 retry)", style="dashed")
dot.edge("verify", "finalize_answerable", label="passed\n(answerable)")
dot.edge("verify", "finalize_escalation", label="passed\n(requires_escalation)")
dot.edge("verify", "finalize_safe_failure", label="failed, retries exhausted")

for n in [
    "finalize_answerable",
    "finalize_escalation",
    "finalize_clarification",
    "finalize_out_of_scope",
    "finalize_safe_failure",
]:
    dot.edge(n, "end")

dot.render("diagram/graph_diagram", cleanup=True)
print("wrote diagram/graph_diagram.png")
