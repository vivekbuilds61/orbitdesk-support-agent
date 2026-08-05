"""Command-line entry point.

Usage:
    python -m src.cli --question "..."
    python -m src.cli --sample-questions          # runs all of data/sample_questions.json
    python -m src.cli --sample-questions --trace   # also prints the node execution trace
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .graph import run_graph

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _run_one(question: str, show_trace: bool) -> dict:
    start = time.time()
    state = run_graph(question)
    elapsed = time.time() - start

    output = {
        "classification": state["classification"],
        "answer": state["answer"],
        "sources": state["sources"],
        "confidence": state["confidence"],
        "requires_human": state["requires_human"],
        "reason": state["reason"],
        "clarification_question": state.get("clarification_question"),
        "warnings": state.get("warnings", []),
    }
    if show_trace:
        output["_trace"] = state.get("trace", [])
        output["_elapsed_seconds"] = round(elapsed, 2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="OrbitDesk support agent")
    parser.add_argument("--question", type=str, help="Ask a single question")
    parser.add_argument("--sample-questions", action="store_true", help="Run data/sample_questions.json")
    parser.add_argument("--trace", action="store_true", help="Include the node execution trace in output")
    parser.add_argument("--out", type=str, help="Write JSON results to this file instead of stdout")
    args = parser.parse_args()

    if not args.question and not args.sample_questions:
        parser.error("Provide --question '...' or --sample-questions")

    results = []
    if args.sample_questions:
        data = json.loads((DATA_DIR / "sample_questions.json").read_text())
        for q in data["questions"]:
            print(f"--- {q['question_id']} ---", file=sys.stderr)
            output = _run_one(q["question"], args.trace)
            results.append({"question_id": q["question_id"], "question": q["question"], **output})
    else:
        output = _run_one(args.question, args.trace)
        results.append({"question": args.question, **output})

    text = json.dumps(results, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
