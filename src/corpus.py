"""Builds a flat corpus of retrievable chunks from the supplied material.

Each chunk carries a stable ``source_id`` (a KB document id or a resolved
case id) and a ``passage`` id so that responses can cite exactly which
section supported an answer, per the assignment's output schema.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KB_DIR = DATA_DIR / "knowledge_base"
RESOLVED_CASES_PATH = DATA_DIR / "resolved_cases.json"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    source_id: str       # e.g. "KB-004" or "CASE-1041"
    passage_id: str      # e.g. "KB-004#Troubleshooting a Missed Export"
    text: str            # the chunk text used for embedding + citation
    kind: str            # "kb" or "case"
    status: str          # "current" | "resolved" | "escalated" | "superseded"


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    fm_text, body = match.groups()
    meta: dict[str, str] = {}
    for line in fm_text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("[]")
    return meta, body


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split a KB doc body into (heading, text) sections on '## ' headers."""
    parts = HEADING_RE.split(body)
    # parts[0] is the intro text before the first "## " heading (if any)
    sections: list[tuple[str, str]] = []
    intro = parts[0].strip()
    if intro:
        # strip the leading '# Title' line from the intro if present
        intro_lines = [l for l in intro.splitlines() if not l.startswith("# ")]
        intro_clean = "\n".join(intro_lines).strip()
        if intro_clean:
            sections.append(("Overview", intro_clean))
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if text:
            sections.append((heading, text))
    return sections


def load_kb_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for md_path in sorted(KB_DIR.glob("*.md")):
        raw = md_path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        doc_id = meta.get("document_id", md_path.stem)
        status = meta.get("status", "current")
        title = meta.get("title", md_path.stem)
        for heading, text in _split_sections(body):
            passage_id = f"{doc_id}#{heading}"
            chunk_text = f"{title} - {heading}\n{text}"
            chunks.append(
                Chunk(
                    source_id=doc_id,
                    passage_id=passage_id,
                    text=chunk_text,
                    kind="kb",
                    status=status,
                )
            )
    return chunks


def load_case_chunks() -> list[Chunk]:
    data = json.loads(RESOLVED_CASES_PATH.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for case in data["cases"]:
        case_id = case["case_id"]
        status = case["status"]
        lines = [f"Case {case_id}: {case['title']} (status: {status})"]
        if case.get("symptoms"):
            lines.append("Symptoms: " + "; ".join(case["symptoms"]))
        if case.get("resolution"):
            lines.append("Resolution steps: " + "; ".join(case["resolution"]))
        if case.get("important_limit"):
            lines.append("Important limit: " + case["important_limit"])
        if case.get("superseded_reason"):
            lines.append("Superseded reason: " + case["superseded_reason"])
        text = "\n".join(lines)
        chunks.append(
            Chunk(
                source_id=case_id,
                passage_id=f"{case_id}#resolution",
                text=text,
                kind="case",
                status=status,
            )
        )
    return chunks


@lru_cache(maxsize=1)
def load_all_chunks() -> tuple[Chunk, ...]:
    return tuple(load_kb_chunks() + load_case_chunks())


if __name__ == "__main__":
    all_chunks = load_all_chunks()
    print(f"Loaded {len(all_chunks)} chunks")
    for c in all_chunks[:5]:
        print(c.passage_id, "->", c.text[:80].replace("\n", " "))
