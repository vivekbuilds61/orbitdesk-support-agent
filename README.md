# OrbitDesk Support Agent

A LangGraph-orchestrated support agent that answers questions about the fictional
"OrbitDesk" product using only its supplied knowledge base and resolved-case history,
running entirely on small local Hugging Face models (no third-party LLM API).

See `diagram/graph_diagram.png` for the graph, or regenerate it with
`python diagram/render_diagram.py`.

## Architecture

```
User question
     |
   triage  -- deterministic safety rules + local-embedding intent scoring
     |------------------------------------------------------------.
  (answerable / requires_escalation)   requires_clarification   out_of_scope
     |                                        |                      |
  retrieve  (embedding top-k search)   finalize_clarification  finalize_out_of_scope
     |                                        |                      |
  generate  (local LLM, grounded prompt)      |                      |
     |                                        |                      |
   verify  (schema + grounding +              |                      |
            banned-action checks)              |                      |
     |----retry once if failed-----.          |                      |
     |                              v          |                      |
  passed -> finalize_*   finalize_safe_failure |                      |
     |________________________________|________|______________________|
                                       |
                                      END  (schema-conformant JSON)
```

**Design principle followed throughout:** deterministic Python code and model
reasoning are kept strictly separate. Models are only ever asked to (a) produce an
embedding vector or (b) write grounded prose. Every structural decision — routing,
which sources get cited, the confidence score, the final JSON shape, and all
safety-critical refusals — is plain, testable Python.

### Nodes

| Node | What it does | Model used |
|---|---|---|
| `triage` | Deterministic regex safety net catches refund/legal/prompt-injection requests first (never model-dependent). Otherwise scores the question against escalation/clarification prototype phrases using the embedding model, combined with retrieval confidence. | local embedding model (shared with retrieval) |
| `retrieve` | Embeds the query, cosine-similarity ranks all KB-doc sections + resolved cases, returns top-5. | local embedding model |
| `generate` | Builds a prompt containing only the retrieved passages + explicit "don't invent, cite sources, don't claim unsupported actions" instructions; asks the local LLM for prose only. | local generation model |
| `verify` | Three independent checks: (1) non-empty/non-placeholder, (2) grounded — embedding similarity between the draft and the retrieved passages must clear a threshold, (3) safe — regex check for claimed unsupported actions and for presenting a `superseded` case as current guidance. | local embedding model (for the grounding check) |
| `finalize_*` | Deterministically assembles the final JSON object matching `data/output_schema.json` for each of the five classifications. | none (pure code) |

### Retry and recursion safety

If `verify` fails, the graph loops back to `generate` **at most once**
(`MAX_GENERATION_ATTEMPTS = 2` in `src/graph.py`), with the failure reason appended
to the prompt. If it fails again, the graph routes to `finalize_safe_failure`
instead of ever returning an ungrounded answer. `run_graph()` also sets LangGraph's
own `recursion_limit` as an independent second safety net.

## Model choices

- **Embedding:** `sentence-transformers/all-MiniLM-L6-v2` — 22M params, CPU-friendly,
  strong general-purpose sentence similarity, used for both retrieval and the
  grounding/intent-scoring checks (one model loaded, multiple uses).
- **Generation:** `google/flan-t5-base` — ~250M params, instruction-tuned, runs
  comfortably on CPU, good at short extractive/grounded answers without needing a
  chat template. (`Qwen/Qwen2.5-0.5B-Instruct` is a reasonable swap if you have a
  GPU and want more fluent prose — change `GENERATION_MODEL_NAME` in `src/models.py`
  and switch the pipeline to `"text-generation"` with its chat template.)

Both are wired through a lazy-loaded, cached wrapper (`src/models.py`) so each model
loads once per process.

## A note on this build environment

This repository was scaffolded inside a sandboxed tool environment whose network
access is restricted to package registries (PyPI, npm, GitHub) — it **cannot reach
`huggingface.co`** to download the actual embedding/generation model weights. So:

- Everything that doesn't require downloading model weights has been built and
  verified here: the LangGraph wiring, all five node implementations, the
  knowledge-base/case parser, the deterministic safety rules, the automated test
  suite (9/9 passing), and the diagram.
- A `stub` model backend (`src/models.py`: `StubEmbedder` / `StubGenerator`) — a
  deterministic, dependency-free embedder and extractive "generator" — was used to
  exercise the **entire graph end-to-end**, including the retry-then-safe-failure
  path, without any network access. `outputs/stub_sample_run.json` is a real run of
  all five sample questions through the full graph in this mode, and every one
  routes to the classification you'd expect (see table below).
- To get natural-language answers from the real models, run this on a machine with
  internet access using the `real` backend (the default) — see **Setup** below. The
  first run will download roughly 90MB (embedder) plus 950MB (generator) from the Hub.

## Setup (on a machine with Hugging Face access)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# single question
python -m src.cli --question "How do Admins change the workspace timezone?"

# all sample questions, with the node trace and timing
python -m src.cli --sample-questions --trace --out outputs/sample_run.json

# force the dependency-free stub backend (no model download, no network)
ORBITDESK_MODEL_BACKEND=stub python -m src.cli --sample-questions
```

## Tests

```bash
ORBITDESK_MODEL_BACKEND=stub pytest tests/ -v
```

9 tests, all passing in this environment:
- Each of the 5 sample questions routes to the expected `classification` /
  `requires_human`, and the output validates against `data/output_schema.json`.
- Out-of-scope requests never reach `retrieve` or `generate` (the safety net fires
  before any model call, not merely happens to produce a refusal).
- Answerable answers always cite at least one real source id.
- A forced always-ungrounded generator triggers exactly one retry, then routes to
  `finalize_safe_failure` with `requires_human=True` — verified structurally, not
  by matching generated wording.
- The graph terminates within the retry budget (recursion is bounded).

## Sample run outcomes (stub backend, this environment)

| Question | Classification | requires_human |
|---|---|---|
| Q-001 (timezone change, missed export) | `answerable` | False |
| Q-002 (Viewer asking to create API credential) | `answerable` | False |
| Q-003 ("sync is not working, fix it") | `requires_clarification` | False |
| Q-004 (two `render_failed` after documented checks) | `requires_escalation` | True |
| Q-005 (prompt-injection refund/legal-advice attempt) | `out_of_scope` | False |

Full JSON in `outputs/stub_sample_run.json`.

## Known limitations / trade-offs

- Triage's escalation/clarification split uses cosine similarity against a small,
  hand-written prototype set plus a couple of regex cues (`already tried`, `still
  fails`, ...). This is deliberately simple and interpretable rather than a trained
  classifier; it will misroute genuinely novel phrasings that don't resemble either
  prototype set or the KB corpus. A production version would want a labeled set of
  historical tickets to either fine-tune thresholds or train a small classifier head.
- The grounding check in `verify` is a single embedding-similarity threshold
  (`GROUNDING_THRESHOLD = 0.35`) over the whole draft vs. the whole retrieved-passage
  block. It won't catch a mostly-grounded answer with one small fabricated clause,
  only wholesale ungrounded drafts. A stronger version would check groundedness
  per-sentence.
- The `superseded`-case caveat check in `verify` is a lexical heuristic (looks for
  words like "historical"/"previously" near a cited superseded source id), not a
  semantic entailment check.
- Retrieval chunks KB docs by `##` section, which works well for this corpus but
  would need re-tuning (e.g. sentence-window or overlapping chunking) for longer or
  differently structured documentation.
- `RealGenerator`/`RealEmbedder` pin `revision="main"` as placeholders — for a real
  production deployment, pin an actual commit hash so behavior can't silently drift
  when the upstream Hub repo is updated.

## Disclosure

This repository (design, code, tests, README, and diagram) was built with the
assistance of Claude (Anthropic), based on the assignment brief and supplied
material. The routing logic, model choices, and trade-offs above reflect decisions
made and verified during that process, including running the full test suite and a
full stub-backend graph execution over all five sample questions in this
environment before this README was written.
