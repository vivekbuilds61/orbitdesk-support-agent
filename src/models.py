"""Local model wrappers.

Two concrete backends are provided for both the embedder and the generator:

- The real backend loads an actual Hugging Face model (sentence-transformers
  for embeddings, transformers for generation) and requires network access
  to the Hugging Face Hub the first time it runs, plus the corresponding
  packages installed.
- The stub backend is a deterministic, dependency-free fallback used by the
  automated tests in ``tests/`` (see the assignment requirement: "at least
  one automated test must verify graph routing without depending on the
  exact wording produced by the model"). It never calls a network and never
  imports torch/transformers, so it runs anywhere, including sandboxes with
  no Hugging Face access.

Which backend is used is controlled by the ``ORBITDESK_MODEL_BACKEND``
environment variable ("real" or "stub"; default "real").
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "main"  # pin a commit hash for reproducibility in production
GENERATION_MODEL_NAME = "google/flan-t5-base"
GENERATION_MODEL_REVISION = "main"


def _backend() -> str:
    return os.environ.get("ORBITDESK_MODEL_BACKEND", "real").lower()


# --------------------------------------------------------------------------- #
# Embedder
# --------------------------------------------------------------------------- #


class Embedder(Protocol):
    def encode(self, text: str) -> list[float]: ...


@dataclass
class LoadStats:
    model_name: str
    revision: str
    load_seconds: float
    device: str


class RealEmbedder:
    """Wraps sentence-transformers/all-MiniLM-L6-v2 (or similar)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, revision: str = EMBEDDING_MODEL_REVISION):
        start = time.time()
        from sentence_transformers import SentenceTransformer  # local import: heavy dep

        self.model = SentenceTransformer(model_name, revision=revision)
        device = str(self.model.device)
        self.stats = LoadStats(model_name, revision, time.time() - start, device)
        print(f"[models] loaded embedder {model_name}@{revision} on {device} in {self.stats.load_seconds:.2f}s")

    def encode(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()


class StubEmbedder:
    """Deterministic bag-of-words hashing embedder. No ML deps, no network.

    Good enough to preserve relative similarity between semantically
    overlapping short strings, which is all the routing tests need.
    """

    DIM = 256

    def __init__(self, *_args, **_kwargs):
        self.stats = LoadStats("stub-hash-embedder", "n/a", 0.0, "cpu")

    def encode(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.DIM
            sign = 1.0 if (h // self.DIM) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return StubEmbedder() if _backend() == "stub" else RealEmbedder()


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


class Generator(Protocol):
    def generate(self, prompt: str, max_new_tokens: int = 220) -> str: ...


class RealGenerator:
    """Wraps a small local seq2seq instruction model via transformers."""

    def __init__(self, model_name: str = GENERATION_MODEL_NAME, revision: str = GENERATION_MODEL_REVISION):
        start = time.time()
        from transformers import pipeline  # local import: heavy dep

        self.pipe = pipeline(
            "text2text-generation",
            model=model_name,
            revision=revision,
        )
        device = str(getattr(self.pipe.model, "device", "cpu"))
        self.stats = LoadStats(model_name, revision, time.time() - start, device)
        print(f"[models] loaded generator {model_name}@{revision} on {device} in {self.stats.load_seconds:.2f}s")

    def generate(self, prompt: str, max_new_tokens: int = 220) -> str:
        out = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return out[0]["generated_text"].strip()


class StubGenerator:
    """Deterministic template generator used for offline routing tests.

    It does not attempt to be a good writer - it mechanically summarizes the
    passages it is given so that verification logic (grounding, source
    citation, banned-action checks) can be exercised without network access.
    """

    def __init__(self, *_args, **_kwargs):
        self.stats = LoadStats("stub-template-generator", "n/a", 0.0, "cpu")

    def generate(self, prompt: str, max_new_tokens: int = 220) -> str:
        # Extract the passages block the node builder embeds in the prompt
        # (see nodes/generate.py PROMPT_TEMPLATE) and turn it into a naive
        # extractive answer so grounding checks have real overlap to find.
        marker = "PASSAGES:\n"
        if marker in prompt:
            passages_block = prompt.split(marker, 1)[1].split("\nQUESTION:", 1)[0]
            sentences = re.split(r"(?<=[.!?])\s+", passages_block.strip())
            sentences = [s.strip() for s in sentences if s.strip()][:4]
            return " ".join(sentences)
        return "No grounded answer could be produced from the supplied passages."


@lru_cache(maxsize=1)
def get_generator() -> Generator:
    return StubGenerator() if _backend() == "stub" else RealGenerator()
