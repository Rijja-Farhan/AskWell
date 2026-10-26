"""Shared fixtures and offline fakes for the RAG tests (no network, no API keys)."""

import hashlib
import math
import re
from pathlib import Path

import pytest

from app.config import Settings
from app.ingestion.models import ParsedDocument
from app.ingestion.pipeline import load_corpus

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Deterministic hashed bag-of-words: each token adds 1 to one of `dim` buckets."""

    model = "fake-hashed-bow"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in TOKEN.findall(text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0], norm = 1.0, 1.0  # avoid a zero vector, which has no cosine direction
        return [v / norm for v in vec]


class FakeLLM:
    """Records the prompts it receives and returns a fixed answer."""

    model = "fake-llm"

    def __init__(self, reply: str = "fake answer") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


@pytest.fixture(scope="session")
def corpus() -> list[ParsedDocument]:
    """The real corpus, parsed in-process (data/processed/ is gitignored)."""
    docs, errors = load_corpus(ROOT, Settings(_env_file=None))
    assert errors == []
    return docs


def make_settings(tmp_path: Path, **overrides) -> Settings:
    """Default settings, isolated from `.env`, with the index under tmp_path."""
    values = {
        "vector_db_path": tmp_path / "chroma",
        "embedding_model": FakeEmbedder.model,
        "llm_model": FakeLLM.model,
        "reranker": "none",  # the real reranker downloads a model; tests opt in with a fake
        "version_policy": "all",  # unfiltered, as before Phase 8; filter tests opt in
    }
    return Settings(_env_file=None, **{**values, **overrides})
