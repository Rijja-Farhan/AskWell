"""Embedding interface and an implementation for OpenAI-compatible APIs (OpenAI, Gemini)."""

import re
import time
from collections.abc import Callable
from typing import Protocol

from openai import OpenAI, RateLimitError

RETRY_HINT = re.compile(r"retry in ([\d.]+)s", re.I)
DEFAULT_WAIT = 30.0  # seconds, when a 429 does not say how long to wait
MAX_WAIT = 90.0


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def rate_limit_wait(exc: RateLimitError) -> float:
    """Seconds to wait before retrying, from the provider's "retry in Ns" hint."""
    match = RETRY_HINT.search(str(exc))
    wait = float(match.group(1)) if match else DEFAULT_WAIT
    return min(wait + 1.0, MAX_WAIT)


class OpenAICompatEmbedder:
    """Calls `/embeddings` on any OpenAI-compatible endpoint, in batches.

    Gemini's free tier counts every text in a batch as one request (100 per minute), so a
    large batch can exhaust the quota for a whole minute. The SDK's own retries back off for
    only a few seconds; when they give up on a 429, wait as long as the provider asks and
    retry the batch, up to `rate_limit_waits` times.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        batch_size: int = 100,
        rate_limit_waits: int = 5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.rate_limit_waits = rate_limit_waits
        self.sleep = sleep

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors += self._embed_batch(texts[i : i + self.batch_size])
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(self.rate_limit_waits + 1):
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
                break
            except RateLimitError as exc:
                if attempt == self.rate_limit_waits:
                    raise
                self.sleep(rate_limit_wait(exc))
        if len(response.data) != len(batch):
            raise RuntimeError(f"got {len(response.data)} embeddings for {len(batch)} texts")
        # `index` gives each item's position. Gemini omits index 0 (returns None), so a missing
        # index means 0.
        ordered = sorted(response.data, key=lambda d: d.index or 0)
        return [item.embedding for item in ordered]
