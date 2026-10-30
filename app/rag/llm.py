"""Chat interface, an OpenAI-compatible implementation, and provider factories.

One OpenAI SDK client serves both providers: Gemini through its OpenAI-compatible endpoint,
and OpenAI through the SDK default. Switching provider is a `.env` change plus a re-index.
The SDK retries HTTP 408/409/429/5xx with exponential backoff, up to `llm_max_retries` times.
"""

from collections.abc import Iterator
from typing import Protocol

from openai import OpenAI

from app.config import Settings
from app.rag.embeddings import OpenAICompatEmbedder
from app.rag.errors import MissingAPIKeyError

SUPPORTED_PROVIDERS = ("gemini", "openai")


class LLM(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...

    def stream(self, system: str, user: str) -> Iterator[str]: ...


class OpenAICompatLLM:
    """Calls `/chat/completions` at temperature 0, so answers are reproducible."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield the reply's text deltas as they arrive (empty deltas are skipped)."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        for event in response:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if delta:
                yield delta


def api_key_variable(provider: str) -> str:
    return f"{provider.upper()}_API_KEY"


def make_client(settings: Settings) -> OpenAI:
    """Build the OpenAI SDK client for `settings.llm_provider`."""
    provider = settings.llm_provider
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"unsupported llm_provider {provider!r}; supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    key = settings.gemini_api_key if provider == "gemini" else settings.openai_api_key
    if key is None:
        raise MissingAPIKeyError(
            f"{api_key_variable(provider)} is not set. Add it to .env (see .env.example)."
        )
    base_url = settings.gemini_base_url if provider == "gemini" else None
    return OpenAI(
        api_key=key.get_secret_value(),
        base_url=base_url,
        max_retries=settings.llm_max_retries,
    )


def get_llm(settings: Settings) -> OpenAICompatLLM:
    return OpenAICompatLLM(make_client(settings), settings.llm_model)


def get_embedder(settings: Settings) -> OpenAICompatEmbedder:
    return OpenAICompatEmbedder(make_client(settings), settings.embedding_model)
