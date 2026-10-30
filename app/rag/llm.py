"""Chat interface, an OpenAI-compatible implementation, and provider factories.

One OpenAI SDK client serves both providers: Gemini through its OpenAI-compatible endpoint,
and OpenAI through the SDK default. Switching provider is a `.env` change plus a re-index.
The SDK retries HTTP 408/409/429/5xx with exponential backoff, up to `llm_max_retries` times.
"""

import logging
import math
import time
from collections.abc import Iterator
from typing import Any, Protocol

import openai
from openai import OpenAI

from app.config import Settings
from app.observability import (
    current_span,
    estimate_tokens,
    record_usage,
    tracing_active,
    usage_events,
)
from app.rag.embeddings import OpenAICompatEmbedder
from app.rag.errors import MissingAPIKeyError

log = logging.getLogger("askwell.llm")

SUPPORTED_PROVIDERS = ("gemini", "openai")


class LLM(Protocol):
    """Chat model. Token usage is optional: an implementation that knows the provider-reported
    usage calls `app.observability.record_usage` (see `OpenAICompatLLM`); otherwise `TracedLLM`
    records an estimate."""

    model: str

    def complete(self, system: str, user: str) -> str: ...

    def stream(self, system: str, user: str) -> Iterator[str]: ...


# Whether the endpoint accepts stream_options={"include_usage": true}. Learned once per process:
# the first rejection turns it off, so later requests do not repeat the failed attempt.
_stream_usage_ok = True


def _report_usage(usage: Any) -> None:
    """Record provider-reported usage on the current span; any malformed shape is ignored."""
    try:
        prompt, completion = int(usage.prompt_tokens), int(usage.completion_tokens)
    except (AttributeError, TypeError, ValueError):
        return
    record_usage(prompt, completion, "reported")


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
        _report_usage(getattr(response, "usage", None))
        return response.choices[0].message.content or ""

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield the reply's text deltas as they arrive (empty deltas are skipped)."""
        global _stream_usage_ok
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        options: dict[str, Any] = {}
        if _stream_usage_ok and tracing_active():
            options["stream_options"] = {"include_usage": True}
        try:
            response = self.client.chat.completions.create(
                model=self.model, temperature=0, stream=True, messages=messages, **options
            )
        except openai.BadRequestError:
            if not options:
                raise
            _stream_usage_ok = False  # the endpoint rejects include_usage: estimate from now on
            log.warning("stream_options include_usage rejected; token usage will be estimated")
            response = self.client.chat.completions.create(
                model=self.model, temperature=0, stream=True, messages=messages
            )
        last_usage = None
        for event in response:
            usage = getattr(event, "usage", None)
            if usage is not None:  # the final usage-only chunk may have no choices
                last_usage = usage
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if delta:
                yield delta
        if last_usage is not None:
            _report_usage(last_usage)


class TracedLLM:
    """Wraps an LLM for one traced request: if the inner call did not record provider-reported
    usage, records an estimate (ceil(chars / 4), `source="estimated"`); marks time to first
    token on the current span for streams. Adds no calls and changes no text."""

    def __init__(self, inner: LLM) -> None:
        self.inner = inner
        self.model = inner.model

    def complete(self, system: str, user: str) -> str:
        before = usage_events()
        text = self.inner.complete(system, user)
        if usage_events() == before:
            record_usage(estimate_tokens(system + user), estimate_tokens(text), "estimated")
        return text

    def stream(self, system: str, user: str) -> Iterator[str]:
        before = usage_events()
        began = time.perf_counter()
        chars = 0
        first = True
        try:
            for delta in self.inner.stream(system, user):
                if first:
                    first = False
                    current_span().set(llm_first_token_ms=(time.perf_counter() - began) * 1000)
                chars += len(delta)
                yield delta
        finally:
            if usage_events() == before:
                record_usage(estimate_tokens(system + user), math.ceil(chars / 4), "estimated")


def traced(llm: LLM) -> LLM:
    """`llm` wrapped for tracing when a trace is active, else `llm` itself."""
    return TracedLLM(llm) if tracing_active() and not isinstance(llm, TracedLLM) else llm


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
