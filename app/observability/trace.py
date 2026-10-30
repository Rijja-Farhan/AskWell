"""A small dependency-free tracer: one `Trace` per request, made of timed `Span`s.

Pipeline code never receives a tracer. It calls `span("name", **attrs)`, which finds the
current request's trace through a `contextvars.ContextVar`; with no active trace (eval runner,
CLI, plain tests) `span()` returns a shared no-op, so behaviour and cost are unchanged. Each
request thread has its own context, so concurrent requests never mix.

Streaming: Starlette iterates the SSE generator in the thread pool, in a context that did not
see the endpoint's `ContextVar.set`. The endpoint therefore keeps the `ActiveTrace` handle and
re-binds it (`ActiveTrace.bind`) at every resumption of the generator. The open-span stack
lives on the trace context object, not in the ContextVar, so a span may stay open across
yields that run in different threads.

Privacy: attributes are ids, counts, scores and timings. `json_safe` caps strings and replaces
unknown objects by their type name. Nothing here knows about question, chunk or answer text.
"""

import contextvars
import math
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

MAX_STR = 200
MAX_DEPTH = 4


class SpanError(BaseModel):
    type: str
    code: str


class Span(BaseModel):
    id: int
    parent: int | None = None
    name: str
    start_ms: float  # offset from the trace start
    duration_ms: float = 0.0
    status: Literal["ok", "error"] = "ok"
    attrs: dict[str, Any] = Field(default_factory=dict)
    error: SpanError | None = None


class Trace(BaseModel):
    trace_id: str
    started_at: str  # UTC ISO 8601
    endpoint: str
    user_id: str | None = None
    session_id: str | None = None
    status: Literal["ok", "error"] = "ok"
    total_ms: float = 0.0
    spans: list[Span] = Field(default_factory=list)
    attrs: dict[str, Any] = Field(default_factory=dict)
    error: SpanError | None = None


class TraceSummary(BaseModel):
    trace_id: str
    started_at: str
    endpoint: str
    user_id: str | None
    session_id: str | None
    status: str
    total_ms: float
    attrs: dict[str, Any]


def summarize(trace: Trace) -> TraceSummary:
    fields = set(TraceSummary.model_fields)
    return TraceSummary(**trace.model_dump(include=fields))


def json_safe(value: Any, depth: int = 0) -> Any:
    """Scalars, lists and dicts of those; long strings are cut; anything else becomes its
    type name (never `str(obj)`, which could carry text)."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:MAX_STR]
    if depth < MAX_DEPTH and isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v, depth + 1) for v in value]
    if depth < MAX_DEPTH and isinstance(value, dict):
        return {str(k)[:MAX_STR]: json_safe(v, depth + 1) for k, v in value.items()}
    return type(value).__name__


def error_info(exc: BaseException, classify: Callable[[BaseException], str] | None) -> SpanError:
    """Exception type and the application's error code only, never the message."""
    try:
        code = classify(exc) if classify else "error"
    except Exception:  # noqa: BLE001
        code = "error"
    return SpanError(type=type(exc).__name__, code=code)


@dataclass
class _Ctx:
    trace: Trace
    clock: Callable[[], float]
    t0: float
    classify: Callable[[BaseException], str] | None
    stack: list["SpanHandle"] = field(default_factory=list)
    next_id: int = 0
    usage_events: int = 0
    finished: bool = False

    def now_ms(self) -> float:
        return (self.clock() - self.t0) * 1000.0


_current: contextvars.ContextVar[_Ctx | None] = contextvars.ContextVar(
    "askwell_trace", default=None
)


class SpanHandle:
    """The live span yielded by `with span(...) as s`."""

    def __init__(self, ctx: _Ctx, model: Span) -> None:
        self._ctx = ctx
        self.model = model

    def set(self, **attrs: Any) -> None:
        for key, value in attrs.items():
            self.model.attrs[key] = json_safe(value)

    def setdefault(self, **attrs: Any) -> None:
        for key, value in attrs.items():
            self.model.attrs.setdefault(key, json_safe(value))

    def elapsed_ms(self) -> float:
        return self._ctx.now_ms() - self.model.start_ms


class _NullSpan:
    """Shared no-op stand-in: context manager and handle in one."""

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def set(self, **attrs: Any) -> None:
        return None

    def setdefault(self, **attrs: Any) -> None:
        return None

    def elapsed_ms(self) -> float:
        return 0.0


NULL_SPAN = _NullSpan()


class _SpanCM:
    def __init__(self, ctx: _Ctx, name: str, attrs: dict[str, Any]) -> None:
        self._ctx, self._name, self._attrs = ctx, name, attrs
        self._handle: SpanHandle | None = None

    def __enter__(self) -> SpanHandle:
        ctx = self._ctx
        parent = ctx.stack[-1].model.id if ctx.stack else None
        model = Span(id=ctx.next_id, parent=parent, name=self._name, start_ms=ctx.now_ms())
        ctx.next_id += 1
        handle = SpanHandle(ctx, model)
        handle.set(**self._attrs)
        ctx.trace.spans.append(model)
        ctx.stack.append(handle)
        self._handle = handle
        return handle

    def __exit__(self, exc_type, exc, tb) -> bool:
        ctx, handle = self._ctx, self._handle
        assert handle is not None
        model = handle.model
        model.duration_ms = max(0.0, ctx.now_ms() - model.start_ms)
        if exc is not None and isinstance(exc, Exception):
            model.status = "error"
            model.error = error_info(exc, ctx.classify)
        if handle in ctx.stack:
            ctx.stack.remove(handle)
        return False


def span(name: str, **attrs: Any):
    """`with span("retrieve", k=5) as s: s.set(...)`. A no-op when no trace is active."""
    ctx = _current.get()
    if ctx is None or ctx.finished:
        return NULL_SPAN
    return _SpanCM(ctx, name, attrs)


def current_span() -> SpanHandle | _NullSpan:
    ctx = _current.get()
    if ctx is None or ctx.finished or not ctx.stack:
        return NULL_SPAN
    return ctx.stack[-1]


def annotate(name: str, **attrs: Any) -> None:
    """Set attributes on the innermost open span called `name` (no-op when there is none)."""
    ctx = _current.get()
    if ctx is None or ctx.finished:
        return
    for handle in reversed(ctx.stack):
        if handle.model.name == name:
            handle.set(**attrs)
            return


def tracing_active() -> bool:
    ctx = _current.get()
    return ctx is not None and not ctx.finished


def current_trace_id() -> str | None:
    ctx = _current.get()
    return ctx.trace.trace_id if ctx is not None else None


def usage_events() -> int:
    """How many usage records this trace has seen (lets a wrapper tell whether the inner call
    recorded provider-reported usage)."""
    ctx = _current.get()
    return ctx.usage_events if ctx is not None else 0


def record_usage(prompt_tokens: int, completion_tokens: int, source: str) -> None:
    """Add one LLM call's token usage to the current span. `source` is `reported` or
    `estimated`. Never raises."""
    try:
        ctx = _current.get()
        if ctx is None or ctx.finished:
            return
        ctx.usage_events += 1
        if not ctx.stack:
            return
        attrs = ctx.stack[-1].model.attrs
        prompt, completion = max(0, int(prompt_tokens)), max(0, int(completion_tokens))
        attrs["llm_calls"] = attrs.get("llm_calls", 0) + 1
        attrs["prompt_tokens"] = attrs.get("prompt_tokens", 0) + prompt
        attrs["completion_tokens"] = attrs.get("completion_tokens", 0) + completion
        attrs["total_tokens"] = attrs["prompt_tokens"] + attrs["completion_tokens"]
        previous = attrs.get("usage_source")
        attrs["usage_source"] = source if previous in (None, source) else "mixed"
    except Exception:  # noqa: BLE001 - usage must never fail a request
        return


def record_embedding_usage(texts: int, tokens: int | None) -> None:
    """Add an embedding call to the current span: texts sent and, if reported, tokens."""
    try:
        ctx = _current.get()
        if ctx is None or ctx.finished or not ctx.stack:
            return
        attrs = ctx.stack[-1].model.attrs
        attrs["embedding_calls"] = attrs.get("embedding_calls", 0) + 1
        if tokens is not None:
            attrs["embedding_tokens"] = attrs.get("embedding_tokens", 0) + int(tokens)
    except Exception:  # noqa: BLE001
        return


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


class ActiveTrace:
    """The request's trace while it runs. Created by `Tracer.start`."""

    def __init__(self, ctx: _Ctx) -> None:
        self._ctx = ctx

    @property
    def trace_id(self) -> str:
        return self._ctx.trace.trace_id

    def bind(self) -> None:
        """Make this the current trace in the calling thread's context (no reset). Used by the
        stream generator at every resumption."""
        _current.set(self._ctx)

    @contextmanager
    def activate(self):
        token = _current.set(self._ctx)
        try:
            yield self
        finally:
            _current.reset(token)

    def now_ms(self) -> float:
        return self._ctx.now_ms()

    def summary(self, **attrs: Any) -> None:
        for key, value in attrs.items():
            self._ctx.trace.attrs[key] = json_safe(value)

    def put_text(self, key: str, text: str, limit: int = 4000) -> None:
        """Store free text in the summary. Only for `trace_store_text=true`."""
        self._ctx.trace.attrs[key] = text[:limit]

    def finish(self, error: BaseException | SpanError | None = None) -> Trace:
        """Close the trace (idempotent) and return it. Derives usage and retrieval totals."""
        ctx = self._ctx
        trace = ctx.trace
        if ctx.finished:
            return trace
        ctx.finished = True
        now = ctx.now_ms()
        for handle in ctx.stack:  # spans left open (client went away): close them
            handle.model.duration_ms = max(0.0, now - handle.model.start_ms)
        ctx.stack.clear()
        trace.total_ms = round(now, 3)
        for s in trace.spans:
            s.start_ms = round(s.start_ms, 3)
            s.duration_ms = round(s.duration_ms, 3)
        if error is not None:
            trace.status = "error"
            trace.error = error if isinstance(error, SpanError) else error_info(error, ctx.classify)
        elif any(s.status == "error" for s in trace.spans):
            trace.status = "error"
            failed = next(s for s in trace.spans if s.status == "error")
            trace.error = failed.error
        _derive_summary(trace)
        return trace


def _derive_summary(trace: Trace) -> None:
    prompt = completion = calls = 0
    for s in trace.spans:
        prompt += int(s.attrs.get("prompt_tokens", 0))
        completion += int(s.attrs.get("completion_tokens", 0))
        calls += int(s.attrs.get("llm_calls", 0))
    chunks: list[dict] = []
    for s in trace.spans:
        if s.name == "retrieve" and isinstance(s.attrs.get("chunks"), list):
            chunks = s.attrs["chunks"]
    top = chunks[0].get("score") if chunks else None
    defaults: dict[str, Any] = {
        "cached": False,
        "refused": False,
        "refusal_reason": None,
        "blocked": False,
    }
    for key, value in defaults.items():
        trace.attrs.setdefault(key, value)
    trace.attrs.update(
        chunks=len(chunks),
        top_score=top,
        prompt_tokens=prompt,
        completion_tokens=completion,
        llm_calls=calls,
    )


class Tracer:
    """Starts traces. `enabled=False` makes `start` return None, so nothing is ever active."""

    def __init__(
        self,
        enabled: bool = True,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        classify: Callable[[BaseException], str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.clock = clock
        self.now = now
        self.classify = classify

    def start(
        self, endpoint: str, user_id: str | None = None, session_id: str | None = None
    ) -> ActiveTrace | None:
        if not self.enabled:
            return None
        trace = Trace(
            trace_id=uuid.uuid4().hex,
            started_at=self.now().astimezone(timezone.utc).isoformat(timespec="milliseconds"),
            endpoint=endpoint,
            user_id=user_id,
            session_id=session_id,
        )
        return ActiveTrace(_Ctx(trace, self.clock, self.clock(), self.classify))

    def span(self, name: str, **attrs: Any):
        return span(name, **attrs)
