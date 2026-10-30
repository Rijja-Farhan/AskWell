"""Tracer, stores and token-usage capture (offline, no network)."""

import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.observability import (
    JsonlTraceStore,
    MemoryTraceStore,
    Tracer,
    TraceStores,
    current_span,
    record_usage,
    span,
)
from app.observability.view import render_html, render_text
from app.rag.llm import OpenAICompatLLM, TracedLLM
from tests.conftest import FakeLLM, make_settings


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t

    def advance(self, ms: float) -> None:
        self.t += ms / 1000


def started(clock=None, **kw):
    clock = clock or FakeClock()
    tracer = Tracer(clock=clock, **kw)
    return clock, tracer.start("/chat", "u01", "s1")


# --- spans --------------------------------------------------------------------------------


def test_span_nesting_and_timing_with_a_fake_clock():
    clock, handle = started()
    with handle.activate():
        with span("retrieve", k=5) as outer:
            clock.advance(10)
            with span("dense") as inner:
                clock.advance(20)
                inner.set(candidates=3)
            clock.advance(5)
            outer.set(cache_hit=False)
        with span("generate"):
            clock.advance(40)
    clock.advance(1)
    trace = handle.finish()
    by_name = {s.name: s for s in trace.spans}
    assert [s.name for s in trace.spans] == ["retrieve", "dense", "generate"]
    assert by_name["dense"].parent == by_name["retrieve"].id and by_name["retrieve"].parent is None
    assert (by_name["retrieve"].start_ms, by_name["retrieve"].duration_ms) == (0, 35)
    assert (by_name["dense"].start_ms, by_name["dense"].duration_ms) == (10, 20)
    assert (by_name["generate"].start_ms, by_name["generate"].duration_ms) == (35, 40)
    assert by_name["retrieve"].attrs == {"k": 5, "cache_hit": False}
    assert trace.total_ms == 76 and trace.status == "ok"
    assert trace.model_dump_json()  # JSON-serialisable


def test_error_in_a_span_marks_it_and_keeps_type_and_code_only():
    _, handle = started(classify=None)
    with handle.activate():
        with pytest.raises(ValueError):
            with span("generate"):
                raise ValueError("secret question text")
    trace = handle.finish()
    assert trace.status == "error"
    failed = trace.spans[0]
    assert failed.status == "error" and failed.error.type == "ValueError"
    assert "secret" not in trace.model_dump_json()


def test_span_is_a_noop_without_an_active_trace():
    with span("anything", x=1) as s:
        s.set(y=2)
        assert current_span() is s
    record_usage(1, 1, "reported")  # nothing to record on, nothing raised
    assert span("a") is span("b")  # the shared no-op object


def test_disabled_tracer_starts_nothing():
    assert Tracer(enabled=False).start("/chat") is None


def test_attrs_are_json_safe_and_never_stringify_objects():
    _, handle = started()
    with handle.activate():
        with span("x", obj=object(), long="a" * 500, items=[1, "b"], nan=float("nan")):
            pass
    attrs = handle.finish().spans[0].attrs
    assert attrs["obj"] == "object" and len(attrs["long"]) == 200
    assert attrs["items"] == [1, "b"] and attrs["nan"] is None


def test_finish_is_idempotent_and_closes_open_spans():
    clock, handle = started()
    with handle.activate():
        cm = span("generate")
        cm.__enter__()
        clock.advance(7)
    trace = handle.finish()
    assert trace.spans[0].duration_ms == 7
    assert handle.finish() is trace


def test_contextvar_isolation_between_concurrent_threads():
    tracer = Tracer()
    results: dict[str, list[str]] = {}
    barrier = threading.Barrier(2)

    def work(name: str) -> None:
        handle = tracer.start("/chat", name)
        with handle.activate():
            barrier.wait()
            with span(f"stage-{name}"):
                barrier.wait()  # both threads are inside a span at the same time
                record_usage(1, 1, "estimated")
        results[name] = [s.name for s in handle.finish().spans]

    threads = [threading.Thread(target=work, args=(n,)) for n in ("a", "b")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == {"a": ["stage-a"], "b": ["stage-b"]}


def test_bind_makes_the_trace_current_in_another_thread():
    _, handle = started()
    seen = []

    def resume() -> None:
        handle.bind()
        with span("late"):
            seen.append(True)

    t = threading.Thread(target=resume)
    t.start()
    t.join()
    assert seen and [s.name for s in handle.finish().spans] == ["late"]


def test_usage_sums_on_the_span_and_in_the_trace_summary():
    _, handle = started()
    with handle.activate():
        with span("rewrite"):
            record_usage(10, 2, "reported")
        with span("generate"):
            record_usage(100, 20, "estimated")
            record_usage(1, 1, "reported")
    trace = handle.finish()
    gen = trace.spans[1].attrs
    assert gen["llm_calls"] == 2 and gen["prompt_tokens"] == 101 and gen["usage_source"] == "mixed"
    assert trace.attrs["prompt_tokens"] == 111 and trace.attrs["llm_calls"] == 3
    assert trace.attrs["completion_tokens"] == 23


# --- stores -------------------------------------------------------------------------------


def make_trace(i: int, day: str = "2026-05-01", status: str = "ok", user: str = "u01"):
    now = datetime.fromisoformat(f"{day}T10:00:{i % 60:02d}+00:00")
    tracer = Tracer(now=lambda: now)
    handle = tracer.start("/chat", user)
    trace = handle.finish(None if status == "ok" else RuntimeError("x"))
    return trace


def test_ring_buffer_evicts_oldest_and_lists_newest_first():
    store = MemoryTraceStore(max_entries=3)
    traces = [make_trace(i) for i in range(5)]
    for t in traces:
        store.add(t)
    assert store.get(traces[0].trace_id) is None and store.get(traces[4].trace_id) is not None
    assert [s.trace_id for s in store.list()] == [t.trace_id for t in reversed(traces[2:])]
    assert store.stats()["stored"] == 3


def test_list_filters_and_limit_and_stats():
    store = MemoryTraceStore()
    for i in range(4):
        store.add(make_trace(i, status="error" if i == 3 else "ok", user="u02" if i < 2 else "u01"))
    assert len(store.list(limit=2)) == 2
    assert {s.user_id for s in store.list(filters={"user_id": "u02"})} == {"u02"}
    assert [s.status for s in store.list(filters={"status": "error"})] == ["error"]
    stats = store.stats()
    assert stats["stored"] == 4 and stats["errors"] == 1 and stats["p50_ms"] is not None


def test_jsonl_round_trip_and_daily_files(tmp_path):
    store = JsonlTraceStore(tmp_path / "traces")
    a, b = make_trace(1, "2026-05-01"), make_trace(2, "2026-05-02")
    store.add(a)
    store.add(b)
    assert sorted(p.name for p in (tmp_path / "traces").iterdir()) == [
        "traces-2026-05-01.jsonl",
        "traces-2026-05-02.jsonl",
    ]
    lines = (tmp_path / "traces" / "traces-2026-05-01.jsonl").read_text().splitlines()
    assert len(lines) == 1  # one trace per line
    assert store.get(a.trace_id) == a and store.get(b.trace_id) == b
    assert [s.trace_id for s in store.list()] == [b.trace_id, a.trace_id]  # newest day first


def test_jsonl_write_failure_is_swallowed(tmp_path, caplog):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    store = JsonlTraceStore(blocker / "traces")
    store.add(make_trace(1))  # must not raise
    assert "trace write failed (" in caplog.text


def test_combined_store_serves_reads_at_once_and_writes_the_file(tmp_path):
    settings = make_settings(tmp_path, trace_max_entries=2)
    stores = TraceStores.from_settings(settings)
    t = make_trace(1)
    stores.add(t)
    assert stores.get(t.trace_id) == t
    stores.flush()
    assert len(list((tmp_path / "traces").glob("*.jsonl"))) == 1
    off = TraceStores.from_settings(make_settings(tmp_path / "x", trace_to_file=False))
    off.add(t)
    off.flush()
    assert not (tmp_path / "x" / "traces").exists()


def test_views_escape_hostile_values():
    _, handle = started()
    with handle.activate():
        with span("<script>alert(1)</script>", note="<img src=x onerror=alert(1)>"):
            pass
    html = render_html(handle.finish())
    assert "<script>" not in html and "<img" not in html and "&lt;script&gt;" in html
    assert "script" in render_text(handle.finish())


# --- token usage --------------------------------------------------------------------------


class FakeCompletions:
    def __init__(self, response=None, stream_events=None, reject_options=False):
        self.response, self.stream_events = response, stream_events
        self.reject_options = reject_options
        self.kwargs: list[dict] = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        if kwargs.get("stream"):
            if self.reject_options and "stream_options" in kwargs:
                raise openai.BadRequestError(
                    "bad",
                    response=httpx.Response(
                        400, request=httpx.Request("POST", "https://x.invalid")
                    ),
                    body=None,
                )
            return iter(self.stream_events)
        return self.response


def llm_with(completions) -> OpenAICompatLLM:
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAICompatLLM(client, "m")  # type: ignore[arg-type]


def reply(text="hi", usage="default"):
    message = SimpleNamespace(content=text)
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=3) if usage == "default" else usage
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def span_attrs_after(fn):
    _, handle = started()
    with handle.activate():
        with span("generate"):
            fn()
    return handle.finish().spans[0].attrs


def test_reported_usage_is_recorded():
    llm = llm_with(FakeCompletions(reply()))
    attrs = span_attrs_after(lambda: TracedLLM(llm).complete("sys", "user"))
    assert (attrs["prompt_tokens"], attrs["completion_tokens"]) == (11, 3)
    assert attrs["usage_source"] == "reported" and attrs["llm_calls"] == 1  # not double counted


def test_missing_usage_is_estimated_at_four_chars_per_token():
    llm = FakeLLM("a" * 9)
    attrs = span_attrs_after(lambda: TracedLLM(llm).complete("s" * 4, "u" * 5))
    assert (attrs["prompt_tokens"], attrs["completion_tokens"]) == (3, 3)
    assert attrs["usage_source"] == "estimated"


@pytest.mark.parametrize(
    "usage", [None, SimpleNamespace(prompt_tokens=None, completion_tokens="x"), "garbage", object()]
)
def test_malformed_usage_never_fails_the_call(usage):
    llm = llm_with(FakeCompletions(reply("ok", usage)))
    holder = []
    attrs = span_attrs_after(lambda: holder.append(TracedLLM(llm).complete("s", "u")))
    assert holder == ["ok"] and attrs["usage_source"] == "estimated"


def event(text=None, usage=None, choices=True):
    delta = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)] if choices else [], usage=usage)


def test_streaming_usage_chunk_with_empty_choices():
    final = event(choices=False, usage=SimpleNamespace(prompt_tokens=50, completion_tokens=7))
    completions = FakeCompletions(stream_events=[event("He"), event("llo"), final])
    llm = llm_with(completions)
    out = []
    attrs = span_attrs_after(lambda: out.extend(TracedLLM(llm).stream("s", "u")))
    assert "".join(out) == "Hello"
    assert completions.kwargs[0]["stream_options"] == {"include_usage": True}
    assert (attrs["prompt_tokens"], attrs["completion_tokens"]) == (50, 7)
    assert attrs["usage_source"] == "reported" and "llm_first_token_ms" in attrs


def test_streaming_without_usage_falls_back_to_estimate():
    llm = llm_with(FakeCompletions(stream_events=[event("abcd"), event("efgh")]))
    attrs = span_attrs_after(lambda: list(TracedLLM(llm).stream("s" * 8, "u" * 8)))
    assert (attrs["prompt_tokens"], attrs["completion_tokens"]) == (4, 2)
    assert attrs["usage_source"] == "estimated"


def test_stream_options_rejection_is_learned_once(monkeypatch):
    import app.rag.llm as llm_module

    monkeypatch.setattr(llm_module, "_stream_usage_ok", True)
    completions = FakeCompletions(stream_events=[event("x")], reject_options=True)
    llm = llm_with(completions)
    attrs = span_attrs_after(lambda: list(TracedLLM(llm).stream("s", "u")))
    assert attrs["usage_source"] == "estimated" and len(completions.kwargs) == 2
    assert "stream_options" not in completions.kwargs[1]
    span_attrs_after(lambda: list(TracedLLM(llm).stream("s", "u")))
    assert len(completions.kwargs) == 3 and "stream_options" not in completions.kwargs[2]


def test_untraced_calls_do_not_ask_for_stream_usage():
    completions = FakeCompletions(stream_events=[event("x")])
    list(llm_with(completions).stream("s", "u"))  # no active trace
    assert "stream_options" not in completions.kwargs[0]


def test_utc_timestamps():
    t = make_trace(1)
    assert datetime.fromisoformat(t.started_at).tzinfo == timezone.utc
