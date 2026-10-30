"""OTLP trace export, offline: an in-memory span exporter stands in for the collector."""

import logging
import sys
import time

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402

from app.observability import MemoryTraceStore, TraceStores  # noqa: E402
from app.observability.otlp import (  # noqa: E402
    OTLPTraceSink,
    attribute_value,
    parse_headers,
    traces_url,
)
from app.observability.trace import Span, SpanError, Trace  # noqa: E402
from tests.conftest import make_settings  # noqa: E402

TRACE_ID = "0123456789abcdef0123456789abcdef"


def sample(**over) -> Trace:
    values = dict(
        trace_id=TRACE_ID,
        started_at="2026-05-01T10:00:00+00:00",
        endpoint="POST /chat",
        user_id="u06",
        session_id="s1",
        total_ms=120.0,
        attrs={"cached": False, "question_chars": 31, "answer_text": "SECRET ANSWER"},
        spans=[
            Span(id=0, parent=None, name="retrieve", start_ms=5.0, duration_ms=60.0,
                 attrs={"k": 5, "chunks": [{"chunk_id": "c1", "score": 0.5}]}),
            Span(id=1, parent=0, name="rerank", start_ms=10.0, duration_ms=40.0,
                 attrs={"candidates": 40}),
            Span(id=2, parent=None, name="generate", start_ms=70.0, duration_ms=45.0,
                 status="error",
                 error=SpanError(type="RateLimitError", code="provider_unavailable")),
        ],
        status="error",
        error=SpanError(type="RateLimitError", code="provider_unavailable"),
    )  # fmt: skip
    return Trace(**{**values, **over})


def make_sink(include_text=False):
    exporter = InMemorySpanExporter()
    return OTLPTraceSink(exporter, "askwell-test", include_text, batch=False), exporter


def test_trace_maps_to_nested_spans_with_ids_and_timing():
    sink, exporter = make_sink()
    sink.add(sample())
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {"POST /chat", "retrieve", "rerank", "generate"}
    root = spans["POST /chat"]
    assert root.parent is None
    assert {s.context.trace_id for s in spans.values()} == {int(TRACE_ID, 16)}
    assert spans["retrieve"].parent.span_id == root.context.span_id
    assert spans["rerank"].parent.span_id == spans["retrieve"].context.span_id
    assert spans["generate"].parent.span_id == root.context.span_id
    assert len({s.context.span_id for s in spans.values()}) == 4
    assert root.end_time - root.start_time == 120_000_000
    assert spans["rerank"].start_time - root.start_time == 10_000_000
    assert spans["rerank"].end_time - spans["rerank"].start_time == 40_000_000
    assert root.resource.attributes["service.name"] == "askwell-test"


def test_attributes_status_and_errors():
    sink, exporter = make_sink()
    sink.add(sample())
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["retrieve"].attributes["k"] == 5
    assert '"chunk_id":"c1"' in spans["retrieve"].attributes["chunks"]  # nested -> JSON string
    assert spans["retrieve"].status.status_code == StatusCode.OK
    assert spans["generate"].status.status_code == StatusCode.ERROR
    assert spans["generate"].attributes["error.type"] == "RateLimitError"
    assert spans["generate"].attributes["error.code"] == "provider_unavailable"
    root = spans["POST /chat"]
    assert root.status.status_code == StatusCode.ERROR
    assert root.attributes["askwell.trace_id"] == TRACE_ID
    assert root.attributes["askwell.user_id"] == "u06"
    assert root.attributes["askwell.cached"] is False


def test_free_text_is_exported_only_with_trace_store_text():
    for include, expected in ((False, False), (True, True)):
        sink, exporter = make_sink(include)
        sink.add(sample())
        flat = repr([dict(s.attributes) for s in exporter.get_finished_spans()])
        assert ("SECRET ANSWER" in flat) is expected


def test_non_hex_trace_id_still_maps():
    sink, exporter = make_sink()
    sink.add(sample(trace_id="not-hex"))
    ids = {s.context.trace_id for s in exporter.get_finished_spans()}
    assert len(ids) == 1 and 0 not in ids


def test_conversion_error_does_not_raise(caplog):
    sink, _ = make_sink()

    def explode(trace):
        raise RuntimeError("conversion exploded with secret")

    sink.to_spans = explode
    with caplog.at_level(logging.WARNING, logger="askwell.otlp"):
        sink.add(sample())
    assert "RuntimeError" in caplog.text and "secret" not in caplog.text


def test_helpers():
    assert parse_headers("a=b, Authorization=Basic abc==,bad,=x") == {
        "a": "b",
        "Authorization": "Basic abc==",
    }
    assert parse_headers("") == {}
    assert traces_url("http://h:4318/") == "http://h:4318/v1/traces"
    assert traces_url("http://h/api/public/otel/v1/traces") == "http://h/api/public/otel/v1/traces"
    assert attribute_value(None) is None
    assert attribute_value([1, 2]) == [1, 2]
    assert attribute_value([1, "a"]) == '[1,"a"]'


def test_batch_sink_exports_through_the_processor():
    exporter = InMemorySpanExporter()
    sink = OTLPTraceSink(exporter, batch=True)
    sink.add(sample())
    assert sink.flush()
    assert len(exporter.get_finished_spans()) == 4
    sink.shutdown()


def test_unreachable_endpoint_neither_blocks_nor_raises(tmp_path):
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    settings = make_settings(tmp_path, otlp_endpoint="http://127.0.0.1:1", trace_to_file=False)
    stores = TraceStores.from_settings(settings)
    assert len(stores.extra_sinks) == 1
    started = time.perf_counter()
    for i in range(20):
        stores.add(sample(trace_id=f"{i:032x}"))
    assert time.perf_counter() - started < 0.5
    stores.flush()
    assert stores.get(f"{3:032x}") is not None


def test_missing_packages_disable_export_with_a_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", None)
    settings = make_settings(tmp_path, otlp_endpoint="http://collector:4318")
    with caplog.at_level(logging.WARNING, logger="askwell.otlp"):
        stores = TraceStores.from_settings(settings)
    assert stores.extra_sinks == []
    assert "not installed" in caplog.text
    stores.add(sample())  # tracing itself still works


def test_sink_is_added_only_when_an_endpoint_is_set(tmp_path):
    assert TraceStores.from_settings(make_settings(tmp_path)).extra_sinks == []
    blank = make_settings(tmp_path, otlp_endpoint="   ")
    assert TraceStores.from_settings(blank).extra_sinks == []
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    on = make_settings(
        tmp_path, otlp_endpoint="http://collector:4318", otlp_headers="x-key=1,y=2=3"
    )
    stores = TraceStores.from_settings(on)
    assert len(stores.extra_sinks) == 1
    assert stores.sink is not None  # the JSONL sink stays alongside


def test_extra_sink_failure_does_not_reach_add(tmp_path):
    class Boom:
        def add(self, trace):
            raise RuntimeError("x")

    stores = TraceStores(MemoryTraceStore(), None, [Boom()])
    stores.add(sample())
    stores.flush()
    assert stores.get(TRACE_ID) is not None
