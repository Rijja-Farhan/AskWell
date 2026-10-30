"""OTLP/HTTP export of finished traces (Phase 17), for Jaeger, Langfuse or any OTel backend.

`OTLPTraceSink` is a `TraceStores` sink: `add(trace)` converts one finished `Trace` into
OpenTelemetry spans and hands them to a `BatchSpanProcessor`. The processor queues them and a
worker thread of its own does the HTTP POST, so a slow or dead collector costs a request
nothing (a full queue drops spans; a failed export is logged by the SDK and forgotten). The
conversion itself runs on the trace-writer thread, never on the request path.

Why `ReadableSpan` and not `tracer.start_span`: Askwell traces are finished objects with their
own ids and timings, recorded by the dependency-free tracer in `trace.py`. We replay them with
the original trace id, start times and durations rather than re-measuring, so the id a client
saw in `X-Trace-Id` is the id to search for in the backend.

The OpenTelemetry packages are imported lazily: with `otlp_endpoint` empty (the default) or the
packages missing, nothing here is imported and the app behaves as before.

Privacy: only what the `Trace` already holds is exported (ids, counts, scores, timings, error
type and code). Nested attributes are JSON-encoded strings. The summary's free text
(`answer_text`, only stored with `trace_store_text=true`) is exported only under that same flag.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.config import Settings
from app.observability.trace import Span, Trace

log = logging.getLogger("askwell.otlp")

TEXT_KEYS = frozenset({"answer_text"})  # summary fields that hold free text


def parse_headers(raw: str) -> dict[str, str]:
    """`"k1=v1,k2=v2"` -> dict. Values may contain `=` (Basic auth is base64). Malformed
    entries are skipped; the value is never logged (it is a credential)."""
    headers: dict[str, str] = {}
    for part in raw.split(","):
        key, sep, value = part.partition("=")
        if sep and key.strip():
            headers[key.strip()] = value.strip()
    return headers


def traces_url(endpoint: str) -> str:
    """The OTLP/HTTP traces URL for a base endpoint (`http://host:4318`)."""
    base = endpoint.strip().rstrip("/")
    return base if base.endswith("/v1/traces") else base + "/v1/traces"


def trace_id_int(trace_id: str) -> int:
    """Askwell ids are 32 hex chars (a UUID4), which is exactly an OTel trace id."""
    try:
        if len(trace_id) == 32:
            value = int(trace_id, 16)
            if value:
                return value
    except ValueError:
        pass
    return int.from_bytes(hashlib.sha256(trace_id.encode()).digest()[:16], "big") or 1


def span_id_int(trace_id: str, local_id: int | str) -> int:
    """A stable non-zero 64-bit span id per (trace, span) so re-exports do not duplicate."""
    digest = hashlib.sha256(f"{trace_id}:{local_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") or 1


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, bool, int, float))


def attribute_value(value: Any) -> Any | None:
    """An OTel attribute value: scalars and homogeneous scalar lists as they are; anything
    nested as compact JSON; None is dropped."""
    if value is None:
        return None
    if _scalar(value):
        return value
    if isinstance(value, list) and value and all(_scalar(v) for v in value):
        if len({type(v) for v in value}) == 1:
            return value
    return json.dumps(value, separators=(",", ":"), default=str)


def attributes(prefix: str, attrs: dict[str, Any], skip: frozenset[str] = frozenset()) -> dict:
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if key in skip:
            continue
        converted = attribute_value(value)
        if converted is not None:
            out[prefix + key] = converted
    return out


def start_ns(trace: Trace) -> int:
    started = datetime.fromisoformat(trace.started_at.replace("Z", "+00:00"))
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return int(started.timestamp() * 1_000_000_000)


class OTLPTraceSink:
    """Converts finished traces to OTel spans and exports them through a batch processor."""

    def __init__(
        self,
        exporter: Any,
        service_name: str = "askwell",
        include_text: bool = False,
        batch: bool = True,
    ) -> None:
        # imported here so the module itself imports without OpenTelemetry
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
        from opentelemetry.sdk.util.instrumentation import InstrumentationScope

        self.include_text = include_text
        self.resource = Resource.create({"service.name": service_name})
        self.scope = InstrumentationScope("askwell")
        self.exporter = exporter
        self.processor = (
            BatchSpanProcessor(
                exporter,
                max_queue_size=2048,
                schedule_delay_millis=2000,
                max_export_batch_size=256,
                export_timeout_millis=5000,
            )
            if batch
            else SimpleSpanProcessor(exporter)
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "OTLPTraceSink | None":
        """The sink for `otlp_endpoint`, or None when export is off or the packages are
        missing (a warning says which; the app keeps running)."""
        if not settings.otlp_endpoint.strip():
            return None
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(
                endpoint=traces_url(settings.otlp_endpoint),
                headers=parse_headers(settings.otlp_headers) or None,
                timeout=5,
            )
            return cls(exporter, settings.otlp_service_name, settings.trace_store_text)
        except ImportError:
            log.warning(
                "OTLP_ENDPOINT is set but the OpenTelemetry packages are not installed; "
                "trace export is off (pip install -r requirements.txt)"
            )
        except Exception as exc:  # noqa: BLE001 - export must never stop the app
            log.warning("OTLP trace export disabled (%s)", type(exc).__name__)
        return None

    def add(self, trace: Trace) -> None:
        """Export one finished trace. Never raises."""
        try:
            for readable in self.to_spans(trace):
                self.processor.on_end(readable)
        except Exception as exc:  # noqa: BLE001 - tracing must never fail a request
            log.warning("otlp export failed (%s)", type(exc).__name__)

    def to_spans(self, trace: Trace) -> list[Any]:
        """The root span (the request) followed by one span per recorded stage."""
        from opentelemetry.trace import SpanContext, TraceFlags

        trace_int = trace_id_int(trace.trace_id)
        root_id = span_id_int(trace.trace_id, "root")
        begin = start_ns(trace)

        def context(span_id: int) -> SpanContext:
            return SpanContext(trace_int, span_id, False, TraceFlags(TraceFlags.SAMPLED))

        skip = frozenset() if self.include_text else TEXT_KEYS
        root_attrs = attributes("askwell.", trace.attrs, skip)
        root_attrs["askwell.trace_id"] = trace.trace_id
        for key, value in (("user_id", trace.user_id), ("session_id", trace.session_id)):
            if value is not None:
                root_attrs[f"askwell.{key}"] = value
        spans = [
            self._readable(
                trace.endpoint,
                context(root_id),
                None,
                root_attrs,
                begin,
                begin + int(trace.total_ms * 1_000_000),
                trace.status == "error",
                trace.error.code if trace.error else None,
                trace.error.type if trace.error else None,
            )
        ]
        for item in trace.spans:
            spans.append(self._span(trace, item, begin, root_id, context))
        return spans

    def _span(self, trace: Trace, item: Span, begin: int, root_id: int, context: Any) -> Any:
        parent_id = root_id if item.parent is None else span_id_int(trace.trace_id, item.parent)
        start = begin + int(item.start_ms * 1_000_000)
        return self._readable(
            item.name,
            context(span_id_int(trace.trace_id, item.id)),
            context(parent_id),
            attributes("", item.attrs),
            start,
            start + int(item.duration_ms * 1_000_000),
            item.status == "error",
            item.error.code if item.error else None,
            item.error.type if item.error else None,
        )

    def _readable(
        self,
        name: str,
        ctx: Any,
        parent: Any,
        attrs: dict[str, Any],
        start: int,
        end: int,
        failed: bool,
        code: str | None,
        error_type: str | None,
    ) -> Any:
        from opentelemetry.sdk.trace import ReadableSpan
        from opentelemetry.trace import SpanKind, Status, StatusCode

        if failed:
            attrs = {**attrs, "error.type": error_type or "error", "error.code": code or "error"}
            status = Status(StatusCode.ERROR, code or "error")
        else:
            status = Status(StatusCode.OK)
        return ReadableSpan(
            name=name,
            context=ctx,
            parent=parent,
            resource=self.resource,
            attributes=attrs,
            kind=SpanKind.SERVER if parent is None else SpanKind.INTERNAL,
            status=status,
            start_time=start,
            end_time=max(start, end),
            instrumentation_scope=self.scope,
        )

    def flush(self, timeout_ms: int = 5000) -> bool:
        return bool(self.processor.force_flush(timeout_ms))

    def shutdown(self) -> None:
        self.processor.shutdown()
