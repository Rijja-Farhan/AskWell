"""Trace storage: an in-memory ring buffer, an append-only JSONL file sink (one file per UTC
day) and `TraceStores`, the pair the server uses.

The file write happens on a single background thread, after the response path has moved on,
under a lock; a failed write is logged (exception type only) and never reaches the request.
"""

import logging
import math
import threading
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings
from app.observability.trace import Trace, TraceSummary, summarize

log = logging.getLogger("askwell.traces")


class TraceStore(Protocol):
    def add(self, trace: Trace) -> None: ...

    def get(self, trace_id: str) -> Trace | None: ...

    def list(
        self, limit: int = 50, filters: dict[str, Any] | None = None
    ) -> list[TraceSummary]: ...

    def stats(self) -> dict[str, Any]: ...


def matches(trace: Trace, filters: dict[str, Any] | None) -> bool:
    return all(
        value is None or getattr(trace, key, None) == value
        for key, value in (filters or {}).items()
    )


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile; None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def compute_stats(traces: list[Trace]) -> dict[str, Any]:
    totals = [t.total_ms for t in traces]
    return {
        "stored": len(traces),
        "errors": sum(1 for t in traces if t.status == "error"),
        "p50_ms": percentile(totals, 50),
        "p95_ms": percentile(totals, 95),
    }


class MemoryTraceStore:
    """Ring buffer of the newest `max_entries` traces."""

    def __init__(self, max_entries: int = 500) -> None:
        self.max_entries = max(1, max_entries)
        self._data: OrderedDict[str, Trace] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, trace: Trace) -> None:
        with self._lock:
            self._data[trace.trace_id] = trace
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def get(self, trace_id: str) -> Trace | None:
        with self._lock:
            return self._data.get(trace_id)

    def _newest_first(self) -> list[Trace]:
        with self._lock:
            return list(reversed(self._data.values()))

    def list(self, limit: int = 50, filters: dict[str, Any] | None = None) -> list[TraceSummary]:
        found = [t for t in self._newest_first() if matches(t, filters)]
        return [summarize(t) for t in found[:limit]]

    def stats(self) -> dict[str, Any]:
        return compute_stats(self._newest_first())


class JsonlTraceStore:
    """Append-only `traces-YYYY-MM-DD.jsonl` files (UTC day of `started_at`), one trace per line."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._lock = threading.Lock()

    def path_for(self, trace: Trace) -> Path:
        return self.directory / f"traces-{trace.started_at[:10]}.jsonl"

    def add(self, trace: Trace) -> None:
        try:
            line = trace.model_dump_json() + "\n"
            with self._lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                with self.path_for(trace).open("a", encoding="utf-8") as fh:
                    fh.write(line)
        except Exception as exc:  # noqa: BLE001 - tracing must never fail a request
            log.warning("trace write failed (%s)", type(exc).__name__)

    def files(self) -> list[Path]:
        if not self.directory.is_dir():
            return []
        return sorted(self.directory.glob("traces-*.jsonl"), reverse=True)

    def iter_traces(self) -> Iterator[Trace]:
        """Newest first (files by day, lines in reverse); unreadable lines are skipped."""
        for path in self.files():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                log.warning("trace read failed (%s)", type(exc).__name__)
                continue
            for line in reversed(lines):
                try:
                    yield Trace.model_validate_json(line)
                except ValueError:
                    continue

    def get(self, trace_id: str) -> Trace | None:
        return next((t for t in self.iter_traces() if t.trace_id == trace_id), None)

    def list(self, limit: int = 50, filters: dict[str, Any] | None = None) -> list[TraceSummary]:
        out: list[TraceSummary] = []
        for trace in self.iter_traces():
            if matches(trace, filters):
                out.append(summarize(trace))
                if len(out) >= limit:
                    break
        return out

    def stats(self) -> dict[str, Any]:
        return compute_stats(list(self.iter_traces()))


class TraceStores:
    """What the server holds: the ring buffer answers reads at once; the file sink (if any)
    is written by a background thread."""

    def __init__(self, memory: MemoryTraceStore, sink: JsonlTraceStore | None = None) -> None:
        self.memory = memory
        self.sink = sink
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trace-writer")

    @classmethod
    def from_settings(cls, settings: Settings) -> "TraceStores":
        sink = JsonlTraceStore(settings.traces_dir) if settings.trace_to_file else None
        return cls(MemoryTraceStore(settings.trace_max_entries), sink)

    def add(self, trace: Trace) -> None:
        self.memory.add(trace)
        if self.sink is not None:
            try:
                self._pool.submit(self.sink.add, trace)
            except RuntimeError:  # pool already shut down
                self.sink.add(trace)

    def flush(self) -> None:
        """Wait for queued file writes (tests, probes, shutdown)."""
        self._pool.submit(lambda: None).result()

    def get(self, trace_id: str) -> Trace | None:
        found = self.memory.get(trace_id)
        if found is None and self.sink is not None:
            self.flush()
            found = self.sink.get(trace_id)
        return found

    def list(self, limit: int = 50, filters: dict[str, Any] | None = None) -> list[TraceSummary]:
        return self.memory.list(limit, filters)

    def stats(self) -> dict[str, Any]:
        return self.memory.stats()
