"""Shared server objects: built once, replaceable by fakes in tests, and the index gate.

Embedder, LLM and reranker are created lazily on first use, so building the app needs no API
key and loads no model. Tests pass fakes to `Deps` and nothing is built.
"""

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

from app.config import Settings
from app.rag.embeddings import Embedder
from app.rag.llm import LLM, get_embedder, get_llm
from app.rag.memory import InMemorySessionStore, SessionStore
from app.rag.rerank import Reranker, get_reranker


class IndexGate:
    """Many queries may read the index at once; an ingest needs it alone.

    `begin_ingest` returns False if an ingest is already running (the caller answers 409).
    While an ingest waits or runs, new queries wait too, so it cannot be starved.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._ingesting = False

    @contextmanager
    def read(self):
        lease = self.acquire_read()
        try:
            yield
        finally:
            lease.release()

    def acquire_read(self) -> "ReadLease":
        with self._cond:
            while self._ingesting:
                self._cond.wait()
            self._readers += 1
        return ReadLease(self)

    def _release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            self._cond.notify_all()

    def begin_ingest(self) -> bool:
        with self._cond:
            if self._ingesting:
                return False
            self._ingesting = True
            while self._readers:
                self._cond.wait()
            return True

    def end_ingest(self) -> None:
        with self._cond:
            self._ingesting = False
            self._cond.notify_all()

    @property
    def ingesting(self) -> bool:
        return self._ingesting


class ReadLease:
    """A held read slot; `release` is safe to call more than once."""

    def __init__(self, gate: IndexGate) -> None:
        self._gate = gate
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._gate._release_read()


@dataclass
class Deps:
    settings: Settings
    embedder: Embedder | None = None
    llm: LLM | None = None
    reranker: Reranker | None = None
    gate: IndexGate = field(default_factory=IndexGate)
    sessions: SessionStore | None = None
    _reranker_ready: bool = field(default=False, repr=False)
    _build_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self._reranker_ready = self.reranker is not None
        if self.sessions is None:
            self.sessions = InMemorySessionStore.from_settings(self.settings)

    def get_embedder(self) -> Embedder:
        with self._build_lock:
            if self.embedder is None:
                self.embedder = get_embedder(self.settings)
            return self.embedder

    def get_llm(self) -> LLM:
        with self._build_lock:
            if self.llm is None:
                self.llm = get_llm(self.settings)
            return self.llm

    def get_reranker(self) -> Reranker | None:
        with self._build_lock:
            if not self._reranker_ready:
                self.reranker = get_reranker(self.settings)  # None when RERANKER=none
                self._reranker_ready = True
            return self.reranker
