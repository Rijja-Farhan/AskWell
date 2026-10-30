"""Shared server objects: built once, replaceable by fakes in tests, and the index gate.

Embedder, LLM and reranker are created lazily on first use, so building the app needs no API
key and loads no model. Tests pass fakes to `Deps` and nothing is built.
"""

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

from fastapi import Request

from app.config import Settings
from app.dataset.schemas import UserProfile
from app.observability import TraceStore, TraceStores
from app.rag.access import GUEST, load_users
from app.rag.cache import CacheBundle
from app.rag.embeddings import Embedder
from app.rag.llm import LLM, get_embedder, get_llm
from app.rag.memory import InMemorySessionStore, ScopedSessionStore, SessionStore
from app.rag.rerank import Reranker, get_reranker

log = logging.getLogger("askwell.api")


class ApiError(Exception):
    """An error with its own status and error code, mapped to the standard error body."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


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
    users: dict[str, UserProfile] | None = None
    caches: CacheBundle | None = None
    traces: TraceStore | None = None
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

    def get_caches(self) -> CacheBundle:
        """The embedding, retrieval and answer caches, built once from the settings (tests
        may pass their own, or `CacheBundle.disabled()`)."""
        with self._build_lock:
            if self.caches is None:
                self.caches = CacheBundle.from_settings(self.settings)
            return self.caches

    def get_traces(self) -> TraceStore:
        """The trace store (ring buffer plus optional JSONL sink), built once from the settings
        (tests may pass their own)."""
        with self._build_lock:
            if self.traces is None:
                self.traces = TraceStores.from_settings(self.settings)
            return self.traces

    def get_users(self) -> dict[str, UserProfile]:
        """Profiles by id, loaded once. A missing or invalid users file fails closed: no
        users, so every caller is the guest and every named user is unknown."""
        with self._build_lock:
            if self.users is None:
                try:
                    self.users = load_users(self.settings)
                except Exception as exc:  # noqa: BLE001
                    log.error("cannot load users (%s): serving guests only", type(exc).__name__)
                    self.users = {}
            return self.users

    def sessions_for(self, user: UserProfile) -> SessionStore:
        """The session store as seen by one user (sessions are owned by their creator)."""
        assert self.sessions is not None
        return ScopedSessionStore(self.sessions, user.user_id)


def current_user(request: Request) -> UserProfile:
    """FastAPI dependency: the profile named by the `X-User-Id` header.

    DEMO IDENTITY SWITCH, NOT AUTHENTICATION: anyone can send any id. It exists so the access
    rules can be exercised and the UI can switch users; real auth is out of scope. No header
    means the least-privilege guest; an unknown id is a 401.
    """
    deps: Deps = request.app.state.deps
    user_id = request.headers.get("x-user-id")
    if not user_id:
        return GUEST
    user = deps.get_users().get(user_id)
    if user is None:
        if not deps.settings.rbac_enabled:
            return GUEST
        raise ApiError(401, "unknown_user", "Unknown user.")
    return user


def is_admin(user: UserProfile) -> bool:
    """May change what everyone can see (ingest): restricted clearance and all departments."""
    return user.clearance == "restricted" and {"HR", "IT", "Operations", "Finance"} <= set(
        user.departments
    )
