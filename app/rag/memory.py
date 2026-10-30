"""Per-session conversation memory.

`SessionStore` is the small interface callers use (get / append / clear / exists);
`InMemorySessionStore` is the default implementation. A persistent backend can replace it
later by implementing the same four methods. Session ids are client-chosen labels, not
identities: nothing here authenticates or authorizes anyone.

Bounds, so memory cannot grow without limit: at most `max_turns` turns per session (oldest
dropped), at most `max_sessions` sessions (least recently used evicted) and sessions idle for
more than `ttl_seconds` expire.
"""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from app.config import Settings


@dataclass(frozen=True)
class Turn:
    question: str  # what the user typed
    standalone_query: str  # what retrieval used (equals question when not rewritten)
    answer: str
    refused: bool = False
    doc_ids: tuple[str, ...] = field(default_factory=tuple)  # cited doc ids


class SessionStore(Protocol):
    def get(self, session_id: str) -> list[Turn]: ...

    def append(self, session_id: str, turn: Turn) -> None: ...

    def clear(self, session_id: str) -> None: ...

    def exists(self, session_id: str) -> bool: ...


class ScopedSessionStore:
    """A view of a store for one user: session ids are stored as `<user_id>:<session_id>`, so
    two users never share a session and the store's interface is unchanged. Valid session ids
    never contain ":", so keys of different users cannot collide."""

    def __init__(self, store: SessionStore, user_id: str) -> None:
        self._store = store
        self._prefix = f"{user_id}:"

    def get(self, session_id: str) -> list[Turn]:
        return self._store.get(self._prefix + session_id)

    def append(self, session_id: str, turn: Turn) -> None:
        self._store.append(self._prefix + session_id, turn)

    def clear(self, session_id: str) -> None:
        self._store.clear(self._prefix + session_id)

    def exists(self, session_id: str) -> bool:
        return self._store.exists(self._prefix + session_id)


class InMemorySessionStore:
    """Thread-safe LRU + TTL store. `clock` is injectable for tests."""

    def __init__(
        self,
        max_turns: int = 10,
        max_sessions: int = 1000,
        ttl_seconds: int = 3600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_turns = max(1, max_turns)
        self.max_sessions = max(1, max_sessions)
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, tuple[float, list[Turn]]] = OrderedDict()

    @classmethod
    def from_settings(cls, settings: Settings) -> "InMemorySessionStore":
        return cls(
            settings.memory_max_turns, settings.memory_max_sessions, settings.memory_ttl_seconds
        )

    def _expire(self, now: float) -> None:
        # OrderedDict is in least-recently-touched order, so expired sessions are at the front
        while self._sessions:
            session_id, (touched, _) = next(iter(self._sessions.items()))
            if now - touched <= self.ttl_seconds:
                break
            del self._sessions[session_id]

    def get(self, session_id: str) -> list[Turn]:
        with self._lock:
            now = self._clock()
            self._expire(now)
            entry = self._sessions.get(session_id)
            if entry is None:
                return []
            self._sessions[session_id] = (now, entry[1])
            self._sessions.move_to_end(session_id)
            return list(entry[1])

    def append(self, session_id: str, turn: Turn) -> None:
        with self._lock:
            now = self._clock()
            self._expire(now)
            turns = self._sessions.pop(session_id, (now, []))[1]
            turns.append(turn)
            del turns[: -self.max_turns]
            self._sessions[session_id] = (now, turns)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def exists(self, session_id: str) -> bool:
        with self._lock:
            self._expire(self._clock())
            return session_id in self._sessions
