"""Caching of query embeddings, retrieval results and final answers (correctness first).

Three layers sit around the pipeline without changing it:

- embeddings: query text -> vector. Key: embedding model + SHA-256 of the exact text. Valid
  across index changes (the model name is in the key).
- retrieval: `cached_retrieve` wraps `retrieve()`. Key: the effective query, k, the user (id and
  every profile field), the full filter incl. the resolved version policy, every retrieval
  setting, every version constant and the *index fingerprint*.
- answers: key = the retrieval key + the ordered chunks actually retrieved (id, text, metadata,
  score) + the sanitized question + every answer setting and version constant.

Invalidation comes from `index_fingerprint`, derived from the Chroma collection itself, so a
re-index by another process (`scripts/build_index.py` while the server runs) changes every
retrieval and answer key. `CacheBundle.clear()` after `POST /ingest` is belt and braces; the
TTL is a safety net only. A miss is always acceptable, a wrong hit never is: when in doubt a
value goes into the key.

Values are stored as deep copies and handed out as deep copies, so a caller cannot poison the
cache. Keys are SHA-256 digests: question text never appears in a key or a log line. The
`Cache` protocol is all a backend needs to implement (Redis in Phase 17).
"""

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from pydantic import BaseModel

from app.config import Settings
from app.dataset.schemas import UserProfile
from app.observability import annotate, span
from app.rag.access import ACCESS_VERSION
from app.rag.answer import ANSWER_VERSION, Answer, answer_from_chunks
from app.rag.bm25 import BM25_VERSION, index_key
from app.rag.embeddings import Embedder
from app.rag.filters import FILTER_VERSION, RetrievalFilter, effective_policy
from app.rag.llm import LLM
from app.rag.rerank import RERANK_VERSION, Reranker
from app.rag.retrieve import RetrievedChunk, retrieve
from app.rag.sanitize import INJECTION_VERSION
from app.rag.store import open_collection, resolve


class Cache(Protocol):
    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any) -> None: ...

    def clear(self) -> None: ...

    def stats(self) -> dict[str, int]: ...


class LRUTTLCache:
    """In-process, thread-safe LRU cache whose entries also expire after `ttl_seconds`."""

    def __init__(
        self,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is not None and entry[0] <= self._clock():
                del self._data[key]  # expired
                entry = None
            if entry is None:
                self._misses += 1
                return None
            self._data.move_to_end(key)
            self._hits += 1
            return entry[1]

    def set(self, key: str, value: Any) -> None:
        if self.max_entries <= 0:
            return
        with self._lock:
            self._data[key] = (self._clock() + self.ttl_seconds, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)  # least recently used

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses, "size": len(self._data)}


class NullCache:
    """A pass-through layer (`cache_enabled=false`): never stores, never hits."""

    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any) -> None:
        return None

    def clear(self) -> None:
        return None

    def stats(self) -> dict[str, int]:
        return {"hits": 0, "misses": 0, "size": 0}


def _canonical(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"cannot build a cache key from {type(value).__name__}")


def make_key(namespace: str, *parts: Any) -> str:
    """Deterministic SHA-256 over a canonical JSON encoding (no Python `hash()`, so it is
    stable across processes). Dict keys are sorted; the parts keep their order."""
    payload = json.dumps(
        [namespace, parts], sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=_canonical,
    )  # fmt: skip
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheBundle:
    embeddings: Cache
    retrieval: Cache
    answers: Cache
    enabled: bool = True

    @classmethod
    def from_settings(
        cls, settings: Settings, clock: Callable[[], float] = time.monotonic
    ) -> "CacheBundle":
        if not settings.cache_enabled:
            return cls.disabled()
        ttl = settings.cache_ttl_seconds
        return cls(
            LRUTTLCache(settings.cache_embedding_entries, ttl, clock),
            LRUTTLCache(settings.cache_retrieval_entries, ttl, clock),
            LRUTTLCache(settings.cache_answer_entries, ttl, clock),
        )

    @classmethod
    def disabled(cls) -> "CacheBundle":
        return cls(NullCache(), NullCache(), NullCache(), enabled=False)

    def clear(self) -> None:
        for layer in self.layers().values():
            layer.clear()

    def layers(self) -> dict[str, Cache]:
        return {"embeddings": self.embeddings, "retrieval": self.retrieval, "answers": self.answers}

    def stats(self) -> dict[str, dict[str, int]]:
        return {name: layer.stats() for name, layer in self.layers().items()}


# --- embeddings ---------------------------------------------------------------------------


class CachedEmbedder:
    """Embedder wrapper for query embeddings: only missed texts go to the inner embedder, in
    one batched call, and vectors come back in the original order."""

    def __init__(self, inner: Embedder, cache: Cache) -> None:
        self.inner = inner
        self.cache = cache
        self.model = inner.model

    def _key(self, text: str) -> str:
        return make_key("embedding", self.model, hashlib.sha256(text.encode("utf-8")).hexdigest())

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        found: dict[str, tuple[float, ...]] = {}
        missing: list[int] = []
        for i, key in enumerate(keys):
            hit = found.get(key) or self.cache.get(key)
            if hit is None:
                missing.append(i)
            else:
                found[key] = hit
        todo = list(dict.fromkeys(keys[i] for i in missing))  # distinct missed texts
        if todo:
            first = {keys[i]: texts[i] for i in reversed(missing)}
            vectors = self.inner.embed([first[k] for k in todo])
            if len(vectors) != len(todo):
                raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(todo)}")
            for key, vector in zip(todo, vectors, strict=True):
                found[key] = tuple(vector)
                self.cache.set(key, found[key])
        annotate("embed_query", cache_hit=not todo)
        return [list(found[k]) for k in keys]


# --- index fingerprint --------------------------------------------------------------------


RACY_SECONDS = 1.0  # a signature this fresh may hide a second write in the same clock tick
_fingerprints: dict[tuple, tuple[tuple, str]] = {}
_fingerprints_lock = threading.Lock()


def _store_signature(settings: Settings) -> tuple[tuple, float] | None:
    """A cheap stat-only signature of the on-disk store (its files' names, sizes, mtimes and
    inodes) and its newest mtime, or None when it cannot be read. Chroma keeps all metadata in
    `chroma.sqlite3`, which every add, update and delete rewrites, and a full rebuild replaces
    the collection directory."""
    try:
        entries = []
        with os.scandir(resolve(settings.vector_db_path)) as it:
            for entry in it:
                st = entry.stat()
                entries.append((entry.name, st.st_mtime_ns, st.st_size, st.st_ino))
        # the WAL may hold a write that the main file does not show yet
        return tuple(sorted(entries)), max((e[1] for e in entries), default=0) / 1e9
    except OSError:
        return None


def _compute_fingerprint(settings: Settings) -> str:
    collection = open_collection(settings)
    stored = collection.get(include=["metadatas"])
    digest = hashlib.sha256()
    digest.update(index_key(str(collection.id), stored["ids"], stored["metadatas"]).encode())
    digest.update(json.dumps(collection.metadata or {}, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def index_fingerprint(settings: Settings) -> str:
    """Changes whenever the collection's content or build config changes: its id (new on a full
    rebuild), the stored index config (embedding model, chunking, CHUNKER_VERSION), and every
    chunk id with its document fingerprint (which covers the file, parsed text, metadata entry
    and chunking config). Read from the collection itself, so a re-index by another process is
    seen, never from an in-memory counter.

    Reading the collection costs ~30 ms (opening the client plus one metadata read), more than a
    warm hit should. So the result is memoized on a stat-only signature of the store's files
    (~0.1 ms). The memo is trusted only when the newest file is older than `RACY_SECONDS`
    (the rule git uses for its index), so two writes in one timestamp tick cannot hide; a store
    that was just written is simply recomputed. Any doubt means recompute."""
    signature = _store_signature(settings)
    memo_key = (
        str(resolve(settings.vector_db_path)),
        settings.vector_collection,
        settings.embedding_model,
        settings.chunk_strategy,
    )
    if signature is not None:
        with _fingerprints_lock:
            memo = _fingerprints.get(memo_key)
        if memo is not None and memo[0] == signature[0]:
            return memo[1]
    fingerprint = _compute_fingerprint(settings)
    if signature is not None and time.time() - signature[1] > RACY_SECONDS:
        # re-read the signature: only memoize if nothing changed while we computed
        if _store_signature(settings) == signature:
            with _fingerprints_lock:
                _fingerprints[memo_key] = (signature[0], fingerprint)
    return fingerprint


# --- retrieval ----------------------------------------------------------------------------


@dataclass
class CachedRetrieval:
    chunks: list[RetrievedChunk]
    key: str  # the retrieval key ("" when caching is off); the answer key builds on it
    cached: bool = False


def _copy(chunks: list[RetrievedChunk] | tuple[RetrievedChunk, ...]) -> list[RetrievedChunk]:
    return [c.model_copy(deep=True) for c in chunks]


def retrieval_key(
    query: str,
    k: int,
    settings: Settings,
    filters: RetrievalFilter | None,
    user: UserProfile | None,
    fingerprint: str,
) -> str:
    return make_key(
        "retrieval",
        query,
        k,
        user,  # id and every access-relevant field
        filters,
        effective_policy(filters, settings.version_policy),
        settings.retrieval_mode,
        settings.reranker,
        settings.rerank_model if settings.reranker != "none" else None,
        settings.rerank_candidates,
        settings.hybrid_candidates,
        settings.rrf_k,
        settings.bm25_weight,
        settings.rbac_enabled,
        settings.embedding_model,
        ACCESS_VERSION,
        FILTER_VERSION,
        BM25_VERSION,
        RERANK_VERSION,
        INJECTION_VERSION,
        fingerprint,
    )


def cached_retrieve(
    question: str,
    k: int,
    embedder: Embedder,
    settings: Settings,
    caches: CacheBundle,
    reranker: Reranker | None = None,
    filters: RetrievalFilter | None = None,
    user: UserProfile | None = None,
) -> CachedRetrieval:
    """`retrieve()` behind the embedding and retrieval caches. `question` is the effective
    retrieval query (rewritten, or the sanitized original). Pass the requesting `user` even
    when it is also inside `filters`; it is part of the key either way."""
    with span("retrieve", mode=settings.retrieval_mode, k=k, cache_hit=False) as traced:
        result = _cached_retrieve(question, k, embedder, settings, caches, reranker, filters, user)
        traced.set(cache_hit=result.cached, chunks=chunk_attrs(result.chunks))
        traced.setdefault(candidates=len(result.chunks))
        return result


def chunk_attrs(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """What a trace records about retrieved chunks: ids, score, version, status; no text."""
    return [
        {
            "rank": c.rank,
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "score": c.score,
            "version": c.metadata.get("version"),
            "status": c.metadata.get("status"),
        }
        for c in chunks
    ]


def _cached_retrieve(
    question: str,
    k: int,
    embedder: Embedder,
    settings: Settings,
    caches: CacheBundle,
    reranker: Reranker | None,
    filters: RetrievalFilter | None,
    user: UserProfile | None,
) -> CachedRetrieval:
    if not caches.enabled:
        return CachedRetrieval(retrieve(question, k, embedder, settings, reranker, filters), "")
    key = retrieval_key(question, k, settings, filters, user, index_fingerprint(settings))
    hit = caches.retrieval.get(key)
    if hit is not None:
        return CachedRetrieval(_copy(hit), key, cached=True)
    chunks = retrieve(
        question, k, CachedEmbedder(embedder, caches.embeddings), settings, reranker, filters
    )
    caches.retrieval.set(key, tuple(_copy(chunks)))
    return CachedRetrieval(chunks, key)


# --- answers ------------------------------------------------------------------------------


def answer_key(retrieval: str, question: str, chunks: list[RetrievedChunk], s: Settings) -> str:
    """`question` is the sanitized original (the text placed in the prompt)."""
    ordered = [c.chunk_id for c in chunks]
    content = hashlib.sha256(
        json.dumps(
            [[c.text, c.metadata, c.score] for c in chunks], sort_keys=True, default=str
        ).encode("utf-8")
    ).hexdigest()
    return make_key(
        "answer",
        retrieval,
        ordered,
        content,
        question,
        ANSWER_VERSION,
        INJECTION_VERSION,
        s.answer_mode,
        s.injection_defense,
        s.refusal_min_score,
        s.llm_model,
        s.llm_provider,
    )


def lookup_answer(caches: CacheBundle, key: str) -> Answer | None:
    hit = caches.answers.get(key) if caches.enabled and key else None
    return hit.model_copy(deep=True) if hit is not None else None


def store_answer(caches: CacheBundle, key: str, answer: Answer) -> None:
    """Remember a successful answer. Output-guard blocks are never cached."""
    if caches.enabled and key and answer.refusal_reason != "blocked":
        caches.answers.set(key, answer.model_copy(deep=True))


def cached_answer(
    question: str,
    retrieval: CachedRetrieval,
    llm: LLM,
    settings: Settings,
    caches: CacheBundle,
) -> tuple[Answer, bool]:
    """`answer_from_chunks` behind the answer cache. Returns (answer, served_from_cache).
    An LLM error propagates and caches nothing."""
    if not caches.enabled:
        return answer_from_chunks(question, retrieval.chunks, llm, settings), False
    key = answer_key(retrieval.key, question, retrieval.chunks, settings)
    hit = lookup_answer(caches, key)
    if hit is not None:
        return hit, True
    answer = answer_from_chunks(question, retrieval.chunks, llm, settings)
    store_answer(caches, key, answer)
    return answer, False
