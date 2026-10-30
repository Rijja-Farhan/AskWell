"""RedisCache against an in-memory fake client (no Redis server, no `redis` package)."""

import ast
from pathlib import Path

import pytest

from app.rag import cache as cache_module
from app.rag.answer import Answer, Citation
from app.rag.cache import (
    CacheBundle,
    LRUTTLCache,
    NullCache,
    RedisCache,
    decode_value,
    encode_value,
)
from app.rag.retrieve import RetrievedChunk
from tests.conftest import make_settings


class FakeRedis:
    """The few commands RedisCache uses, with expiry on an injectable clock."""

    def __init__(self, clock=None) -> None:
        self.data: dict[str, tuple[str, float | None]] = {}
        self.clock = clock or (lambda: 0.0)
        self.fail = False
        self.flushdb_called = False

    def _check(self) -> None:
        if self.fail:
            raise ConnectionError("redis is down: secret-key-value")

    def _live(self, key: str) -> bool:
        item = self.data.get(key)
        if item is None:
            return False
        if item[1] is not None and item[1] <= self.clock():
            del self.data[key]
            return False
        return True

    def get(self, key):
        self._check()
        return self.data[key][0].encode() if self._live(key) else None

    def set(self, key, value, ex=None):
        self._check()
        self.data[key] = (value, None if ex is None else self.clock() + ex)

    def scan_iter(self, match="*", count=None):
        self._check()
        prefix = match.rstrip("*")
        return iter([k.encode() for k in list(self.data) if self._live(k) and k.startswith(prefix)])

    def delete(self, *keys):
        self._check()
        for key in keys:
            self.data.pop(key.decode() if isinstance(key, bytes) else key, None)

    def flushdb(self):
        self.flushdb_called = True
        self.data.clear()


def chunk(i: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{i}",
        doc_id="d1",
        rank=i,
        score=0.1 + i / 3,  # a float with a long repr
        text="text é\n",
        metadata={"version": "2.1", "year": 2024, "flag": True, "f": 0.30000000000000004},
        dense_rank=2,
    )


def answer() -> Answer:
    return Answer(
        question="q?",
        answer="a [1]",
        chunks=[chunk(1), chunk(2)],
        model="m",
        citations=[Citation(number=1, chunk_id="c1", doc_id="d1", title="t", section="s")],
        refused=False,
    )


def make(clock=None, ttl=60, retry=5.0):
    client = FakeRedis(clock)
    return client, RedisCache(client, "askwell:answers:", ttl, clock or (lambda: 0.0), retry)


def test_get_set_and_miss():
    _, c = make()
    assert c.get("k") is None
    c.set("k", (0.1, 0.2, 1 / 3))
    assert c.get("k") == (0.1, 0.2, 1 / 3)
    assert c.stats() == {"hits": 1, "misses": 1, "size": 1}


def test_stats_have_the_same_keys_as_the_lru_cache():
    _, c = make()
    assert set(c.stats()) == set(LRUTTLCache(1, 1).stats())


def test_ttl_sets_expiry_and_entries_expire():
    now = [0.0]
    client, c = make(clock=lambda: now[0], ttl=60)
    c.set("k", (1.0,))
    assert client.data["askwell:answers:k"][1] == 60
    now[0] = 59
    assert c.get("k") == (1.0,)
    now[0] = 61
    assert c.get("k") is None


def test_zero_ttl_stores_nothing():
    client, c = make(ttl=0)
    c.set("k", (1.0,))
    assert client.data == {}


def test_clear_only_touches_own_prefix_and_never_flushes():
    client = FakeRedis()
    mine = RedisCache(client, "askwell:answers:", 60)
    other = RedisCache(client, "askwell:retrieval:", 60)
    client.data["unrelated"] = ("x", None)
    mine.set("a", (1.0,))
    other.set("b", (2.0,))
    mine.clear()
    assert mine.get("a") is None
    assert other.get("b") == (2.0,)
    assert "unrelated" in client.data
    assert not client.flushdb_called


def test_clear_handles_many_keys():
    client, c = make()
    for i in range(1200):
        c.set(f"k{i}", (float(i),))
    c.clear()
    assert client.data == {}


def test_redis_errors_are_misses_and_noops(caplog):
    client, c = make()
    c.set("k", (1.0,))
    client.fail = True
    with caplog.at_level("WARNING", logger="askwell.cache"):
        assert c.get("k") is None
        c.set("other", (2.0,))
        c.clear()
        assert c.stats()["size"] == 0
    logged = caplog.text
    assert "ConnectionError" in logged
    assert "secret-key-value" not in logged and "askwell:answers:k" not in logged


def test_a_down_server_is_skipped_until_the_retry_time():
    now = [0.0]
    client, c = make(clock=lambda: now[0], retry=5.0)
    c.set("k", (1.0,))
    client.fail = True
    assert c.get("k") is None  # error, circuit opens
    client.fail = False
    assert c.get("k") is None  # skipped without calling Redis
    now[0] = 6.0
    assert c.get("k") == (1.0,)


def test_corrupt_or_foreign_entry_is_a_miss():
    client, c = make()
    client.data["askwell:answers:k"] = ("not json", None)
    assert c.get("k") is None
    client.data["askwell:answers:k"] = ('{"t":"pickle","v":1}', None)
    assert c.get("k") is None


def test_round_trip_vector_chunks_and_answer():
    vector = (0.1, 1 / 3, -2.5e-10, 1.0, 3)
    assert decode_value(encode_value(vector)) == vector
    chunks = (chunk(1), chunk(2))
    back = decode_value(encode_value(chunks))
    assert isinstance(back, tuple) and list(back) == list(chunks)
    assert back[0].metadata == chunks[0].metadata
    ans = answer()
    assert decode_value(encode_value(ans)) == ans
    assert decode_value(encode_value(())) == ()


def test_cannot_encode_other_types():
    with pytest.raises(TypeError):
        encode_value({"a": 1})
    with pytest.raises(TypeError):
        encode_value(object())


def test_values_are_copies_not_shared_objects():
    _, c = make()
    ans = answer()
    c.set("a", ans)
    got = c.get("a")
    got.answer = "poisoned"
    assert c.get("a").answer == "a [1]"


def test_cache_module_does_not_import_pickle():
    tree = ast.parse(Path(cache_module.__file__).read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {m for m in imported if m and "pickle" in m}
    assert "pickle" not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def test_bundle_default_is_memory(tmp_path):
    bundle = CacheBundle.from_settings(make_settings(tmp_path))
    assert all(isinstance(layer, LRUTTLCache) for layer in bundle.layers().values())


def test_bundle_redis_backend_uses_prefixed_layers(tmp_path, monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr(cache_module, "redis_client", lambda settings: client)
    settings = make_settings(tmp_path, cache_backend="redis", redis_key_prefix="t:")
    bundle = CacheBundle.from_settings(settings)
    assert all(isinstance(layer, RedisCache) for layer in bundle.layers().values())
    bundle.embeddings.set("k", (1.0,))
    bundle.retrieval.set("k", (chunk(1),))
    bundle.answers.set("k", answer())
    assert sorted(client.data) == ["t:answers:k", "t:embeddings:k", "t:retrieval:k"]
    bundle.retrieval.clear()
    assert sorted(client.data) == ["t:answers:k", "t:embeddings:k"]
    assert bundle.stats()["answers"]["size"] == 1


def test_bundle_redis_respects_disabled_and_zero_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "redis_client", lambda settings: FakeRedis())
    off = CacheBundle.from_settings(
        make_settings(tmp_path, cache_backend="redis", cache_enabled=False)
    )
    assert not off.enabled and isinstance(off.answers, NullCache)
    zero = CacheBundle.from_settings(
        make_settings(tmp_path, cache_backend="redis", cache_answer_entries=0)
    )
    assert isinstance(zero.answers, NullCache) and isinstance(zero.retrieval, RedisCache)


def test_unreachable_server_never_fails_a_lookup(tmp_path):
    pytest.importorskip("redis")
    settings = make_settings(tmp_path, cache_backend="redis", redis_url="redis://127.0.0.1:1/0")
    bundle = CacheBundle.from_settings(settings)
    assert bundle.answers.get("k") is None
    bundle.answers.set("k", (1.0,))
    bundle.clear()


# --- the Phase 14 pipeline behaviour on the Redis backend ----------------------------------


def test_pipeline_hits_and_equality_match_the_memory_backend(tmp_path_factory, corpus, monkeypatch):
    from app.rag.cache import cached_answer, cached_retrieve
    from app.rag.store import sync_index
    from tests.conftest import FakeEmbedder, FakeLLM

    settings = make_settings(tmp_path_factory.mktemp("redis_pipe"), retrieval_mode="dense")
    sync_index(corpus, FakeEmbedder(), settings)
    question = "How many days of annual leave do new employees get?"
    results = {}
    for backend in ("memory", "redis"):
        monkeypatch.setattr(cache_module, "redis_client", lambda s: FakeRedis())
        caches = CacheBundle.from_settings(settings.model_copy(update={"cache_backend": backend}))
        emb, llm = FakeEmbedder(), FakeLLM("Twenty days [1].")
        first = cached_retrieve(question, 3, emb, settings, caches)
        second = cached_retrieve(question, 3, emb, settings, caches)
        a1, hit1 = cached_answer(question, second, llm, settings, caches)
        a2, hit2 = cached_answer(question, second, llm, settings, caches)
        assert (first.cached, second.cached, hit1, hit2) == (False, True, False, True)
        assert len(llm.calls) == 1 and a1 == a2 and first.chunks == second.chunks
        results[backend] = (first.key, [c.model_dump() for c in second.chunks], a2.model_dump())
    assert results["memory"] == results["redis"]
