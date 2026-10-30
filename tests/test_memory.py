"""SessionStore: caps, LRU eviction, TTL and thread safety."""

import json
import threading
from pathlib import Path

from app.rag.memory import InMemorySessionStore, Turn

ROOT = Path(__file__).resolve().parents[1]


def turn(n: int) -> Turn:
    return Turn(f"q{n}", f"sq{n}", f"a{n}", False, (f"doc-{n}",))


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_append_get_exists_clear():
    store = InMemorySessionStore()
    assert store.get("s") == [] and not store.exists("s")
    store.append("s", turn(1))
    assert store.exists("s") and store.get("s") == [turn(1)]
    store.clear("s")
    assert not store.exists("s") and store.get("s") == []
    store.clear("s")  # idempotent


def test_get_returns_a_copy():
    store = InMemorySessionStore()
    store.append("s", turn(1))
    store.get("s").append(turn(2))
    assert len(store.get("s")) == 1


def test_per_session_turn_cap_drops_oldest():
    store = InMemorySessionStore(max_turns=3)
    for n in range(5):
        store.append("s", turn(n))
    assert [t.question for t in store.get("s")] == ["q2", "q3", "q4"]


def test_session_cap_evicts_least_recently_used():
    store = InMemorySessionStore(max_sessions=2)
    store.append("a", turn(1))
    store.append("b", turn(2))
    store.get("a")  # touch a, so b is now the oldest
    store.append("c", turn(3))
    assert store.exists("a") and store.exists("c") and not store.exists("b")


def test_ttl_expires_idle_sessions_and_touch_renews():
    clock = Clock()
    store = InMemorySessionStore(ttl_seconds=100, clock=clock)
    store.append("old", turn(1))
    store.append("live", turn(2))
    clock.now = 80
    store.get("live")  # renews
    clock.now = 150
    assert not store.exists("old") and store.exists("live")
    assert store.get("old") == []
    clock.now = 300
    assert not store.exists("live")


def test_concurrent_appends_are_safe_and_bounded():
    store = InMemorySessionStore(max_turns=1000, max_sessions=5)

    def work(worker: int) -> None:
        for n in range(200):
            store.append(f"s{n % 8}", turn(worker * 1000 + n))
            store.get(f"s{(n + 1) % 8}")

    threads = [threading.Thread(target=work, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(store.exists(f"s{i}") for i in range(8)) <= 5

    one = InMemorySessionStore(max_turns=10_000)
    threads = [
        threading.Thread(target=lambda: [one.append("x", turn(n)) for n in range(250)])
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(one.get("x")) == 2000


def test_multiturn_probe_gold_docs_exist_and_shape():
    docs = {d["doc_id"] for d in json.loads((ROOT / "data/metadata/documents.json").read_text())}
    rows = [
        json.loads(line)
        for line in (ROOT / "eval/probes/multiturn.jsonl").read_text().split("\n")
        if line
    ]
    assert len(rows) >= 12 and len({r["id"] for r in rows}) == len(rows)
    for row in rows:
        assert 2 <= len(row["turns"]) <= 3
        for t in row["turns"]:
            assert t["question"] and t["scripted_query"] and t["gold_doc_ids"]
            assert set(t["gold_doc_ids"]) <= docs
