"""Offline latency probe for the caches (Phase 14).

Usage: python scripts/run_cache_probe.py [--delay-ms 50] [--mode hybrid|dense]
Builds a temporary index from a temporary copy of the real corpus (`data/` and `chroma_db/`
are never touched) with a hashed fake embedder, runs the real API app in-process (RBAC on)
with an embedder and an LLM that each sleep `--delay-ms` per call, sends the same 10 questions
twice and compares the cold and warm passes. Checks:
  * warm answers are byte-identical to cold ones (apart from latency and the `cached` flag);
  * the warm pass made no embedder or LLM call;
  * warm median latency is at least 5x lower than cold (end to end, retrieval and answer layer);
  * after one document is changed and `POST /ingest` runs, the next request is a miss and shows
    the new content;
  * two users with different access never share an answer.
Writes eval/probes/results/cache_probe.json.
Exit codes: 0 all checks pass, 1 at least one failure, 2 setup problem.
"""

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.api import main as api_main  # noqa: E402
from app.api.deps import Deps  # noqa: E402
from app.api.main import create_app  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.rag.cache import index_fingerprint  # noqa: E402
from app.rag.ingest import ingest_corpus  # noqa: E402
from tests.conftest import FakeEmbedder  # noqa: E402

OUT = ROOT / "eval/probes/results/cache_probe.json"
SPEEDUP = 5.0
USER, ADMIN, OTHER = "u06", "u06", "u05"  # u06 reads everything and may ingest; u05 is public
QUESTIONS = [
    "How many days of annual leave do new employees get?",
    "What is the minimum password length?",
    "How do I request parental leave?",
    "What is the hotel limit for business travel in New York?",
    "What is the VPN gateway address?",
    "How do I report a lost laptop?",
    "What is the process for submitting an expense report?",
    "What are the working hours for warehouse staff?",
    "How are overtime hours paid?",
    "Who approves purchase orders above the standard limit?",
]
PASSWORD_Q = "What is the minimum password length?"


class SlowEmbedder(FakeEmbedder):
    """Hashed fake embedder that sleeps per call and records the time spent."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay
        self.spent = 0.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        began = time.perf_counter()
        time.sleep(self.delay)
        out = super().embed(texts)
        self.spent += time.perf_counter() - began
        return out


class SlowLLM:
    model = "probe-fake"

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        time.sleep(self.delay)
        return "Per the sources, here is the answer. [1]"

    def stream(self, system: str, user: str):
        yield self.complete(system, user)


class Timer:
    """Accumulates the time one layer's function took during the current request."""

    def __init__(self, fn):
        self.fn, self.spent = fn, 0.0

    def __call__(self, *args, **kwargs):
        began = time.perf_counter()
        try:
            return self.fn(*args, **kwargs)
        finally:
            self.spent += time.perf_counter() - began


def comparable(body: dict) -> str:
    return json.dumps(
        {k: v for k, v in body.items() if k not in ("latency_ms", "cached")}, sort_keys=True
    )


def median_ms(values: list[float]) -> float:
    return round(statistics.median(values) * 1000, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--delay-ms", type=float, default=50.0, help="sleep per embed/LLM call")
    parser.add_argument("--mode", choices=("hybrid", "dense"), default="hybrid")
    args = parser.parse_args()
    delay = args.delay_ms / 1000

    work = Path(tempfile.mkdtemp(prefix="askwell-cache-probe-"))
    try:
        return run(work, delay, args)
    except Exception as exc:  # noqa: BLE001
        print(f"setup problem: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run(work: Path, delay: float, args) -> int:
    root = work / "root"
    shutil.copytree(ROOT / "data" / "raw", root / "data" / "raw")
    shutil.copytree(ROOT / "data" / "metadata", root / "data" / "metadata")
    base = get_settings()
    settings = base.model_copy(
        update={
            "vector_db_path": work / "chroma",
            "embedding_model": FakeEmbedder.model,
            "llm_model": SlowLLM.model,
            "retrieval_mode": args.mode,
            "reranker": "none",
            "rbac_enabled": True,
            "memory_enabled": False,
            "answer_mode": "grounded",
            "injection_defense": True,
            "cache_enabled": True,
        }
    )
    ingest_corpus(root, FakeEmbedder, settings)  # build with the raw, undelayed embedder
    api_main.REPO_ROOT = root  # POST /ingest reads the temporary copy, never data/

    embedder, llm = SlowEmbedder(delay), SlowLLM(delay)
    retrieval_t, answer_t = Timer(api_main.cached_retrieve), Timer(api_main.cached_answer)
    api_main.cached_retrieve, api_main.cached_answer = retrieval_t, answer_t
    client = TestClient(create_app(settings, Deps(settings, embedder=embedder, llm=llm)))

    def ask(question: str, user: str = USER) -> tuple[dict, dict[str, float]]:
        retrieval_t.spent = answer_t.spent = embedder.spent = 0.0
        began = time.perf_counter()
        r = client.post("/chat", json={"question": question, "k": 5}, headers={"X-User-Id": user})
        if r.status_code != 200:
            raise RuntimeError(f"/chat returned {r.status_code}")
        total = time.perf_counter() - began
        return r.json(), {
            "total": total,
            "retrieval": retrieval_t.spent,
            "answer": answer_t.spent,
            "embedding": embedder.spent,
        }

    passes: dict[str, list[tuple[dict, dict[str, float]]]] = {}
    counts = {}
    for name in ("cold", "warm"):
        before = (embedder.calls, llm.calls)
        passes[name] = [ask(q) for q in QUESTIONS]
        counts[name] = {"embedder": embedder.calls - before[0], "llm": llm.calls - before[1]}

    layers = ("total", "retrieval", "answer", "embedding")
    medians = {
        name: {layer: median_ms([t[layer] for _, t in rows]) for layer in layers}
        for name, rows in passes.items()
    }
    speedup = {  # None when the warm pass spent no time in the layer at all
        layer: round(medians["cold"][layer] / medians["warm"][layer], 1)
        if medians["warm"][layer] >= 0.01
        else None
        for layer in layers
    }
    health = client.get("/health").json()["cache"]

    checks: dict[str, bool] = {}
    checks["warm_answers_identical"] = all(
        comparable(c) == comparable(w)
        for (c, _), (w, _) in zip(passes["cold"], passes["warm"], strict=True)
    )
    checks["cold_not_cached_warm_cached"] = all(not b["cached"] for b, _ in passes["cold"]) and all(
        b["cached"] for b, _ in passes["warm"]
    )
    checks["warm_made_no_embedder_or_llm_call"] = counts["warm"] == {"embedder": 0, "llm": 0}
    checks["warm_total_at_least_5x_faster"] = (speedup["total"] or 0) >= SPEEDUP
    checks["warm_retrieval_at_least_5x_faster"] = (speedup["retrieval"] or 0) >= SPEEDUP
    checks["warm_answer_at_least_5x_faster"] = (speedup["answer"] or 0) >= SPEEDUP

    # a change to one document, then POST /ingest: the next identical request is a miss
    first_body = passes["cold"][0][0]
    target = first_body["sources"][0]["doc_id"]
    meta_path = root / "data" / "metadata" / "documents.json"
    docs = json.loads(meta_path.read_text(encoding="utf-8"))
    entry = next(d for d in docs if d["doc_id"] == target)
    entry["title"] += " (revised, cache probe)"
    meta_path.write_text(json.dumps(docs), encoding="utf-8")
    fp_before = index_fingerprint(settings)
    ingest = client.post("/ingest", headers={"X-User-Id": ADMIN})
    llm_before = llm.calls
    after, _ = ask(QUESTIONS[0])
    checks["ingest_updated_the_document"] = ingest.status_code == 200 and ingest.json()[
        "updated"
    ] == [target]
    checks["fingerprint_changed"] = index_fingerprint(settings) != fp_before
    checks["miss_after_ingest"] = after["cached"] is False and llm.calls == llm_before + 1
    checks["new_content_after_ingest"] = any(
        "revised, cache probe" in s["title"] for s in after["sources"]
    ) and not any("revised, cache probe" in s["title"] for s in first_body["sources"])
    again, _ = ask(QUESTIONS[0])
    checks["rewarmed_after_ingest"] = again["cached"] is True

    # two users with different access never share an answer
    admin_a, _ = ask(PASSWORD_Q, ADMIN)
    other_a, _ = ask(PASSWORD_Q, OTHER)
    admin_b, _ = ask(PASSWORD_Q, ADMIN)
    other_b, _ = ask(PASSWORD_Q, OTHER)
    admin_docs = {s["doc_id"] for s in admin_a["sources"]}
    other_docs = {s["doc_id"] for s in other_a["sources"]}
    checks["users_do_not_share"] = (
        admin_b["cached"] is True
        and other_b["cached"] is True
        and not (other_a["cached"] and admin_a["cached"])
        and admin_docs != other_docs
        and {s["doc_id"] for s in other_b["sources"]} == other_docs
    )

    fp_times = []
    for _ in range(20):
        began = time.perf_counter()
        index_fingerprint(settings)
        fp_times.append(time.perf_counter() - began)

    print(f"mode={args.mode} delay={args.delay_ms:g} ms per embed/LLM call, 10 questions")
    print(f"\n{'layer':<12}{'cold ms':>10}{'warm ms':>10}{'speedup':>10}")
    for layer in layers:
        ratio = f"{speedup[layer]:.1f}x" if speedup[layer] is not None else "no calls"
        print(
            f"{layer:<12}{medians['cold'][layer]:>10.2f}{medians['warm'][layer]:>10.2f}{ratio:>10}"
        )
    print(f"\ncalls cold {counts['cold']}, warm {counts['warm']}")
    print("cache stats " + json.dumps(health))
    print(f"index_fingerprint median {median_ms(fp_times):.2f} ms")
    print()
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")

    passed = all(checks.values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "passed": passed,
                "mode": args.mode,
                "delay_ms": args.delay_ms,
                "questions": len(QUESTIONS),
                "median_ms": medians,
                "speedup": speedup,
                "calls": counts,
                "cache_stats": health,
                "index_fingerprint_median_ms": median_ms(fp_times),
                "checks": checks,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n{'ALL CHECKS PASSED' if passed else 'FAILURES'} -> {OUT.relative_to(ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
