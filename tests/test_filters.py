import json
from datetime import date

import pytest

from app.evaluation.report import retrieval_label
from app.evaluation.runner import ConfigMismatchError, run_config, run_eval
from app.rag.bm25 import get_index
from app.rag.filters import (
    FILTER_VERSION,
    RetrievalFilter,
    allowed_doc_ids,
    effective_policy,
    family,
    prefer_active,
)
from app.rag.retrieve import RetrievedChunk, retrieve
from app.rag.store import build_index, open_collection, resolve
from tests.conftest import ROOT, FakeEmbedder, make_settings
from tests.test_rerank import FakeReranker

DOCS = json.loads((ROOT / "data" / "metadata" / "documents.json").read_text())
SUPERSEDED = {d["doc_id"] for d in DOCS if d["status"] == "superseded"}
MEAL = "What is the daily meal allowance when traveling?"
MODES = ["dense", "bm25", "hybrid"]


@pytest.fixture(scope="module")
def indexed(corpus, tmp_path_factory):
    settings = make_settings(tmp_path_factory.mktemp("filters"), eval_chat_interval=0)
    build_index(corpus, FakeEmbedder(), settings)
    return settings


def with_mode(settings, mode, **overrides):
    return settings.model_copy(update={"retrieval_mode": mode, **overrides})


def doc_ids(hits) -> set[str]:
    return {h.doc_id for h in hits}


# ---------------------------------------------------------------- filters on a real index


def test_corpus_has_eight_superseded_docs():
    assert len(SUPERSEDED) == 8


@pytest.mark.parametrize("mode", MODES)
def test_department_filter(indexed, mode):
    hits = retrieve(
        "How do I reset my password?",
        10,
        FakeEmbedder(),
        with_mode(indexed, mode),
        filters=RetrievalFilter(departments=["HR"]),
    )
    assert hits and {h.metadata["department"] for h in hits} == {"HR"}


@pytest.mark.parametrize("mode", MODES)
def test_active_only_drops_every_superseded_doc(indexed, mode):
    settings = with_mode(indexed, mode, version_policy="active_only")
    for question in [MEAL, "How many days a week can I work from home?", "corporate card limit"]:
        hits = retrieve(question, 20, FakeEmbedder(), settings)
        assert hits and not doc_ids(hits) & SUPERSEDED


def test_allowed_doc_ids_active_only_is_exactly_the_active_docs(indexed):
    metadatas = get_index(open_collection(indexed)).metadatas
    allowed = allowed_doc_ids(metadatas, None, "active_only")
    assert allowed == {d["doc_id"] for d in DOCS} - SUPERSEDED
    assert allowed_doc_ids(metadatas, None, "all") is None
    assert allowed_doc_ids(metadatas, None, "prefer_active") is None


@pytest.mark.parametrize("mode", MODES)
def test_as_of_picks_the_version_in_force(indexed, mode):
    settings = with_mode(indexed, mode)
    mid_2024 = RetrievalFilter(as_of=date(2024, 6, 1), version_policy="active_only")
    families = {h.doc_id for h in retrieve(MEAL, 20, FakeEmbedder(), settings, filters=mid_2024)}
    assert "fin-pol-001-v2" in families
    assert not families & {"fin-pol-001-v1", "fin-pol-001-v3"}

    early = RetrievalFilter(as_of=date(2023, 1, 15))  # before fin-pol-001-v1 took effect
    hits = retrieve(MEAL, 20, FakeEmbedder(), settings, filters=early)
    assert not any(h.doc_id.startswith("fin-pol-001") for h in hits)


def test_as_of_replaces_the_version_policy():
    assert effective_policy(RetrievalFilter(as_of=date(2024, 1, 1)), "active_only") == "all"
    assert effective_policy(RetrievalFilter(version_policy="all"), "active_only") == "all"
    assert effective_policy(None, "prefer_active") == "prefer_active"


def test_no_match_returns_empty(indexed):
    embedder = FakeEmbedder()
    nothing = RetrievalFilter(as_of=date(2000, 1, 1))
    assert retrieve(MEAL, 5, embedder, indexed, filters=nothing) == []
    assert embedder.calls == 0  # no embedding call when nothing can match


def test_bm25_scores_unchanged_by_filtering(indexed):
    settings = with_mode(indexed, "bm25")
    plain = {h.chunk_id: h.score for h in retrieve(MEAL, 50, FakeEmbedder(), settings)}
    hr_fin = RetrievalFilter(departments=["Finance"], version_policy="active_only")
    filtered = retrieve(MEAL, 50, FakeEmbedder(), settings, filters=hr_fin)
    assert filtered
    for h in filtered:
        assert h.score == pytest.approx(plain[h.chunk_id])


def test_all_equals_phase7_output(indexed):
    settings = indexed.model_copy(update={"reranker": "cross-encoder", "version_policy": "all"})
    first = retrieve(MEAL, 10, FakeEmbedder(), settings, FakeReranker())
    again = retrieve(
        MEAL, 10, FakeEmbedder(), settings, FakeReranker(), RetrievalFilter(version_policy="all")
    )
    assert first == again
    assert doc_ids(first) & SUPERSEDED  # unfiltered: old versions can still rank


def test_prefer_active_on_the_index(indexed):
    settings = indexed.model_copy(update={"version_policy": "prefer_active"})
    hits = retrieve(MEAL, 20, FakeEmbedder(), settings)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    for i, h in enumerate(hits):
        if h.metadata["status"] == "superseded":
            below = hits[i + 1 :]
            assert not any(
                family(b.metadata) == family(h.metadata) and b.metadata["status"] == "active"
                for b in below
            )


# ---------------------------------------------------------------- prefer_active ordering


def chunk(doc_id: str, status: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}::{rank}",
        doc_id=doc_id,
        rank=rank,
        score=0.0,
        text="",
        metadata={"doc_id": doc_id, "status": status},  # no family_id: derived from doc_id
    )


def test_prefer_active_ordering():
    hits = [
        chunk("a-v1", "superseded", 1),
        chunk("x-v1", "active", 2),
        chunk("b-v1", "superseded", 3),  # family b has no active chunk: stays in place
        chunk("a-v2", "active", 4),
        chunk("a-v2", "active", 5),
        chunk("a-v1", "superseded", 6),  # already below the last active a chunk: stays
        chunk("y-v1", "active", 7),
    ]
    order = [c.chunk_id for c in prefer_active(hits)]
    assert order == [
        "x-v1::2",
        "b-v1::3",
        "a-v2::4",
        "a-v2::5",
        "a-v1::1",
        "a-v1::6",
        "y-v1::7",
    ]


def test_family_falls_back_to_doc_id():
    assert family({"doc_id": "fin-pol-001-v3"}) == "fin-pol-001"
    assert family({"doc_id": "x-v2", "family_id": "fam"}) == "fam"


# ---------------------------------------------------------------- eval config


@pytest.fixture
def settings(indexed, tmp_path):
    return indexed.model_copy(update={"eval_reports_dir": tmp_path / "reports"})


def test_run_config_records_version_policy(settings):
    eval_set = resolve(settings.eval_set_path)
    plain = run_config(settings, eval_set)
    assert plain["version_policy"] == "all" and "filter_version" not in plain
    active = run_config(settings.model_copy(update={"version_policy": "active_only"}), eval_set)
    assert active["filter_version"] == FILTER_VERSION
    assert "active_only" in retrieval_label(active)
    assert "versions" not in retrieval_label(plain)


def test_manifest_without_version_policy_resumes_as_all(settings):
    run_dir = settings.eval_reports_dir / "pre8"
    run_dir.mkdir(parents=True)
    config = run_config(settings, resolve(settings.eval_set_path))
    del config["version_policy"]  # as written before Phase 8
    (run_dir / "manifest.json").write_text(json.dumps({"name": "pre8", "config": config}))

    assert run_eval("pre8", ["retrieval"], FakeEmbedder(), None, None, settings, limit=2).complete
    active = settings.model_copy(update={"version_policy": "active_only"})
    with pytest.raises(ConfigMismatchError, match="version_policy"):
        run_eval("pre8", ["retrieval"], FakeEmbedder(), None, None, active, limit=2)
