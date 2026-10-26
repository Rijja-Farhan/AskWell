import json
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from app.evaluation.report import compare_reports, percentile, retrieval_label, summarize_latency
from app.evaluation.runner import ConfigMismatchError, load_results, run_config, run_eval
from app.rag import rerank as rerank_module
from app.rag.rerank import MAX_LENGTH, RERANK_VERSION, CrossEncoderReranker, get_reranker, rerank
from app.rag.retrieve import RetrievedChunk, _first_stage, retrieve
from app.rag.store import build_index, open_collection, resolve
from tests.conftest import FakeEmbedder, make_settings


class FakeReranker:
    """Scores each text with `scorer` (default: its length) and records every call."""

    model = "fake-reranker"

    def __init__(self, scorer=len) -> None:
        self.scorer = scorer
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, question: str, texts: list[str]) -> list[float]:
        self.calls.append((question, texts))
        return [float(self.scorer(t)) for t in texts]


def chunk(chunk_id: str, rank: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, doc_id=chunk_id, rank=rank, score=0.0, text=text, metadata={}
    )


# ---------------------------------------------------------------- rerank()


def test_rerank_orders_by_score_and_truncates():
    candidates = [chunk("a", 1, "x"), chunk("b", 2, "xxx"), chunk("c", 3, "xx")]
    out = rerank("q", candidates, FakeReranker(), k=2)
    assert [(c.chunk_id, c.rank, c.score, c.first_stage_rank) for c in out] == [
        ("b", 1, 3.0, 2),
        ("c", 2, 2.0, 3),
    ]


def test_rerank_ties_keep_first_stage_order():
    candidates = [chunk("z", 1, "aa"), chunk("a", 2, "bb"), chunk("m", 3, "c")]
    out = rerank("q", candidates, FakeReranker(), k=3)
    assert [c.chunk_id for c in out] == ["z", "a", "m"]


def test_rerank_empty_makes_no_call():
    reranker = FakeReranker()
    assert rerank("q", [], reranker, k=5) == []
    assert reranker.calls == []


# ---------------------------------------------------------------- retrieve() with a reranker


@pytest.fixture(scope="module")
def indexed(corpus, tmp_path_factory):
    settings = make_settings(tmp_path_factory.mktemp("rerank"), eval_chat_interval=0)
    build_index(corpus, FakeEmbedder(), settings)
    return settings


def reranked(settings, **overrides):
    return settings.model_copy(update={"reranker": "cross-encoder", **overrides})


def test_reranker_none_is_phase6_output(indexed):
    question = "What does error NX-2150 mean?"
    reranker = FakeReranker()
    plain = retrieve(question, 10, FakeEmbedder(), indexed)
    assert retrieve(question, 10, FakeEmbedder(), indexed, reranker) == plain
    assert reranker.calls == []  # reranker="none" ignores a passed reranker
    assert all(c.first_stage_rank is None for c in plain)


def test_missing_reranker_raises(indexed):
    with pytest.raises(ValueError, match="get_reranker"):
        retrieve("anything", 5, FakeEmbedder(), reranked(indexed))


def test_bm25_only_chunk_can_reach_rank_1(indexed):
    question = "What does error NX-2150 mean?"
    collection = open_collection(indexed)
    pool = _first_stage(collection, question, 10, 40, FakeEmbedder(), indexed, None)  # 20 + 20
    assert len(pool) <= 40 and len(pool) > 20
    late = [c for c in pool if c.rank > 10 and c.dense_rank is None]
    assert late, "expected a BM25-only chunk outside the fused top 10"
    target = late[-1]
    reranker = FakeReranker(lambda text: 1.0 if text == target.text else 0.0)

    hits = retrieve(question, 10, FakeEmbedder(), reranked(indexed), reranker)
    assert len(hits) == 10
    assert hits[0].chunk_id == target.chunk_id
    assert (hits[0].first_stage_rank, hits[0].bm25_rank) == (target.rank, target.bm25_rank)
    assert len(reranker.calls[0][1]) == len(pool)

    narrow = reranked(indexed, rerank_candidates=10)  # the target is not a candidate
    hits = retrieve(question, 10, FakeEmbedder(), narrow, reranker)
    assert target.chunk_id not in {h.chunk_id for h in hits}


def test_get_reranker_none(tmp_path):
    assert get_reranker(make_settings(tmp_path)) is None


# ---------------------------------------------------------------- CrossEncoderReranker, offline


def tiny_tokenizer() -> Tokenizer:
    """A real word-level tokenizer with BERT-style pair template, so truncation and
    type ids behave as with the downloaded tokenizer.json."""
    words = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "warm", "up", "q", "word", "other"]
    tokenizer = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return tokenizer


class StubSession:
    """Stands in for an ONNX session: the logit is the number of non-padding tokens."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.feeds: list[dict] = []

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in self.names]

    def run(self, _outputs, feeds):
        self.feeds.append(feeds)
        return [feeds["attention_mask"].sum(axis=1, keepdims=True).astype(np.float32)]


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    downloads: list[tuple] = []
    session = StubSession(["input_ids", "attention_mask", "token_type_ids"])

    def fake_download(repo, filename, revision, cache_dir):
        downloads.append((repo, filename, revision, cache_dir))
        return str(tmp_path / filename)

    monkeypatch.setattr(rerank_module, "hf_hub_download", fake_download)
    monkeypatch.setattr(
        rerank_module, "Tokenizer", SimpleNamespace(from_file=lambda _: tiny_tokenizer())
    )
    monkeypatch.setattr(rerank_module.ort, "InferenceSession", lambda *_a, **_k: session)
    reranker = CrossEncoderReranker("org/model", "abc123", "onnx/model.onnx", tmp_path)
    return reranker, session, downloads


def test_cross_encoder_downloads_pinned_files(stubbed, tmp_path):
    reranker, session, downloads = stubbed
    assert reranker.model == "org/model"
    assert {d[1] for d in downloads} == {"tokenizer.json", "onnx/model.onnx"}
    assert all(d[0] == "org/model" and d[2] == "abc123" and d[3] == tmp_path for d in downloads)
    assert len(session.feeds) == 1  # the warm-up run


def test_cross_encoder_scores_pairs_in_batches(stubbed):
    reranker, session, _ = stubbed
    texts = ["word"] * 17 + ["word word other"] * 3
    scores = reranker.score("q", texts)
    # [CLS] q [SEP] text [SEP]: 5 tokens for "word", 7 for "word word other"
    assert scores == [5.0] * 17 + [7.0] * 3
    batches = session.feeds[1:]
    assert [b["input_ids"].shape[0] for b in batches] == [16, 4]
    feed = batches[1]
    assert set(feed) == {"input_ids", "attention_mask", "token_type_ids"}
    assert feed["input_ids"].dtype == np.int64
    assert feed["token_type_ids"][-1].tolist() == [0, 0, 0, 1, 1, 1, 1]


def test_cross_encoder_truncates_only_the_chunk(stubbed):
    reranker, session, _ = stubbed
    question = "q " * 20
    [score] = reranker.score(question, ["word " * 2000])
    ids = session.feeds[-1]["input_ids"][0].tolist()
    assert score == MAX_LENGTH and len(ids) == MAX_LENGTH
    assert ids[:22] == [2] + [6] * 20 + [3]  # the whole question survives


def test_cross_encoder_skips_inputs_the_model_lacks(stubbed):
    reranker, session, _ = stubbed
    session.names = ["input_ids", "attention_mask"]  # XLM-R style: no token_type_ids
    reranker.input_names = set(session.names)
    reranker.score("q", ["word"])
    assert set(session.feeds[-1]) == {"input_ids", "attention_mask"}


# ---------------------------------------------------------------- eval runner and report


@pytest.fixture
def settings(indexed, tmp_path):
    return indexed.model_copy(update={"eval_reports_dir": tmp_path / "reports"})


def test_run_config_records_reranker(settings):
    eval_set = resolve(settings.eval_set_path)
    plain = run_config(settings, eval_set)
    assert plain["reranker"] == "none" and "rerank_model" not in plain
    config = run_config(reranked(settings), eval_set)
    assert config["rerank_version"] == RERANK_VERSION
    assert {
        "rerank_model",
        "rerank_model_revision",
        "rerank_onnx_file",
        "rerank_candidates",
    } <= set(config)
    assert "reranked by `" in retrieval_label(config)
    assert "reranked" not in retrieval_label(plain)


def test_eval_records_rerank_latency(settings):
    reranker = FakeReranker()
    s = reranked(settings)
    summary = run_eval(
        "rr", ["retrieval"], FakeEmbedder(), None, None, s, limit=3, reranker=reranker
    )
    assert summary.complete and len(reranker.calls) == 3
    results = load_results(settings.eval_reports_dir / "rr" / "results.jsonl").values()
    for r in results:
        assert 0 <= r.latency["rerank"] <= r.latency["retrieval"]
        assert r.retrieved[0].first_stage_rank is not None
    latency = summary.report["latency"]
    assert latency["rerank"]["n"] == latency["retrieval"]["n"] == 3
    assert "## Latency" in (settings.eval_reports_dir / "rr" / "report.md").read_text()


def test_eval_without_reranker_records_no_rerank_latency(settings):
    summary = run_eval("plain", ["retrieval"], FakeEmbedder(), None, None, settings, limit=2)
    assert "rerank" not in summary.report["latency"]
    assert summary.report["latency"]["retrieval"]["n"] == 2


def test_manifest_without_reranker_resumes_as_none(settings):
    run_dir = settings.eval_reports_dir / "pre7"
    run_dir.mkdir(parents=True)
    config = run_config(settings, resolve(settings.eval_set_path))
    del config["reranker"]  # as written before Phase 7
    (run_dir / "manifest.json").write_text(json.dumps({"name": "pre7", "config": config}))

    assert run_eval("pre7", ["retrieval"], FakeEmbedder(), None, None, settings, limit=2).complete
    with pytest.raises(ConfigMismatchError, match="reranker"):
        run_eval(
            "pre7",
            ["retrieval"],
            FakeEmbedder(),
            None,
            None,
            reranked(settings),
            limit=2,
            reranker=FakeReranker(),
        )


def test_percentile_nearest_rank():
    values = [float(v) for v in range(20, 0, -1)]
    assert percentile(values, 50) == 10.0
    assert percentile(values, 95) == 19.0
    assert percentile([3.0], 95) == 3.0


def test_latency_summary_and_deltas():
    def result(seconds: float, ok: bool = True):
        state = "ok" if ok else "error"
        return SimpleNamespace(
            latency={"retrieval": seconds, "rerank": seconds / 2},
            ok=lambda stage, state=state: stage == "retrieval" and state == "ok",
        )

    latency = summarize_latency([result(1.0), result(3.0), result(2.0), result(99.0, ok=False)])
    assert latency["retrieval"] == {"n": 3, "median": 2.0, "p95": 3.0}
    assert latency["rerank"] == {"n": 3, "median": 1.0, "p95": 1.5}

    new = {"name": "new", "categories": {}, "latency": latency}
    old = {"name": "old", "categories": {}, "latency": {"retrieval": {"median": 0.5, "p95": 1.0}}}
    assert compare_reports(new, old)["latency"] == {"retrieval": {"median": 1.5, "p95": 2.0}}
    assert compare_reports(new, {"name": "pre7", "categories": {}})["latency"] == {}
