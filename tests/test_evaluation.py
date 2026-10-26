import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.dataset.schemas import EvalQuestion
from app.evaluation.judge import JudgeError, judge_answer, parse_judge_reply
from app.evaluation.metrics import doc_ranks, recall_at_k, reciprocal_rank, superseded_hit
from app.evaluation.runner import (
    ConfigMismatchError,
    Throttle,
    load_results,
    run_config,
    run_eval,
)
from app.rag.answer import answer_from_chunks, answer_question
from app.rag.retrieve import retrieve
from app.rag.store import build_index, resolve
from tests.conftest import ROOT, FakeEmbedder, FakeLLM, make_settings

# ---------------------------------------------------------------- metrics


def hits(*doc_ids: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(doc_id=d, rank=i) for i, d in enumerate(doc_ids, start=1)]


def test_doc_ranks_keeps_best_rank_per_doc():
    assert doc_ranks(hits("a", "a", "b", "a", "c")) == {"a": 1, "b": 3, "c": 5}


def test_recall_counts_chunks_not_docs():
    ranks = doc_ranks(hits("a", "a", "a", "b"))  # b is the 4th chunk, though only the 2nd doc
    assert recall_at_k(["b"], ranks, 3) == 0.0
    assert recall_at_k(["b"], ranks, 4) == 1.0
    assert recall_at_k(["a", "b", "z"], ranks, 4) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "a"], ranks, 1) == 1.0  # duplicate gold ids count once


def test_reciprocal_rank():
    ranks = doc_ranks(hits("x", "y", "b", "a"))
    assert reciprocal_rank(["a", "b"], ranks) == pytest.approx(1 / 3)
    assert reciprocal_rank(["z"], ranks) == 0.0


def test_superseded_hit():
    ranks = doc_ranks(hits("new", "x", "x", "x", "x", "old"))
    assert superseded_hit(["old"], ranks, 5) is False
    assert superseded_hit(["old"], ranks, 6) is True


def test_empty_gold_gives_none():
    ranks = doc_ranks(hits("a"))
    assert recall_at_k([], ranks, 5) is None
    assert reciprocal_rank([], ranks) is None
    assert superseded_hit([], ranks, 5) is None


# ---------------------------------------------------------------- judge

VERDICT = {"faithfulness": 0.5, "relevance": 1, "behavior_ok": True, "reason": "fine"}


class ScriptedLLM:
    """Returns the given replies in order (the last one repeats) and records each call."""

    model = "fake-judge"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.replies[min(len(self.calls), len(self.replies)) - 1]


EVAL_Q = EvalQuestion(
    id="q999",
    question="How many PTO days carry over?",
    category="simple",
    user_id="u01",
    gold_doc_ids=["hr-pol-002-v2"],
    reference_answer="Five days.",
    expected_behavior="answer",
)


def test_parse_plain_and_fenced_json():
    plain = parse_judge_reply(json.dumps(VERDICT))
    fenced = parse_judge_reply("Here you go:\n```json\n" + json.dumps(VERDICT) + "\n```")
    assert plain == fenced
    assert plain.faithfulness == 0.5 and plain.relevance == 1.0 and plain.behavior_ok


@pytest.mark.parametrize(
    "reply",
    ["not json", '{"faithfulness": 2, "relevance": 1, "behavior_ok": true, "reason": ""}',
     '{"faithfulness": 1, "relevance": 0.7, "behavior_ok": true, "reason": ""}'],
)  # fmt: skip
def test_parse_rejects_invalid(reply):
    with pytest.raises(ValueError):
        parse_judge_reply(reply)


def test_judge_retries_once_then_succeeds():
    llm = ScriptedLLM("oops", json.dumps(VERDICT))
    result = judge_answer(EVAL_Q.question, "Five.", [], EVAL_Q, llm)
    assert result.reason == "fine"
    assert len(llm.calls) == 2
    assert "not a valid JSON" in llm.calls[1][1]


def test_judge_invalid_twice_raises():
    llm = ScriptedLLM("oops", "still not json")
    with pytest.raises(JudgeError):
        judge_answer(EVAL_Q.question, "Five.", [], EVAL_Q, llm)
    assert len(llm.calls) == 2


def test_judge_message_has_context_answer_and_expectation(indexed):
    settings = indexed
    chunks = retrieve(EVAL_Q.question, 2, FakeEmbedder(), settings)
    llm = ScriptedLLM(json.dumps(VERDICT))
    judge_answer(EVAL_Q.question, "Five days.", chunks, EVAL_Q, llm)
    [(_, user)] = llm.calls
    for c in chunks:
        assert c.text in user
    assert "Five days." in user and "Expected behavior (answer)" in user


# ---------------------------------------------------------------- throttle


def test_throttle_spaces_calls():
    now, sleeps = [0.0], []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    throttle = Throttle(6.0, clock=lambda: now[0], sleep=sleep)
    throttle.wait()  # first call: no wait
    now[0] += 2.0
    throttle.wait()
    now[0] += 10.0
    throttle.wait()  # already 10 s apart
    assert sleeps == [4.0]
    assert throttle.waited == 4.0


# ---------------------------------------------------------------- runner


@pytest.fixture(scope="module")
def indexed(corpus, tmp_path_factory):
    settings = make_settings(tmp_path_factory.mktemp("index"), eval_chat_interval=0)
    build_index(corpus, FakeEmbedder(), settings)
    return settings


@pytest.fixture
def settings(indexed, tmp_path):
    return indexed.model_copy(update={"eval_reports_dir": tmp_path / "reports"})


ALL = ["retrieval", "generate", "judge"]


def judge_llm() -> ScriptedLLM:
    return ScriptedLLM(json.dumps(VERDICT))


def unavailable(*_args) -> str:
    request = httpx.Request("POST", "https://x/chat/completions")
    raise openai.InternalServerError(
        "high demand", response=httpx.Response(503, request=request), body=None
    )


def test_full_run_writes_all_files(settings):
    llm, judge = FakeLLM(), judge_llm()
    summary = run_eval("t", ALL, FakeEmbedder(), llm, judge, settings, limit=5)

    run_dir = settings.eval_reports_dir / "t"
    assert {p.name for p in run_dir.iterdir()} == {
        "manifest.json",
        "results.jsonl",
        "report.json",
        "report.md",
    }
    assert summary.complete and summary.done == {"retrieval": 5, "generate": 5, "judge": 5}
    assert len(llm.calls) == 5 and len(judge.calls) == 5

    results = list(load_results(run_dir / "results.jsonl").values())
    assert [r.id for r in results] == ["q001", "q002", "q003", "q004", "q005"]
    r = results[0]
    assert len(r.retrieved) == settings.eval_retrieval_k
    assert "text" not in r.retrieved[0].model_dump()
    assert set(r.metrics) >= {"recall@1", "recall@5", "recall@10", "mrr"}
    assert r.answer == "fake answer" and r.model == FakeLLM.model
    assert r.judge is not None and r.judge.faithfulness == 0.5
    assert set(r.latency) == set(ALL)

    # Generation saw the first retrieval_top_k chunks, in rank order.
    question = "What is the phone number for the ethics hotline?"
    top = retrieve(question, settings.retrieval_top_k, FakeEmbedder(), settings)
    assert all(c.text in llm.calls[0][1] for c in top)

    report = json.loads((run_dir / "report.json").read_text())
    assert report["categories"]["overall"]["n"] == 5
    assert report["categories"]["overall"]["behavior_pass"] == 1.0
    assert "# Eval report: t" in (run_dir / "report.md").read_text()


def test_rerun_makes_no_calls(settings):
    run_eval("t", ALL, FakeEmbedder(), FakeLLM(), judge_llm(), settings, limit=5)
    embedder, llm, judge = FakeEmbedder(), FakeLLM(), judge_llm()
    summary = run_eval("t", ALL, embedder, llm, judge, settings, limit=5)
    assert summary.complete
    assert embedder.calls == 0 and llm.calls == [] and judge.calls == []


def test_retrieval_stage_needs_no_llm(settings):
    summary = run_eval("t", ["retrieval"], FakeEmbedder(), None, None, settings, limit=5)
    assert summary.complete and summary.done["generate"] == 0
    overall = summary.report["categories"]["overall"]
    assert overall["n_scored"] == 5 and overall["faithfulness"] is None


def test_outage_stops_stage_and_resumes(settings):
    down = FakeLLM()
    down.complete = unavailable  # type: ignore[method-assign]
    summary = run_eval("t", ALL, FakeEmbedder(), down, judge_llm(), settings, limit=5)
    assert summary.stopped_stage == "generate"
    assert summary.done == {"retrieval": 5, "generate": 0, "judge": 0}
    results = load_results(settings.eval_reports_dir / "t" / "results.jsonl")
    states = [r.status.get("generate") for r in results.values()]
    assert [s.state for s in states if s] == ["error"] * 3  # stopped after 3, 2 untouched
    assert "InternalServerError" in states[0].message
    report = summary.report["categories"]["overall"]
    assert report["n_judged"] == 0 < report["n"]

    embedder, llm, judge = FakeEmbedder(), FakeLLM(), judge_llm()
    summary = run_eval("t", ALL, embedder, llm, judge, settings, limit=5)
    assert summary.complete and summary.stopped_stage is None
    assert embedder.calls == 0  # retrieval was kept
    assert len(llm.calls) == 5 and len(judge.calls) == 5


def test_judge_error_is_recorded_and_retried(settings):
    summary = run_eval("t", ALL, FakeEmbedder(), FakeLLM(), ScriptedLLM("no"), settings, limit=2)
    assert summary.done["judge"] == 0 and not summary.complete
    assert summary.stopped_stage is None  # a parse failure is not an outage
    assert [e["stage"] for e in summary.report["errors"]] == ["judge", "judge"]
    summary = run_eval("t", ALL, FakeEmbedder(), FakeLLM(), judge_llm(), settings, limit=2)
    assert summary.complete and summary.report["errors"] == []


def test_config_change_requires_fresh(settings):
    run_eval("t", ["retrieval"], FakeEmbedder(), None, None, settings, limit=2)
    changed = settings.model_copy(update={"llm_model": "other-model"})
    with pytest.raises(ConfigMismatchError, match="--fresh"):
        run_eval("t", ["retrieval"], FakeEmbedder(), None, None, changed, limit=2)
    summary = run_eval("t", ["retrieval"], FakeEmbedder(), None, None, changed, fresh=True, limit=2)
    assert summary.complete


def test_compare_adds_deltas(settings):
    run_eval("a", ALL, FakeEmbedder(), FakeLLM(), judge_llm(), settings, limit=5)
    summary = run_eval(
        "a", ALL, FakeEmbedder(), FakeLLM(), judge_llm(), settings, limit=5, compare="a"
    )
    deltas = summary.report["comparison"]["deltas"]["overall"]
    assert deltas["recall@5"] == 0.0 and deltas["faithfulness"] == 0.0
    assert "(+0.00)" in (settings.eval_reports_dir / "a" / "report.md").read_text()


def test_answer_from_chunks_matches_answer_question(indexed):
    question = "How many PTO days can I carry over?"
    via_question = answer_question(question, 5, FakeEmbedder(), FakeLLM(), indexed)
    chunks = retrieve(question, 5, FakeEmbedder(), indexed)
    assert answer_from_chunks(question, chunks, FakeLLM()) == via_question


# ---------------------------------------------------------------- CLI


def test_run_eval_without_key_exits_2(tmp_path: Path):
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env["LLM_PROVIDER"] = "gemini"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_eval.py"), "--name", "x"],
        cwd=tmp_path,  # no .env here
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 2
    assert "GEMINI_API_KEY" in result.stderr


def test_manifest_without_chunk_strategy_resumes_as_fixed(corpus, settings):
    # A fixed-chunk run as written before Phase 5, whose manifest has no chunk_strategy key.
    fixed = settings.model_copy(update={"chunk_strategy": "fixed", "vector_collection": "old"})
    build_index(corpus, FakeEmbedder(), fixed)
    run_dir = settings.eval_reports_dir / "old"
    run_dir.mkdir(parents=True)
    config = run_config(fixed, resolve(fixed.eval_set_path))
    del config["chunk_strategy"]
    (run_dir / "manifest.json").write_text(json.dumps({"name": "old", "config": config}))

    summary = run_eval("old", ["retrieval"], FakeEmbedder(), None, None, fixed, limit=2)
    assert summary.complete
    saved = json.loads((run_dir / "manifest.json").read_text())["config"]
    assert saved["chunk_strategy"] == "fixed"
    structured = fixed.model_copy(update={"chunk_strategy": "structure"})
    with pytest.raises(ConfigMismatchError, match="chunk_strategy"):
        run_eval("old", ["retrieval"], FakeEmbedder(), None, None, structured, limit=2)


def test_manifest_without_retrieval_mode_resumes_as_dense(corpus, settings):
    dense = settings.model_copy(update={"retrieval_mode": "dense", "vector_collection": "old6"})
    build_index(corpus, FakeEmbedder(), dense)
    run_dir = settings.eval_reports_dir / "pre6"
    run_dir.mkdir(parents=True)
    config = run_config(dense, resolve(dense.eval_set_path))
    del config["retrieval_mode"]  # as written before Phase 6
    (run_dir / "manifest.json").write_text(json.dumps({"name": "pre6", "config": config}))

    assert run_eval("pre6", ["retrieval"], FakeEmbedder(), None, None, dense, limit=2).complete
    hybrid = dense.model_copy(update={"retrieval_mode": "hybrid"})
    with pytest.raises(ConfigMismatchError, match="retrieval_mode"):
        run_eval("pre6", ["retrieval"], FakeEmbedder(), None, None, hybrid, limit=2)


def test_hybrid_config_records_fusion_settings(settings):
    config = run_config(settings, resolve(settings.eval_set_path))
    assert config["retrieval_mode"] == "hybrid"
    assert {"bm25_weight", "rrf_k", "hybrid_candidates", "bm25_version"} <= set(config)
    dense = run_config(
        settings.model_copy(update={"retrieval_mode": "dense"}), resolve(settings.eval_set_path)
    )
    assert "bm25_weight" not in dense
