"""Run the eval set through retrieval, generation and judging, saving after every question.

Stages run in order over all questions: `retrieval` (embeddings only), `generate`, `judge`.
Results live in `<eval_reports_dir>/<name>/results.jsonl`, rewritten after every question, so an
interrupted run resumes where it stopped: a rerun skips every stage already `ok`. Results store
chunk ids, not texts; later stages read the texts back from Chroma, with no embedding call.
"""

import hashlib
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import openai
from pydantic import BaseModel

from app.config import Settings
from app.dataset.schemas import EvalQuestion
from app.evaluation import metrics
from app.evaluation.judge import JudgeError, judge_answer
from app.evaluation.report import build_report, load_report, write_report
from app.rag.answer import ANSWER_VERSION, answer_from_chunks
from app.rag.bm25 import BM25_VERSION
from app.rag.embeddings import Embedder
from app.rag.filters import FILTER_VERSION
from app.rag.llm import LLM
from app.rag.rerank import RERANK_VERSION, Reranker
from app.rag.retrieve import RetrievedChunk, retrieve
from app.rag.store import open_collection, resolve

STAGES = ("retrieval", "generate", "judge")
RECALL_KS = (1, 3, 5, 10)
SUPERSEDED_K = 5
# Errors that mean "the provider is down or rate limited", after the SDK's own retries.
UNAVAILABLE = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)
MAX_CONSECUTIVE_UNAVAILABLE = 3
# Runs made before Phase 5 have no chunk_strategy in their manifest; they used fixed chunks.
# Runs made before Phase 6 have no retrieval_mode; they used dense retrieval.
# Runs made before Phase 7 have no reranker; they were not reranked.
# Runs made before Phase 8 have no version_policy; they kept every version.
# Runs made before Phase 9 have no answer_mode; they used the minimal baseline prompt.
MANIFEST_DEFAULTS = {
    "chunk_strategy": "fixed",
    "retrieval_mode": "dense",
    "reranker": "none",
    "version_policy": "all",
    "answer_mode": "baseline",
}

Stage = Literal["retrieval", "generate", "judge"]


class ConfigMismatchError(RuntimeError):
    """An existing run was made with a different configuration."""


class StageStatus(BaseModel):
    state: Literal["ok", "error"]
    message: str = ""


class ChunkRef(BaseModel):
    chunk_id: str
    doc_id: str
    rank: int
    score: float
    first_stage_rank: int | None = None  # rank before reranking; None if not reranked


class JudgeFields(BaseModel):
    faithfulness: float
    relevance: float
    behavior_ok: bool
    reason: str
    model: str


class QuestionResult(BaseModel):
    id: str
    category: str
    question: str
    expected_behavior: str
    gold_doc_ids: list[str]
    superseded_doc_ids: list[str]
    retrieved: list[ChunkRef] = []
    doc_ranks: dict[str, int] = {}
    metrics: dict[str, float | bool | None] = {}
    answer: str | None = None
    model: str | None = None
    refused: bool | None = None  # None: not answered yet, or an answer made before Phase 9
    refusal_reason: str | None = None
    citations: list[str] | None = None  # chunk ids of the valid cited sources
    invalid_citations: int | None = None
    judge: JudgeFields | None = None
    # Seconds per stage, excluding throttle waits. "rerank" is the reranker's share of
    # "retrieval" (which times the whole retrieve() call).
    latency: dict[str, float] = {}
    status: dict[str, StageStatus] = {}

    def ok(self, stage: str) -> bool:
        status = self.status.get(stage)
        return status is not None and status.state == "ok"


class RunSummary(BaseModel):
    name: str
    run_dir: Path
    n: int
    stages: list[str]
    done: dict[str, int]  # questions with the stage `ok`
    stopped_stage: str | None = None  # stage stopped because the provider stayed unavailable
    report: dict

    @property
    def complete(self) -> bool:
        return all(self.done[s] == self.n for s in self.stages)


class Throttle:
    """Keeps at least `interval` seconds between calls to `wait()`."""

    def __init__(
        self,
        interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = interval
        self.clock = clock
        self.sleep = sleep
        self.last: float | None = None
        self.waited = 0.0  # total seconds slept, so latencies can leave the waits out

    def wait(self) -> None:
        if self.last is not None:
            remaining = self.interval - (self.clock() - self.last)
            if remaining > 0:
                self.sleep(remaining)
                self.waited += remaining
        self.last = self.clock()


class ThrottledLLM:
    """An LLM whose every call first waits on a shared throttle."""

    def __init__(self, llm: LLM, throttle: Throttle) -> None:
        self.llm = llm
        self.model = llm.model
        self.throttle = throttle

    def complete(self, system: str, user: str) -> str:
        self.throttle.wait()
        return self.llm.complete(system, user)


class TimedReranker:
    """A reranker that adds the seconds spent in each `score()` call to `seconds`."""

    def __init__(self, reranker: Reranker) -> None:
        self.reranker = reranker
        self.model = reranker.model
        self.seconds = 0.0

    def score(self, question: str, texts: list[str]) -> list[float]:
        start = time.perf_counter()
        try:
            return self.reranker.score(question, texts)
        finally:
            self.seconds += time.perf_counter() - start


def load_eval_set(path: Path) -> list[EvalQuestion]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalQuestion.model_validate_json(line) for line in lines if line.strip()]


def judge_model(settings: Settings) -> str:
    return settings.eval_judge_model or settings.llm_model


def run_config(settings: Settings, eval_set: Path) -> dict:
    """Everything that changes the results. A resumed run must match it exactly."""
    config = {
        "collection": settings.vector_collection,
        "embedding_model": settings.embedding_model,
        "chunk_strategy": settings.chunk_strategy,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "eval_retrieval_k": settings.eval_retrieval_k,
        "retrieval_top_k": settings.retrieval_top_k,
        "chat_model": settings.llm_model,
        "judge_model": judge_model(settings),
        "eval_set_sha256": hashlib.sha256(eval_set.read_bytes()).hexdigest(),
        "retrieval_mode": settings.retrieval_mode,
    }
    if settings.retrieval_mode != "dense":  # fusion settings only matter for bm25/hybrid
        config |= {
            "hybrid_candidates": settings.hybrid_candidates,
            "rrf_k": settings.rrf_k,
            "bm25_weight": settings.bm25_weight,
            "bm25_version": BM25_VERSION,
        }
    config["reranker"] = settings.reranker
    if settings.reranker != "none":
        config |= {
            "rerank_model": settings.rerank_model,
            "rerank_model_revision": settings.rerank_model_revision,
            "rerank_onnx_file": settings.rerank_onnx_file,
            "rerank_candidates": settings.rerank_candidates,
            "rerank_version": RERANK_VERSION,
        }
    config["version_policy"] = settings.version_policy
    if settings.version_policy != "all":
        config["filter_version"] = FILTER_VERSION
    config["answer_mode"] = settings.answer_mode
    if settings.answer_mode == "grounded":
        config |= {
            "answer_version": ANSWER_VERSION,
            "refusal_min_score": settings.refusal_min_score,
        }
    return config


def check_manifest(run_dir: Path, name: str, config: dict) -> None:
    path = run_dir / "manifest.json"
    if path.exists():
        # Keys added after a run was made get the value that run implicitly used.
        saved = {**MANIFEST_DEFAULTS, **json.loads(path.read_text(encoding="utf-8"))["config"]}
        changed = sorted(k for k in config.keys() | saved.keys() if config.get(k) != saved.get(k))
        if changed:
            details = ", ".join(f"{k}: {saved.get(k)!r} -> {config.get(k)!r}" for k in changed)
            raise ConfigMismatchError(
                f"run {name!r} was made with a different config ({details}). "
                f"Use --fresh to start it over, or pick another --name."
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(path, {"name": name, "config": config})


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_results(path: Path) -> dict[str, QuestionResult]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    results = [QuestionResult.model_validate_json(line) for line in lines if line.strip()]
    return {r.id: r for r in results}


def save_results(path: Path, results: list[QuestionResult]) -> None:
    """Rewrite the whole file through a temp file, so a crash never leaves half a line."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8")
    tmp.replace(path)


def new_result(q: EvalQuestion) -> QuestionResult:
    return QuestionResult(
        id=q.id,
        category=q.category,
        question=q.question,
        expected_behavior=q.expected_behavior,
        gold_doc_ids=q.gold_doc_ids,
        superseded_doc_ids=q.superseded_doc_ids,
    )


def score_retrieval(q: EvalQuestion, ranks: dict[str, int]) -> dict[str, float | bool | None]:
    scores: dict[str, float | bool | None] = {
        f"recall@{k}": metrics.recall_at_k(q.gold_doc_ids, ranks, k) for k in RECALL_KS
    }
    scores["mrr"] = metrics.reciprocal_rank(q.gold_doc_ids, ranks)
    scores[f"superseded_hit@{SUPERSEDED_K}"] = metrics.superseded_hit(
        q.superseded_doc_ids, ranks, SUPERSEDED_K
    )
    return scores


def load_chunks(collection, refs: list[ChunkRef]) -> list[RetrievedChunk]:
    """Rebuild retrieved chunks (with texts) from their ids, keeping rank and score."""
    if not refs:
        return []
    got = collection.get(ids=[r.chunk_id for r in refs], include=["documents", "metadatas"])
    by_id = dict(zip(got["ids"], zip(got["documents"], got["metadatas"], strict=True), strict=True))
    missing = [r.chunk_id for r in refs if r.chunk_id not in by_id]
    if missing:
        raise ConfigMismatchError(
            f"chunks {missing[:3]} are no longer in the index; the index was rebuilt with "
            "different chunks. Use --fresh."
        )
    return [
        RetrievedChunk(
            chunk_id=r.chunk_id,
            doc_id=r.doc_id,
            rank=r.rank,
            score=r.score,
            text=by_id[r.chunk_id][0],
            metadata=dict(by_id[r.chunk_id][1]),
        )
        for r in refs
    ]


def run_eval(
    name: str,
    stages: list[str],
    embedder: Embedder | None,
    llm: LLM | None,
    judge_llm: LLM | None,
    settings: Settings,
    fresh: bool = False,
    limit: int | None = None,
    compare: str | None = None,
    log: Callable[[str], None] = lambda _: None,
    reranker: Reranker | None = None,
) -> RunSummary:
    """Run the requested stages on the eval set, resuming any earlier progress.
    `reranker` is needed for the retrieval stage unless `settings.reranker` is "none"."""
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; choose from {', '.join(STAGES)}")
    stages = [s for s in STAGES if s in stages]
    reports_dir = resolve(settings.eval_reports_dir)
    run_dir = reports_dir / name
    if compare:
        load_report(reports_dir / compare, compare)  # fail before any API call, not after
    if fresh and run_dir.exists():
        shutil.rmtree(run_dir)

    eval_path = resolve(settings.eval_set_path)
    questions = load_eval_set(eval_path)[:limit]
    check_manifest(run_dir, name, run_config(settings, eval_path))
    results_path = run_dir / "results.jsonl"
    saved = load_results(results_path)
    results = [saved.get(q.id) or new_result(q) for q in questions]

    # One throttle for every chat call, generation and judge alike.
    throttle = Throttle(settings.eval_chat_interval)
    chat = {
        "generate": ThrottledLLM(llm, throttle) if llm else None,
        "judge": ThrottledLLM(judge_llm, throttle) if judge_llm else None,
    }
    timed = TimedReranker(reranker) if reranker else None
    collection = None

    def run_one(stage: str, q: EvalQuestion, r: QuestionResult) -> None:
        nonlocal collection
        if stage == "retrieval":
            assert embedder is not None, "the retrieval stage needs an embedder"
            before = timed.seconds if timed else 0.0
            hits = retrieve(q.question, settings.eval_retrieval_k, embedder, settings, timed)
            if timed and settings.reranker != "none":
                r.latency["rerank"] = round(timed.seconds - before, 3)
            r.retrieved = [
                ChunkRef(**h.model_dump(include=set(ChunkRef.model_fields))) for h in hits
            ]
            r.doc_ranks = metrics.doc_ranks(hits)
            r.metrics = score_retrieval(q, r.doc_ranks)
            return
        collection = collection or open_collection(settings)
        chunks = load_chunks(collection, r.retrieved[: settings.retrieval_top_k])
        if stage == "generate":
            assert chat["generate"] is not None, "the generate stage needs an LLM"
            result = answer_from_chunks(q.question, chunks, chat["generate"], settings)
            r.answer, r.model = result.answer, result.model
            if settings.answer_mode == "grounded":
                r.refused, r.refusal_reason = result.refused, result.refusal_reason
                r.citations = [c.chunk_id for c in result.citations]
                r.invalid_citations = result.invalid_citations
        else:
            assert chat["judge"] is not None, "the judge stage needs a judge LLM"
            verdict = judge_answer(q.question, r.answer or "", chunks, q, chat["judge"])
            r.judge = JudgeFields(**verdict.model_dump(), model=chat["judge"].model)

    prerequisite = {"generate": "retrieval", "judge": "generate"}
    stopped_stage = None
    for stage in stages:
        if stopped_stage:
            break
        todo = [
            (q, r)
            for q, r in zip(questions, results, strict=True)
            if not r.ok(stage) and (stage not in prerequisite or r.ok(prerequisite[stage]))
        ]
        log(f"{stage}: {len(todo)} to run")
        consecutive = 0
        for i, (q, r) in enumerate(todo, start=1):
            start, waited = time.perf_counter(), throttle.waited
            try:
                run_one(stage, q, r)
                r.status[stage] = StageStatus(state="ok")
                consecutive = 0
            except UNAVAILABLE as exc:
                r.status[stage] = StageStatus(state="error", message=error_message(exc))
                consecutive += 1
            except (openai.APIError, JudgeError) as exc:
                r.status[stage] = StageStatus(state="error", message=error_message(exc))
                consecutive = 0
            elapsed = time.perf_counter() - start - (throttle.waited - waited)
            r.latency[stage] = round(elapsed, 3)
            save_results(results_path, results)
            log(f"  {stage} {i}/{len(todo)} {q.id}: {r.status[stage].state}")
            if consecutive >= MAX_CONSECUTIVE_UNAVAILABLE:
                stopped_stage = stage
                log(f"{stage}: provider unavailable {consecutive} times in a row, stopping")
                break

    report = build_report(name, results, run_config(settings, eval_path), reports_dir, compare)
    write_report(run_dir, report)
    return RunSummary(
        name=name,
        run_dir=run_dir,
        n=len(results),
        stages=stages,
        done={s: sum(r.ok(s) for r in results) for s in STAGES},
        stopped_stage=stopped_stage,
        report=report,
    )


def error_message(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:200]}"
