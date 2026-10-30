import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ingestion.models import ParsedDocument
from app.rag.answer import SYSTEM_PROMPT, answer_question
from app.rag.chunking import chunk_document
from app.rag.embeddings import OpenAICompatEmbedder
from app.rag.errors import IndexNotReadyError, MissingAPIKeyError
from app.rag.llm import get_embedder, get_llm
from app.rag.retrieve import retrieve
from app.rag.store import build_index
from tests.conftest import ROOT, FakeEmbedder, FakeLLM, make_settings

# ---------------------------------------------------------------- chunking


def by_id(corpus: list[ParsedDocument], doc_id: str) -> ParsedDocument:
    return next(d for d in corpus if d.meta.doc_id == doc_id)


@pytest.mark.parametrize(("size", "overlap"), [(800, 100), (300, 0), (500, 499), (5000, 50)])
def test_chunks_cover_text_with_overlap(corpus, size, overlap):
    for doc in corpus:
        chunks = chunk_document(doc, size, overlap)
        assert chunks, doc.meta.doc_id
        assert chunks[0].start_char == 0
        assert chunks[-1].end_char == len(doc.text)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            assert c.chunk_id == f"{doc.meta.doc_id}::{i:04d}"
            assert c.doc_id == doc.meta.doc_id
            assert c.text == doc.text[c.start_char : c.end_char]
            assert 0 < len(c.text) <= size
            if i < len(chunks) - 1:
                assert len(c.text) == size
                assert chunks[i + 1].start_char == c.end_char - overlap


def test_chunk_metadata_is_scalar(corpus):
    doc = by_id(corpus, "hr-pol-001-v1")
    meta = chunk_document(doc, 800, 100)[0].metadata
    assert meta == {
        "doc_id": "hr-pol-001-v1",
        "title": "Remote and Hybrid Work Policy",
        "department": "HR",
        "doc_type": "policy",
        "version": 1,
        "status": "superseded",
        "access_level": "internal",
        "effective_date": "2023-03-01",
        "path": "data/raw/hr/hr-pol-001-v1.pdf",
    }
    assert all(isinstance(v, str | int | float | bool) for v in meta.values())


@pytest.mark.parametrize(("size", "overlap"), [(100, 100), (100, 150), (100, -1), (0, 0)])
def test_bad_chunk_settings(corpus, size, overlap):
    with pytest.raises(ValueError):
        chunk_document(corpus[0], size, overlap)


def test_chunking_is_deterministic(corpus):
    first = [c for d in corpus for c in chunk_document(d, 800, 100)]
    second = [c for d in corpus for c in chunk_document(d, 800, 100)]
    assert first == second


# ---------------------------------------------------------------- index and retrieval


@pytest.fixture(scope="module")
def indexed(corpus, tmp_path_factory):
    # These tests check Phase 3 fixed-size chunks with dense (cosine) scores; structured chunks
    # have test_chunking.py and BM25/hybrid have test_bm25.py.
    settings = make_settings(
        tmp_path_factory.mktemp("index"), chunk_strategy="fixed", retrieval_mode="dense"
    )
    stats = build_index(corpus, FakeEmbedder(), settings)
    return settings, stats


def test_build_index_stats(indexed, corpus):
    settings, stats = indexed
    expected = sum(len(chunk_document(d, 800, 100)) for d in corpus)
    assert stats.documents == len(corpus) == 52
    assert stats.chunks == expected
    assert stats.dimension == 512
    assert stats.embedding_model == FakeEmbedder.model


def test_rebuild_gives_same_count(corpus, tmp_path):
    settings = make_settings(tmp_path, chunk_strategy="fixed", retrieval_mode="dense")
    first = build_index(corpus, FakeEmbedder(), settings)
    second = build_index(corpus, FakeEmbedder(), settings)  # deletes and recreates
    assert first.chunks == second.chunks
    hits = retrieve("ethics hotline", first.chunks + 10, FakeEmbedder(), settings)
    assert len(hits) == first.chunks  # no duplicates left over from the first build


def test_retrieve_ranks_and_scores(indexed):
    settings, _ = indexed
    hits = retrieve("How many PTO days can I carry over?", 5, FakeEmbedder(), settings)
    assert len(hits) == 5
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)
    for h in hits:
        assert h.chunk_id.startswith(f"{h.doc_id}::")
        assert h.metadata["doc_id"] == h.doc_id


@pytest.mark.parametrize(
    ("question", "doc_id"),
    [
        ("Ethics Hotline 1-800-555-0142 anonymously ethics.nimbus.example", "hr-pol-003-v1"),
        ("Tomasz Nowak Hannah Lindqvist Monique Grant site manager", "ops-tbl-001-v1"),
        ("break-glass nx-bg-01 nx-bg-02 fire safe envelopes", "it-man-005-v1"),
    ],
)
def test_distinctive_phrase_ranks_its_doc_first(indexed, question, doc_id):
    settings, _ = indexed
    hits = retrieve(question, 3, FakeEmbedder(), settings)
    assert hits[0].doc_id == doc_id


def test_retrieve_without_index(tmp_path):
    with pytest.raises(IndexNotReadyError, match="build_index.py"):
        retrieve("anything", 3, FakeEmbedder(), make_settings(tmp_path))


def test_retrieve_with_other_embedding_model(indexed):
    settings, _ = indexed
    other = settings.model_copy(update={"embedding_model": "another-model"})
    with pytest.raises(IndexNotReadyError, match="build_index.py"):
        retrieve("anything", 3, FakeEmbedder(), other)


# ---------------------------------------------------------------- answering


def test_answer_question_prompt(indexed):
    settings, _ = indexed
    settings = settings.model_copy(update={"answer_mode": "baseline"})
    llm = FakeLLM("The hotline is 1-800-555-0142.")
    question = "What is the phone number for the ethics hotline?"
    result = answer_question(question, 4, FakeEmbedder(), llm, settings)

    assert result.answer == "The hotline is 1-800-555-0142."
    assert result.model == FakeLLM.model
    assert result.chunks == retrieve(question, 4, FakeEmbedder(), settings)
    [(system, user)] = llm.calls
    assert system == SYSTEM_PROMPT
    assert question in user
    for c in result.chunks:
        assert c.text in user
        assert f"[{c.doc_id}] {c.metadata['title']}" in user


# ---------------------------------------------------------------- provider factories


@pytest.mark.parametrize(
    ("provider", "variable"), [("gemini", "GEMINI_API_KEY"), ("openai", "OPENAI_API_KEY")]
)
def test_missing_key_names_variable(tmp_path, provider, variable):
    settings = make_settings(tmp_path, llm_provider=provider)
    for factory in (get_llm, get_embedder):
        with pytest.raises(MissingAPIKeyError, match=variable):
            factory(settings)


def test_unsupported_provider(tmp_path):
    settings = make_settings(tmp_path, llm_provider="anthropic", anthropic_api_key="x")
    with pytest.raises(ValueError, match="gemini, openai"):
        get_llm(settings)


def test_factories_pick_base_url(tmp_path):
    gemini = make_settings(tmp_path, llm_provider="gemini", gemini_api_key="test-key")
    openai_ = make_settings(tmp_path, llm_provider="openai", openai_api_key="test-key")
    for factory in (get_llm, get_embedder):
        g, o = factory(gemini), factory(openai_)
        assert str(g.client.base_url) == gemini.gemini_base_url
        assert str(o.client.base_url) == "https://api.openai.com/v1/"
        assert g.client.max_retries == gemini.llm_max_retries
    assert get_llm(gemini).model == gemini.llm_model
    assert get_embedder(gemini).model == gemini.embedding_model


def test_unsupported_vector_db(corpus, tmp_path):
    with pytest.raises(ValueError, match="chroma"):
        build_index(corpus[:1], FakeEmbedder(), make_settings(tmp_path, vector_db="qdrant"))


# ---------------------------------------------------------------- CLI


def test_ask_without_key_exits_2(tmp_path: Path):
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env["LLM_PROVIDER"] = "gemini"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ask.py"), "test"],
        cwd=tmp_path,  # no .env here
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 2
    assert "GEMINI_API_KEY" in result.stderr


# ---------------------------------------------------------------- OpenAI-compatible embedder


class FakeEmbeddingsAPI:
    """Stands in for `client.embeddings`: fails with 429 `fail_times` times, then answers
    with items in reverse order and index 0 missing (as Gemini does)."""

    def __init__(self, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def create(self, model: str, input: list[str]):
        import httpx
        import openai

        self.calls += 1
        if self.calls <= self.fail_times:
            response = httpx.Response(429, request=httpx.Request("POST", "https://x/embeddings"))
            raise openai.RateLimitError(
                "Quota exceeded. Please retry in 2.5s.", response=response, body=None
            )
        items = [
            SimpleNamespace(index=i or None, embedding=[float(len(text))])
            for i, text in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(items)))


def make_embedder(api: FakeEmbeddingsAPI, waits: list[float], **kwargs) -> OpenAICompatEmbedder:
    client = SimpleNamespace(embeddings=api)
    return OpenAICompatEmbedder(client, "m", sleep=waits.append, **kwargs)  # type: ignore[arg-type]


def test_embedder_orders_by_index_and_batches():
    api, waits = FakeEmbeddingsAPI(), []
    embedder = make_embedder(api, waits, batch_size=2)
    assert embedder.embed(["a", "bb", "ccc", "dddd", "eeeee"]) == [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ]
    assert api.calls == 3
    assert waits == []


def test_embedder_waits_on_rate_limit():
    api, waits = FakeEmbeddingsAPI(fail_times=2), []
    embedder = make_embedder(api, waits, rate_limit_waits=3)
    assert embedder.embed(["a", "bb"]) == [[1.0], [2.0]]
    assert waits == [3.5, 3.5]  # the provider's "retry in 2.5s" plus a 1s margin


def test_embedder_gives_up_after_waits():
    import openai

    api, waits = FakeEmbeddingsAPI(fail_times=10), []
    with pytest.raises(openai.RateLimitError):
        make_embedder(api, waits, rate_limit_waits=2).embed(["a"])
    assert len(waits) == 2
