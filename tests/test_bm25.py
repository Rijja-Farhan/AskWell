import json
import re

import pytest

from app.dataset.schemas import EvalQuestion
from app.ingestion.models import Block, ParsedDocument
from app.rag.bm25 import BM25Index, tokenize
from app.rag.retrieve import RetrievedChunk, fuse, retrieve
from app.rag.store import build_index, sync_index
from tests.conftest import ROOT, FakeEmbedder, make_settings

PROBES = ROOT / "eval" / "probes" / "exact_terms.jsonl"

# ---------------------------------------------------------------- tokenizer and scoring


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("What does error NX-2150 mean?", ["error", "nx-2150", "mean"]),
        ("WH-DAL-02 | WMS 8.2, grade G5.", ["wh-dal-02", "wms", "8.2", "grade", "g5"]),
        (
            "Document code: HR-POL-005 · Version 1",
            ["document", "code", "hr-pol-005", "version", "1"],
        ),
        ("snake_case_id and e-mail", ["snake_case_id", "e-mail"]),
        ("The policy.", ["policy"]),
    ],
)
def test_tokenize(text, tokens):
    assert tokenize(text) == tokens


def test_bm25_prefers_rare_terms_and_short_chunks():
    index = BM25Index(
        ["a", "b", "c"],
        ["error NX-2150 printer", "error error error printer printer laptop", "laptop bag"],
        [{}, {}, {}],
    )
    assert [h.chunk_id for h in index.search("NX-2150 error", 3)] == ["a", "b"]
    assert index.search("unknownword", 3) == []  # only chunks with a score above 0


def chunk(chunk_id: str, rank: int) -> RetrievedChunk:
    doc_id = chunk_id.split("::")[0]
    return RetrievedChunk(
        chunk_id=chunk_id, doc_id=doc_id, rank=rank, score=0.0, text="", metadata={}
    )


def test_rrf_arithmetic():
    dense = [chunk("a::0", 1), chunk("b::0", 2), chunk("c::0", 3)]
    lexical = [chunk("c::0", 1), chunk("d::0", 2)]
    fused = fuse(dense, lexical, k=4, rrf_k=60, bm25_weight=0.5)
    expected = {
        "a::0": 1 / 61,
        "b::0": 1 / 62,
        "c::0": 1 / 63 + 0.5 / 61,
        "d::0": 0.5 / 62,
    }
    assert [c.chunk_id for c in fused] == sorted(expected, key=lambda c: -expected[c])
    for c in fused:
        assert c.score == pytest.approx(expected[c.chunk_id])
    c = next(x for x in fused if x.chunk_id == "c::0")
    assert (c.rank, c.dense_rank, c.bm25_rank) == (1, 3, 1)
    d = next(x for x in fused if x.chunk_id == "d::0")
    assert (d.dense_rank, d.bm25_rank) == (None, 2)


def test_rrf_ties_break_by_chunk_id():
    fused = fuse([chunk("b::0", 1)], [chunk("a::0", 1)], k=2, rrf_k=60, bm25_weight=1.0)
    assert [c.chunk_id for c in fused] == ["a::0", "b::0"]


# ---------------------------------------------------------------- retrieval modes on a real index


@pytest.fixture(scope="module")
def indexed(corpus, tmp_path_factory):
    settings = make_settings(tmp_path_factory.mktemp("bm25"))  # structured chunks
    build_index(corpus, FakeEmbedder(), settings)
    return settings


@pytest.mark.parametrize("term", ["NX-2150", "GL-6115", "WH-CHI-03"])
def test_bm25_ranks_identifier_chunk_first(indexed, term):
    hits = retrieve(term, 5, FakeEmbedder(), indexed.model_copy(update={"retrieval_mode": "bm25"}))
    assert term in hits[0].text
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_bm25_mode_makes_no_embedding_call(indexed):
    embedder = FakeEmbedder()
    retrieve("NX-2150", 5, embedder, indexed.model_copy(update={"retrieval_mode": "bm25"}))
    assert embedder.calls == 0


def test_hybrid_reports_both_ranks(indexed):
    hits = retrieve("What does error NX-2150 mean?", 10, FakeEmbedder(), indexed)
    assert indexed.retrieval_mode == "hybrid"
    assert len(hits) == 10
    assert all(h.dense_rank or h.bm25_rank for h in hits)
    assert any(h.dense_rank and h.bm25_rank for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_dense_mode_is_unchanged(indexed):
    dense = indexed.model_copy(update={"retrieval_mode": "dense"})
    hits = retrieve("ethics hotline", 5, FakeEmbedder(), dense)
    assert all(h.dense_rank is None and h.bm25_rank is None for h in hits)
    assert all(-1.0 <= h.score <= 1.0 for h in hits)


def with_note(doc: ParsedDocument, note: str) -> ParsedDocument:
    last = doc.blocks[-1]
    block = Block(kind="paragraph", text=note, section_path=last.section_path, page=last.page)
    return doc.model_copy(update={"blocks": [*doc.blocks, block], "text": f"{doc.text}\n\n{note}"})


def test_cache_follows_sync(corpus, tmp_path):
    settings = make_settings(tmp_path, retrieval_mode="bm25")
    sync_index(corpus, FakeEmbedder(), settings)
    assert retrieve("quokkafrobnicator", 5, FakeEmbedder(), settings) == []  # warms the cache

    edited = [
        with_note(d, "Note: the quokkafrobnicator rule.") if d.meta.doc_id == "hr-mtg-001-v1" else d
        for d in corpus
    ]
    assert sync_index(edited, FakeEmbedder(), settings).updated == ["hr-mtg-001-v1"]
    hits = retrieve("quokkafrobnicator", 5, FakeEmbedder(), settings)
    assert [h.doc_id for h in hits] == ["hr-mtg-001-v1"]

    sync_index(corpus, FakeEmbedder(), settings)  # and back again
    assert retrieve("quokkafrobnicator", 5, FakeEmbedder(), settings) == []


# ---------------------------------------------------------------- probe set


def test_probe_set(corpus):
    rows = [EvalQuestion.model_validate_json(line) for line in PROBES.read_text().splitlines()]
    assert len(rows) == 24 and [r.id for r in rows] == [f"q{i}" for i in range(101, 125)]
    main = [
        json.loads(line) for line in (ROOT / "eval" / "eval_set.jsonl").read_text().splitlines()
    ]
    main_text = " ".join(q["question"] + " " + " ".join(q["exact_terms"]) for q in main)
    docs = {d.meta.doc_id: d for d in corpus}
    for r in rows:
        assert r.category == "exact-term" and r.expected_behavior == "answer"
        [term] = r.exact_terms
        assert term in r.question
        assert not re.search(rf"\b{re.escape(term)}\b", main_text), term
        containing = {
            d.meta.doc_id
            for d in corpus
            if d.meta.status == "active" and re.search(rf"\b{re.escape(term)}\b", d.text)
        }
        assert set(r.gold_doc_ids) == containing, term
        assert 1 <= len(containing) <= 3
        assert all(docs[g].meta.status == "active" for g in r.gold_doc_ids)
