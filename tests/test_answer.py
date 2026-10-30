"""Grounded answering: source numbering, citations, refusals and baseline compatibility."""

import pytest

from app.rag.answer import (
    GROUNDED_SYSTEM_PROMPT,
    REFUSAL_TEXT,
    SYSTEM_PROMPT,
    answer_from_chunks,
    build_grounded_message,
    build_user_message,
    is_refusal,
    parse_citations,
)
from app.rag.retrieve import RetrievedChunk
from tests.conftest import FakeLLM, make_settings


def chunk(n: int, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc-{n}#0",
        doc_id=f"doc-{n}",
        rank=n,
        score=score,
        text=f"Text of chunk {n}.",
        metadata={
            "title": f"Title {n}",
            "section": f"Section {n}",
            "version": "2",
            "status": "active",
            "effective_date": "2025-01-01",
        },
    )


@pytest.fixture
def chunks() -> list[RetrievedChunk]:
    return [chunk(1), chunk(2), chunk(3)]


def settings(tmp_path, **overrides):
    return make_settings(tmp_path, **overrides)


def test_sources_are_numbered_and_labelled(chunks):
    message = build_grounded_message("What?", chunks)
    lines = message.splitlines()
    assert lines[0] == "Sources:"
    assert (
        "[1] Title 1 | section: Section 1 | doc_id: doc-1 | version 2 (active) | "
        "effective 2025-01-01"
    ) in message
    assert (
        message.index("[1] Title 1") < message.index("[2] Title 2") < message.index("[3] Title 3")
    )
    assert "Text of chunk 2." in message and message.endswith("Question: What?")


def test_citations_in_first_cited_order_with_adjacent_and_repeated(chunks):
    text, citations, invalid = parse_citations("A [3]. B [2][1]. C [3].", chunks)
    assert text == "A [3]. B [2][1]. C [3]."
    assert [c.number for c in citations] == [3, 2, 1]
    assert citations[0].chunk_id == "doc-3#0" and citations[0].title == "Title 3"
    assert citations[0].section == "Section 3" and citations[0].doc_id == "doc-3"
    assert invalid == 0


def test_out_of_range_markers_are_removed_and_counted(chunks):
    text, citations, invalid = parse_citations("Yes [1] and more [9]. Also [0].", chunks)
    assert text == "Yes [1] and more. Also."
    assert [c.number for c in citations] == [1]
    assert invalid == 2


def test_no_markers_means_no_citations(chunks):
    assert parse_citations("Plain.", chunks) == ("Plain.", [], 0)


def test_refusal_detection():
    assert is_refusal(REFUSAL_TEXT)
    assert is_refusal(f"  {REFUSAL_TEXT}\n")
    assert is_refusal(f'"{REFUSAL_TEXT}"')
    assert not is_refusal("I don't know the answer [1].")
    assert not is_refusal(REFUSAL_TEXT + " But maybe 5 days [1].")


def test_grounded_answer_has_citations(tmp_path, chunks):
    llm = FakeLLM("Five days [2]. Approved by HR [1][2][7].")
    result = answer_from_chunks("q", chunks, llm, settings(tmp_path))
    assert result.answer == "Five days [2]. Approved by HR [1][2]."
    assert [c.number for c in result.citations] == [2, 1]
    assert result.invalid_citations == 1 and not result.refused
    [(system, user)] = llm.calls
    assert system == GROUNDED_SYSTEM_PROMPT and REFUSAL_TEXT in system
    assert user == build_grounded_message("q", chunks)


def test_model_refusal(tmp_path, chunks):
    result = answer_from_chunks("q", chunks, FakeLLM(REFUSAL_TEXT), settings(tmp_path))
    assert result.refused and result.refusal_reason == "model"
    assert result.answer == REFUSAL_TEXT and result.citations == []


def test_no_context_refuses_without_llm_call(tmp_path):
    llm = FakeLLM()
    result = answer_from_chunks("q", [], llm, settings(tmp_path))
    assert result.refused and result.refusal_reason == "no_context"
    assert result.answer == REFUSAL_TEXT and llm.calls == []


def test_low_score_refuses_without_llm_call(tmp_path):
    llm = FakeLLM()
    s = settings(tmp_path, reranker="cross-encoder")
    result = answer_from_chunks("q", [chunk(1, score=-4.5), chunk(2, score=-6)], llm, s)
    assert result.refused and result.refusal_reason == "low_score" and llm.calls == []


def test_score_at_threshold_is_answered(tmp_path):
    llm = FakeLLM("Yes [1].")
    s = settings(tmp_path, reranker="cross-encoder")
    result = answer_from_chunks("q", [chunk(1, score=-4.0)], llm, s)
    assert not result.refused and len(llm.calls) == 1


def test_gate_skipped_without_reranker_or_threshold(tmp_path):
    low = [chunk(1, score=-50.0)]
    for overrides in (
        {"reranker": "none"},
        {"reranker": "cross-encoder", "refusal_min_score": None},
    ):
        llm = FakeLLM("Yes [1].")
        result = answer_from_chunks("q", low, llm, settings(tmp_path, **overrides))
        assert not result.refused and len(llm.calls) == 1


def test_baseline_sends_the_old_messages(tmp_path, chunks):
    llm = FakeLLM("plain")
    s = settings(tmp_path, answer_mode="baseline", reranker="cross-encoder")
    result = answer_from_chunks("q", [chunk(1, score=-50.0)], llm, s)  # gate does not apply
    assert llm.calls == [(SYSTEM_PROMPT, build_user_message("q", [chunk(1, score=-50.0)]))]
    assert result.answer == "plain" and not result.refused and result.citations == []
    assert result.refusal_reason is None and result.invalid_citations == 0
    assert build_user_message("q", chunks).startswith("Context:\n\n[doc-1] Title 1\n")
    # even with no chunks baseline still asks the model, as before Phase 9
    empty = FakeLLM()
    answer_from_chunks("q", [], empty, s)
    assert len(empty.calls) == 1
