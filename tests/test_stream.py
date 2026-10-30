"""Streaming answers: holdback of possible refusals, final Answer, and OpenAICompatLLM.stream."""

from types import SimpleNamespace

import pytest

from app.rag.answer import REFUSAL_TEXT, Answer, answer_from_chunks, stream_answer
from app.rag.llm import OpenAICompatLLM
from app.rag.retrieve import RetrievedChunk
from tests.conftest import FakeLLM, make_settings


def chunk(n: int, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc-{n}#0",
        doc_id=f"doc-{n}",
        rank=n,
        score=score,
        text=f"Text {n}.",
        metadata={"title": f"T{n}", "section": f"S{n}"},
    )


def run(llm, chunks, settings):
    items = list(stream_answer("q?", chunks, llm, settings))
    *deltas, final = items
    assert isinstance(final, Answer) and all(isinstance(d, str) for d in deltas)
    return deltas, final


def test_streamed_text_matches_final_answer_and_complete(tmp_path):
    settings = make_settings(tmp_path)
    reply = "Twenty days [1]. See also [2]."
    llm = FakeLLM(reply)
    deltas, final = run(llm, [chunk(1), chunk(2)], settings)
    assert len(deltas) > 1 and "".join(deltas) == reply
    assert final == answer_from_chunks("q?", [chunk(1), chunk(2)], FakeLLM(reply), settings)
    assert [c.number for c in final.citations] == [1, 2]


def test_invalid_marker_is_streamed_but_removed_from_final(tmp_path):
    deltas, final = run(FakeLLM("Fact [1]. Other [9]."), [chunk(1)], make_settings(tmp_path))
    assert "[9]" in "".join(deltas)
    assert final.answer == "Fact [1]. Other." and final.invalid_citations == 1


def test_model_refusal_never_leaks_as_tokens(tmp_path):
    for reply in (REFUSAL_TEXT, f'"{REFUSAL_TEXT}"', f"  {REFUSAL_TEXT}\n"):
        deltas, final = run(FakeLLM(reply), [chunk(1)], make_settings(tmp_path))
        assert deltas == []
        assert final.refused and final.refusal_reason == "model" and final.answer == REFUSAL_TEXT


def test_reply_starting_like_refusal_is_released_when_it_diverges(tmp_path):
    reply = "I don't know of any other policy, but leave is 20 days [1]."
    deltas, final = run(FakeLLM(reply), [chunk(1)], make_settings(tmp_path))
    assert "".join(deltas) == reply and not final.refused


def test_reply_ending_as_refusal_prefix_is_still_emitted(tmp_path):
    deltas, final = run(FakeLLM("I don't"), [chunk(1)], make_settings(tmp_path))
    assert "".join(deltas) == "I don't" and not final.refused


@pytest.mark.parametrize(
    "chunks, settings_kw, reason",
    [
        ([], {}, "no_context"),
        ([chunk(1, score=-9.0)], {"reranker": "cross-encoder"}, "low_score"),
    ],
)
def test_refusals_before_llm_make_no_call(tmp_path, chunks, settings_kw, reason):
    llm = FakeLLM("x")
    deltas, final = run(llm, chunks, make_settings(tmp_path, **settings_kw))
    assert deltas == [] and final.refusal_reason == reason and llm.calls == []


def test_baseline_streams_directly_without_citation_handling(tmp_path):
    settings = make_settings(tmp_path, answer_mode="baseline")
    deltas, final = run(FakeLLM(REFUSAL_TEXT + " [9]"), [chunk(1)], settings)
    assert "".join(deltas) == final.answer == REFUSAL_TEXT + " [9]"
    assert not final.refused and final.citations == []


def delta(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        yield SimpleNamespace(choices=[])
        yield delta("Hel")
        yield delta(None)
        yield delta("")
        yield delta("lo")


def test_openai_compat_stream_skips_empty_deltas():
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm = OpenAICompatLLM(client, "m")
    assert list(llm.stream("sys", "user")) == ["Hel", "lo"]
    assert completions.kwargs["stream"] is True and completions.kwargs["temperature"] == 0
