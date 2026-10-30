"""Follow-up rewriting: no call without history, safe fallbacks, sanitizing, prompt window."""

from app.rag.memory import Turn
from app.rag.rewrite import MAX_QUERY_CHARS, rewrite_query
from tests.conftest import FakeLLM, make_settings


class BoomLLM(FakeLLM):
    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        raise RuntimeError("quota exceeded: secret-details")


def history(n: int) -> list[Turn]:
    return [Turn(f"question {i}", f"sq {i}", f"answer {i}", False, ()) for i in range(n)]


def test_empty_history_makes_no_llm_call(tmp_path):
    llm = FakeLLM("ignored")
    result = rewrite_query("What about contractors?", [], llm, make_settings(tmp_path))
    assert result.query == "What about contractors?" and not result.rewritten and llm.calls == []


def test_rewrites_with_history(tmp_path):
    llm = FakeLLM("annual leave for contractors")
    result = rewrite_query("And contractors?", history(1), llm, make_settings(tmp_path))
    assert result.query == "annual leave for contractors" and result.rewritten
    assert len(llm.calls) == 1 and "New question: And contractors?" in llm.calls[0][1]


def test_llm_error_falls_back(tmp_path):
    result = rewrite_query("And contractors?", history(2), BoomLLM(), make_settings(tmp_path))
    assert result.query == "And contractors?" and not result.rewritten


def test_empty_output_falls_back(tmp_path):
    for reply in ("", "   \n ", '""'):
        result = rewrite_query(
            "And contractors?", history(1), FakeLLM(reply), make_settings(tmp_path)
        )
        assert result.query == "And contractors?" and not result.rewritten


def test_output_is_single_line_trimmed_and_capped(tmp_path):
    settings = make_settings(tmp_path)
    llm = FakeLLM('  "Query: annual leave\n for   contractors"  \n')
    assert rewrite_query("q", history(1), llm, settings).query == "annual leave for contractors"
    long = rewrite_query("q", history(1), FakeLLM("word " * 500), settings).query
    assert len(long) <= MAX_QUERY_CHARS and "\n" not in long


def test_unchanged_query_is_not_flagged_rewritten(tmp_path):
    result = rewrite_query("Same?", history(1), FakeLLM(" Same? "), make_settings(tmp_path))
    assert result.query == "Same?" and not result.rewritten


def test_prompt_has_only_the_last_n_turns(tmp_path):
    llm = FakeLLM("x")
    settings = make_settings(tmp_path, memory_rewrite_turns=2)
    rewrite_query("follow-up", history(5), llm, settings)
    prompt = llm.calls[0][1]
    assert "question 3" in prompt and "question 4" in prompt
    assert "question 2" not in prompt and "question 0" not in prompt


def test_refused_turn_is_shown_as_unanswered(tmp_path):
    llm = FakeLLM("x")
    past = [Turn("old q", "old q", "I don't know. long refusal text", True, ())]
    rewrite_query("more?", past, llm, make_settings(tmp_path))
    assert "no answer was found" in llm.calls[0][1] and "long refusal" not in llm.calls[0][1]
