"""Rewrite a follow-up question into a standalone search query.

Retrieval sees the rewritten query; the answer step still sees the user's original question.
A rewrite failure must never fail the request: any LLM error or empty output falls back to the
original question.

There is deliberately no heuristic "is it already standalone?" shortcut: a cheap check (for
example "no pronouns") cannot tell "And for contractors?" from a new question without false
negatives, and a wrong skip silently breaks retrieval. With history, the LLM decides.
"""

import logging
import re
from dataclasses import dataclass

from app.config import Settings
from app.rag.llm import LLM
from app.rag.memory import Turn

log = logging.getLogger("askwell.rewrite")

MAX_QUERY_CHARS = 300
MAX_HISTORY_ANSWER_CHARS = 300

REWRITE_SYSTEM_PROMPT = """\
You rewrite a follow-up question from an employee into one standalone search query for a \
company document search.

Rules:
- Use the conversation only to resolve pronouns, ellipsis and implicit subjects \
("it", "that policy", "and for contractors?").
- If the new question is already self-contained or changes topic, return it unchanged. Do not \
add details from the conversation that the new question does not depend on.
- Keep exact terms (policy names, ids, error codes, numbers, site codes) verbatim.
- Output only the query, on one line, with no quotes, labels or explanation."""


@dataclass(frozen=True)
class RewriteResult:
    query: str
    rewritten: bool


def format_history(history: list[Turn]) -> str:
    lines = []
    for turn in history:
        answer = "(no answer was found)" if turn.refused else turn.answer
        lines.append(f"User: {turn.question}\nAssistant: {answer[:MAX_HISTORY_ANSWER_CHARS]}")
    return "\n".join(lines)


def build_rewrite_message(question: str, history: list[Turn]) -> str:
    return f"Conversation so far:\n{format_history(history)}\n\nNew question: {question}"


def clean_query(text: str) -> str:
    """One line, no wrapping quotes or label, capped in length."""
    line = re.sub(r"\s+", " ", text).strip().strip("\"'`* ")
    line = re.sub(r"^(standalone )?(search )?query\s*:\s*", "", line, flags=re.IGNORECASE)
    line = line.strip("\"'`* ")
    return line[:MAX_QUERY_CHARS].strip()


def rewrite_query(
    question: str, history: list[Turn], llm: LLM, settings: Settings
) -> RewriteResult:
    if not history:
        return RewriteResult(question, False)
    recent = history[-max(1, settings.memory_rewrite_turns) :]
    try:
        raw = llm.complete(REWRITE_SYSTEM_PROMPT, build_rewrite_message(question, recent))
    except Exception as exc:  # noqa: BLE001 - never fail the request over a rewrite
        log.warning("rewrite failed (%s), using the original question", type(exc).__name__)
        return RewriteResult(question, False)
    query = clean_query(raw or "")
    if not query:
        log.warning("rewrite returned nothing, using the original question")
        return RewriteResult(question, False)
    return RewriteResult(query, query != question.strip())
