"""Shared "retrieve with memory" logic, used by both /chat and /chat/stream so they cannot drift.

Without a session (or with `memory_enabled=false`) nothing is read, rewritten or stored. Only
the retrieval query changes; the answer step gets the original question and the retrieved
context, never the history. Log ids and counts only, never question or history text.
"""

import logging

from app.config import Settings
from app.observability import span
from app.rag.answer import Answer
from app.rag.llm import LLM
from app.rag.memory import SessionStore, Turn
from app.rag.rewrite import rewrite_query

log = logging.getLogger("askwell.conversation")


def memory_active(session_id: str | None, settings: Settings) -> bool:
    return bool(session_id) and settings.memory_enabled


def standalone_query(
    question: str, session_id: str | None, store: SessionStore, llm: LLM, settings: Settings
) -> str | None:
    """The query to retrieve with when memory applies, else None (stateless: use the question)."""
    if not memory_active(session_id, settings):
        return None
    history = store.get(session_id)  # type: ignore[arg-type]
    with span("rewrite", used=bool(history), history_turns=len(history)) as traced:
        result = rewrite_query(question, history, llm, settings)
        traced.set(rewritten=result.rewritten)
    log.info("session turns=%d rewritten=%s", len(history), result.rewritten)
    return result.query


def record_turn(
    store: SessionStore, session_id: str | None, question: str, query: str | None, answer: Answer
) -> None:
    """Store a successfully answered turn (refusals included, flagged)."""
    if session_id is None or query is None:
        return
    doc_ids = tuple(dict.fromkeys(c.doc_id for c in answer.citations))
    store.append(
        session_id,
        Turn(
            question=question,
            standalone_query=query,
            answer=answer.answer,
            refused=answer.refused,
            doc_ids=doc_ids,
        ),
    )
