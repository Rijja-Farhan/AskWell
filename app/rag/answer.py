"""Answer a question: retrieve top-k chunks, then ask the LLM to answer from them.

Two modes (`settings.answer_mode`). `baseline` is the Phase 3 minimal prompt, kept so older eval
runs resume and so grounding has a measured baseline. `grounded` numbers the sources `[1]..[k]`,
requires a `[n]` marker after each claim and a fixed refusal reply when the sources do not
answer, and refuses without an LLM call when retrieval found nothing or the top reranker score
is below `refusal_min_score`. The `[n]` markers and `Citation` fields are a contract for the API
and UI phases: keep them stable.
"""

import re
from typing import Literal

from pydantic import BaseModel

from app.config import Settings
from app.rag.embeddings import Embedder
from app.rag.filters import RetrievalFilter
from app.rag.llm import LLM
from app.rag.rerank import Reranker
from app.rag.retrieve import RetrievedChunk, retrieve

ANSWER_VERSION = 1  # bump when the grounded prompt or citation handling changes

REFUSAL_TEXT = (
    "I don't know. The company documents I can access do not answer this question. "
    "Please check with the relevant department."
)

SYSTEM_PROMPT = (
    "You are Askwell, the internal knowledge assistant for Nimbus Logistics. "
    "Answer the employee's question using the provided context."
)

GROUNDED_SYSTEM_PROMPT = f"""\
You are Askwell, the internal knowledge assistant for Nimbus Logistics.

Answer the employee's question using only the numbered sources provided in the user message. \
Do not use outside knowledge and do not guess.

Rules:
- Put a citation marker such as [1] right after every sentence or claim that comes from a source. \
Use the number of the source that supports it. If several sources support a claim, write one \
marker after another, like [1][2]. Never cite a number that is not in the source list.
- Be concise and state facts exactly as the sources do (numbers, dates, names, codes).
- If sources conflict, prefer the one with the later effective date and the active status, and \
say which version you used.
- If the sources do not contain the answer, or only touch on the topic without answering the \
question, reply with exactly this text and nothing else: {REFUSAL_TEXT}"""

MARKER = re.compile(r"\[(\d+)\]")


class Citation(BaseModel):
    number: int
    chunk_id: str
    doc_id: str
    title: str
    section: str


class Answer(BaseModel):
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    model: str
    citations: list[Citation] = []
    refused: bool = False
    refusal_reason: Literal["no_context", "low_score", "model"] | None = None
    invalid_citations: int = 0


def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(f"[{c.doc_id}] {c.metadata.get('title', '')}\n{c.text}" for c in chunks)
    return f"Context:\n\n{context}\n\nQuestion: {question}"


def source_label(number: int, c: RetrievedChunk) -> str:
    m = c.metadata
    return (
        f"[{number}] {m.get('title', '')} | section: {m.get('section', '')} | "
        f"doc_id: {c.doc_id} | version {m.get('version', '')} ({m.get('status', '')}) | "
        f"effective {m.get('effective_date', '')}"
    )


def build_grounded_message(question: str, chunks: list[RetrievedChunk]) -> str:
    sources = "\n\n".join(f"{source_label(i, c)}\n{c.text}" for i, c in enumerate(chunks, 1))
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def is_refusal(text: str) -> bool:
    """True when the reply is `REFUSAL_TEXT` (ignoring surrounding whitespace and quotes)."""
    return text.strip().strip("\"'*` \n") == REFUSAL_TEXT


def parse_citations(text: str, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation], int]:
    """Return the text without markers that point to no source, the valid citations in
    first-cited order, and how many invalid markers were removed."""
    cited: dict[int, Citation] = {}
    invalid = 0

    def check(match: re.Match) -> str:
        nonlocal invalid
        n = int(match.group(1))
        if not 1 <= n <= len(chunks):
            invalid += 1
            return ""
        if n not in cited:
            c = chunks[n - 1]
            cited[n] = Citation(
                number=n,
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                title=str(c.metadata.get("title", "")),
                section=str(c.metadata.get("section", "")),
            )
        return match.group(0)

    cleaned = MARKER.sub(check, text)
    if invalid:  # tidy the gaps left by removed markers
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    return cleaned, list(cited.values()), invalid


def answer_question(
    question: str,
    k: int,
    embedder: Embedder,
    llm: LLM,
    settings: Settings,
    reranker: Reranker | None = None,
    filters: RetrievalFilter | None = None,
) -> Answer:
    chunks = retrieve(question, k, embedder, settings, reranker, filters)
    return answer_from_chunks(question, chunks, llm, settings)


def refusal(question: str, chunks: list[RetrievedChunk], llm: LLM, reason: str) -> Answer:
    return Answer(
        question=question,
        answer=REFUSAL_TEXT,
        chunks=chunks,
        model=llm.model,
        refused=True,
        refusal_reason=reason,  # type: ignore[arg-type]
    )


def answer_from_chunks(
    question: str, chunks: list[RetrievedChunk], llm: LLM, settings: Settings
) -> Answer:
    """Answer from chunks that were already retrieved (the eval runner reuses its retrieval)."""
    if settings.answer_mode == "baseline":
        text = llm.complete(SYSTEM_PROMPT, build_user_message(question, chunks))
        return Answer(question=question, answer=text, chunks=chunks, model=llm.model)

    if not chunks:
        return refusal(question, chunks, llm, "no_context")
    gate = settings.refusal_min_score
    if settings.reranker == "cross-encoder" and gate is not None and chunks[0].score < gate:
        return refusal(question, chunks, llm, "low_score")

    text = llm.complete(GROUNDED_SYSTEM_PROMPT, build_grounded_message(question, chunks))
    if is_refusal(text):
        return refusal(question, chunks, llm, "model")
    cleaned, citations, invalid = parse_citations(text, chunks)
    return Answer(
        question=question,
        answer=cleaned,
        chunks=chunks,
        model=llm.model,
        citations=citations,
        invalid_citations=invalid,
    )
