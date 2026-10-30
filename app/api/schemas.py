"""Wire format of the API. `ChatResponse` is the body of `POST /chat` and the `done` event of
`POST /chat/stream`; Phases 11, 12 and 16 build on these fields, so keep them stable."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, SerializeAsAny, field_validator

from app.dataset.schemas import AccessLevel, Department
from app.rag.answer import Answer, Citation
from app.rag.filters import VersionPolicy
from app.rag.memory import Turn
from app.rag.retrieve import RetrievedChunk

MAX_QUESTION_CHARS = 2000


class ChatRequest(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_CHARS)
    k: int | None = Field(default=None, ge=1, le=20)  # None means settings.retrieval_top_k
    departments: list[Department] | None = None
    as_of: date | None = None
    version_policy: VersionPolicy | None = None
    include_text: bool = False  # add each source's chunk text
    # Client-chosen conversation label (not an identity). Omitted: stateless, no rewrite.
    session_id: str | None = Field(
        default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"
    )

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value


class UserPublic(BaseModel):
    """What `GET /users` shows per demo user: identity and access scope, never documents."""

    user_id: str
    name: str
    role: str
    department: Department
    clearance: AccessLevel
    departments: list[Department]


class Source(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    title: str
    section: str
    score: float
    version: str
    status: str
    effective_date: str


class SourceWithText(Source):
    text: str


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: Literal["no_context", "low_score", "model", "blocked"] | None
    citations: list[Citation]
    invalid_citations: int
    sources: list[SerializeAsAny[Source]]
    model: str
    latency_ms: int
    session_id: str | None = None
    standalone_query: str | None = None  # the query retrieval used, when memory applied
    cached: bool = False  # the answer came from the answer cache (no LLM call)
    trace_id: str | None = None  # the request's trace (null when tracing is off)


class TurnOut(BaseModel):
    question: str
    standalone_query: str
    answer: str
    refused: bool
    doc_ids: list[str]


class SessionResponse(BaseModel):
    session_id: str
    turns: list[TurnOut]


class TokenEvent(BaseModel):
    text: str


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    index_present: bool
    documents: int
    chunks: int
    provider: str
    api_key_present: bool
    answer_mode: str
    retrieval_mode: str
    reranker: str
    problems: list[str]


def to_source(chunk: RetrievedChunk, include_text: bool) -> Source:
    m = chunk.metadata
    fields = dict(
        rank=chunk.rank,
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        title=str(m.get("title", "")),
        section=str(m.get("section", "")),
        score=chunk.score,
        version=str(m.get("version", "")),
        status=str(m.get("status", "")),
        effective_date=str(m.get("effective_date", "")),
    )
    return SourceWithText(**fields, text=chunk.text) if include_text else Source(**fields)


def to_turn(turn: Turn) -> TurnOut:
    return TurnOut(
        question=turn.question,
        standalone_query=turn.standalone_query,
        answer=turn.answer,
        refused=turn.refused,
        doc_ids=list(turn.doc_ids),
    )


def to_response(
    answer: Answer,
    include_text: bool,
    latency_ms: int,
    session_id: str | None = None,
    standalone_query: str | None = None,
    cached: bool = False,
    trace_id: str | None = None,
) -> ChatResponse:
    return ChatResponse(
        answer=answer.answer,
        refused=answer.refused,
        refusal_reason=answer.refusal_reason,
        citations=answer.citations,
        invalid_citations=answer.invalid_citations,
        sources=[to_source(c, include_text) for c in answer.chunks],
        model=answer.model,
        latency_ms=latency_ms,
        session_id=session_id,
        standalone_query=standalone_query,
        cached=cached,
        trace_id=trace_id,
    )
