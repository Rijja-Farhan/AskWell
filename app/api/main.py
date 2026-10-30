"""FastAPI app: chat (JSON and SSE), ingestion and health.

`create_app(settings, deps)` builds the app; tests pass fakes through `deps`. Nothing heavy is
created at import or build time: the embedder, LLM and reranker are built on first use (the
lifespan warms them when it can, and only logs if it cannot, so the server still starts
without an API key and /health/ready explains what is missing).

All endpoints are plain `def`, so Starlette runs them in its thread pool and blocking work
(embedding, Chroma, reranking, LLM calls) never blocks the event loop. The SSE endpoint hands
Starlette a sync generator, which it iterates in the thread pool as well.
"""

import json
import logging
import re
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import openai
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.deps import ApiError, Deps, current_user, is_admin
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ReadyResponse,
    SessionResponse,
    to_response,
    to_source,
    to_turn,
)
from app.config import Settings, get_settings
from app.dataset.schemas import UserProfile
from app.rag.answer import Answer, answer_from_chunks, empty_question_answer, stream_answer
from app.rag.conversation import memory_active, record_turn, standalone_query
from app.rag.errors import CorpusParseError, IndexNotReadyError, MissingAPIKeyError
from app.rag.filters import RetrievalFilter
from app.rag.ingest import ingest_corpus
from app.rag.retrieve import retrieve
from app.rag.sanitize import prepare_question
from app.rag.store import REPO_ROOT, SyncStats, open_collection

log = logging.getLogger("askwell.api")

UNAVAILABLE = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)
DEFAULT_RETRY_AFTER = 30
SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
CurrentUser = Annotated[UserProfile, Depends(current_user)]


class IngestBusyError(RuntimeError):
    """Another ingest is already running."""


class IngestRequest(BaseModel):
    full: bool = False


def error_body(code: str, message: str, **extra: Any) -> dict:
    return {"error": {"code": code, "message": message, **extra}}


def classify(exc: BaseException) -> tuple[int, str, str]:
    """Map an exception to (status, code, safe message). Messages never include exception
    text from the provider SDK, which can echo request details."""
    if isinstance(exc, MissingAPIKeyError):
        return 503, "missing_api_key", str(exc)  # names the variable, never the key
    if isinstance(exc, IndexNotReadyError):
        return 503, "index_not_ready", str(exc)
    if isinstance(exc, UNAVAILABLE):
        return 503, "provider_unavailable", "The model provider is unavailable. Try again later."
    if isinstance(exc, openai.APIStatusError):
        return 502, "provider_error", "The model provider returned an error."
    if isinstance(exc, IngestBusyError):
        return 409, "ingest_busy", "An ingest is already running."
    if isinstance(exc, CorpusParseError):
        return 422, "parse_errors", "Some files failed to parse; the index was not changed."
    return 500, "internal_error", "Internal server error."


def retry_after(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    return value if value and value.isdigit() else str(DEFAULT_RETRY_AFTER)


def error_response(exc: BaseException) -> JSONResponse:
    status, code, message = classify(exc)
    log.warning("request failed: %s -> %s %s", type(exc).__name__, status, code)
    extra = {"errors": exc.errors} if isinstance(exc, CorpusParseError) else {}
    headers = {"Retry-After": retry_after(exc)} if isinstance(exc, UNAVAILABLE) else None
    return JSONResponse(error_body(code, message, **extra), status_code=status, headers=headers)


def sse(event: str, data: BaseModel | dict) -> str:
    payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def filters_from(req: ChatRequest, user: UserProfile | None = None) -> RetrievalFilter | None:
    """The request's filters plus the user's access (None when access is not enforced)."""
    if (
        user is None
        and req.departments is None
        and req.as_of is None
        and req.version_policy is None
    ):
        return None
    return RetrievalFilter(
        departments=req.departments, as_of=req.as_of, version_policy=req.version_policy, user=user
    )


def valid_session_id(value: str) -> bool:
    return SESSION_ID.fullmatch(value) is not None


def elapsed_ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def create_app(settings: Settings | None = None, deps: Deps | None = None) -> FastAPI:
    settings = settings or (deps.settings if deps else get_settings())
    deps = deps or Deps(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Warm the heavy objects once so the first request is not slow. Failures (no key, no
        # model download) are not fatal: they surface per request and in /health/ready.
        for build in (deps.get_embedder, deps.get_llm, deps.get_reranker):
            try:
                build()
            except Exception as exc:  # noqa: BLE001
                log.warning("startup: %s failed (%s)", build.__name__, type(exc).__name__)
        yield

    app = FastAPI(title="Askwell", version="0.12.0", lifespan=lifespan)
    app.state.deps = deps

    origins = [o.strip() for o in settings.api_cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["*"],
        )

    # Error mapping: one body shape, {"error": {"code", "message"}}.
    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        details = [
            {"loc": [str(p) for p in e["loc"]], "msg": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        return JSONResponse(
            error_body("validation_error", "Invalid request.", details=details), status_code=422
        )

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return JSONResponse(error_body(exc.code, exc.message), status_code=exc.status)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            error_body(f"http_{exc.status_code}", str(exc.detail)), status_code=exc.status_code
        )

    for exc_type in (
        MissingAPIKeyError,
        IndexNotReadyError,
        openai.APIConnectionError,
        openai.APIStatusError,
        IngestBusyError,
        CorpusParseError,
    ):
        app.add_exception_handler(exc_type, lambda request, exc: error_response(exc))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.error("unhandled %s", type(exc).__name__)
        return error_response(exc)

    # Health
    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/health/ready", response_model=ReadyResponse)
    def ready():
        provider = settings.llm_provider
        key = {"gemini": settings.gemini_api_key, "openai": settings.openai_api_key}.get(provider)
        problems: list[str] = []
        documents = chunks = 0
        index_present = False
        if key is None:
            problems.append(f"{provider.upper()}_API_KEY is not set")
        try:
            collection = open_collection(settings)
            chunks = collection.count()
            metas = collection.get(include=["metadatas"])["metadatas"]
            documents = len({m["doc_id"] for m in metas})
            index_present = True
        except IndexNotReadyError as exc:
            problems.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            log.warning("readiness check failed: %s", type(exc).__name__)
            problems.append(f"cannot open the index ({type(exc).__name__})")
        body = ReadyResponse(
            status="ready" if not problems else "not_ready",
            index_present=index_present,
            documents=documents,
            chunks=chunks,
            provider=provider,
            api_key_present=key is not None,
            answer_mode=settings.answer_mode,
            retrieval_mode=settings.retrieval_mode,
            reranker=settings.rerank_model if settings.reranker != "none" else "none",
            problems=problems,
        )
        return JSONResponse(body.model_dump(), status_code=503 if problems else 200)

    # Chat
    def session_for(req: ChatRequest) -> str | None:
        """The session id when memory applies; a supplied id is ignored if memory is off."""
        return req.session_id if memory_active(req.session_id, settings) else None

    def access_filter(req: ChatRequest, user: UserProfile) -> RetrievalFilter | None:
        return filters_from(req, user if settings.rbac_enabled else None)

    @app.get("/me", response_model=UserProfile)
    def me(user: CurrentUser) -> UserProfile:
        return user

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, user: CurrentUser) -> ChatResponse:
        start = time.perf_counter()
        sessions = deps.sessions_for(user)
        session_id = session_for(req)
        clean = prepare_question(req.question, settings)  # every later step sees only this
        if settings.injection_defense and not clean.text:  # nothing but an attack
            log.info("chat user=%s pure-attack removed=%d", user.user_id, clean.removed)
            answer = empty_question_answer(clean.text, settings)
            return to_response(answer, req.include_text, elapsed_ms(start), session_id, None)
        embedder, llm, reranker = deps.get_embedder(), deps.get_llm(), deps.get_reranker()
        with deps.gate.read():
            query = standalone_query(clean.text, session_id, sessions, llm, settings)
            chunks = retrieve(
                query or clean.text,
                req.k or settings.retrieval_top_k,
                embedder,
                settings,
                reranker,
                access_filter(req, user),
            )
            # the answer step sees the original question, never the history
            answer = answer_from_chunks(clean.text, chunks, llm, settings)
        record_turn(sessions, session_id, clean.text, query, answer)
        log.info(
            "chat user=%s chunks=%d refused=%s removed=%d",
            user.user_id,
            len(chunks),
            answer.refused,
            clean.removed,
        )
        return to_response(answer, req.include_text, elapsed_ms(start), session_id, query)

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest, user: CurrentUser):
        start = time.perf_counter()
        sessions = deps.sessions_for(user)
        session_id = session_for(req)
        clean = prepare_question(req.question, settings)  # same input as /chat
        if settings.injection_defense and not clean.text:  # nothing but an attack
            log.info("chat user=%s pure-attack removed=%d", user.user_id, clean.removed)
            refused = empty_question_answer(clean.text, settings)
            return StreamingResponse(
                iter(
                    [
                        sse("sources", {"sources": [], "standalone_query": None}),
                        sse(
                            "done",
                            to_response(
                                refused, req.include_text, elapsed_ms(start), session_id, None
                            ),
                        ),
                    ]
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        embedder, llm, reranker = deps.get_embedder(), deps.get_llm(), deps.get_reranker()
        lease = deps.gate.acquire_read()
        try:  # retrieval errors are still plain HTTP errors: nothing has been sent yet
            query = standalone_query(clean.text, session_id, sessions, llm, settings)
            chunks = retrieve(
                query or clean.text,
                req.k or settings.retrieval_top_k,
                embedder,
                settings,
                reranker,
                access_filter(req, user),
            )
        except BaseException:
            lease.release()
            raise

        def events() -> Iterator[str]:
            try:
                yield sse(
                    "sources",
                    {
                        "sources": [to_source(c, req.include_text).model_dump() for c in chunks],
                        "standalone_query": query,
                    },
                )
                for item in stream_answer(clean.text, chunks, llm, settings):
                    if isinstance(item, Answer):
                        done = sse(
                            "done",
                            to_response(
                                item, req.include_text, elapsed_ms(start), session_id, query
                            ),
                        )
                        record_turn(sessions, session_id, clean.text, query, item)
                        yield done
                    else:
                        yield sse("token", {"text": item})
            except Exception as exc:  # noqa: BLE001 - the response has started: emit an event
                _, code, message = classify(exc)
                log.warning("stream failed: %s -> %s", type(exc).__name__, code)
                yield sse("error", {"code": code, "message": message})
            finally:
                lease.release()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            background=BackgroundTask(lease.release),  # covers a stream that never started
        )

    # Sessions: owned by the user who created them; another user's id is simply "not found"
    @app.get("/sessions/{session_id}", response_model=SessionResponse)
    def get_session(session_id: str, user: CurrentUser) -> SessionResponse:
        sessions = deps.sessions_for(user)
        if not valid_session_id(session_id) or not sessions.exists(session_id):
            raise StarletteHTTPException(404, "Session not found.")
        return SessionResponse(
            session_id=session_id, turns=[to_turn(t) for t in sessions.get(session_id)]
        )

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str, user: CurrentUser) -> Response:
        sessions = deps.sessions_for(user)
        if not valid_session_id(session_id) or not sessions.exists(session_id):
            raise StarletteHTTPException(404, "Session not found.")
        sessions.clear(session_id)
        return Response(status_code=204)

    # Ingest
    @app.post("/ingest", response_model=SyncStats)
    def ingest(user: CurrentUser, body: IngestRequest | None = None):
        # Admin only: an ingest changes what everyone can see.
        if settings.rbac_enabled and not is_admin(user):
            log.warning("ingest denied user=%s", user.user_id)
            raise ApiError(403, "forbidden", "Only an administrator may run an ingest.")
        if not deps.gate.begin_ingest():
            raise IngestBusyError
        try:
            return ingest_corpus(
                REPO_ROOT, deps.get_embedder, settings, full=body.full if body else False
            )
        finally:
            deps.gate.end_ingest()

    return app


app = create_app()
