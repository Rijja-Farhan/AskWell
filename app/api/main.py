"""FastAPI app: chat (JSON and SSE), ingestion and health.

`create_app(settings, deps)` builds the app; tests pass fakes through `deps`. Nothing heavy is
created at import or build time: the embedder, LLM and reranker are built on first use (the
lifespan warms them when it can, and only logs if it cannot, so the server still starts
without an API key and /health/ready explains what is missing).

All endpoints are plain `def`, so Starlette runs them in its thread pool and blocking work
(embedding, Chroma, reranking, LLM calls) never blocks the event loop. The SSE endpoint hands
Starlette a sync generator, which it iterates in the thread pool as well.
"""

import hashlib
import json
import logging
import re
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager, nullcontext
from pathlib import Path
from typing import Annotated, Any

import openai
from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.deps import ApiError, Deps, current_user, is_admin
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ReadyResponse,
    SessionResponse,
    UserPublic,
    to_response,
    to_source,
    to_turn,
)
from app.config import Settings, get_settings
from app.dataset.schemas import UserProfile
from app.observability import (
    ActiveTrace,
    Trace,
    Tracer,
    current_trace_id,
    span,
)
from app.observability.trace import SpanError
from app.observability.view import render_html
from app.rag.answer import Answer, empty_question_answer, stream_answer
from app.rag.cache import (
    answer_key,
    cached_answer,
    cached_retrieve,
    lookup_answer,
    store_answer,
)
from app.rag.conversation import memory_active, record_turn, standalone_query
from app.rag.errors import CorpusParseError, IndexNotReadyError, MissingAPIKeyError
from app.rag.filters import RetrievalFilter
from app.rag.ingest import ingest_corpus
from app.rag.llm import traced as trace_llm
from app.rag.sanitize import prepare_question
from app.rag.store import REPO_ROOT, SyncStats, open_collection

log = logging.getLogger("askwell.api")

UI_DIR = Path(__file__).resolve().parents[1] / "ui"

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


def error_response(exc: BaseException, extra_headers: dict[str, str] | None = None) -> JSONResponse:
    status, code, message = classify(exc)
    log.warning("request failed: %s -> %s %s", type(exc).__name__, status, code)
    extra = {"errors": exc.errors} if isinstance(exc, CorpusParseError) else {}
    headers = dict(extra_headers or {})
    if isinstance(exc, UNAVAILABLE):
        headers["Retry-After"] = retry_after(exc)
    return JSONResponse(
        error_body(code, message, **extra), status_code=status, headers=headers or None
    )


def trace_ref() -> str:
    return current_trace_id() or "-"


def trace_header(request: Request) -> dict[str, str]:
    """`X-Trace-Id` for an error response, when the failed request had a trace."""
    trace_id = getattr(request.state, "trace_id", None)
    return {"X-Trace-Id": trace_id} if trace_id else {}


def trace_error_code(exc: BaseException) -> str:
    if isinstance(exc, ApiError):
        return exc.code
    if isinstance(exc, StarletteHTTPException):
        return f"http_{exc.status_code}"
    return classify(exc)[1]


def question_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


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


REPLAY_CHARS = 24


def replay(answer: Answer) -> Iterator[str | Answer]:
    """A cached answer as the stream would have produced it: text pieces, then the Answer.
    Refusals stream no text (as in `stream_answer`)."""
    if not answer.refused:
        for i in range(0, len(answer.answer), REPLAY_CHARS):
            yield answer.answer[i : i + REPLAY_CHARS]
    yield answer


def generate_attrs(answer: Answer, cached: bool) -> dict[str, Any]:
    return {
        "cache_hit": cached,
        "refused": answer.refused,
        "refusal_reason": answer.refusal_reason,
        "citations": len(answer.citations),
        "invalid_citations": answer.invalid_citations,
    }


def bind_trace(handle: ActiveTrace | None, stream: Iterator[str]) -> Iterator[str]:
    """Iterate `stream` with the trace bound to the current thread's context at every step:
    Starlette resumes a sync generator in whichever pool thread is free, in a context that
    never saw the endpoint's binding."""
    if handle is None:
        yield from stream
        return
    try:
        while True:
            handle.bind()
            try:
                item = next(stream)
            except StopIteration:
                return
            yield item
    finally:
        stream.close()


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
    tracer = Tracer(enabled=settings.trace_enabled, classify=trace_error_code)
    traces = deps.get_traces()

    def finish_trace(handle: ActiveTrace | None, error: BaseException | SpanError | None = None):
        """Close a request's trace and store it. Storing never fails the request."""
        if handle is None:
            return None
        trace = handle.finish(error)
        try:
            traces.add(trace)
        except Exception as exc:  # noqa: BLE001
            log.warning("trace store failed (%s)", type(exc).__name__)
        return trace

    @contextmanager
    def request_trace(
        endpoint: str,
        user: UserProfile,
        session_id: str | None,
        request: Request,
        response: Response | None = None,
    ):
        """Trace one request for the length of the `with` block (None when tracing is off)."""
        handle = tracer.start(endpoint, user.user_id, session_id)
        if handle is None:
            yield None
            return
        request.state.trace_id = handle.trace_id
        if response is not None:
            response.headers["X-Trace-Id"] = handle.trace_id
        failure: BaseException | None = None
        try:
            with handle.activate():
                yield handle
        except BaseException as exc:
            failure = exc
            raise
        finally:
            finish_trace(handle, failure)

    def summarize_answer(handle: ActiveTrace | None, answer: Answer, cached: bool) -> None:
        if handle is None:
            return
        handle.summary(
            cached=cached,
            refused=answer.refused,
            refusal_reason=answer.refusal_reason,
            blocked=answer.refusal_reason == "blocked",
        )
        if settings.trace_store_text:
            handle.put_text("answer_text", answer.answer)

    def sanitized(handle: ActiveTrace | None, question: str):
        with span("sanitize") as s:
            clean = prepare_question(question, settings)  # every later step sees only this
            s.set(removed=clean.removed, empty=not clean.text)
        if handle is not None:
            handle.summary(question_chars=len(clean.text), question_hash=question_hash(clean.text))
        return clean

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
        return JSONResponse(
            error_body(exc.code, exc.message),
            status_code=exc.status,
            headers=trace_header(request) or None,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            error_body(f"http_{exc.status_code}", str(exc.detail)),
            status_code=exc.status_code,
            headers=trace_header(request) or None,
        )

    for exc_type in (
        MissingAPIKeyError,
        IndexNotReadyError,
        openai.APIConnectionError,
        openai.APIStatusError,
        IngestBusyError,
        CorpusParseError,
    ):
        app.add_exception_handler(
            exc_type, lambda request, exc: error_response(exc, trace_header(request))
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.error("unhandled %s", type(exc).__name__)
        return error_response(exc, trace_header(request))

    # Health
    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "cache": deps.get_caches().stats(),
            "traces": traces.stats(),
        }

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

    @app.get("/users", response_model=list[UserPublic])
    def list_users() -> list[UserPublic]:
        """The demo users for the UI's switcher. Public fields only; no header needed."""
        users = deps.get_users()
        return [UserPublic(**users[k].model_dump()) for k in sorted(users)]

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, user: CurrentUser, request: Request, response: Response):
        start = time.perf_counter()
        sessions = deps.sessions_for(user)
        session_id = session_for(req)
        with request_trace("/chat", user, session_id, request, response) as handle:
            trace_id = handle.trace_id if handle else None
            clean = sanitized(handle, req.question)
            if settings.injection_defense and not clean.text:  # nothing but an attack
                log.info(
                    "chat user=%s pure-attack removed=%d trace=%s",
                    user.user_id,
                    clean.removed,
                    trace_ref(),
                )
                answer = empty_question_answer(clean.text, settings)
                summarize_answer(handle, answer, False)
                if handle:
                    handle.summary(blocked=True)
                return to_response(
                    answer, req.include_text, elapsed_ms(start), session_id, None, trace_id=trace_id
                )
            embedder, reranker = deps.get_embedder(), deps.get_reranker()
            llm = trace_llm(deps.get_llm())
            caches = deps.get_caches()
            with deps.gate.read():
                query = standalone_query(clean.text, session_id, sessions, llm, settings)
                retrieval = cached_retrieve(
                    query or clean.text,
                    req.k or settings.retrieval_top_k,
                    embedder,
                    settings,
                    caches,
                    reranker,
                    access_filter(req, user),
                    user,
                )
                chunks = retrieval.chunks
                # the answer step sees the original question, never the history
                with span("generate", answer_mode=settings.answer_mode, llm_calls=0) as gen:
                    answer, cached = cached_answer(clean.text, retrieval, llm, settings, caches)
                    gen.set(**generate_attrs(answer, cached))
            summarize_answer(handle, answer, cached)
            record_turn(sessions, session_id, clean.text, query, answer)
            log.info(
                "chat user=%s chunks=%d refused=%s removed=%d trace=%s",
                user.user_id,
                len(chunks),
                answer.refused,
                clean.removed,
                trace_ref(),
            )
            return to_response(
                answer, req.include_text, elapsed_ms(start), session_id, query, cached, trace_id
            )

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest, user: CurrentUser, request: Request):
        start = time.perf_counter()
        sessions = deps.sessions_for(user)
        session_id = session_for(req)
        # The trace starts here, before the response object exists, and is finished by the
        # generator's `finally` (or here, when the request fails before streaming).
        handle = tracer.start("/chat/stream", user.user_id, session_id)
        trace_id = handle.trace_id if handle else None
        trace_headers = {"X-Trace-Id": trace_id} if trace_id else {}
        if trace_id:
            request.state.trace_id = trace_id
        lease = None
        try:
            with handle.activate() if handle else nullcontext():
                clean = sanitized(handle, req.question)  # same input as /chat
                if settings.injection_defense and not clean.text:  # nothing but an attack
                    log.info(
                        "chat user=%s pure-attack removed=%d trace=%s",
                        user.user_id,
                        clean.removed,
                        trace_ref(),
                    )
                    refused = empty_question_answer(clean.text, settings)
                    summarize_answer(handle, refused, False)
                    if handle:
                        handle.summary(blocked=True)
                    finish_trace(handle)
                    return StreamingResponse(
                        iter(
                            [
                                sse("sources", {"sources": [], "standalone_query": None}),
                                sse(
                                    "done",
                                    to_response(
                                        refused,
                                        req.include_text,
                                        elapsed_ms(start),
                                        session_id,
                                        None,
                                        trace_id=trace_id,
                                    ),
                                ),
                            ]
                        ),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no",
                            **trace_headers,
                        },
                    )
                embedder, reranker = deps.get_embedder(), deps.get_reranker()
                llm = trace_llm(deps.get_llm())
                caches = deps.get_caches()
                lease = deps.gate.acquire_read()
                # retrieval errors are still plain HTTP errors: nothing has been sent yet
                query = standalone_query(clean.text, session_id, sessions, llm, settings)
                retrieval = cached_retrieve(
                    query or clean.text,
                    req.k or settings.retrieval_top_k,
                    embedder,
                    settings,
                    caches,
                    reranker,
                    access_filter(req, user),
                    user,
                )
                chunks = retrieval.chunks
                key = (
                    answer_key(retrieval.key, clean.text, chunks, settings)
                    if caches.enabled
                    else ""
                )
                hit = lookup_answer(caches, key)
        except BaseException as exc:
            if lease is not None:
                lease.release()
            finish_trace(handle, exc)
            raise

        def events() -> Iterator[str]:
            failure: BaseException | SpanError | None = None
            completed = False
            first_token = True
            try:
                yield sse(
                    "sources",
                    {
                        "sources": [to_source(c, req.include_text).model_dump() for c in chunks],
                        "standalone_query": query,
                    },
                )
                with span("generate", answer_mode=settings.answer_mode, llm_calls=0) as gen:
                    items = (
                        replay(hit)
                        if hit is not None
                        else stream_answer(clean.text, chunks, llm, settings)
                    )
                    for item in items:
                        if isinstance(item, Answer):
                            if hit is None:
                                store_answer(caches, key, item)
                            gen.set(**generate_attrs(item, hit is not None))
                            summarize_answer(handle, item, hit is not None)
                            done = sse(
                                "done",
                                to_response(
                                    item,
                                    req.include_text,
                                    elapsed_ms(start),
                                    session_id,
                                    query,
                                    hit is not None,
                                    trace_id,
                                ),
                            )
                            record_turn(sessions, session_id, clean.text, query, item)
                            completed = True
                            yield done
                        else:
                            if first_token:
                                first_token = False
                                gen.set(ttft_ms=round(gen.elapsed_ms(), 3))
                                if handle:
                                    handle.summary(ttft_ms=round(handle.now_ms(), 3))
                            yield sse("token", {"text": item})
            except Exception as exc:  # noqa: BLE001 - the response has started: emit an event
                failure = exc
                _, code, message = classify(exc)
                log.warning("stream failed: %s -> %s trace=%s", type(exc).__name__, code, trace_id)
                yield sse("error", {"code": code, "message": message})
            finally:
                lease.release()
                if not completed and failure is None:  # closed before the done event
                    failure = SpanError(type="ClientDisconnect", code="client_closed")
                finish_trace(handle, failure)

        return StreamingResponse(
            bind_trace(handle, events()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **trace_headers},
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
    def ingest(
        user: CurrentUser,
        request: Request,
        response: Response,
        body: IngestRequest | None = None,
    ):
        with request_trace("/ingest", user, None, request, response):
            # Admin only: an ingest changes what everyone can see.
            if settings.rbac_enabled and not is_admin(user):
                log.warning("ingest denied user=%s", user.user_id)
                raise ApiError(403, "forbidden", "Only an administrator may run an ingest.")
            if not deps.gate.begin_ingest():
                raise IngestBusyError
            try:
                full = body.full if body else False
                with span("ingest", full=full) as s:
                    stats = ingest_corpus(REPO_ROOT, deps.get_embedder, settings, full=full)
                    s.set(
                        added=len(stats.added),
                        updated=len(stats.updated),
                        removed=len(stats.removed),
                        unchanged=len(stats.unchanged),
                        chunks_embedded=stats.chunks_embedded,
                        full_rebuild=stats.full_rebuild,
                    )
                deps.get_caches().clear()  # belt and braces: the index fingerprint already changed
                return stats
            finally:
                deps.gate.end_ingest()

    # Traces: they describe other users' requests, so administrators only
    def require_trace_access(user: UserProfile) -> None:
        if settings.rbac_enabled and not is_admin(user):
            log.warning("trace access denied user=%s", user.user_id)
            raise ApiError(403, "forbidden", "Only an administrator may read traces.")

    def trace_or_404(trace_id: str) -> Trace:
        trace = traces.get(trace_id)
        if trace is None:
            raise StarletteHTTPException(404, "Trace not found.")
        return trace

    @app.get("/traces")
    def list_traces(
        user: CurrentUser,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        status: str | None = None,
        user_id: str | None = None,
    ) -> list[dict]:
        require_trace_access(user)
        filters = {"status": status, "user_id": user_id}
        return [t.model_dump(mode="json") for t in traces.list(limit, filters)]

    @app.get("/traces/{trace_id}")
    def get_trace(trace_id: str, user: CurrentUser) -> dict:
        require_trace_access(user)
        return trace_or_404(trace_id).model_dump(mode="json")

    @app.get("/traces/{trace_id}/view", response_class=HTMLResponse)
    def view_trace(trace_id: str, user: CurrentUser) -> HTMLResponse:
        require_trace_access(user)
        return HTMLResponse(render_html(trace_or_404(trace_id)))

    # Web UI: plain static files from the same origin (no CORS, no build step).
    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

    return app


app = create_app()
