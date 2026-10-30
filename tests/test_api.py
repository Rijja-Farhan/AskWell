"""HTTP API with fakes: no network, no API key, no reranker model."""

import json
import threading

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api import main as api_main
from app.api.deps import Deps
from app.api.main import create_app
from app.rag import rerank
from app.rag.answer import REFUSAL_TEXT
from app.rag.errors import CorpusParseError
from app.rag.store import build_index
from tests.conftest import FakeEmbedder, FakeLLM, make_settings

QUESTION = "How many days of annual leave do new employees get?"


class FakeReranker:
    model = "fake-reranker"

    def __init__(self, score: float = 5.0) -> None:
        self.value = score

    def score(self, question: str, texts: list[str]) -> list[float]:
        return [self.value] * len(texts)


@pytest.fixture(scope="module")
def index_settings(corpus, tmp_path_factory):
    settings = make_settings(
        tmp_path_factory.mktemp("api_index"), gemini_api_key=SecretStr("sk-test-value")
    )
    build_index(corpus, FakeEmbedder(), settings)
    return settings


def make_client(settings, reply="Twenty days [1].", llm=None, reranker=None, **kw):
    llm = llm or FakeLLM(reply)
    deps = Deps(settings, embedder=FakeEmbedder(), llm=llm, reranker=reranker, **kw)
    return TestClient(create_app(settings, deps)), llm


@pytest.fixture
def client(index_settings):
    return make_client(index_settings)[0]


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        assert lines[0].startswith("event: ") and lines[1].startswith("data: ")
        events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


def provider_error(cls, status, headers=None):
    request = httpx.Request("POST", "https://example.invalid")
    response = httpx.Response(status, request=request, headers=headers)
    return cls("provider said secret-text", response=response, body=None)


# --- health -------------------------------------------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_ready_200(index_settings):
    client, _ = make_client(index_settings)
    r = client.get("/health/ready")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ready" and body["problems"] == []
    assert body["documents"] == 52 and body["chunks"] > 52 and body["index_present"]
    assert body["provider"] == "gemini" and body["api_key_present"] is True
    assert body["answer_mode"] == "grounded" and body["retrieval_mode"] == "hybrid"
    assert body["reranker"] == "none"
    assert "sk-test-value" not in r.text


def test_ready_503_without_index_or_key(tmp_path):
    client, _ = make_client(make_settings(tmp_path, gemini_api_key=None))
    r = client.get("/health/ready")
    body = r.json()
    assert r.status_code == 503 and body["status"] == "not_ready"
    assert body["api_key_present"] is False and body["index_present"] is False
    assert len(body["problems"]) == 2 and body["chunks"] == 0


# --- chat ---------------------------------------------------------------------------------


def test_chat_answer_with_citations(index_settings):
    client, llm = make_client(index_settings, "Twenty days [1]. Also [2].")
    r = client.post("/chat", json={"question": QUESTION, "k": 3})
    body = r.json()
    assert r.status_code == 200 and not body["refused"] and body["refusal_reason"] is None
    assert body["answer"] == "Twenty days [1]. Also [2]."
    assert [c["number"] for c in body["citations"]] == [1, 2]
    assert set(body["citations"][0]) == {"number", "chunk_id", "doc_id", "title", "section"}
    assert body["invalid_citations"] == 0 and body["model"] == "fake-llm"
    assert body["latency_ms"] >= 0 and len(body["sources"]) == 3 and len(llm.calls) == 1
    assert body["citations"][0]["chunk_id"] == body["sources"][0]["chunk_id"]
    source = body["sources"][0]
    assert set(source) == {
        "rank", "chunk_id", "doc_id", "title", "section", "score", "version", "status",
        "effective_date",
    }  # fmt: skip


def test_chat_include_text(client):
    r = client.post("/chat", json={"question": QUESTION, "k": 2, "include_text": True})
    assert all(s["text"] for s in r.json()["sources"])


def test_chat_default_k_and_invalid_marker(index_settings):
    client, _ = make_client(index_settings, "Fact [1]. Bogus [42].")
    body = client.post("/chat", json={"question": QUESTION}).json()
    assert len(body["sources"]) == index_settings.retrieval_top_k
    assert body["answer"] == "Fact [1]. Bogus." and body["invalid_citations"] == 1
    assert [c["number"] for c in body["citations"]] == [1]


def test_chat_no_context_refusal_makes_no_llm_call(index_settings):
    client, llm = make_client(index_settings)
    r = client.post("/chat", json={"question": QUESTION, "as_of": "1990-01-01"})
    body = r.json()
    assert body["refused"] and body["refusal_reason"] == "no_context"
    assert body["answer"] == REFUSAL_TEXT and body["sources"] == [] and llm.calls == []


def test_chat_low_score_refusal_makes_no_llm_call(index_settings):
    settings = index_settings.model_copy(update={"reranker": "cross-encoder"})
    client, llm = make_client(settings, reranker=FakeReranker(-9.0))
    body = client.post("/chat", json={"question": QUESTION}).json()
    assert body["refused"] and body["refusal_reason"] == "low_score" and llm.calls == []
    assert body["sources"]  # the weak evidence is still reported


def test_chat_model_refusal(index_settings):
    client, llm = make_client(index_settings, REFUSAL_TEXT)
    body = client.post("/chat", json={"question": QUESTION}).json()
    assert body["refused"] and body["refusal_reason"] == "model" and len(llm.calls) == 1
    assert body["citations"] == []


def test_chat_filters_apply(index_settings):
    client, _ = make_client(index_settings)
    body = client.post("/chat", json={"question": QUESTION, "departments": ["IT"], "k": 10}).json()
    assert body["sources"] and all(s["doc_id"].startswith("it-") for s in body["sources"])
    body = client.post(
        "/chat", json={"question": QUESTION, "version_policy": "active_only", "k": 20}
    ).json()
    assert body["sources"] and all(s["status"] == "active" for s in body["sources"])


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "q", "k": 0},
        {"question": "q", "k": 21},
        {"question": "q", "departments": ["Legal"]},
        {"question": "q", "as_of": "not-a-date"},
        {"question": "q", "version_policy": "newest"},
    ],
)
def test_validation_errors(client, payload):
    for path in ("/chat", "/chat/stream"):
        r = client.post(path, json=payload)
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "validation_error"
        assert set(r.json()) == {"error"}


def test_unknown_route_uses_error_body(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "http_404"


# --- streaming ----------------------------------------------------------------------------


def test_stream_event_order_and_done_matches_chat(index_settings):
    reply = "Twenty days [1]. Also [2]."
    client, llm = make_client(index_settings, reply)
    r = client.post("/chat/stream", json={"question": QUESTION, "k": 3})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [name for name, _ in events]
    assert names[0] == "sources" and names[-1] == "done"
    assert set(names[1:-1]) == {"token"} and len(names) > 3
    assert "".join(d["text"] for n, d in events if n == "token") == reply
    sources, done = events[0][1], events[-1][1]
    assert len(sources["sources"]) == 3 and done["sources"] == sources["sources"]
    plain = client.post("/chat", json={"question": QUESTION, "k": 3}).json()
    assert set(done) == set(plain)
    assert {k: v for k, v in done.items() if k != "latency_ms"} == {
        k: v for k, v in plain.items() if k != "latency_ms"
    }
    assert len(llm.calls) == 2


def test_stream_done_removes_invalid_markers(index_settings):
    client, _ = make_client(index_settings, "Fact [1]. Bogus [42].")
    events = parse_sse(client.post("/chat/stream", json={"question": QUESTION}).text)
    done = events[-1][1]
    streamed = "".join(d["text"] for n, d in events if n == "token")
    assert "[42]" in streamed and "[42]" not in done["answer"]
    assert done["invalid_citations"] == 1


@pytest.mark.parametrize(
    "payload, reason, rerank_score",
    [
        ({"question": QUESTION, "as_of": "1990-01-01"}, "no_context", None),
        ({"question": QUESTION}, "low_score", -9.0),
    ],
)
def test_stream_refusal_before_llm(index_settings, payload, reason, rerank_score):
    settings = index_settings
    kw = {}
    if rerank_score is not None:
        settings = settings.model_copy(update={"reranker": "cross-encoder"})
        kw["reranker"] = FakeReranker(rerank_score)
    client, llm = make_client(settings, **kw)
    events = parse_sse(client.post("/chat/stream", json=payload).text)
    assert [n for n, _ in events] == ["sources", "done"]
    assert events[-1][1]["refused"] and events[-1][1]["refusal_reason"] == reason
    assert events[-1][1]["answer"] == REFUSAL_TEXT and llm.calls == []


def test_stream_model_refusal_has_no_tokens(index_settings):
    client, _ = make_client(index_settings, REFUSAL_TEXT)
    events = parse_sse(client.post("/chat/stream", json={"question": QUESTION}).text)
    assert [n for n, _ in events] == ["sources", "done"]
    assert events[-1][1]["refused"] and events[-1][1]["refusal_reason"] == "model"


class FailingLLM(FakeLLM):
    def stream(self, system, user, size=4):
        yield "partial "
        raise RuntimeError("boom sk-secret-key traceback")


def test_stream_error_event_mid_stream(index_settings):
    client, _ = make_client(index_settings, llm=FailingLLM())
    r = client.post("/chat/stream", json={"question": QUESTION})
    assert r.status_code == 200
    events = parse_sse(r.text)
    assert [n for n, _ in events] == ["sources", "token", "error"]
    assert events[-1][1] == {"code": "internal_error", "message": "Internal server error."}
    assert "sk-secret" not in r.text and "Traceback" not in r.text


class RateLimitedLLM(FakeLLM):
    def stream(self, system, user, size=4):
        raise provider_error(openai.RateLimitError, 429)
        yield  # pragma: no cover


def test_stream_provider_failure_becomes_error_event(index_settings):
    client, _ = make_client(index_settings, llm=RateLimitedLLM())
    events = parse_sse(client.post("/chat/stream", json={"question": QUESTION}).text)
    assert events[-1][0] == "error" and events[-1][1]["code"] == "provider_unavailable"
    assert "secret-text" not in json.dumps(events)


# --- error mapping ------------------------------------------------------------------------


class RaisingLLM(FakeLLM):
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def complete(self, system, user):
        raise self.exc


def test_rate_limit_maps_to_503_with_retry_after(index_settings):
    exc = provider_error(openai.RateLimitError, 429, {"retry-after": "7"})
    client, _ = make_client(index_settings, llm=RaisingLLM(exc))
    r = client.post("/chat", json={"question": QUESTION})
    assert r.status_code == 503 and r.headers["retry-after"] == "7"
    assert r.json()["error"]["code"] == "provider_unavailable"
    assert "secret-text" not in r.text


def test_connection_error_maps_to_503_default_retry_after(index_settings):
    exc = openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
    client, _ = make_client(index_settings, llm=RaisingLLM(exc))
    r = client.post("/chat", json={"question": QUESTION})
    assert r.status_code == 503 and r.headers["retry-after"] == "30"


def test_other_provider_status_maps_to_502(index_settings):
    exc = provider_error(openai.AuthenticationError, 401)
    client, _ = make_client(index_settings, llm=RaisingLLM(exc))
    r = client.post("/chat", json={"question": QUESTION})
    assert r.status_code == 502 and r.json()["error"]["code"] == "provider_error"
    assert "retry-after" not in r.headers


def test_missing_key_maps_to_503(index_settings):
    settings = index_settings.model_copy(update={"gemini_api_key": None})
    client = TestClient(create_app(settings, Deps(settings)))
    for path in ("/chat", "/chat/stream"):
        r = client.post(path, json={"question": QUESTION})
        assert r.status_code == 503 and r.json()["error"]["code"] == "missing_api_key"
        assert "GEMINI_API_KEY" in r.json()["error"]["message"]


def test_index_not_ready_maps_to_503(tmp_path):
    client, _ = make_client(make_settings(tmp_path))
    r = client.post("/chat", json={"question": QUESTION})
    assert r.status_code == 503 and r.json()["error"]["code"] == "index_not_ready"
    r = client.post("/chat/stream", json={"question": QUESTION})
    assert r.status_code == 503 and r.json()["error"]["code"] == "index_not_ready"


# --- ingest -------------------------------------------------------------------------------


def test_ingest_success_and_idempotent(tmp_path):
    settings = make_settings(tmp_path)
    client, _ = make_client(settings)
    r = client.post("/ingest")
    body = r.json()
    assert r.status_code == 200 and body["full_rebuild"] and len(body["added"]) == 52
    assert body["documents"] == 52 and body["chunks"] == body["chunks_embedded"]
    body = client.post("/ingest", json={"full": False}).json()
    assert (
        not body["full_rebuild"] and body["chunks_embedded"] == 0 and len(body["unchanged"]) == 52
    )
    assert client.post("/ingest", json={"full": True}).json()["full_rebuild"]
    assert client.post("/chat", json={"question": QUESTION}).status_code == 200


def test_ingest_parse_errors_422(tmp_path, monkeypatch):
    def broken(root, embedder_factory, settings, full=False):
        raise CorpusParseError(["data/raw/hr/x.pdf: cannot parse"])

    monkeypatch.setattr(api_main, "ingest_corpus", broken)
    client, _ = make_client(make_settings(tmp_path))
    r = client.post("/ingest")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "parse_errors"
    assert r.json()["error"]["errors"] == ["data/raw/hr/x.pdf: cannot parse"]
    assert client.get("/health/ready").status_code == 503  # index unchanged (still absent)


def test_ingest_parse_errors_leave_index_unchanged(index_settings, monkeypatch):
    client, _ = make_client(index_settings)
    before = client.get("/health/ready").json()
    monkeypatch.setattr(
        "app.rag.ingest.load_corpus", lambda root, settings: ([], ["a.pdf: broken"])
    )
    assert client.post("/ingest").status_code == 422
    assert client.get("/health/ready").json() == before


def test_concurrent_ingest_gets_409(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def slow(root, embedder_factory, settings, full=False):
        started.set()
        release.wait(5)
        return api_main.SyncStats(
            documents=0, chunks=0, dimension=0, embedding_model="m", collection="c", path="p"
        )

    monkeypatch.setattr(api_main, "ingest_corpus", slow)
    client, _ = make_client(make_settings(tmp_path))
    first: list = []
    thread = threading.Thread(target=lambda: first.append(client.post("/ingest")))
    thread.start()
    try:
        assert started.wait(5)
        r = client.post("/ingest")
        assert r.status_code == 409 and r.json()["error"]["code"] == "ingest_busy"
    finally:
        release.set()
        thread.join(5)
    assert first[0].status_code == 200
    assert client.post("/ingest").status_code == 200  # the lock was released


def test_chat_waits_for_running_ingest(index_settings, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def slow(root, embedder_factory, settings, full=False):
        started.set()
        release.wait(5)
        return api_main.SyncStats(
            documents=0, chunks=0, dimension=0, embedding_model="m", collection="c", path="p"
        )

    monkeypatch.setattr(api_main, "ingest_corpus", slow)
    client, _ = make_client(index_settings)
    results: dict = {}
    t1 = threading.Thread(target=lambda: results.update(ingest=client.post("/ingest")))
    t1.start()
    assert started.wait(5)
    t2 = threading.Thread(
        target=lambda: results.update(chat=client.post("/chat", json={"question": QUESTION}))
    )
    t2.start()
    t2.join(0.3)
    assert t2.is_alive() and "chat" not in results  # serialized behind the ingest
    release.set()
    t1.join(5)
    t2.join(5)
    assert results["chat"].status_code == 200


# --- app construction ---------------------------------------------------------------------


def test_create_app_loads_no_reranker_and_needs_no_key(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("reranker must not load while building the app")

    monkeypatch.setattr(rerank.CrossEncoderReranker, "__init__", boom)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = make_settings(tmp_path, reranker="cross-encoder", gemini_api_key=None)
    app = create_app(settings)
    assert app.title == "Askwell"
    assert TestClient(app).get("/health").status_code == 200  # no lifespan: nothing built
    with TestClient(app) as started:  # lifespan warm-up failures are logged, not fatal
        assert started.get("/health").json() == {"status": "ok"}


def test_module_level_app_exists():
    from app.api.main import app

    assert app.state.deps is not None


def test_cors_disabled_by_default_and_enabled_by_setting(tmp_path):
    headers = {"Origin": "http://ui.test", "Access-Control-Request-Method": "POST"}
    off, _ = make_client(make_settings(tmp_path))
    assert "access-control-allow-origin" not in off.options("/chat", headers=headers).headers
    on, _ = make_client(make_settings(tmp_path, api_cors_origins="http://ui.test, http://b.test"))
    assert on.options("/chat", headers=headers).headers["access-control-allow-origin"] == (
        "http://ui.test"
    )
