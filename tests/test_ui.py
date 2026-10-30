"""Web UI and GET /users, offline with fakes."""

import re

import pytest
from fastapi.testclient import TestClient

from app.api.deps import Deps
from app.api.main import UI_DIR, create_app
from tests.conftest import FakeEmbedder, FakeLLM, make_settings

PUBLIC_FIELDS = {"user_id", "name", "role", "department", "clearance", "departments"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    settings = make_settings(tmp_path_factory.mktemp("ui"))
    deps = Deps(settings, embedder=FakeEmbedder(), llm=FakeLLM())
    return TestClient(create_app(settings, deps))


def test_users_shape_and_order(client):
    res = client.get("/users")
    assert res.status_code == 200
    users = res.json()
    assert [u["user_id"] for u in users] == [f"u{i:02d}" for i in range(1, 11)]
    assert all(set(u) == PUBLIC_FIELDS for u in users)
    assert {u["user_id"]: u["clearance"] for u in users}["u06"] == "restricted"


def test_users_with_and_without_header(client):
    base = client.get("/users").json()
    assert client.get("/users", headers={"X-User-Id": "u05"}).json() == base
    assert client.get("/users", headers={"X-User-Id": "nobody"}).json() == base


def test_root_redirects_and_ui_serves_page(client):
    res = client.get("/", follow_redirects=False)
    assert res.status_code in (302, 307) and res.headers["location"] == "/ui/"
    for path in ("/ui/", "/ui/index.html"):
        page = client.get(path)
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert "<title>Askwell</title>" in page.text


@pytest.mark.parametrize(
    "path,kind",
    [("/ui/app.js", "javascript"), ("/ui/style.css", "text/css")],
)
def test_static_assets(client, path, kind):
    res = client.get(path)
    assert res.status_code == 200
    assert kind in res.headers["content-type"]


def test_api_still_reachable_next_to_ui(client):
    assert client.get("/health").status_code == 200
    assert client.get("/me").json()["user_id"]


def test_no_external_hosts():
    for name in ("index.html", "style.css", "app.js"):
        text = (UI_DIR / name).read_text()
        assert not re.search(r"(https?:)?//[a-z0-9.-]+\.[a-z]{2,}", text, re.I), name


REQUIRED_IDS = ["user-select", "citation-panel", "trace-link-template"]


def test_required_hooks_exist():
    html = (UI_DIR / "index.html").read_text()
    js = (UI_DIR / "app.js").read_text()
    for hook in REQUIRED_IDS:
        assert f'id="{hook}"' in html, hook
        assert f'"{hook}"' in js, hook


def test_js_never_uses_innerhtml():
    js = (UI_DIR / "app.js").read_text()
    assert "innerHTML" not in js and "outerHTML" not in js and "insertAdjacentHTML" not in js
