"""Static checks of the deployment files (no Docker, no network)."""

import re

import yaml

from app.config import Settings
from tests.conftest import ROOT

NEW_SETTINGS = [
    "CACHE_BACKEND",
    "REDIS_URL",
    "REDIS_KEY_PREFIX",
    "OTLP_ENDPOINT",
    "OTLP_HEADERS",
    "OTLP_SERVICE_NAME",
]


def compose() -> dict:
    return yaml.safe_load((ROOT / "compose.yaml").read_text())


def test_compose_has_the_four_services():
    assert set(compose()["services"]) == {"api", "indexer", "redis", "tracing"}


def test_images_are_pinned_never_latest():
    data = compose()
    for name, service in data["services"].items():
        image = service["image"]
        assert ":" in image and not image.endswith(":latest"), name
    dockerfile = (ROOT / "Dockerfile").read_text()
    base = re.search(r"^FROM (\S+)", dockerfile, re.M).group(1)
    assert base.startswith("python:3.10-slim") and not base.endswith(":latest")


def test_named_volumes_and_mounts():
    data = compose()
    assert {"chroma_db", "model_cache", "traces"} <= set(data["volumes"])
    api = data["services"]["api"]["volumes"]
    assert "chroma_db:/app/chroma_db" in api
    assert "model_cache:/app/.cache/models" in api
    assert "traces:/app/traces" in api
    assert "chroma_db:/app/chroma_db" in data["services"]["indexer"]["volumes"]


def test_api_is_bound_to_loopback_only():
    data = compose()
    for name in ("api", "tracing"):
        for port in data["services"][name]["ports"]:
            assert str(port).startswith("127.0.0.1:"), (name, port)
    assert "127.0.0.1:8000:8000" in data["services"]["api"]["ports"]


def test_startup_order_and_health():
    api = compose()["services"]["api"]
    deps = api["depends_on"]
    assert deps["indexer"]["condition"] == "service_completed_successfully"
    assert deps["redis"]["condition"] == "service_healthy"
    assert deps["tracing"]["condition"] == "service_healthy"
    assert "healthcheck" in compose()["services"]["redis"]
    assert "healthcheck" in compose()["services"]["tracing"]
    assert int(api["healthcheck"]["start_period"].rstrip("s")) >= 120
    assert compose()["services"]["indexer"]["restart"] == "no"


def test_redis_is_a_bounded_cache():
    command = " ".join(compose()["services"]["redis"]["command"])
    assert "allkeys-lru" in command and "maxmemory" in command and "appendonly no" in command


def test_environment_keys_exist_in_settings():
    fields = set(Settings.model_fields)
    for name, service in compose()["services"].items():
        for key in service.get("environment", {}) or {}:
            if name == "tracing":
                continue  # Jaeger's own variables
            assert key.lower() in fields, (name, key)
    env = compose()["services"]["api"]["environment"]
    assert env["CACHE_BACKEND"] == "redis" and env["REDIS_URL"].startswith("redis://redis:")
    assert env["OTLP_ENDPOINT"].startswith("http://tracing:")


def test_env_file_is_optional_and_no_secret_is_in_compose():
    api = compose()["services"]["api"]
    assert api["env_file"] == [{"path": ".env", "required": False}]
    text = (ROOT / "compose.yaml").read_text() + (ROOT / "Dockerfile").read_text()
    assert not re.search(r"(API_KEY|SECRET|PASSWORD)\s*[:=]\s*\S", text)


def test_dockerfile_basics():
    text = (ROOT / "Dockerfile").read_text()
    assert "requirements.txt" in text and "requirements-dev" not in text
    assert re.search(r"^USER \w+", text, re.M) and "USER root" not in text
    assert "API_HOST=0.0.0.0" in text and "EXPOSE 8000" in text
    assert "HEALTHCHECK" in text and "/health" in text
    assert 'CMD ["python", "scripts/serve.py"]' in text
    assert "COPY . " not in text and "COPY .env" not in text


def test_dockerignore_keeps_secrets_and_state_out():
    lines = {
        line.strip() for line in (ROOT / ".dockerignore").read_text().splitlines() if line.strip()
    }
    for entry in (".env", ".git", ".venv", "chroma_db/", ".cache/", "traces/", "tests/"):
        assert entry in lines, entry
    for entry in ("Notes/", "prompts/", ".claude/", "__pycache__/"):
        assert entry in lines, entry


def test_env_example_documents_every_new_setting():
    text = (ROOT / ".env.example").read_text()
    keys = {m.group(1) for m in re.finditer(r"^([A-Z0-9_]+)=", text, re.M)}
    assert set(NEW_SETTINGS) <= keys
    assert {k.upper() for k in Settings.model_fields} <= keys  # still fully mirrored


def test_new_settings_default_to_the_old_behaviour():
    s = Settings(_env_file=None)
    assert s.cache_backend == "memory" and s.otlp_endpoint == ""


def test_requirements_pin_the_new_packages():
    text = (ROOT / "requirements.txt").read_text()
    for package in ("redis", "opentelemetry-sdk", "opentelemetry-exporter-otlp-proto-http"):
        assert re.search(rf"^{package}>=[\d.]+,<\d+", text, re.M), package
