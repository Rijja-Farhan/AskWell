from pathlib import Path

import pytest

from app.config import Settings, get_settings

FIELDS = list(Settings.model_fields)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Run from an empty dir with no Settings env vars, so no `.env` or shell value leaks in."""
    for name in FIELDS:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_loads_defaults_without_env_file(clean_env):
    settings = Settings()
    assert settings.env == "dev"
    assert settings.log_level == "INFO"
    assert settings.data_raw_dir == Path("data/raw")
    assert settings.llm_provider == "gemini"
    assert settings.embedding_model == "gemini-embedding-2-preview"
    assert settings.gemini_api_key is None
    assert settings.vector_db == "chroma"
    assert settings.openai_api_key is None


def test_env_var_overrides_default(clean_env, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("VECTOR_DB", "qdrant")
    settings = Settings()
    assert settings.log_level == "DEBUG"
    assert settings.vector_db == "qdrant"


def test_reads_dotenv_file(clean_env, tmp_path):
    (tmp_path / ".env").write_text("ENV=prod\nLLM_MODEL=some-model\n")
    settings = Settings()
    assert settings.env == "prod"
    assert settings.llm_model == "some-model"


def test_blank_values_use_defaults(clean_env, tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=\nLOG_LEVEL=\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    settings = Settings()
    assert settings.openai_api_key is None
    assert settings.anthropic_api_key is None
    assert settings.log_level == "INFO"


def test_secrets_hidden_in_repr(clean_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-456")
    settings = Settings()
    assert "sk-test-secret-123" not in repr(settings)
    assert "sk-ant-secret-456" not in str(settings)
    assert settings.openai_api_key.get_secret_value() == "sk-test-secret-123"


def test_get_settings_is_cached(clean_env):
    assert get_settings() is get_settings()


def test_env_example_lists_every_field():
    example = Path(__file__).resolve().parents[1] / ".env.example"
    keys = {
        line.split("=", 1)[0].strip()
        for line in example.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert keys == {name.upper() for name in FIELDS}
