"""Application settings.

All configuration is read here, from environment variables and an optional `.env` file.
Other modules call `get_settings()` instead of reading `os.environ` directly.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # blank values (e.g. `OPENAI_API_KEY=`) fall back to defaults
        extra="ignore",
    )

    # App
    env: str = "dev"
    log_level: str = "INFO"

    # Data
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    parsed_documents_path: Path = Path("data/processed/documents.jsonl")
    data_metadata_dir: Path = Path("data/metadata")
    eval_set_path: Path = Path("eval/eval_set.jsonl")

    # LLM and embeddings (one provider serves both: "gemini" or "openai")
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.6-flash"
    embedding_model: str = "gemini-embedding-2-preview"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_max_retries: int = 5

    # Secrets
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    # Vector store
    vector_db: str = "chroma"
    vector_db_path: Path = Path("chroma_db")
    vector_collection: str = "askwell_structured"

    # Chunking and retrieval (characters / number of chunks). "structure" follows headings,
    # sections and tables, with chunk_size as the maximum; "fixed" is the Phase 3 baseline.
    chunk_strategy: Literal["fixed", "structure"] = "structure"
    chunk_size: int = 800
    chunk_overlap: int = 100
    retrieval_top_k: int = 5

    # Retrieval mode: "dense" (embeddings), "bm25" (keywords) or "hybrid" (both, merged with
    # weighted reciprocal rank fusion over hybrid_candidates chunks from each list)
    retrieval_mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    hybrid_candidates: int = 20
    rrf_k: int = 60
    bm25_weight: float = 0.3

    # Reranking: "cross-encoder" rescores the first-stage top rerank_candidates with a local
    # ONNX cross-encoder (pinned Hugging Face revision, cached in rerank_model_dir); "none"
    # keeps the first-stage order
    reranker: Literal["none", "cross-encoder"] = "cross-encoder"
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_model_revision: str = "2cfc18c9415c912f9d8155881c133215df768a70"
    rerank_onnx_file: str = "onnx/model.onnx"
    rerank_candidates: int = 40
    rerank_model_dir: Path = Path(".cache/models")

    # Version policy for every retrieval: "active_only" drops superseded documents,
    # "prefer_active" ranks them below their family's active version, "all" keeps everything
    version_policy: Literal["active_only", "prefer_active", "all"] = "active_only"

    # Answers: "grounded" answers only from numbered sources with [n] citations and refuses
    # with a fixed reply when evidence is weak; "baseline" is the Phase 3 minimal prompt.
    # refusal_min_score: with the cross-encoder, refuse without an LLM call when the top chunk
    # scores below it (bge-reranker logit); None disables the gate.
    answer_mode: Literal["baseline", "grounded"] = "grounded"
    refusal_min_score: float | None = -4.0

    # Conversation memory (per session_id): history kept per session, and how many recent turns
    # the follow-up rewriter sees. With memory_enabled=false a session_id is accepted but ignored.
    memory_enabled: bool = True
    memory_max_turns: int = 10
    memory_rewrite_turns: int = 4
    memory_max_sessions: int = 1000
    memory_ttl_seconds: int = 3600

    # Role-based access: when true every retrieval is restricted to what the request's user may
    # read (see app/rag/access.py). false disables filtering (eval comparison, legacy tests).
    rbac_enabled: bool = True

    # Prompt-injection defense (Phase 13): sanitize questions, neutralize retrieved text, wrap
    # untrusted text in delimiters under a hardened prompt, guard the output. false restores the
    # exact pre-Phase-13 prompts and inputs (eval comparison, legacy tests). Questions are cut
    # to max_question_chars before anything else.
    injection_defense: bool = True
    max_question_chars: int = 2000

    # Caching (Phase 14): in-process LRU caches for query embeddings, retrieval results and final
    # answers. Invalidation is by index fingerprint; the TTL is only a safety net.
    # cache_enabled=false makes every layer a pass-through (the Phase 13 behaviour).
    cache_enabled: bool = True
    cache_embedding_entries: int = 2048
    cache_retrieval_entries: int = 512
    cache_answer_entries: int = 256
    cache_ttl_seconds: int = 3600
    # Phase 17: "redis" keeps the three layers in Redis (shared, survives restarts; entry-count
    # settings only switch a layer on/off, Redis `maxmemory` bounds the size). Each layer gets
    # its own key prefix. Any Redis error is a cache miss.
    cache_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "askwell:"

    # Tracing (Phase 15): one trace per chat/ingest request with per-stage spans. Traces hold ids,
    # counts, scores and timings only; answer text is stored just when trace_store_text=true.
    # trace_enabled=false restores the Phase 14 behaviour (no trace, no X-Trace-Id header).
    trace_enabled: bool = True
    trace_max_entries: int = 500
    trace_to_file: bool = True
    traces_dir: Path = Path("traces")
    trace_store_text: bool = False
    # Phase 17: export finished traces as OpenTelemetry spans over OTLP/HTTP (empty = off).
    # otlp_endpoint is the base URL (e.g. http://localhost:4318; /v1/traces is appended);
    # otlp_headers is a comma list of k=v for hosted backends such as Langfuse.
    otlp_endpoint: str = ""
    otlp_headers: str = ""
    otlp_service_name: str = "askwell"

    # API server: bind address and CORS origins (comma list; empty disables CORS)
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_cors_origins: str = ""

    # Evaluation: report folder, chunks retrieved per eval question, judge model (empty means
    # llm_model) and the minimum seconds between chat calls (free-tier rate limits)
    eval_reports_dir: Path = Path("eval/reports")
    eval_retrieval_k: int = 10
    eval_judge_model: str = ""
    eval_chat_interval: float = 6.0

    @field_validator("refusal_min_score", mode="before")
    @classmethod
    def _score_off(cls, value):
        """`REFUSAL_MIN_SCORE=none` (or `null`) disables the gate."""
        if isinstance(value, str) and value.strip().lower() in ("none", "null"):
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the shared settings instance (loaded once per process)."""
    return Settings()
