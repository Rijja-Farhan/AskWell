# Askwell API + UI image. Build context is filtered by .dockerignore: no .env, no local index,
# no model cache, no traces, no tests. Secrets are read at runtime (compose env_file), never baked.
FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    API_HOST=0.0.0.0

WORKDIR /app

# Runtime dependencies only (no dev or test tooling). Copied first so the
# layer is cached until the pins change.
COPY requirements.txt .
RUN pip install --default-timeout=100 --retries=5 -r requirements.txt

# Non-root user. The state directories are created and owned here so that named volumes mounted
# on them start out writable by this user.
RUN useradd --create-home --uid 10001 askwell \
    && mkdir -p /app/chroma_db /app/.cache/models /app/traces \
    && chown -R askwell:askwell /app

# Only what runtime needs: code, scripts (serve + build_index), the corpus and its metadata, the
# eval questions.
COPY --chown=askwell:askwell app/ app/
COPY --chown=askwell:askwell scripts/ scripts/
COPY --chown=askwell:askwell data/ data/
COPY --chown=askwell:askwell eval/eval_set.jsonl eval/eval_set.jsonl

USER askwell
EXPOSE 8000

# Liveness only: /health does not load models, so it answers while the reranker still downloads.
# (python:slim has no curl; urllib is enough.)
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["python", "scripts/serve.py"]
