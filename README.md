# Askwell

Askwell is an internal knowledge assistant for Nimbus Logistics, a fictional company with HR, IT, Operations and Finance departments. It answers employee questions from a synthetic corpus of 52 documents using retrieval-augmented generation, and it does so with citations, version awareness and per-user access control. It ships with a FastAPI backend, a web chat UI, an evaluation harness and a Docker Compose stack.

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (how it works), [docs/DEMO.md](docs/DEMO.md) (a 10 to 15 minute demo script), [docs/PENDING_CHECKS.md](docs/PENDING_CHECKS.md) (what is not yet verified with a live model, and how to finish it).

## Features

- **Ingestion**: PDF, DOCX, Markdown, HTML, CSV and text loaders; structure-aware chunking (headings kept with content, tables split by rows); incremental indexing into embedded Chroma.
- **Retrieval**: dense + BM25 hybrid search merged with weighted reciprocal rank fusion, then a local cross-encoder reranker (`BAAI/bge-reranker-base`, ONNX, CPU).
- **Filters**: metadata and version filters (`active_only` hides superseded documents, `--as-of` picks the version in force on a date) and role-based access control (a user only ever retrieves what they may read).
- **Answers**: grounded, with `[n]` citations; refuses when the evidence is weak or missing.
- **Conversation**: per-user sessions and follow-up question rewriting.
- **Safety**: prompt-injection defense (input sanitizing, prompt isolation, output guard).
- **Performance and operations**: embedding, retrieval and answer caches (in-process or Redis); a trace for every request (JSONL, optional OTLP to Jaeger or Langfuse); SSE streaming.
- **Evaluation**: an 80-question eval set, retrieval metrics, an LLM judge, resumable runs and a baseline-to-final comparison.

## Quick start (local)

Prerequisites: Python 3.10+, `pip`, the `venv` module, and a Gemini API key (or an OpenAI key with `LLM_PROVIDER=openai`).

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # set GEMINI_API_KEY
python scripts/build_index.py # parse data/raw/ and embed the documents (a rerun embeds only changes)
python scripts/serve.py       # API at http://127.0.0.1:8000 , chat UI at /ui/
```

On Debian/Ubuntu `python3 -m venv` needs `python3.10-venv` (`sudo apt install python3.10-venv`); without sudo use `pip3 install --user virtualenv && python3 -m virtualenv .venv`. All settings have defaults, so the code loads without a `.env`; only the key is needed to embed and answer.

Ask from the command line:

```bash
python scripts/ask.py "What is the phone number for the ethics hotline?"
python scripts/ask.py "When is open enrollment?" -k 3 --show-chunks
python scripts/ask.py "How many vacation days do I get?" --repeat 2   # the second ask shows the cache
```

`ask.py` exits 0 for answers and refusals, 2 if the key or the index is missing, 3 if the provider is unavailable after retries (for example a 429 or 503; try another `LLM_MODEL`).

## Quick start (Docker)

```bash
cp .env.example .env          # set GEMINI_API_KEY (or OPENAI_API_KEY with LLM_PROVIDER=openai)
docker compose up --build
```

- UI: http://localhost:8000/ui/ , API docs: http://localhost:8000/docs
- Traces: http://localhost:16686 (Jaeger, service `askwell`)

A one-shot `indexer` runs `scripts/build_index.py` into the `chroma_db` volume, then `api` starts once Redis and Jaeger are healthy. Without an API key the indexer exits with a message and the api does not start: add the key and run `docker compose up` again. The first question downloads the 1.1 GB reranker into the `model_cache` volume. Ports are published on `127.0.0.1` only, because `X-User-Id` is a demo switch, not authentication. `docker compose down` keeps the index, model cache and traces; `docker compose down -v` deletes them. Langfuse instead of Jaeger: set `OTLP_ENDPOINT` and `OTLP_HEADERS` (see `.env.example`).

The Docker checks with a live key have not been run yet; see [docs/PENDING_CHECKS.md](docs/PENDING_CHECKS.md).

## Try it

Open the UI, pick a user in the switcher (for example Sam Okafor `u05`, a public-clearance contractor, then Dana Whitfield `u06`, the COO) and click "Salary band G5": the same question gives different answers. [docs/DEMO.md](docs/DEMO.md) is the full walkthrough: citations, refusals, conflicting versions, access differences, memory, injection, cache hits and traces.

## Configuration

Everything is an environment variable read by `app/config.py` (and an optional `.env`). [.env.example](.env.example) documents every variable with its default, grouped by area: LLM and keys, vector store, chunking, retrieval, reranking, version policy, answers, RBAC, memory, injection defense, caching, tracing, API and evaluation. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) lists the groups and the reason for each default.

Changing `EMBEDDING_MODEL`, the provider's embeddings or the chunk settings needs a reindex (`python scripts/build_index.py --full`); changing only `LLM_MODEL` does not.

## How retrieval and answers work

- **Hybrid search** (`RETRIEVAL_MODE=hybrid`): dense and BM25 each return `HYBRID_CANDIDATES` chunks, merged by weighted RRF (`BM25_WEIGHT`, `RRF_K`). BM25 keeps identifiers such as `NX-2150` or `HR-POL-005` as single tokens. `dense` and `bm25` modes use one list.
- **Reranking** (`RERANKER=cross-encoder`): the top `RERANK_CANDIDATES` (default 40, the whole hybrid union) are rescored by the cross-encoder and the best k are kept. The model is pinned to `RERANK_MODEL_REVISION` and downloaded once to `RERANK_MODEL_DIR` (`.cache/models/`, gitignored). It costs about 4 to 7 s per question on CPU. `RERANKER=none` keeps the first-stage order.
- **Filters**: `VERSION_POLICY=active_only` (default) drops superseded documents; `prefer_active` ranks them below the active version; `all` keeps everything. Filters apply before each list's top-n cut.

```bash
python scripts/ask.py "What is the daily meal allowance when traveling?" --as-of 2024-06-01   # the version in force then
python scripts/ask.py "How do I request equipment?" --department IT --department Operations
python scripts/ask.py "How many days a week can I work from home?" --version-policy all --show-chunks
```

- **Grounded answers** (`ANSWER_MODE=grounded`): sources reach the model as `[1] title | section | doc_id | version | effective date`; each claim must carry a `[n]` marker. Invalid markers are removed and counted; valid ones become `citations`. When the sources do not answer, the reply is the fixed refusal text. Two refusals need no LLM call: no chunks (`no_context`) and a top reranker score below `REFUSAL_MIN_SCORE` (default `-4.0`, `low_score`). `ANSWER_MODE=baseline` keeps the minimal Phase 3 prompt.
- **Access control** (`RBAC_ENABLED=true`): each request runs as the `X-User-Id` profile from `data/metadata/users.json`; no header means `guest` (public documents only). Restricted chunks are filtered out before ranking, so the model never sees them.
- **Memory** (`MEMORY_ENABLED=true`): send a `session_id` and follow-ups such as "Who is eligible for it?" are rewritten into standalone queries; the answer step never sees the history.
- **Injection defense** (`INJECTION_DEFENSE=true`): questions and retrieved text are sanitized and isolated in the prompt, and replies that leak the system prompt are withheld.
- **Caching** (`CACHE_ENABLED=true`): entries are keyed by user, filters, settings, versions and the index fingerprint. `CACHE_BACKEND=redis` shares them across processes.
- **Tracing** (`TRACE_ENABLED=true`): ids, counts, timings and chunk ids are stored, not question or chunk text. Inspect with `python scripts/show_trace.py --list` and `python scripts/show_trace.py <trace_id>`.

## API

`python scripts/serve.py [--host 127.0.0.1] [--port 8000]` runs `app.api.main:app` (defaults from `API_HOST` and `API_PORT`). Run one server at a time and no other script on `chroma_db/` meanwhile (the local store is single-writer). `API_CORS_ORIGINS` is a comma list of allowed origins (empty = CORS off). Identity is the `X-User-Id` header, a demo switch, not authentication.

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness, always `200 {"status":"ok"}` |
| `GET /health/ready` | No network calls. 200 when ready, else 503 with a `problems` list (key present as a boolean, index, documents, chunks, provider, modes) |
| `POST /chat` | JSON answer |
| `POST /chat/stream` | The same answer as Server-Sent Events |
| `POST /ingest` | Parse `data/raw/` and sync the index; body `{"full": false}` optional. One ingest at a time (409 otherwise) |
| `GET /me`, `GET /users` | The caller's profile; the demo users for the UI switcher |
| `GET /sessions/{id}`, `DELETE /sessions/{id}` | Read or delete the caller's conversation |
| `GET /traces`, `GET /traces/{id}` | Stored traces (admin users only) |
| `GET /ui/` | The chat UI (`/` redirects there) |

Chat request body: `question` (1 to 2000 chars), optional `k` (1 to 20), `departments` (`HR|IT|Operations|Finance`), `as_of` (`YYYY-MM-DD`), `version_policy`, `include_text`, `session_id`. Response (and the `done` event): `answer, refused, refusal_reason (null|no_context|low_score|model|blocked), citations, invalid_citations, sources, model, latency_ms, session_id, standalone_query, cached, trace_id`. `[n]` in `answer` is `sources[n-1]`.

`/chat/stream` events: `sources` (before generation), `token` (`{"text"}`, zero or more), `done` (the final payload; replace the streamed text with its `answer`), `error` (`{"code","message"}`, then the stream ends). In grounded mode text is held back while the reply could still be the fixed refusal.

Errors are `{"error": {"code", "message"}}`: 422 `validation_error` or `parse_errors`, 409 `ingest_busy`, 503 `missing_api_key`, `index_not_ready` or `provider_unavailable` (with `Retry-After`), 502 `provider_error`. After a stream has started they arrive as the `error` event. Messages never contain keys or stack traces.

```bash
curl -s localhost:8000/health/ready
curl -s -X POST localhost:8000/chat -H 'content-type: application/json' -H 'X-User-Id: u06' \
  -d '{"question":"How many days of annual leave do new employees get?","departments":["HR"]}'
curl -N -s -X POST localhost:8000/chat/stream -H 'content-type: application/json' \
  -d '{"question":"How many days of annual leave do new employees get?"}'
curl -s -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{"full": false}'
```

## Data pipeline

The corpus is generated, so you can rebuild it:

```bash
python scripts/generate_dataset.py   # write the synthetic corpus, metadata and eval set
python scripts/validate_dataset.py   # check coverage and label consistency
python scripts/parse_corpus.py       # parse the corpus into data/processed/documents.jsonl
```

`build_index.py` is incremental: each chunk stores a fingerprint of its document (file bytes, parsed text, metadata entry, chunking config), so a rerun embeds only added or changed documents and deletes chunks of removed ones. If any file fails to parse, nothing changes (exit 1). `--full` re-embeds everything.

## Tests and lint

```bash
pytest -q
ruff check .
ruff format --check .   # or `ruff format .` to fix
```

All tests run offline with fakes; no key is needed.

## Evaluate

`scripts/run_eval.py` scores the pipeline on the 80-question eval set (`eval/eval_set.jsonl`; categories simple, exact-term, multi-doc, conflicting-versions, unanswerable, access-denied, injection) and writes `manifest.json`, `results.jsonl`, `report.json` and `report.md` to `eval/reports/<name>/`.

```bash
python scripts/run_eval.py --name my-run --stages retrieval        # retrieval metrics only, no chat calls
python scripts/run_eval.py --name my-run                           # + generate answers + LLM judge
python scripts/run_eval.py --name my-change --compare my-run       # add deltas versus another run
python scripts/run_eval.py --name test --limit 5 --fresh           # quick check on 5 questions
```

Exit codes: 0 complete, 1 incomplete (errors or a config mismatch), 2 setup problem, 3 provider unavailable. Progress is saved after every question, so rerunning the same command resumes; chat calls are spaced `EVAL_CHAT_INTERVAL` seconds apart. A run keeps its config in `manifest.json`; changing the models, chunking, retrieval, reranker, version, access or answer settings needs `--fresh` or a new `--name`. Use `LLM_MODEL=...` or `EVAL_JUDGE_MODEL=...` on the command line to try other models.

The historical runs are `baseline` (fixed chunks, dense), `phase05` (structured chunks), `phase06` (hybrid), `phase07` (+ reranker), `phase08` (+ active versions only) and `phase09` (+ grounded answers). Each earlier run resumes only with the settings it was made with, for example `ANSWER_MODE=baseline VERSION_POLICY=all python scripts/run_eval.py --name phase07`; see the "Finding" section of [docs/PENDING_CHECKS.md](docs/PENDING_CHECKS.md) for why `phase08` and `phase09` no longer resume as they are.

Probes (no model needed unless noted):

```bash
python scripts/run_access_probe.py        # 0 leaked chunks for every user
python scripts/run_injection_probe.py
python scripts/run_cache_probe.py
python scripts/run_trace_probe.py
python scripts/run_multiturn_probe.py --offline   # without --offline: live, needs a model
python scripts/run_ui_probe.py            # live: needs a running server and a model
EVAL_SET_PATH=eval/probes/exact_terms.jsonl python scripts/run_eval.py --name probe-exact --stages retrieval   # 24 identifier lookups
```

## Results

`python scripts/compare_reports.py` writes [eval/reports/FINAL.md](eval/reports/FINAL.md): per-category recall@k, MRR, faithfulness, answer relevance, refusal rate and latency across `baseline` to `phase09` (plus `phase12` and `phase13` when they exist). A metric that was never measured shows "pending".

Retrieval, overall over the 68 questions that have a gold document:

| Run | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| `baseline` (fixed chunks, dense) | 0.699 | 0.963 | 0.856 |
| `phase05` (structured chunks) | 0.676 | 0.985 | 0.870 |
| `phase06` (+ hybrid) | 0.699 | 0.993 | 0.885 |
| `phase07` (+ reranker) | 0.676 | 0.971 | 0.874 |
| `phase08` (+ active versions only) | 0.794 | 1.000 | 0.937 |

These come from `eval/reports/`. Answer quality (faithfulness, relevance, refusal behavior) has no numbers yet: the judged runs were blocked by chat quota.

## Known limitations and pending checks

- **No judged answer metrics yet.** `phase08` has 42 of 80 answers and no judging, `phase09` was interrupted, and `phase12` and `phase13` (access and injection evals) were never run, so faithfulness, relevance and refusal rate are pending.
- **Live checks not run**: multi-turn live probe, API smoke with a session, injection smoke, `ask.py --repeat 2`, provider token usage, the UI probe and manual browser check, and the Docker stack with a real key.
- `phase08` and `phase09` manifests no longer match the code, so finishing them restarts them (`--fresh-mismatched`).
- `X-User-Id` is a demo switch, not authentication; run the stack on loopback only.
- Chroma is embedded and single-writer: one server at a time. The refusal gate (`-4.0`) is calibrated on the eval set and may overfit.
- The reranker first-run download is 1.1 GB and each rerank costs seconds on CPU.

Everything pending is listed in [docs/PENDING_CHECKS.md](docs/PENDING_CHECKS.md). After switching to a working model, one command runs it all and records the result:

```bash
python scripts/run_pending_checks.py          # --list shows the plan, --offline-only needs no model
```

## Layout

| Path | Holds |
|---|---|
| `app/` | Application code |
| `app/ingestion/` | Loaders that parse every corpus format into one `ParsedDocument` |
| `app/rag/` | Chunking, embeddings, incremental Chroma index, hybrid retrieval, reranking, filters, access control, answers, memory, injection defense, caching |
| `app/api/` | FastAPI app: chat (JSON and SSE), ingest, health, sessions, traces |
| `app/observability/` | Tracing: spans, JSONL store, OTLP export |
| `app/evaluation/` | Retrieval metrics, LLM judge, resumable runner, reports |
| `app/ui/` | The static chat UI served at `/ui/` |
| `docs/` | `ARCHITECTURE.md`, `DEMO.md`, `PENDING_CHECKS.md` |
| `data/raw/` | Source documents (tracked) |
| `data/processed/` | Generated output (ignored) |
| `data/metadata/` | Document metadata and user profiles (tracked) |
| `eval/` | Eval set, probe sets (`probes/`), run reports (`reports/<name>/`, `FINAL.md`) and pending-check status (`pending_checks/`) |
| `scripts/` | CLI scripts: `ask.py`, `build_index.py`, `serve.py`, `run_eval.py`, `compare_reports.py`, `run_pending_checks.py`, `show_trace.py`, probes, dataset tools |
| `tests/` | Automated tests (offline) |
| `Dockerfile`, `compose.yaml` | The container image and the Compose stack |
