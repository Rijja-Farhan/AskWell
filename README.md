# Askwell

Askwell is an internal knowledge assistant for Nimbus Logistics, a fictional company with four departments: HR, IT, Operations and Finance. It answers employee questions from a synthetic document corpus using retrieval-augmented generation: hybrid search, reranking, metadata filtering, grounded answers with citations, role-based access control and evaluation. It is served by a FastAPI backend with a simple chat UI.

## Prerequisites

- Python 3.10+
- `pip` and the `venv` module

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env    # then fill in API keys as needed
```

On Debian/Ubuntu, `python3 -m venv` fails with "ensurepip is not available" unless `python3.10-venv` is installed (`sudo apt install python3.10-venv`). Without sudo, use `pip3 install --user virtualenv && python3 -m virtualenv .venv` instead.

All settings have safe defaults, so the app loads without a `.env`. Configuration is read only through `app.config.get_settings()`.

## Tests and lint

```bash
pytest -q
ruff check .
ruff format --check .   # or `ruff format .` to fix
```

## Data pipeline

```bash
python scripts/generate_dataset.py   # write the synthetic corpus, metadata and eval set
python scripts/validate_dataset.py   # check coverage and label consistency
python scripts/parse_corpus.py       # parse the corpus into data/processed/documents.jsonl
```

## Build the index and ask a question

Needs `GEMINI_API_KEY` in `.env` (or `LLM_PROVIDER=openai` with `OPENAI_API_KEY`).

```bash
python scripts/build_index.py                                   # parse data/raw/, embed new or changed docs
python scripts/build_index.py --full                            # re-embed everything
python scripts/ask.py "What is the phone number for the ethics hotline?"
python scripts/ask.py "When is open enrollment?" -k 3 --show-chunks
```

`ask.py` exits 2 if the API key or the index is missing, and 3 if the provider is unavailable after retries (for example a 503 on the free tier; set `LLM_MODEL` in `.env` to another model). After changing `EMBEDDING_MODEL` or the provider, rerun `build_index.py`.

`build_index.py` is incremental. Each chunk stores a fingerprint of its document (file bytes, parsed text, metadata entry, chunking config), so a rerun embeds only documents that were added or changed and deletes chunks of removed ones. Changing the embedding model or chunk settings triggers a full rebuild. If any file fails to parse, nothing is changed (exit 1).

Chunks follow the document structure by default (`CHUNK_STRATEGY=structure`, collection `askwell_structured`): one top-level section at most per chunk, headings kept with their content, tables split by rows. The Phase 3 baseline index stays available as `VECTOR_COLLECTION=askwell_baseline CHUNK_STRATEGY=fixed`.

Retrieval is hybrid by default (`RETRIEVAL_MODE=hybrid`): dense (embedding) search and BM25 keyword search each return `HYBRID_CANDIDATES` chunks, merged with weighted reciprocal rank fusion (`BM25_WEIGHT`, `RRF_K`). BM25 keeps identifiers such as `NX-2150` or `HR-POL-005` as single tokens. Its index is built in memory from the Chroma collection, so it always matches the dense index. `RETRIEVAL_MODE=dense` or `bm25` uses one list only; `ask.py --show-chunks` prints each hit's rank in both lists.

The first-stage results are then reranked (`RERANKER=cross-encoder`). A cross-encoder reads the question and each of the top `RERANK_CANDIDATES` chunks together, scores them, and the best k by that score are kept. The default 40 covers the whole hybrid union (20 + 20), so a chunk only BM25 found can still reach rank 1. The model runs locally on ONNX Runtime (CPU, no API quota). `RERANK_MODEL` is pinned to `RERANK_MODEL_REVISION` and downloaded once to `RERANK_MODEL_DIR` (`.cache/models/`, gitignored). The default model is `BAAI/bge-reranker-base` (1.1 GB download, about 4-7 s per question on CPU); `RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2` with `RERANK_MODEL_REVISION=233902d25c440f23af6f7d6e94d2946bac0bee0a` is about 10x faster but scored lower. `RERANKER=none` keeps the first-stage order. `ask.py --show-chunks` also prints each hit's rank before reranking.

Retrieval only ranks chunks the metadata filters allow (`app/rag/filters.py`). `VERSION_POLICY=active_only` (default) drops superseded documents, so an outdated policy never reaches the answer; `prefer_active` keeps them but always below the active version of the same document; `all` keeps everything. Filters apply before each list's top-n cut, and BM25 statistics stay corpus-wide. `ask.py` takes the filters as flags:

```bash
python scripts/ask.py "What is the daily meal allowance when traveling?" --as-of 2024-06-01   # the version in force then
python scripts/ask.py "How do I request equipment?" --department IT --department Operations
python scripts/ask.py "How many days a week can I work from home?" --version-policy all --show-chunks
```

`--as-of` picks, per document, the version in force on that date and replaces the version policy. `--show-chunks` prints each hit's version, status and effective date. If no document matches, `ask.py` says so and makes no LLM call.

## Grounded answers

`ANSWER_MODE=grounded` (default) answers only from the retrieved chunks. They reach the LLM as numbered sources (`[1] title | section | doc_id | version | effective date`), and the model must put a `[n]` marker after each claim. Markers that point to no source are removed and counted (`invalid_citations`); the valid ones become `Citation` objects (`number, chunk_id, doc_id, title, section`) in first-cited order. When the sources do not answer, the model replies with the fixed `REFUSAL_TEXT` ("I don't know. ...").

Two refusals happen without any LLM call: no chunks at all (`no_context`, for example filters that match nothing), and `REFUSAL_MIN_SCORE` (default `-4.0`): with `RERANKER=cross-encoder`, a top chunk whose reranker score is below it means the evidence is too weak (`low_score`). `REFUSAL_MIN_SCORE=none` disables the gate; it is skipped when `RERANKER=none`, because first-stage scores are not comparable. A reply equal to `REFUSAL_TEXT` is marked `model`. `ANSWER_MODE=baseline` keeps the Phase 3 minimal prompt, with no citations or refusals.

```text
$ python scripts/ask.py "<answerable question>"
<answer text with [1] and [2] markers>

Sources:
  [1] <title>, <section> (<doc_id>)
  [2] <title>, <section> (<doc_id>)

Retrieved (hybrid, reranked by BAAI/bge-reranker-base, versions active_only):
  1. ...

$ python scripts/ask.py "What does error NX-6100 mean?"
I don't know. The company documents I can access do not answer this question. Please check with the relevant department.

(Refused: low_score, best match scored below -4.0, no LLM call)

Retrieved (hybrid, reranked by BAAI/bge-reranker-base, versions active_only):
  1. it-mtg-001-v1::0004 (-5.936)
  ...
```

Only cited sources are listed under `Sources:`. `ask.py` exits 0 for refusals.

The same `answer_from_chunks` serves the CLI and the eval runner. Access rules and prompt-injection defenses are not part of this phase.

## API

`python scripts/serve.py [--host 127.0.0.1] [--port 8000]` runs the FastAPI app (`app.api.main:app`; defaults from `API_HOST` / `API_PORT`). Run one server at a time and no other script on `chroma_db/` meanwhile (the local store is single-writer). `API_CORS_ORIGINS` is a comma list of allowed origins (empty = CORS off). There is no authentication yet (Phase 12); `/ingest` is open.

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness, always `200 {"status":"ok"}` |
| `GET /health/ready` | No network calls. 200 when ready, else 503 with the same body and a `problems` list: `status, index_present, documents, chunks, provider, api_key_present (boolean only), answer_mode, retrieval_mode, reranker, problems` |
| `POST /chat` | JSON answer |
| `POST /chat/stream` | The same answer as Server-Sent Events |
| `POST /ingest` | Parse `data/raw/` and sync the index (same code as `build_index.py`); body `{"full": false}` is optional. Returns the sync stats. One ingest at a time (409 otherwise); chat waits while it runs |

Request body of both chat endpoints: `question` (1 to 2000 chars), optional `k` (1 to 20, default `RETRIEVAL_TOP_K`), `departments` (list of `HR|IT|Operations|Finance`), `as_of` (`YYYY-MM-DD`), `version_policy`, `include_text` (add each source's chunk text). No history yet (Phase 11).

Response of `/chat`, and the `done` event: `answer, refused, refusal_reason (null|no_context|low_score|model), citations [{number, chunk_id, doc_id, title, section}], invalid_citations, sources [{rank, chunk_id, doc_id, title, section, score, version, status, effective_date}], model, latency_ms`. `[n]` in `answer` is `sources[n-1]`.

`/chat/stream` sends `text/event-stream`, each event as `event: <name>` plus one JSON `data:` line:

| Event | Data |
|---|---|
| `sources` | `{"sources": [...]}` as above, sent before generation |
| `token` | `{"text": "<delta>"}`, zero or more |
| `done` | The final payload, same shape as `/chat`. Its `answer` has invalid markers removed and `citations` are validated: replace the streamed text with it |
| `error` | `{"code", "message"}`, then the stream ends |

Refusals before the LLM (`no_context`, `low_score`) send `sources` then `done`, no `token`. In grounded mode text is held back while the reply could still be the fixed refusal, so a model refusal never appears as partial tokens; it ends with `done` (`refused: true`, `refusal_reason: "model"`).

Errors always have the body `{"error": {"code", "message"}}` (validation adds `details`, parse errors add `errors`):

| Status | Code | When |
|---|---|---|
| 422 | `validation_error` | Bad request body |
| 422 | `parse_errors` | `/ingest`: a file failed to parse; index unchanged |
| 409 | `ingest_busy` | `/ingest` while another ingest runs |
| 503 | `missing_api_key`, `index_not_ready` | Setup problem (message says what to do) |
| 503 | `provider_unavailable` | Connection error, 429 or 5xx after retries; has a `Retry-After` header |
| 502 | `provider_error` | Other provider status error |

After a stream has started, the same codes arrive as the `error` event instead of a status. Messages never contain keys or stack traces.

```bash
python scripts/serve.py &
curl -s localhost:8000/health/ready
curl -s -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question":"How many days of annual leave do new employees get?","departments":["HR"]}'
curl -N -s -X POST localhost:8000/chat/stream -H 'content-type: application/json' \
  -d '{"question":"How many days of annual leave do new employees get?"}'
curl -s -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{"full": false}'
```

Phase 10 added no eval run; the eval documentation below is unchanged.

## Docker

One command brings up the API and UI, a Redis cache and a Jaeger trace UI.

Prerequisites: Docker with Compose v2 and a Gemini or OpenAI API key.

```bash
cp .env.example .env     # set GEMINI_API_KEY (or OPENAI_API_KEY with LLM_PROVIDER=openai)
docker compose up --build
```

- UI: http://localhost:8000/ui/ , API docs: http://localhost:8000/docs
- Traces: http://localhost:16686 (Jaeger; service `askwell`, search by the `X-Trace-Id` of a response)

Start order: a one-shot `indexer` runs `scripts/build_index.py` (incremental, embeds nothing on a restart) into the `chroma_db` volume, then `api` starts once Redis and Jaeger are healthy. Without an API key the indexer exits with a message and the api does not start; add the key to `.env` and run `docker compose up` again. The first question downloads the ~1.1 GB reranker into the `model_cache` volume (the first answer is slow, later ones are not).

The vector DB is embedded Chroma on the `chroma_db` volume (not a separate server: the single-writer store and the cache's index fingerprint need a local directory). Ports are published on `127.0.0.1` only because `X-User-Id` is a demo switch, not authentication.

`docker compose down` stops the containers and keeps the index, model cache and traces; `docker compose down -v` also deletes those volumes (the next start re-indexes and re-downloads). Langfuse instead of Jaeger: set `OTLP_ENDPOINT` and `OTLP_HEADERS` in `.env` (see `.env.example`). Outside Docker nothing changes: `CACHE_BACKEND=memory` and `OTLP_ENDPOINT` empty are the defaults.

## Evaluate

Scores the pipeline on the 80-question eval set and writes `manifest.json`, `results.jsonl`, `report.json` and `report.md` to `eval/reports/<name>/`.

```bash
python scripts/run_eval.py --name baseline --stages retrieval   # retrieval metrics only, no chat calls
python scripts/run_eval.py --name baseline                      # + generate answers + LLM judge
python scripts/run_eval.py --name my-change --compare baseline  # add deltas versus the baseline
python scripts/run_eval.py --name test --limit 5 --fresh        # quick check on 5 questions
```

Reports so far: `baseline` (Phase 3, fixed chunks, dense), `phase05` (structured chunks, dense), `phase06` (structured chunks, hybrid) and `phase07` (hybrid + reranker) and `phase08` (+ active versions only, minimal prompt) and `phase09` (+ grounded answers). Reports include median and p95 latency per stage and, for grounded runs, the `Refused` and `Cited` columns plus a false-refusal count. Resume older runs with the settings they were made with; runs before Phase 7 were not reranked, runs before Phase 8 kept every version and runs before Phase 9 used the minimal prompt (`ANSWER_MODE=baseline`):

```bash
ANSWER_MODE=baseline VERSION_POLICY=all python scripts/run_eval.py --name phase07
ANSWER_MODE=baseline RERANKER=none VERSION_POLICY=all python scripts/run_eval.py --name phase06
ANSWER_MODE=baseline RERANKER=none VERSION_POLICY=all RETRIEVAL_MODE=dense python scripts/run_eval.py --name phase05
ANSWER_MODE=baseline VECTOR_COLLECTION=askwell_baseline CHUNK_STRATEGY=fixed RETRIEVAL_MODE=dense RERANKER=none VERSION_POLICY=all \
  LLM_MODEL=gemini-3-flash-preview python scripts/run_eval.py --name baseline
```

`eval/probes/exact_terms.jsonl` holds 24 identifier lookups ("NX-4020", "GL-7100 account") that the main set does not cover. Score them with `EVAL_SET_PATH`:

```bash
EVAL_SET_PATH=eval/probes/exact_terms.jsonl ANSWER_MODE=baseline RERANKER=none VERSION_POLICY=all RETRIEVAL_MODE=dense python scripts/run_eval.py --name probe06_dense --stages retrieval
EVAL_SET_PATH=eval/probes/exact_terms.jsonl ANSWER_MODE=baseline RERANKER=none VERSION_POLICY=all python scripts/run_eval.py --name probe06_hybrid --stages retrieval --compare probe06_dense
EVAL_SET_PATH=eval/probes/exact_terms.jsonl ANSWER_MODE=baseline VERSION_POLICY=all python scripts/run_eval.py --name probe07 --stages retrieval --compare probe06_hybrid
EVAL_SET_PATH=eval/probes/exact_terms.jsonl ANSWER_MODE=baseline python scripts/run_eval.py --name probe08 --stages retrieval --compare probe07
```

Progress is saved after every question: rerunning the same command skips finished work, so an interrupted run (exit 3, provider unavailable) resumes later. Chat calls are spaced `EVAL_CHAT_INTERVAL` seconds apart for free-tier limits. A run keeps its config in `manifest.json`; changing the models, chunk, retrieval, reranker or version-policy settings needs `--fresh` or a new `--name`. Use `LLM_MODEL=...` or `EVAL_JUDGE_MODEL=...` on the command line to try other models.

## Layout

| Path | Holds |
|---|---|
| `app/` | Application code |
| `app/ingestion/` | Loaders that parse every corpus format into one `ParsedDocument` |
| `app/rag/` | RAG pipeline: structure-aware and fixed chunking, embeddings, incremental Chroma index, dense/BM25/hybrid retrieval, cross-encoder reranking, metadata filters, LLM answers |
| `app/api/` | FastAPI app: chat (JSON and SSE), ingest, health |
| `app/evaluation/` | Eval harness: retrieval metrics, LLM judge, resumable runner, reports |
| `data/raw/` | Source documents (tracked) |
| `data/processed/` | Generated output (ignored) |
| `data/metadata/` | Document metadata and user profiles (tracked) |
| `eval/` | Eval set (`eval_set.jsonl`), probe sets (`probes/`) and run reports in `eval/reports/<name>/` (tracked) |
| `scripts/` | CLI scripts |
| `tests/` | Automated tests |
| `Notes/` | Short explainers, one per implemented topic |
