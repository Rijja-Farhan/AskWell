# Eval report: tune07_minilm20

- Collection `askwell_structured`, embeddings `gemini-embedding-2-preview`, structure chunks 800/100 chars
- Retrieval k = 10 chunks; the answer uses the top 5
- Chat model `gemini-3.6-flash`, judge model `gemini-3.6-flash`
- Retrieval mode `hybrid` (BM25 v1, weight 0.3, rrf_k 60, 20 candidates per list), reranked by `cross-encoder/ms-marco-MiniLM-L6-v2` (v1, top 20 candidates)
- Stages done: retrieval 80/80, generate 0/80, judge 0/80
- Deltas in brackets are versus run `phase06`

Retrieval metrics are doc-level: each document counts at its best chunk rank, and k counts chunks. Questions without gold documents (unanswerable, some injection) show `n/a`. `scored` = questions with retrieval metrics, `judged` = questions with answer metrics. Sup@5 = share of conflicting-versions questions where a superseded version reaches the top 5.

Access-denied recall is computed on purpose: until Phase 12 adds access control, a high value means restricted content reaches the LLM.

## Results by category

| Category | n | scored | judged | R@1 | R@3 | R@5 | R@10 | MRR | Sup@5 | Faith | Rel | Behav |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple | 16 | 16 | 0 | 1.00 (+0.00) | 1.00 (+0.00) | 1.00 (+0.00) | 1.00 (+0.00) | 1.00 (+0.00) | n/a | n/a | n/a | n/a |
| exact-term | 12 | 12 | 0 | 0.58 (-0.17) | 0.92 (-0.08) | 0.92 (-0.08) | 0.92 (-0.08) | 0.79 (-0.12) | n/a | n/a | n/a | n/a |
| multi-doc | 12 | 12 | 0 | 0.33 (-0.04) | 0.79 (-0.08) | 0.88 (-0.08) | 0.96 (-0.04) | 0.81 (-0.06) | n/a | n/a | n/a | n/a |
| conflicting-versions | 10 | 10 | 0 | 0.50 (+0.10) | 1.00 (+0.00) | 1.00 (+0.00) | 1.00 (+0.00) | 0.73 (+0.05) | 1.00 (+0.00) | n/a | n/a | n/a |
| unanswerable | 10 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| access-denied | 10 | 10 | 0 | 0.90 (+0.10) | 1.00 (+0.00) | 1.00 (+0.00) | 1.00 (+0.00) | 0.95 (+0.05) | n/a | n/a | n/a | n/a |
| injection | 10 | 8 | 0 | 0.81 (+0.06) | 0.94 (+0.00) | 1.00 (+0.00) | 1.00 (+0.00) | 0.94 (+0.08) | n/a | n/a | n/a | n/a |
| **overall** | 80 | 68 | 0 | 0.70 (+0.00) | 0.94 (-0.03) | 0.96 (-0.03) | 0.98 (-0.02) | 0.88 (-0.01) | 1.00 (+0.00) | n/a | n/a | n/a |

## Latency (seconds per question)

Throttle waits are excluded; provider retries are not. `rerank` is part of `retrieval`.

| Stage | n | median | p95 |
|---|---:|---:|---:|
| retrieval | 80 | 1.176 (+0.617) | 1.611 (+0.864) |
| rerank | 80 | 0.514 | 0.855 |

## Behavior failures (`behavior_ok` false)

None.

## Retrieval misses (recall@5 = 0)

| id | question | first gold rank |
|---|---|---:|
| q020 | What expenses are coded to GL-6115? | > 10 |
