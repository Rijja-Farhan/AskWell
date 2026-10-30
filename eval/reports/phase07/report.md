# Eval report: phase07

- Collection `askwell_structured`, embeddings `gemini-embedding-2-preview`, structure chunks 800/100 chars
- Retrieval k = 10 chunks; the answer uses the top 5
- Chat model `gemini-3.6-flash`, judge model `gemini-3.6-flash`
- Retrieval mode `hybrid` (BM25 v1, weight 0.3, rrf_k 60, 20 candidates per list), reranked by `BAAI/bge-reranker-base` (v1, top 40 candidates)
- Answer mode `baseline`
- Stages done: retrieval 80/80, generate 0/80, judge 0/80

Retrieval metrics are doc-level: each document counts at its best chunk rank, and k counts chunks. Questions without gold documents (unanswerable, some injection) show `n/a`. `scored` = questions with retrieval metrics, `judged` = questions with answer metrics. Sup@5 = share of conflicting-versions questions where a superseded version reaches the top 5.

Access-denied recall is computed on purpose: until Phase 12 adds access control, a high value means restricted content reaches the LLM.

## Results by category

| Category | n | scored | judged | R@1 | R@3 | R@5 | R@10 | MRR | Sup@5 | Faith | Rel | Behav | Refused | Cited |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple | 16 | 16 | 0 | 0.94 | 1.00 | 1.00 | 1.00 | 0.97 | n/a | n/a | n/a | n/a | n/a | n/a |
| exact-term | 12 | 12 | 0 | 0.50 | 0.92 | 0.92 | 1.00 | 0.76 | n/a | n/a | n/a | n/a | n/a | n/a |
| multi-doc | 12 | 12 | 0 | 0.46 | 0.75 | 0.92 | 1.00 | 0.93 | n/a | n/a | n/a | n/a | n/a | n/a |
| conflicting-versions | 10 | 10 | 0 | 0.50 | 1.00 | 1.00 | 1.00 | 0.73 | 1.00 | n/a | n/a | n/a | n/a | n/a |
| unanswerable | 10 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| access-denied | 10 | 10 | 0 | 0.90 | 1.00 | 1.00 | 1.00 | 0.95 | n/a | n/a | n/a | n/a | n/a | n/a |
| injection | 10 | 8 | 0 | 0.69 | 0.81 | 1.00 | 1.00 | 0.84 | n/a | n/a | n/a | n/a | n/a | n/a |
| **overall** | 80 | 68 | 0 | 0.68 | 0.92 | 0.97 | 1.00 | 0.87 | 1.00 | n/a | n/a | n/a | n/a | n/a |

## Latency (seconds per question)

Throttle waits are excluded; provider retries are not. `rerank` is part of `retrieval`.

| Stage | n | median | p95 |
|---|---:|---:|---:|
| retrieval | 80 | 7.641 | 12.170 |
| rerank | 80 | 6.765 | 10.609 |

## Behavior failures (`behavior_ok` false)

None.

## Retrieval misses (recall@5 = 0)

| id | question | first gold rank |
|---|---|---:|
| q020 | What expenses are coded to GL-6115? | 6 |
