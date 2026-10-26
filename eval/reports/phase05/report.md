# Eval report: phase05

- Collection `askwell_structured`, embeddings `gemini-embedding-2-preview`, structure chunks 800/100 chars
- Retrieval k = 10 chunks; the answer uses the top 5
- Chat model `gemini-3.6-flash`, judge model `gemini-3.6-flash`
- Retrieval mode `dense`
- Stages done: retrieval 80/80, generate 0/80, judge 0/80

Retrieval metrics are doc-level: each document counts at its best chunk rank, and k counts chunks. Questions without gold documents (unanswerable, some injection) show `n/a`. `scored` = questions with retrieval metrics, `judged` = questions with answer metrics. Sup@5 = share of conflicting-versions questions where a superseded version reaches the top 5.

Access-denied recall is computed on purpose: until Phase 12 adds access control, a high value means restricted content reaches the LLM.

## Results by category

| Category | n | scored | judged | R@1 | R@3 | R@5 | R@10 | MRR | Sup@5 | Faith | Rel | Behav |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple | 16 | 16 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a |
| exact-term | 12 | 12 | 0 | 0.75 | 1.00 | 1.00 | 1.00 | 0.92 | n/a | n/a | n/a | n/a |
| multi-doc | 12 | 12 | 0 | 0.33 | 0.79 | 0.92 | 0.96 | 0.83 | n/a | n/a | n/a | n/a |
| conflicting-versions | 10 | 10 | 0 | 0.40 | 1.00 | 1.00 | 1.00 | 0.68 | 1.00 | n/a | n/a | n/a |
| unanswerable | 10 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| access-denied | 10 | 10 | 0 | 0.70 | 1.00 | 1.00 | 1.00 | 0.85 | n/a | n/a | n/a | n/a |
| injection | 10 | 8 | 0 | 0.75 | 0.94 | 1.00 | 1.00 | 0.85 | n/a | n/a | n/a | n/a |
| **overall** | 80 | 68 | 0 | 0.68 | 0.96 | 0.99 | 0.99 | 0.87 | 1.00 | n/a | n/a | n/a |

## Latency (seconds per question)

Throttle waits are excluded; provider retries are not. `rerank` is part of `retrieval`.

| Stage | n | median | p95 |
|---|---:|---:|---:|
| retrieval | 80 | 0.590 | 1.255 |

## Behavior failures (`behavior_ok` false)

None.

## Retrieval misses (recall@5 = 0)

None.
