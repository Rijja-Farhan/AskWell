# Eval report: probe06_bm25

- Collection `askwell_structured`, embeddings `gemini-embedding-2-preview`, structure chunks 800/100 chars
- Retrieval k = 10 chunks; the answer uses the top 5
- Chat model `gemini-3.6-flash`, judge model `gemini-3.6-flash`
- Retrieval mode `bm25` (BM25 v1, weight 0.5, rrf_k 60, 20 candidates per list)
- Stages done: retrieval 24/24, generate 0/24, judge 0/24
- Deltas in brackets are versus run `probe06_dense`

Retrieval metrics are doc-level: each document counts at its best chunk rank, and k counts chunks. Questions without gold documents (unanswerable, some injection) show `n/a`. `scored` = questions with retrieval metrics, `judged` = questions with answer metrics. Sup@5 = share of conflicting-versions questions where a superseded version reaches the top 5.

Access-denied recall is computed on purpose: until Phase 12 adds access control, a high value means restricted content reaches the LLM.

## Results by category

| Category | n | scored | judged | R@1 | R@3 | R@5 | R@10 | MRR | Sup@5 | Faith | Rel | Behav |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact-term | 24 | 24 | 0 | 0.74 (+0.21) | 0.94 (+0.17) | 0.99 (+0.10) | 1.00 (+0.10) | 0.98 (+0.17) | n/a | n/a | n/a | n/a |
| **overall** | 24 | 24 | 0 | 0.74 (+0.21) | 0.94 (+0.17) | 0.99 (+0.10) | 1.00 (+0.10) | 0.98 (+0.17) | n/a | n/a | n/a | n/a |

## Behavior failures (`behavior_ok` false)

None.

## Retrieval misses (recall@5 = 0)

None.
