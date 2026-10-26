# Eval report: phase06_bm25

- Collection `askwell_structured`, embeddings `gemini-embedding-2-preview`, structure chunks 800/100 chars
- Retrieval k = 10 chunks; the answer uses the top 5
- Chat model `gemini-3.6-flash`, judge model `gemini-3.6-flash`
- Retrieval mode `bm25` (BM25 v1, weight 0.5, rrf_k 60, 20 candidates per list)
- Stages done: retrieval 80/80, generate 0/80, judge 0/80
- Deltas in brackets are versus run `phase05`

Retrieval metrics are doc-level: each document counts at its best chunk rank, and k counts chunks. Questions without gold documents (unanswerable, some injection) show `n/a`. `scored` = questions with retrieval metrics, `judged` = questions with answer metrics. Sup@5 = share of conflicting-versions questions where a superseded version reaches the top 5.

Access-denied recall is computed on purpose: until Phase 12 adds access control, a high value means restricted content reaches the LLM.

## Results by category

| Category | n | scored | judged | R@1 | R@3 | R@5 | R@10 | MRR | Sup@5 | Faith | Rel | Behav |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple | 16 | 16 | 0 | 1.00 (+0.00) | 1.00 (+0.00) | 1.00 (+0.00) | 1.00 (+0.00) | 1.00 (+0.00) | n/a | n/a | n/a | n/a |
| exact-term | 12 | 12 | 0 | 0.33 (-0.42) | 0.88 (-0.12) | 0.88 (-0.12) | 1.00 (+0.00) | 0.65 (-0.27) | n/a | n/a | n/a | n/a |
| multi-doc | 12 | 12 | 0 | 0.38 (+0.04) | 0.79 (+0.00) | 0.88 (-0.04) | 1.00 (+0.04) | 0.84 (+0.01) | n/a | n/a | n/a | n/a |
| conflicting-versions | 10 | 10 | 0 | 0.20 (-0.20) | 1.00 (+0.00) | 1.00 (+0.00) | 1.00 (+0.00) | 0.55 (-0.13) | 1.00 (+0.00) | n/a | n/a | n/a |
| unanswerable | 10 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| access-denied | 10 | 10 | 0 | 0.80 (+0.10) | 0.90 (-0.10) | 1.00 (+0.00) | 1.00 (+0.00) | 0.87 (+0.02) | n/a | n/a | n/a | n/a |
| injection | 10 | 8 | 0 | 0.62 (-0.12) | 0.94 (+0.00) | 1.00 (+0.00) | 1.00 (+0.00) | 0.79 (-0.06) | n/a | n/a | n/a | n/a |
| **overall** | 80 | 68 | 0 | 0.58 (-0.10) | 0.92 (-0.04) | 0.96 (-0.03) | 1.00 (+0.01) | 0.80 (-0.07) | 1.00 (+0.00) | n/a | n/a | n/a |

## Behavior failures (`behavior_ok` false)

None.

## Retrieval misses (recall@5 = 0)

| id | question | first gold rank |
|---|---|---:|
| q020 | What expenses are coded to GL-6115? | 9 |
