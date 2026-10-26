# Eval report: baseline

- Collection `askwell_baseline`, embeddings `gemini-embedding-2-preview`, fixed chunks 800/100 chars
- Retrieval k = 10 chunks; the answer uses the top 5
- Chat model `gemini-3-flash-preview`, judge model `gemini-3-flash-preview`
- Retrieval mode `dense`
- Stages done: retrieval 80/80, generate 3/80, judge 0/80

Retrieval metrics are doc-level: each document counts at its best chunk rank, and k counts chunks. Questions without gold documents (unanswerable, some injection) show `n/a`. `scored` = questions with retrieval metrics, `judged` = questions with answer metrics. Sup@5 = share of conflicting-versions questions where a superseded version reaches the top 5.

Access-denied recall is computed on purpose: until Phase 12 adds access control, a high value means restricted content reaches the LLM.

## Results by category

| Category | n | scored | judged | R@1 | R@3 | R@5 | R@10 | MRR | Sup@5 | Faith | Rel | Behav |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple | 16 | 16 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a |
| exact-term | 12 | 12 | 0 | 0.75 | 1.00 | 1.00 | 1.00 | 0.90 | n/a | n/a | n/a | n/a |
| multi-doc | 12 | 12 | 0 | 0.33 | 0.71 | 0.88 | 0.96 | 0.79 | n/a | n/a | n/a | n/a |
| conflicting-versions | 10 | 10 | 0 | 0.20 | 0.70 | 0.90 | 1.00 | 0.45 | 0.90 | n/a | n/a | n/a |
| unanswerable | 10 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| access-denied | 10 | 10 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a |
| injection | 10 | 8 | 0 | 0.81 | 0.94 | 1.00 | 1.00 | 0.92 | n/a | n/a | n/a | n/a |
| **overall** | 80 | 68 | 0 | 0.70 | 0.90 | 0.96 | 0.99 | 0.86 | 0.90 | n/a | n/a | n/a |

## Behavior failures (`behavior_ok` false)

None.

## Retrieval misses (recall@5 = 0)

| id | question | first gold rank |
|---|---|---:|
| q044 | When does the peak season PTO blackout start for warehouse staff? | 7 |
