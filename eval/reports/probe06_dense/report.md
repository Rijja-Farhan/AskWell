# Eval report: probe06_dense

- Collection `askwell_structured`, embeddings `gemini-embedding-2-preview`, structure chunks 800/100 chars
- Retrieval k = 10 chunks; the answer uses the top 5
- Chat model `gemini-3.6-flash`, judge model `gemini-3.6-flash`
- Retrieval mode `dense`
- Stages done: retrieval 24/24, generate 0/24, judge 0/24

Retrieval metrics are doc-level: each document counts at its best chunk rank, and k counts chunks. Questions without gold documents (unanswerable, some injection) show `n/a`. `scored` = questions with retrieval metrics, `judged` = questions with answer metrics. Sup@5 = share of conflicting-versions questions where a superseded version reaches the top 5.

Access-denied recall is computed on purpose: until Phase 12 adds access control, a high value means restricted content reaches the LLM.

## Results by category

| Category | n | scored | judged | R@1 | R@3 | R@5 | R@10 | MRR | Sup@5 | Faith | Rel | Behav |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact-term | 24 | 24 | 0 | 0.53 | 0.76 | 0.89 | 0.90 | 0.81 | n/a | n/a | n/a | n/a |
| **overall** | 24 | 24 | 0 | 0.53 | 0.76 | 0.89 | 0.90 | 0.81 | n/a | n/a | n/a | n/a |

## Behavior failures (`behavior_ok` false)

None.

## Retrieval misses (recall@5 = 0)

| id | question | first gold rank |
|---|---|---:|
| q106 | IT-MAN-005 | > 10 |
| q121 | What is HR-POL-003? | > 10 |
