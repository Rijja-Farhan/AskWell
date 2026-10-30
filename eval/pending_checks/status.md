# Pending checks status

Updated 2026-10-30T14:39:35+00:00 by `python scripts/run_pending_checks.py`. pass 6, fail 0, pending_provider 1, skipped 11.

| Step | Phase | Status | Exit | Seconds | Reason |
|---|---|---|---|---|---|
| `preflight` | 18 | pending_provider | - | 3.2 | chat call to gemini-3.6-flash failed: RateLimitError 429 |
| `access_probe` | 12 | pass | 0 | 49.4 | exit 0 |
| `injection_probe` | 13 | pass | 0 | 2.4 | exit 0 |
| `cache_probe` | 14 | pass | 0 | 7.6 | exit 0 |
| `trace_probe` | 15 | pass | 0 | 10.5 | exit 0 |
| `multiturn_offline` | 11 | pass | 0 | 6.5 | exit 0 |
| `eval_phase08` | 8 | skipped | - | 0 | not run yet |
| `eval_phase09` | 9 | skipped | - | 0 | not run yet |
| `eval_phase12` | 12 | skipped | - | 0 | not run yet |
| `eval_phase13` | 13 | skipped | - | 0 | not run yet |
| `multiturn_live` | 11 | skipped | - | 0 | not run yet |
| `ask_repeat` | 14 | skipped | - | 0 | not run yet |
| `injection_smoke` | 13 | skipped | - | 0 | not run yet |
| `api_smoke` | 10-12 | skipped | - | 0 | not run yet |
| `ui_probe` | 16 | skipped | - | 0 | not run yet |
| `trace_usage` | 15 | skipped | - | 0 | not run yet |
| `docker_stack` | 17 | skipped | - | 0 | not run yet |
| `compare_reports` | 18 | pass | 0 | 0.0 | exit 0 |
