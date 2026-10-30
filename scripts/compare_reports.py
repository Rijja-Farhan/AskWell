"""Build eval/reports/FINAL.md: the baseline-to-final comparison of the eval reports.

Usage: python scripts/compare_reports.py [--reports-dir eval/reports] [--out PATH]
       [--names baseline,phase05,...]
Reads report.json (and manifest.json) of each run and writes one Markdown file with, per metric,
a category x report table: recall@k, MRR, faithfulness, answer relevance, refusal rate, and
median/p95 latency per stage. A metric a report does not have renders as "pending", never as 0
(n/a only where a metric cannot exist, e.g. recall on questions with no gold document). Reports
that are missing or incomplete are listed. Existing reports are only read.
Exit codes: 0 written, 1 no report found.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_NAMES = ["baseline", "phase05", "phase06", "phase07", "phase08", "phase09"]
OPTIONAL_NAMES = ["phase12", "phase13"]  # included only when they exist
CATEGORIES = [
    "simple",
    "exact-term",
    "multi-doc",
    "conflicting-versions",
    "unanswerable",
    "access-denied",
    "injection",
    "overall",
]
STAGES = ("retrieval", "generate", "judge")
RETRIEVAL_METRICS = {"recall@1", "recall@3", "recall@5", "recall@10", "mrr"}
SECTIONS = [
    ("Recall@1", "recall@1"),
    ("Recall@3", "recall@3"),
    ("Recall@5", "recall@5"),
    ("Recall@10", "recall@10"),
    ("MRR", "mrr"),
    ("Faithfulness (LLM judge, 0-1)", "faithfulness"),
    ("Answer relevance (LLM judge, 0-1)", "relevance"),
    ("Refusal rate (share of answers that are the fixed refusal)", "refused"),
]


def load_run(reports_dir: Path, name: str) -> dict | None:
    """The run's report.json and manifest.json, or None when the folder does not exist."""
    run = reports_dir / name
    if not run.is_dir():
        return None
    out: dict = {"name": name, "report": None, "manifest": None}
    for key, file in (("report", "report.json"), ("manifest", "manifest.json")):
        path = run / file
        if path.exists():
            try:
                out[key] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    return out


def cell(run: dict | None, category: str, metric: str) -> str:
    report = run["report"] if run else None
    if not report:
        return "pending"
    row = report.get("categories", {}).get(category)
    if row is None:
        return "pending"
    value = row.get(metric)
    if value is None:
        if metric in RETRIEVAL_METRICS and row.get("n_scored") == 0:
            return "n/a"  # no gold document to find (unanswerable questions)
        return "pending"
    return f"{value:.3f}"


def stage_progress(run: dict) -> dict[str, tuple[int, int]]:
    """(done, total) per stage from the report, else the manifest, else zeros."""
    for key in ("report", "manifest"):
        data = run.get(key)
        if data and data.get("stages"):
            return {s: (v["ok"], v["n"]) for s, v in data["stages"].items()}
    return {}


def incomplete_reason(run: dict | None) -> str | None:
    """None when every stage the run covers is finished and it has a report."""
    if run is None:
        return "no run folder"
    if not run["report"]:
        return "no report.json (run was interrupted before a report was written)"
    progress = stage_progress(run)
    gaps = [f"{s} {ok}/{n}" for s, (ok, n) in progress.items() if ok < n]
    if gaps:
        return "unfinished stages: " + ", ".join(gaps)
    if run["report"].get("errors"):
        return f"{len(run['report']['errors'])} question error(s)"
    return None


def answer_stages_run(run: dict | None) -> bool:
    """Whether the run was meant to produce answer metrics (generate or judge was attempted)."""
    if not run:
        return False
    return any(ok for stage, (ok, _) in stage_progress(run).items() if stage != "retrieval")


def latency_cell(run: dict | None, stage: str, key: str) -> str:
    report = run["report"] if run else None
    row = (report or {}).get("latency", {}).get(stage)
    return f"{row[key]:.2f}" if row else "pending"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def config_label(run: dict | None) -> str:
    data = run and (run["report"] or run["manifest"])
    config = (data or {}).get("config", {})
    if not config:
        return "unknown"
    retrieval = config.get("retrieval_mode", "dense")
    if config.get("reranker", "none") != "none":
        retrieval += "+rerank"
    parts = [
        config.get("chunk_strategy", "fixed"),
        retrieval,
        f"versions {config.get('version_policy', 'all')}",
        f"{config.get('answer_mode', 'baseline')} prompt",
    ]
    if config.get("rbac_enabled"):
        parts.append("RBAC")
    if config.get("injection_enabled"):
        parts.append("injection defense")
    return ", ".join(parts)


def build(reports_dir: Path, names: list[str]) -> str:
    runs = {n: load_run(reports_dir, n) for n in names}
    lines = [
        "# Askwell: baseline to final",
        "",
        "Generated by `python scripts/compare_reports.py` from the run folders in "
        f"`{_rel(reports_dir)}/`. Nothing here is typed by hand: a metric a run does not have "
        "shows **pending**, and **n/a** marks a metric that cannot exist (recall on questions "
        "with no gold document). Re-run the script after a pending eval finishes.",
        "",
        "## Runs",
        "",
    ]
    rows = []
    for name in names:
        run = runs[name]
        progress = stage_progress(run) if run else {}
        done = ", ".join(f"{s} {ok}/{n}" for s, (ok, n) in progress.items()) or "-"
        reason = incomplete_reason(run)
        rows.append([f"`{name}`", config_label(run), done, "complete" if not reason else "no"])
    lines += table(["Run", "Configuration", "Stages done", "Complete"], rows)

    lines += ["", "## Incomplete or missing reports", ""]
    incomplete = [(n, incomplete_reason(runs[n])) for n in names if incomplete_reason(runs[n])]
    if incomplete:
        lines += [f"- `{n}`: {why}" for n, why in incomplete]
        lines += [
            "",
            "Retrieval-only runs (`phase05` to `phase07`) have no answer metrics by design; "
            "their answer columns read pending. Finishing the owed runs is described in "
            "`docs/PENDING_CHECKS.md`.",
        ]
    else:
        lines.append("None: every run is complete.")

    for title, metric in SECTIONS:
        lines += ["", f"## {title}", ""]
        body = [[c] + [cell(runs[n], c, metric) for n in names] for c in CATEGORIES]
        lines += table(["Category"] + [f"`{n}`" for n in names], body)

    lines += ["", "## Latency (seconds, per stage over finished questions)", ""]
    body = []
    for stage in ("retrieval", "rerank", "generate"):
        for key, label in (("median", "median"), ("p95", "p95")):
            body.append([f"{stage} {label}"] + [latency_cell(runs[n], stage, key) for n in names])
    lines += table(["Stage"] + [f"`{n}`" for n in names], body)
    lines += [
        "",
        "Retrieval latency includes the embedding call and (from `phase07`) the reranker, "
        "which runs on CPU. Generate latency includes provider retries, so p95 can be large "
        "when the free tier throttles.",
        "",
    ]
    return "\n".join(lines)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "eval/reports")
    parser.add_argument("--out", type=Path, help="default: <reports-dir>/FINAL.md")
    parser.add_argument("--names", help="comma-separated run names (default: baseline..phase09)")
    args = parser.parse_args(argv)
    if args.names:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
    else:
        names = DEFAULT_NAMES + [n for n in OPTIONAL_NAMES if (args.reports_dir / n).is_dir()]
    if not any((args.reports_dir / n).is_dir() for n in names):
        print(f"no report folders found under {args.reports_dir}", file=sys.stderr)
        return 1
    out = args.out or args.reports_dir / "FINAL.md"
    out.write_text(build(args.reports_dir, names), encoding="utf-8")
    print(f"wrote {_rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
