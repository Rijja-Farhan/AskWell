"""Aggregate per-question results into `report.json` and `report.md`, optionally with deltas."""

import json
import math
from pathlib import Path
from typing import get_args

from app.dataset.schemas import Category
from app.evaluation.metrics import first_gold_rank

RETRIEVAL_KEYS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr", "superseded_hit@5")
ANSWER_KEYS = ("faithfulness", "relevance", "behavior_pass", "refused", "cited")
METRIC_KEYS = RETRIEVAL_KEYS + ANSWER_KEYS
STAGE_NAMES = ("retrieval", "generate", "judge")
LATENCY_STAGES = ("retrieval", "rerank", "generate", "judge")  # rerank is part of retrieval
COLUMNS = {
    "recall@1": "R@1",
    "recall@3": "R@3",
    "recall@5": "R@5",
    "recall@10": "R@10",
    "mrr": "MRR",
    "superseded_hit@5": "Sup@5",
    "faithfulness": "Faith",
    "relevance": "Rel",
    "behavior_pass": "Behav",
    "refused": "Refused",
    "cited": "Cited",
}


def mean(values: list) -> float | None:
    values = [float(v) for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def summarize(results: list) -> dict:
    retrieved = [r for r in results if r.ok("retrieval")]
    judged = [r for r in results if r.ok("judge") and r.judge is not None]
    row: dict = {
        "n": len(results),
        "n_scored": sum(r.metrics.get("recall@1") is not None for r in retrieved),
        "n_answered": sum(r.ok("generate") for r in results),
        "n_judged": len(judged),
    }
    for key in RETRIEVAL_KEYS:
        row[key] = mean([r.metrics.get(key) for r in retrieved])
    row["faithfulness"] = mean([r.judge.faithfulness for r in judged])
    row["relevance"] = mean([r.judge.relevance for r in judged])
    row["behavior_pass"] = mean([r.judge.behavior_ok for r in judged])
    # Grounded answers only: share refused, and share of non-refused with a valid citation
    grounded = [r for r in results if r.ok("generate") and r.refused is not None]
    row["refused"] = mean([r.refused for r in grounded])
    row["cited"] = mean([bool(r.citations) for r in grounded if not r.refused])
    return row


def summarize_answers(results: list) -> dict:
    """Refusal details: false refusals on `answer` questions, refusals by reason, and the
    invalid citation markers removed. Empty when no answer was made in grounded mode."""
    grounded = [r for r in results if r.ok("generate") and r.refused is not None]
    if not grounded:
        return {}
    answerable = [r for r in results if r.expected_behavior == "answer"]
    return {
        "false_refusals": {
            "n": sum(bool(r.refused) for r in answerable if r.ok("generate")),
            "of": len(answerable),
            "ids": [r.id for r in answerable if r.refused],
        },
        "refusals": [
            {"id": r.id, "category": r.category, "reason": r.refusal_reason}
            for r in grounded
            if r.refused
        ],
        "invalid_citations": sum(r.invalid_citations or 0 for r in grounded),
    }


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: the smallest value with at least p% of values at or below it."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def summarize_latency(results: list) -> dict:
    """Median and p95 seconds per stage, over questions whose stage is `ok`. Throttle waits
    are excluded; provider retries (e.g. 429 backoff) are not, which is why p95 matters."""
    latency = {}
    for stage in LATENCY_STAGES:
        source = "retrieval" if stage == "rerank" else stage
        values = [r.latency[stage] for r in results if r.ok(source) and stage in r.latency]
        if values:
            latency[stage] = {
                "n": len(values),
                "median": round(percentile(values, 50), 3),
                "p95": round(percentile(values, 95), 3),
            }
    return latency


def build_report(
    name: str, results: list, config: dict, reports_dir: Path, compare: str | None = None
) -> dict:
    by_category = {
        cat: summarize([r for r in results if r.category == cat]) for cat in get_args(Category)
    }
    by_category = {cat: row for cat, row in by_category.items() if row["n"]}
    by_category["overall"] = summarize(results)
    report: dict = {
        "name": name,
        "config": config,
        "stages": {
            s: {"ok": sum(r.ok(s) for r in results), "n": len(results)} for s in STAGE_NAMES
        },
        "categories": by_category,
        "latency": summarize_latency(results),
        "answers": summarize_answers(results),
        "behavior_failures": [
            {"id": r.id, "category": r.category, "reason": r.judge.reason}
            for r in results
            if r.ok("judge") and r.judge is not None and not r.judge.behavior_ok
        ],
        "retrieval_misses": [
            {
                "id": r.id,
                "question": r.question,
                "first_gold_rank": first_gold_rank(r.gold_doc_ids, r.doc_ranks),
            }
            for r in results
            if r.ok("retrieval") and r.metrics.get("recall@5") == 0
        ],
        "errors": [
            {"id": r.id, "stage": stage, "message": status.message}
            for r in results
            for stage, status in r.status.items()
            if status.state == "error"
        ],
    }
    if compare:
        report["comparison"] = compare_reports(report, load_report(reports_dir / compare, compare))
    return report


def load_report(run_dir: Path, name: str) -> dict:
    path = run_dir / "report.json"
    if not path.exists():
        raise FileNotFoundError(f"no report for run {name!r} at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def compare_reports(report: dict, other: dict) -> dict:
    """Deltas (this run minus the other) for every metric both runs have."""
    deltas = {}
    for cat, row in report["categories"].items():
        before = other["categories"].get(cat, {})
        deltas[cat] = {
            key: round(row[key] - before[key], 4)
            if row.get(key) is not None and before.get(key) is not None
            else None
            for key in METRIC_KEYS
        }
    before = other.get("latency", {})  # reports made before Phase 7 have no latency
    latency = {
        stage: {key: round(row[key] - before[stage][key], 3) for key in ("median", "p95")}
        for stage, row in report.get("latency", {}).items()
        if stage in before
    }
    return {"against": other["name"], "deltas": deltas, "latency": latency}


def write_report(run_dir: Path, report: dict) -> None:
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def retrieval_label(config: dict) -> str:
    mode = config.get("retrieval_mode", "dense")
    label = f"`{mode}`"
    if mode != "dense":
        label += (
            f" (BM25 v{config['bm25_version']}, weight {config['bm25_weight']}, "
            f"rrf_k {config['rrf_k']}, {config['hybrid_candidates']} candidates per list)"
        )
    if config.get("reranker", "none") != "none":
        label += (
            f", reranked by `{config['rerank_model']}` (v{config['rerank_version']}, "
            f"top {config['rerank_candidates']} candidates)"
        )
    policy = config.get("version_policy", "all")
    if policy != "all":
        label += f", versions `{policy}` (filter v{config['filter_version']})"
    return label


def render_markdown(report: dict) -> str:
    config = report["config"]
    comparison = report.get("comparison")
    lines = [
        f"# Eval report: {report['name']}",
        "",
        f"- Collection `{config['collection']}`, embeddings `{config['embedding_model']}`, "
        f"{config.get('chunk_strategy', 'fixed')} chunks "
        f"{config['chunk_size']}/{config['chunk_overlap']} chars",
        f"- Retrieval k = {config['eval_retrieval_k']} chunks; the answer uses the top "
        f"{config['retrieval_top_k']}",
        f"- Chat model `{config['chat_model']}`, judge model `{config['judge_model']}`",
        f"- Retrieval mode {retrieval_label(config)}",
        f"- Answer mode `{config.get('answer_mode', 'baseline')}`"
        + (
            f" (v{config['answer_version']}, refusal gate {config['refusal_min_score']})"
            if config.get("answer_mode") == "grounded"
            else ""
        ),
        "- Stages done: "
        + ", ".join(f"{s} {v['ok']}/{v['n']}" for s, v in report["stages"].items()),
    ]
    if comparison:
        lines.append(f"- Deltas in brackets are versus run `{comparison['against']}`")
    lines += [
        "",
        "Retrieval metrics are doc-level: each document counts at its best chunk rank, and k "
        "counts chunks. Questions without gold documents (unanswerable, some injection) show "
        "`n/a`. `scored` = questions with retrieval metrics, `judged` = questions with answer "
        "metrics. Sup@5 = share of conflicting-versions questions where a superseded version "
        "reaches the top 5.",
        "",
        "Access-denied recall is computed on purpose: until Phase 12 adds access control, a high "
        "value means restricted content reaches the LLM.",
        "",
        "## Results by category",
        "",
        "| Category | n | scored | judged | " + " | ".join(COLUMNS.values()) + " |",
        "|---|---:|---:|---:|" + "---:|" * len(COLUMNS),
    ]
    for cat, row in report["categories"].items():
        values = []
        for key in COLUMNS:
            text = fmt(row[key])
            delta = comparison["deltas"].get(cat, {}).get(key) if comparison else None
            if delta is not None:
                text += f" ({delta:+.2f})"
            values.append(text)
        label = f"**{cat}**" if cat == "overall" else cat
        lines.append(
            f"| {label} | {row['n']} | {row['n_scored']} | {row['n_judged']} | "
            + " | ".join(values)
            + " |"
        )

    answers = report.get("answers")
    if answers:
        fr = answers["false_refusals"]
        lines += [
            "",
            "## Refusals",
            "",
            f"`Refused` = share of answers that are the fixed refusal reply; `Cited` = share of "
            f"non-refused answers with at least one valid citation. False refusals on `answer` "
            f"questions: {fr['n']} of {fr['of']}"
            + (f" ({', '.join(fr['ids'])})" if fr["ids"] else "")
            + f". Invalid citation markers removed: {answers['invalid_citations']}.",
            "",
        ]
        if answers["refusals"]:
            lines += ["| id | category | reason |", "|---|---|---|"]
            lines += [
                f"| {x['id']} | {x['category']} | {x['reason']} |" for x in answers["refusals"]
            ]
        else:
            lines.append("No refusals.")

    lines += ["", "## Latency (seconds per question)", ""]
    if report.get("latency"):
        deltas = comparison.get("latency", {}) if comparison else {}
        lines += [
            "Throttle waits are excluded; provider retries are not. `rerank` is part of "
            "`retrieval`.",
            "",
            "| Stage | n | median | p95 |",
            "|---|---:|---:|---:|",
        ]
        for stage, row in report["latency"].items():
            cells = []
            for key in ("median", "p95"):
                text = f"{row[key]:.3f}"
                if stage in deltas:
                    text += f" ({deltas[stage][key]:+.3f})"
                cells.append(text)
            lines.append(f"| {stage} | {row['n']} | " + " | ".join(cells) + " |")
    else:
        lines.append("None recorded.")

    lines += ["", "## Behavior failures (`behavior_ok` false)", ""]
    if report["behavior_failures"]:
        lines += ["| id | category | judge reason |", "|---|---|---|"]
        lines += [
            f"| {f['id']} | {f['category']} | {cell(f['reason'])} |"
            for f in report["behavior_failures"]
        ]
    else:
        lines.append("None.")

    lines += ["", "## Retrieval misses (recall@5 = 0)", ""]
    if report["retrieval_misses"]:
        k = config["eval_retrieval_k"]
        lines += ["| id | question | first gold rank |", "|---|---|---:|"]
        lines += [
            f"| {m['id']} | {cell(m['question'])} | "
            f"{m['first_gold_rank'] if m['first_gold_rank'] else f'> {k}'} |"
            for m in report["retrieval_misses"]
        ]
    else:
        lines.append("None.")

    if report["errors"]:
        lines += ["", "## Errors", "", "| id | stage | message |", "|---|---|---|"]
        lines += [f"| {e['id']} | {e['stage']} | {cell(e['message'])} |" for e in report["errors"]]
    return "\n".join(lines) + "\n"
