"""Phase 18: the pending-checks runner, the final report and the docs (offline, no network)."""

import json
import re

import pytest

from scripts import compare_reports as cr
from scripts import run_pending_checks as rpc
from tests.conftest import ROOT

LIVE_KINDS = ("judged", "live", "server", "docker")


# ---- runner plan -------------------------------------------------------------------------


def test_plan_has_stable_unique_names_in_the_documented_order():
    names = [s.name for s in rpc.build_plan()]
    assert len(names) == len(set(names))
    assert names == [
        "preflight",
        "access_probe",
        "injection_probe",
        "cache_probe",
        "trace_probe",
        "multiturn_offline",
        "eval_phase08",
        "eval_phase09",
        "eval_phase12",
        "eval_phase13",
        "multiturn_live",
        "ask_repeat",
        "injection_smoke",
        "api_smoke",
        "ui_probe",
        "trace_usage",
        "docker_stack",
        "compare_reports",
    ]


def test_every_step_has_a_command_an_expected_result_and_a_phase():
    for step in rpc.build_plan():
        assert step.command and step.expected and step.phase, step.name
        assert step.args or step.run, step.name


def test_step_scripts_exist():
    for step in rpc.build_plan():
        if step.args:
            assert (ROOT / step.args[0]).is_file(), step.name


def test_judged_evals_resume_with_fixed_names_and_never_fresh():
    judged = [s for s in rpc.build_plan() if s.kind == "judged"]
    assert [s.args[2] for s in judged] == ["phase08", "phase09", "phase12", "phase13"]
    assert all("--fresh" not in s.args for s in judged)
    # phase08/09 predate RBAC and injection defense, so they must run without them
    assert judged[0].env["ANSWER_MODE"] == "baseline" and judged[0].env["RBAC_ENABLED"] == "false"
    assert judged[1].env["ANSWER_MODE"] == "grounded" and judged[1].env["RBAC_ENABLED"] == "false"
    assert judged[2].env["RBAC_ENABLED"] == "true" and judged[2].env["INJECTION_DEFENSE"] == "false"
    assert judged[3].env["INJECTION_DEFENSE"] == "true"


def test_list_prints_every_step_and_runs_nothing(capsys, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("--list must not run anything")

    monkeypatch.setattr(rpc.subprocess, "run", boom)
    assert rpc.main(["--list"]) == 0
    out = capsys.readouterr().out
    for step in rpc.build_plan():
        assert step.name in out and step.command in out and step.expected in out
    assert out.index("preflight") < out.index("eval_phase08") < out.index("compare_reports")


def test_select_only_skip_and_offline_only():
    plan = rpc.build_plan()

    def pick(*argv):
        parser_args = rpc.argparse.Namespace(
            only=None, skip=None, offline_only=False, with_docker=False
        )
        for flag, value in zip(argv[::2], argv[1::2], strict=True):
            setattr(parser_args, flag, value)
        return [s.name for s in rpc.select(plan, parser_args)]

    assert "docker_stack" not in pick()
    assert "docker_stack" in pick("with_docker", True)
    assert pick("only", ["access_probe", "cache_probe"]) == ["access_probe", "cache_probe"]
    assert "access_probe" not in pick("skip", ["access_probe"])
    offline = pick("offline_only", True)
    assert offline[-1] == "compare_reports" and "preflight" not in offline
    assert not any(s.kind in LIVE_KINDS for s in plan if s.name in offline)
    with pytest.raises(SystemExit):
        pick("only", ["nope"])


# ---- classification and exit codes -------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "output", "status"),
    [
        (0, "ok", "pass"),
        (3, "", "pending_provider"),
        (1, "RateLimitError: Error code: 429 - quota", "pending_provider"),
        (1, "503 provider_unavailable", "pending_provider"),
        (2, "GEMINI_API_KEY is not set", "fail"),
        (1, "AssertionError: 3 leaks", "fail"),
        (124, "timed out after 10s", "fail"),
    ],
)
def test_classify(code, output, status):
    assert rpc.classify(code, output)[0] == status


def test_exit_code_logic():
    assert rpc.overall_exit(["pass", "pass"]) == 0
    assert rpc.overall_exit(["pass", "pending_provider"]) == 3
    assert rpc.overall_exit(["pending_provider", "fail", "pass"]) == 1
    assert rpc.overall_exit(["skipped", "pass"]) == 0
    assert rpc.overall_exit(["skipped"]) == 0


def test_secrets_are_redacted():
    text = "key AIzaSyA1234567890abcdefghijklmnopqrstu and sk-abcdefghijklmnopqrstuvwxyz"
    assert "AIza" not in rpc.redact(text) and "sk-abc" not in rpc.redact(text)


def test_dead_model_marks_every_live_step_pending_without_running(tmp_path, monkeypatch):
    monkeypatch.setattr(rpc, "OUT_DIR", tmp_path)
    runner = rpc.Runner()
    runner.live_ok, runner.live_reason = False, "chat model unavailable (RateLimitError 429)"
    monkeypatch.setattr(runner, "manifest_mismatch", lambda *a: None)
    monkeypatch.setattr(runner, "python", lambda *a, **k: pytest.fail("a step ran"))
    records = [runner.execute(s) for s in rpc.build_plan() if s.kind in LIVE_KINDS]
    assert records and all(r["status"] == "pending_provider" for r in records)
    assert rpc.overall_exit([r["status"] for r in records]) == 3


def test_failed_check_and_provider_exit_code_are_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(rpc, "OUT_DIR", tmp_path)
    (tmp_path / "logs").mkdir()
    runner = rpc.Runner()
    steps = {s.name: s for s in rpc.build_plan()}
    monkeypatch.setattr(runner, "python", lambda *a, **k: (3, "provider down"))
    rec = runner.execute(steps["multiturn_live"])
    assert rec["status"] == "pending_provider" and rec["exit_code"] == 3
    monkeypatch.setattr(runner, "python", lambda *a, **k: (0, "[ask 1/2] cache miss"))
    rec = runner.execute(steps["ask_repeat"])
    assert rec["status"] == "fail" and "cache hit" in rec["reason"]


def test_mismatched_judged_run_fails_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(rpc, "OUT_DIR", tmp_path)
    (tmp_path / "logs").mkdir()
    runner = rpc.Runner()
    monkeypatch.setattr(runner, "manifest_mismatch", lambda *a: "filter_version: 1 -> 2")
    monkeypatch.setattr(runner, "python", lambda *a, **k: pytest.fail("must not run"))
    step = next(s for s in rpc.build_plan() if s.name == "eval_phase08")
    rec = runner.execute(step)
    assert rec["status"] == "fail" and "filter_version" in rec["reason"]


def test_status_files_are_written_and_merged(tmp_path, monkeypatch):
    monkeypatch.setattr(rpc, "OUT_DIR", tmp_path)
    plan = rpc.build_plan()
    first = {"access_probe": {**_rec("access_probe"), "status": "pass"}}
    rpc.write_status(plan, first, offline_only=True)
    second = {"cache_probe": {**_rec("cache_probe"), "status": "fail", "reason": "boom"}}
    rows = rpc.write_status(plan, second, offline_only=False)
    by_name = {r["name"]: r for r in rows}
    assert by_name["access_probe"]["status"] == "pass"  # kept from the earlier run
    assert by_name["cache_probe"]["status"] == "fail"
    assert by_name["eval_phase09"]["status"] == "skipped"
    data = json.loads((tmp_path / "status.json").read_text())
    assert data["counts"]["pass"] == 1 and len(data["steps"]) == len(plan)
    assert "cache_probe" in (tmp_path / "status.md").read_text()


def _rec(name):
    return {
        "name": name,
        "phase": "1",
        "kind": "offline",
        "command": "c",
        "expected": "e",
        "exit_code": 0,
        "duration_s": 1.0,
        "reason": "",
        "finished_at": None,
    }


# ---- compare_reports ---------------------------------------------------------------------


def _write_run(root, name, report=True, stages=None):
    run = root / name
    run.mkdir(parents=True)
    config = {"chunk_strategy": "structure", "retrieval_mode": "hybrid", "answer_mode": "grounded"}
    (run / "manifest.json").write_text(json.dumps({"name": name, "config": config}))
    if not report:
        return
    row = {
        "n": 10,
        "n_scored": 10,
        "recall@1": 0.5,
        "recall@3": 0.7,
        "recall@5": 0.8,
        "recall@10": 0.9,
        "mrr": 0.6123,
        "faithfulness": 0.95,
        "relevance": 0.9,
        "refused": None,
    }
    empty = {k: (None if k != "n_scored" else 0) for k in row} | {"n": 5}
    stages = stages or {"retrieval": {"ok": 15, "n": 15}}
    data = {
        "name": name,
        "config": config,
        "stages": stages,
        "categories": {"simple": row, "unanswerable": empty, "overall": row},
        "latency": {"retrieval": {"n": 15, "median": 1.5, "p95": 3.25}},
        "errors": [],
    }
    (run / "report.json").write_text(json.dumps(data))


def test_compare_reports_renders_numbers_and_pending(tmp_path):
    _write_run(tmp_path, "done")
    _write_run(tmp_path, "partial", report=False)
    _write_run(
        tmp_path, "cut", stages={"retrieval": {"ok": 15, "n": 15}, "generate": {"ok": 2, "n": 15}}
    )
    text = cr.build(tmp_path, ["done", "partial", "cut", "ghost"])
    assert "0.612" in text and "1.50" in text and "3.25" in text
    simple_mrr = next(ln for ln in text.splitlines() if ln.startswith("| simple | 0.612"))
    assert simple_mrr.count("pending") == 2  # the run with no report and the missing folder
    refused = text.split("## Refusal rate")[1]
    assert "pending" in refused and "| simple | pending |" in refused  # never 0
    assert "| unanswerable | n/a |" in text.split("## Recall@1")[1]
    incomplete = text.split("## Incomplete or missing reports")[1].split("##")[0]
    assert "`partial`" in incomplete and "`ghost`" in incomplete and "generate 2/15" in incomplete
    assert "`done`" not in incomplete


def test_compare_reports_cli_writes_final_md(tmp_path):
    _write_run(tmp_path, "baseline")
    assert cr.main(["--reports-dir", str(tmp_path), "--names", "baseline,phase05"]) == 0
    assert (tmp_path / "FINAL.md").read_text().startswith("# Askwell")
    assert cr.main(["--reports-dir", str(tmp_path / "empty"), "--names", "x"]) == 1


def test_real_reports_give_a_final_md_without_invented_numbers():
    text = cr.build(ROOT / "eval/reports", cr.DEFAULT_NAMES)
    assert "`phase09`" in text and "pending" in text
    phase09 = ROOT / "eval/reports/phase09/report.json"
    if not phase09.exists():  # phase09 never finished: its column must be all pending
        row = next(ln for ln in text.splitlines() if ln.startswith("| overall |"))
        assert row.split("|")[-2].strip() == "pending"


# ---- docs reference only things that exist -----------------------------------------------

DOC_FILES = [
    "README.md",
    *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").glob("*.md")),
]
PATH_RE = re.compile(r"(?<![\w/.-])((?:scripts|docs|eval|app|data|tests|Notes)/[\w./*<>-]+)")
LINK_RE = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")
CREATED_AT_RUNTIME = ("eval/pending_checks", "eval/probes/results/ui_probe.json")


def _doc_text(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_the_three_docs_exist():
    for name in ("PENDING_CHECKS.md", "ARCHITECTURE.md", "DEMO.md"):
        assert (ROOT / "docs" / name).is_file()
    readme = _doc_text("README.md")
    for name in ("PENDING_CHECKS", "ARCHITECTURE", "DEMO"):
        assert f"docs/{name}.md" in readme


@pytest.mark.parametrize("doc", DOC_FILES)
def test_referenced_paths_exist(doc):
    text = _doc_text(doc)
    missing = []
    for raw in PATH_RE.findall(text):
        path = raw.rstrip(".,:;)`")
        if "<" in path or "*" in path or path.startswith(CREATED_AT_RUNTIME):
            continue
        if path.startswith("eval/reports/") and path.count("/") > 2:
            continue  # per-run files exist only for runs that were made
        if not (ROOT / path).exists():
            missing.append(path)
    assert not missing, f"{doc} references missing paths: {sorted(set(missing))}"


@pytest.mark.parametrize("doc", DOC_FILES)
def test_relative_links_resolve(doc):
    base = (ROOT / doc).parent
    bad = []
    for target in LINK_RE.findall(_doc_text(doc)):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (base / target).resolve().exists():
            bad.append(target)
    assert not bad, f"{doc} has broken links: {bad}"


def test_demo_users_and_questions_exist():
    users = {u["user_id"] for u in json.loads((ROOT / "data/metadata/users.json").read_text())}
    questions = {
        json.loads(line)["question"]
        for line in (ROOT / "eval/eval_set.jsonl").read_text().splitlines()
        if line.strip()
    }
    demo = _doc_text("docs/DEMO.md")
    assert set(re.findall(r"\bu\d{2}\b", demo)) <= users
    quoted = re.findall(r'"([A-Z][^"\n]{15,}\?)"', demo)
    assert quoted
    known = questions | {
        json.loads(line)["turns"][i]["question"]
        for line in (ROOT / "eval/probes/multiturn.jsonl").read_text().splitlines()
        if line.strip()
        for i in range(2)
    }
    assert [q for q in quoted if q not in known] == []
