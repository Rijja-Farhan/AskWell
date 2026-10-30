"""UI probe (Phase 16): replay one question as two users through /chat/stream of a running server.

Usage: python scripts/run_ui_probe.py [--base-url http://127.0.0.1:8000] [--users u05 u06]
                                       [--question "..."]
Pure HTTP (httpx), no browser. Needs a running server (python scripts/serve.py) with a working
chat model: it makes real LLM calls. Writes eval/probes/results/ui_probe.json with, per user,
the sources, refusal flag, cached flag and trace id. Exit codes: 0 done, 1 a request failed,
2 server unreachable.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

OUT = ROOT / "eval/probes/results/ui_probe.json"
QUESTION = "What is the salary range for grade G5?"


def parse_sse(lines) -> list[tuple[str, dict]]:
    events, event, data = [], "message", []
    for line in lines:
        if line == "":
            if data:
                events.append((event, json.loads("\n".join(data))))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        events.append((event, json.loads("\n".join(data))))
    return events


def ask(client: httpx.Client, user_id: str | None, question: str) -> dict:
    headers = {"X-User-Id": user_id} if user_id else {}
    body = {"question": question, "include_text": False}
    with client.stream("POST", "/chat/stream", json=body, headers=headers) as res:
        if res.status_code != 200:
            res.read()
            return {"user_id": user_id, "http_status": res.status_code, "body": res.text[:300]}
        events = parse_sse(res.iter_lines())
    done = next((d for e, d in events if e == "done"), None)
    sources = next((d["sources"] for e, d in events if e == "sources"), [])
    error = next((d for e, d in events if e == "error"), None)
    return {
        "user_id": user_id,
        "http_status": 200,
        "sources": [
            {k: s[k] for k in ("rank", "doc_id", "title", "version", "status")} for s in sources
        ],
        "refused": done["refused"] if done else None,
        "refusal_reason": done.get("refusal_reason") if done else None,
        "cached": done.get("cached") if done else None,
        "citations": [c["doc_id"] for c in done["citations"]] if done else [],
        "trace_id": done.get("trace_id") if done else None,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", nargs="+", default=["u05", "u06"])
    parser.add_argument("--question", default=QUESTION)
    args = parser.parse_args()
    results = []
    try:
        with httpx.Client(base_url=args.base_url, timeout=120) as client:
            for user_id in args.users:
                results.append(ask(client, user_id, args.question))
    except httpx.ConnectError:
        print(f"cannot reach {args.base_url}; start it with: python scripts/serve.py")
        return 2
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"question": args.question, "results": results}, indent=2) + "\n")
    for r in results:
        docs = sorted({s["doc_id"] for s in r.get("sources", [])})
        print(r["user_id"], r["http_status"], "refused=", r.get("refused"), "sources=", docs)
    print("wrote", OUT.relative_to(ROOT))
    ok = all(r["http_status"] == 200 and not r.get("error") for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
