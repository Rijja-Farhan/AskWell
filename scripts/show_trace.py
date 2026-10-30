"""Inspect stored traces (Phase 15).

Usage:
  python scripts/show_trace.py --list [--limit N] [--dir traces]
  python scripts/show_trace.py <trace_id> [--dir traces]

Reads the JSONL files written by the server (`TRACES_DIR`, default `traces/`). `--list` prints
one row per trace, newest first; a trace id (or a unique prefix of at least 6 characters)
prints an indented text waterfall with attributes and the retrieved chunks.
Exit codes: 0 ok, 1 trace not found, 2 setup problem (no traces directory).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.observability import JsonlTraceStore  # noqa: E402
from app.observability.view import render_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("trace_id", nargs="?", help="trace id (or unique prefix) to print")
    parser.add_argument("--list", action="store_true", help="list recent traces")
    parser.add_argument("--limit", type=int, default=20, help="rows for --list (default 20)")
    parser.add_argument("--dir", type=Path, help="traces directory (default: TRACES_DIR)")
    args = parser.parse_args()
    if not args.list and not args.trace_id:
        parser.error("give a trace id or --list")

    directory = args.dir or get_settings().traces_dir
    if not directory.is_absolute():
        directory = ROOT / directory
    if not directory.is_dir():
        print(f"no traces directory at {directory}", file=sys.stderr)
        return 2
    store = JsonlTraceStore(directory)

    if args.list:
        print(
            f"{'time (UTC)':<20}{'trace_id':<34}{'endpoint':<14}{'user':<8}{'status':<7}"
            f"{'total_ms':>9}{'tokens':>8}  {'cached':<7}{'refused'}"
        )
        for t in store.list(args.limit):
            a = t.attrs
            tokens = int(a.get("prompt_tokens") or 0) + int(a.get("completion_tokens") or 0)
            print(
                f"{t.started_at[:19].replace('T', ' '):<20}{t.trace_id:<34}{t.endpoint:<14}"
                f"{str(t.user_id):<8}{t.status:<7}{t.total_ms:>9.1f}{tokens:>8}  "
                f"{str(a.get('cached', False)):<7}{a.get('refused', False)}"
            )
        return 0

    trace = store.get(args.trace_id)
    if trace is None and len(args.trace_id) >= 6:
        matches = [t for t in store.iter_traces() if t.trace_id.startswith(args.trace_id)]
        trace = matches[0] if len(matches) == 1 else None
    if trace is None:
        print(f"trace {args.trace_id} not found in {directory}", file=sys.stderr)
        return 1
    print(render_text(trace))
    return 0


if __name__ == "__main__":
    sys.exit(main())
