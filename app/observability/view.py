"""Render a trace: a self-contained HTML waterfall (no scripts, no external assets, every value
HTML-escaped) and an indented text waterfall for the CLI."""

import json
from html import escape

from app.observability.trace import Span, Trace

CSS = """
body{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#1b1f23;background:#fff}
h1{font-size:18px;margin:0 0 4px}h2{font-size:15px;margin:22px 0 6px}
.meta{color:#57606a;margin-bottom:12px}.err{color:#b42318;font-weight:600}
table{border-collapse:collapse;width:100%}td,th{padding:3px 8px;text-align:left;
vertical-align:top;border-bottom:1px solid #eaecef}
.bar{position:relative;height:14px;background:#f1f3f5;min-width:240px}
.bar i{position:absolute;top:0;height:14px;background:#2f6fed;min-width:2px}
.bar i.error{background:#d92d20}code{font:12px ui-monospace,monospace;color:#3b4252}
@media (prefers-color-scheme:dark){body{background:#0d1117;color:#e6edf3}
.meta{color:#9ba7b4}td,th{border-color:#30363d}.bar{background:#21262d}code{color:#c9d1d9}}
"""


def _depths(spans: list[Span]) -> dict[int, int]:
    depth: dict[int, int] = {}
    for s in spans:
        depth[s.id] = 0 if s.parent is None else depth.get(s.parent, 0) + 1
    return depth


def _attrs_text(attrs: dict) -> str:
    return ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in attrs.items())


def _public_attrs(span: Span) -> dict:
    return {k: v for k, v in span.attrs.items() if k != "chunks"}


def render_html(trace: Trace) -> str:
    total = max(trace.total_ms, 0.001)
    depth = _depths(trace.spans)
    rows = []
    for s in trace.spans:
        left = min(100.0, max(0.0, s.start_ms / total * 100))
        width = min(100.0 - left, max(0.0, s.duration_ms / total * 100))
        err = (
            f' <span class="err">{escape(s.error.type)} / {escape(s.error.code)}</span>'
            if (s.error)
            else ""
        )
        rows.append(
            f"<tr><td style='padding-left:{8 + 16 * depth[s.id]}px'>{escape(s.name)}{err}</td>"
            f"<td>{s.start_ms:.1f}</td><td>{s.duration_ms:.1f}</td>"
            f"<td><div class='bar'><i class='{escape(s.status)}' "
            f"style='left:{left:.2f}%;width:{width:.2f}%'></i></div></td>"
            f"<td><code>{escape(_attrs_text(_public_attrs(s)))}</code></td></tr>"
        )
    chunk_rows = []
    for s in trace.spans:
        for c in s.attrs.get("chunks") or []:
            if isinstance(c, dict):
                cells = "".join(
                    f"<td>{escape(str(c.get(k, '')))}</td>"
                    for k in ("rank", "chunk_id", "doc_id", "score", "version", "status")
                )
                chunk_rows.append(f"<tr><td>{escape(s.name)}</td>{cells}</tr>")
    a = trace.attrs
    usage = (
        f"prompt {escape(str(a.get('prompt_tokens', 0)))} + completion "
        f"{escape(str(a.get('completion_tokens', 0)))} tokens over "
        f"{escape(str(a.get('llm_calls', 0)))} LLM call(s)"
    )
    error = (
        f"<p class='err'>Error: {escape(trace.error.type)} / {escape(trace.error.code)}</p>"
        if trace.error
        else ""
    )
    summary = escape(_attrs_text(a))
    chunks_html = (
        "<h2>Retrieved chunks</h2><table><tr><th>span</th><th>rank</th><th>chunk</th><th>doc</th>"
        "<th>score</th><th>version</th><th>status</th></tr>" + "".join(chunk_rows) + "</table>"
        if chunk_rows
        else ""
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Trace {escape(trace.trace_id[:12])}</title><style>{CSS}</style></head><body>"
        f"<h1>Trace {escape(trace.trace_id)}</h1>"
        f"<div class='meta'>{escape(trace.endpoint)} &middot; user {escape(str(trace.user_id))}"
        f" &middot; session {escape(str(trace.session_id))} &middot; {escape(trace.started_at)}"
        f" &middot; status <b>{escape(trace.status)}</b> &middot; {trace.total_ms:.1f} ms</div>"
        f"{error}<p>{usage}</p><p><code>{summary}</code></p>"
        "<h2>Spans</h2><table><tr><th>span</th><th>start ms</th><th>ms</th><th>timeline</th>"
        f"<th>attributes</th></tr>{''.join(rows)}</table>{chunks_html}</body></html>"
    )


def render_text(trace: Trace, width: int = 40) -> str:
    total = max(trace.total_ms, 0.001)
    depth = _depths(trace.spans)
    err = f"  ERROR {trace.error.type}/{trace.error.code}" if trace.error else ""
    lines = [
        f"trace {trace.trace_id}  {trace.endpoint}  user={trace.user_id}  "
        f"session={trace.session_id}  {trace.started_at}",
        f"status={trace.status}  total={trace.total_ms:.1f} ms{err}",
        f"summary: {_attrs_text(trace.attrs)}",
        "",
    ]
    chunk_lines: list[str] = []
    for s in trace.spans:
        left = int(s.start_ms / total * width)
        size = max(1, int(s.duration_ms / total * width))
        bar = " " * left + ("!" if s.status == "error" else "#") * size
        label = "  " * depth[s.id] + s.name
        lines.append(f"{label:<22}{s.start_ms:>9.1f}{s.duration_ms:>9.1f}  |{bar:<{width + 1}}|")
        attrs = _attrs_text(_public_attrs(s))
        if attrs:
            lines.append("  " * depth[s.id] + f"  {attrs}")
        if s.error:
            lines.append("  " * depth[s.id] + f"  error: {s.error.type} / {s.error.code}")
        for c in s.attrs.get("chunks") or []:
            if isinstance(c, dict):
                chunk_lines.append(
                    f"  [{s.name}] #{c.get('rank')} {c.get('chunk_id')} doc={c.get('doc_id')} "
                    f"score={c.get('score')} v={c.get('version')} {c.get('status')}"
                )
    if chunk_lines:
        lines += ["", "chunks:", *chunk_lines]
    return "\n".join(lines)
