"""Generate the synthetic Nimbus Logistics dataset.

Reads hand-written content from `scripts/dataset_content/` and (re)writes:
- corpus files in `data/raw/<dept>/<doc_id>.<ext>` (pdf, docx, md, txt, html, csv)
- `data/metadata/documents.json` and `data/metadata/users.json`
- `eval/eval_set.jsonl`

No network or LLM calls: two runs produce byte-identical files (including PDF and DOCX).

Content file format (`dataset_content/<dept>/<doc_id>.txt`): a `---` front-matter block of
`key: value` lines, then a body in light markdown: `## h2`, `### h3`, paragraphs, `- ` bullets,
`1. ` numbered items, ``` fenced code, and `| a | b |` pipe tables (first row = header).
"""

import csv
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from html import escape
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docx import Document  # noqa: E402
from docx.shared import Pt  # noqa: E402
from reportlab import rl_config  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import LETTER  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.config import get_settings  # noqa: E402
from app.dataset.schemas import (  # noqa: E402
    DEPT_DIRS,
    DocumentMeta,
    EvalQuestion,
    UserProfile,
)
from app.dataset.validate import PREFIX_DEPT, PREFIX_TYPE  # noqa: E402

CONTENT_DIR = ROOT / "scripts" / "dataset_content"
KEY_TERM_RE = re.compile(
    r"\b(?:(?:HR|IT|OPS|FIN)-(?:POL|MAN|MTG|FAQ|TBL)-\d{3}|NX-\d{4}|GL-\d{4}|WH-[A-Z]{3}-\d{2})\b"
)
FIXED_ZIP_TIME = (2024, 1, 1, 0, 0, 0)
rl_config.invariant = 1  # no timestamps or random ids in PDFs

Block = tuple[str, object]


@dataclass
class Source:
    doc_id: str
    front: dict[str, str]
    blocks: list[Block]

    @property
    def family_id(self) -> str:
        return self.doc_id.rsplit("-v", 1)[0]

    @property
    def version(self) -> int:
        return int(self.doc_id.rsplit("-v", 1)[1])

    @property
    def code(self) -> str:
        return self.family_id.upper()


# ---------------------------------------------------------------- parsing


def parse_source(path: Path) -> Source:
    text = path.read_text(encoding="utf-8")
    _, front_text, body = text.split("---\n", 2)
    front = {}
    for line in front_text.strip().splitlines():
        key, value = line.split(":", 1)
        front[key.strip()] = value.strip()
    return Source(doc_id=path.stem, front=front, blocks=parse_body(body))


def parse_body(body: str) -> list[Block]:
    blocks: list[Block] = []
    para: list[str] = []
    lines = body.splitlines()
    i = 0

    def flush() -> None:
        if para:
            blocks.append(("p", " ".join(para)))
            para.clear()

    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("```"):
            flush()
            code: list[str] = []
            i += 1
            while not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            blocks.append(("code", "\n".join(code)))
        elif line.startswith("### "):
            flush()
            blocks.append(("h3", line[4:].strip()))
        elif line.startswith("## "):
            flush()
            blocks.append(("h2", line[3:].strip()))
        elif line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
            continue
        elif re.match(r"^(- |\d+\. )", line):
            flush()
            kind = "ul" if line.startswith("- ") else "ol"
            items = []
            while i < len(lines) and re.match(r"^(- |\d+\. )", lines[i]):
                items.append(re.sub(r"^(- |\d+\. )", "", lines[i]).strip())
                i += 1
            blocks.append((kind, items))
            continue
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
        i += 1
    flush()
    return blocks


# ---------------------------------------------------------------- rendering


def meta_line(src: Source, access: str) -> str:
    f = src.front
    return (
        f"Document code: {src.code} · Version {src.version} · "
        f"Effective date: {f['effective_date']} · Owner: {f['owner']} · "
        f"Classification: {access.capitalize()}"
    )


def table_text(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(r) for r in rows)


def render_txt(src: Source, title: str, access: str) -> str:
    out = ["NIMBUS LOGISTICS", title.upper(), meta_line(src, access), "=" * 72, ""]
    for kind, val in src.blocks:
        if kind == "h2":
            out += [val, "-" * len(val), ""]
        elif kind == "h3":
            out += [val, ""]
        elif kind == "p":
            out += [val, ""]
        elif kind == "ul":
            out += [f"- {x}" for x in val] + [""]
        elif kind == "ol":
            out += [f"{n}. {x}" for n, x in enumerate(val, 1)] + [""]
        elif kind == "table":
            out += [table_text(val), ""]
        elif kind == "code":
            out += [val, ""]
    return "\n".join(out).rstrip() + "\n"


def render_md(src: Source, title: str, access: str) -> str:
    out = [f"# {title}", "", f"_{meta_line(src, access)}_", ""]
    for kind, val in src.blocks:
        if kind == "h2":
            out += [f"## {val}", ""]
        elif kind == "h3":
            out += [f"### {val}", ""]
        elif kind == "p":
            out += [val, ""]
        elif kind == "ul":
            out += [f"- {x}" for x in val] + [""]
        elif kind == "ol":
            out += [f"{n}. {x}" for n, x in enumerate(val, 1)] + [""]
        elif kind == "table":
            header, *rows = val
            out.append("| " + " | ".join(header) + " |")
            out.append("|" + "---|" * len(header))
            out += ["| " + " | ".join(r) + " |" for r in rows] + [""]
        elif kind == "code":
            out += ["```", val, "```", ""]
    return "\n".join(out).rstrip() + "\n"


def render_html(src: Source, title: str, access: str) -> str:
    dept = PREFIX_DEPT[src.doc_id.split("-")[0]]
    out = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{escape(title)} | Nimbus Intranet</title>",
        "</head>",
        "<body>",
        '<header class="site-header">',
        '<div class="logo">Nimbus Logistics Intranet</div>',
        '<nav class="main-nav"><ul>',
        '<li><a href="/">Home</a></li>',
        '<li><a href="/hr">HR</a></li>',
        '<li><a href="/it">IT</a></li>',
        '<li><a href="/operations">Operations</a></li>',
        '<li><a href="/finance">Finance</a></li>',
        '<li><a href="/search">Search</a></li>',
        "</ul></nav>",
        f'<div class="breadcrumb">Home &gt; {escape(dept)} &gt; {escape(title)}</div>',
        "</header>",
        "<main>",
        "<article>",
        f"<h1>{escape(title)}</h1>",
        f'<p class="doc-meta">{escape(meta_line(src, access))}</p>',
    ]
    for kind, val in src.blocks:
        if kind in ("h2", "h3"):
            out.append(f"<{kind}>{escape(val)}</{kind}>")
        elif kind == "p":
            out.append(f"<p>{escape(val)}</p>")
        elif kind in ("ul", "ol"):
            out.append(f"<{kind}>")
            out += [f"<li>{escape(x)}</li>" for x in val]
            out.append(f"</{kind}>")
        elif kind == "table":
            header, *rows = val
            out.append("<table>")
            out.append("<tr>" + "".join(f"<th>{escape(c)}</th>" for c in header) + "</tr>")
            for r in rows:
                out.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>")
            out.append("</table>")
        elif kind == "code":
            out.append(f"<pre><code>{escape(val)}</code></pre>")
    out += [
        "</article>",
        '<aside class="related"><h4>Need more help?</h4>',
        "<p>Open a ticket on the Service Portal or call the help line at extension 4357.</p>",
        "</aside>",
        "</main>",
        '<footer class="site-footer">',
        "<p>&copy; 2026 Nimbus Logistics, Inc. All rights reserved.</p>",
        '<p><a href="/privacy">Privacy</a> | <a href="/accessibility">Accessibility</a> | '
        '<a href="/contact">Contact the intranet team</a></p>',
        "</footer>",
        "</body>",
        "</html>",
    ]
    return "\n".join(out) + "\n"


def render_csv(src: Source) -> str:
    tables = [val for kind, val in src.blocks if kind == "table"]
    if len(tables) != 1:
        raise ValueError(f"{src.doc_id}: CSV documents need exactly one table")
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(tables[0])
    return buf.getvalue()


def write_pdf(path: Path, src: Source, title: str, access: str) -> None:
    styles = getSampleStyleSheet()
    cell = styles["BodyText"].clone("cell", fontSize=8.5, leading=10.5)
    story: list = [Paragraph(xml_escape(title), styles["Title"])]
    story.append(Paragraph(xml_escape(meta_line(src, access)), styles["Italic"]))
    story.append(Spacer(1, 10))
    for kind, val in src.blocks:
        if kind == "h2":
            story.append(Paragraph(xml_escape(val), styles["Heading2"]))
        elif kind == "h3":
            story.append(Paragraph(xml_escape(val), styles["Heading3"]))
        elif kind == "p":
            story.append(Paragraph(xml_escape(val), styles["BodyText"]))
        elif kind in ("ul", "ol"):
            items = [ListItem(Paragraph(xml_escape(x), styles["BodyText"])) for x in val]
            story.append(
                ListFlowable(items, bulletType="bullet" if kind == "ul" else "1", leftIndent=14)
            )
        elif kind == "table":
            data = [[Paragraph(xml_escape(c), cell) for c in row] for row in val]
            table = Table(data, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story += [table, Spacer(1, 8)]
        elif kind == "code":
            story.append(Preformatted(val, styles["Code"]))

    footer = f"Nimbus Logistics · {src.code} · {access.capitalize()}"

    def on_page(canvas, doc) -> None:
        canvas.setFont("Helvetica", 8)
        canvas.drawString(54, 30, footer)
        canvas.drawRightString(LETTER[0] - 54, 30, f"Page {doc.page}")

    doc = SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        title=title,
        author="Nimbus Logistics",
        subject=src.code,
        invariant=1,
    )
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def write_docx(path: Path, src: Source, title: str, access: str) -> None:
    doc = Document()
    fixed = datetime.combine(date.fromisoformat(src.front["effective_date"]), datetime.min.time())
    props = doc.core_properties
    props.title = title
    props.author = "Nimbus Logistics"
    props.last_modified_by = "Nimbus Logistics"
    props.created = props.modified = props.last_printed = fixed
    props.revision = 1

    doc.add_heading(title, level=0)
    doc.add_paragraph(meta_line(src, access)).italic = True
    for kind, val in src.blocks:
        if kind == "h2":
            doc.add_heading(val, level=1)
        elif kind == "h3":
            doc.add_heading(val, level=2)
        elif kind == "p":
            doc.add_paragraph(val)
        elif kind in ("ul", "ol"):
            style = "List Bullet" if kind == "ul" else "List Number"
            for x in val:
                doc.add_paragraph(x, style=style)
        elif kind == "table":
            table = doc.add_table(rows=len(val), cols=len(val[0]), style="Table Grid")
            for r, row in enumerate(val):
                for c, text in enumerate(row):
                    table.cell(r, c).text = text
        elif kind == "code":
            run = doc.add_paragraph().add_run(val)
            run.font.name = "Courier New"
            run.font.size = Pt(9)

    buf = io.BytesIO()
    doc.save(buf)
    path.write_bytes(normalize_zip(buf.getvalue()))


def normalize_zip(data: bytes) -> bytes:
    """Rewrite a zip with fixed timestamps so identical content gives identical bytes."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            fixed = zipfile.ZipInfo(info.filename, date_time=FIXED_ZIP_TIME)
            fixed.compress_type = zipfile.ZIP_DEFLATED
            fixed.external_attr = 0o644 << 16
            dst.writestr(fixed, src.read(info.filename))
    return out.getvalue()


# ---------------------------------------------------------------- build


def build_metadata(sources: list[Source], raw_rel: str) -> list[DocumentMeta]:
    families: dict[str, list[Source]] = defaultdict(list)
    for s in sources:
        families[s.family_id].append(s)
    for members in families.values():
        members.sort(key=lambda s: s.version)

    docs = []
    for s in sources:
        f = s.front
        prefix, type_code = s.doc_id.split("-")[:2]
        department = PREFIX_DEPT[prefix]
        members = families[s.family_id]
        idx = members.index(s)
        title = f["title"]
        canonical = render_csv(s) if f["format"] == "csv" else render_txt(s, title, f["access"])
        tables = [v for k, v in s.blocks if k == "table"]
        docs.append(
            DocumentMeta(
                doc_id=s.doc_id,
                family_id=s.family_id,
                title=title,
                department=department,
                doc_type=PREFIX_TYPE[type_code],
                format=f["format"],
                path=f"{raw_rel}/{DEPT_DIRS[department]}/{s.doc_id}.{f['format']}",
                version=s.version,
                status="active" if s is members[-1] else "superseded",
                effective_date=f["effective_date"],
                last_updated=f.get("last_updated", f["effective_date"]),
                supersedes=members[idx - 1].doc_id if idx > 0 else None,
                superseded_by=members[idx + 1].doc_id if s is not members[-1] else None,
                access_level=f["access"],
                owner=f["owner"],
                tags=[t.strip() for t in f.get("tags", "").split(",") if t.strip()],
                key_terms=sorted(set(KEY_TERM_RE.findall(canonical))),
                has_tables=bool(tables),
                contains_injection=f.get("contains_injection", "false") == "true",
                section_headings=[v for k, v in s.blocks if k == "h2"],
                max_table_rows=max((len(t) - 1 for t in tables), default=0),
                word_count=len(canonical.split()),
            )
        )
    return sorted(docs, key=lambda d: d.doc_id)


def write_corpus(sources: list[Source], docs: list[DocumentMeta]) -> None:
    by_id = {s.doc_id: s for s in sources}
    wanted = {ROOT / d.path for d in docs}
    for dept_dir in DEPT_DIRS.values():
        folder = ROOT / get_settings().data_raw_dir / dept_dir
        folder.mkdir(parents=True, exist_ok=True)
        for stale in folder.iterdir():
            if stale.is_file() and stale not in wanted:
                stale.unlink()
    for d in docs:
        src, path = by_id[d.doc_id], ROOT / d.path
        if d.format == "pdf":
            write_pdf(path, src, d.title, d.access_level)
        elif d.format == "docx":
            write_docx(path, src, d.title, d.access_level)
        else:
            render = {"md": render_md, "txt": render_txt, "html": render_html}
            text = (
                render_csv(src)
                if d.format == "csv"
                else render[d.format](src, d.title, d.access_level)
            )
            path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    settings = get_settings()
    sources = [parse_source(p) for p in sorted(CONTENT_DIR.glob("*/*.txt"))]
    docs = build_metadata(sources, settings.data_raw_dir.as_posix())
    write_corpus(sources, docs)

    metadata_dir = ROOT / settings.data_metadata_dir
    write_json(metadata_dir / "documents.json", [d.model_dump(mode="json") for d in docs])

    users = json.loads((CONTENT_DIR / "users.json").read_text(encoding="utf-8"))
    users = [UserProfile.model_validate(u).model_dump(mode="json") for u in users]
    write_json(metadata_dir / "users.json", users)

    raw_questions = json.loads((CONTENT_DIR / "eval_questions.json").read_text(encoding="utf-8"))
    eval_path = ROOT / settings.eval_set_path
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for n, q in enumerate(raw_questions, start=1):
        question = EvalQuestion.model_validate({"id": f"q{n:03d}", **q})
        lines.append(json.dumps(question.model_dump(mode="json"), ensure_ascii=False))
    eval_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(docs)} documents, {len(users)} users, {len(lines)} eval questions")
    print("By department:", dict(sorted(Counter(d.department for d in docs).items())))
    print("By format:    ", dict(sorted(Counter(d.format for d in docs).items())))
    short = [d.doc_id for d in docs if d.format != "csv" and d.word_count < 300]
    if short:
        print("Under 300 words:", ", ".join(short))


if __name__ == "__main__":
    main()
