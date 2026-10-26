"""PDF loader built on pdfplumber.

Structure is recovered from layout:
- the largest bold line(s) on page 1 are the title (skipped); other bold lines larger than the
  body font are headings, levels assigned by size (largest = level 1);
- lines are grouped into paragraphs by vertical gap; Courier lines are code;
- a line starting with a bullet glyph or a number in a bullet column starts a list item, and
  indented lines that follow continue it;
- tables come from pdfplumber's table finder, and their text is removed from the line stream;
- the page footer band (and any stray `Page N` line) is dropped.
"""

import io
import re
from collections import Counter
from dataclasses import dataclass, field

import pdfplumber

from app.dataset.schemas import DocumentMeta
from app.ingestion.blocks import BlockBuilder, clean, table_text
from app.ingestion.models import Block

FOOTER_BAND = 60  # points from the bottom edge that hold only the page footer
PARAGRAPH_GAP = 4  # vertical gap (points) between lines that starts a new block
INDENT = 5  # points right of the body margin that mark a list-item continuation
BULLET = re.compile(r"^(?:\(cid:\d+\)|[•●▪◦‣])\s*")
LIST_GAP = 4  # gap (points) between a list number and its text; a normal space is ~3
NUMBERED = re.compile(r"^(\d+)[.)]?\s+")
PAGE_NUMBER = re.compile(r"^Page \d+$")
SENTENCE_END = (".", "!", "?", ":", ";")


@dataclass
class Line:
    text: str
    top: float
    bottom: float
    x0: float
    size: float
    bold: bool
    italic: bool
    mono: bool
    page: int
    chars: list = field(repr=False, default_factory=list)


@dataclass
class TableItem:
    rows: list[list[str]]
    top: float
    page: int


def _line_font(chars: list[dict]) -> tuple[str, float]:
    """Dominant (fontname, size) of a line, ignoring spaces and unmapped bullet glyphs."""
    counts = Counter(
        (c["fontname"], round(c["size"], 1))
        for c in chars
        if c["text"].strip() and not c["text"].startswith("(cid:")
    )
    if not counts:
        c = chars[0]
        return c["fontname"], round(c["size"], 1)
    return counts.most_common(1)[0][0]


def _page_items(page, page_no: int) -> list[Line | TableItem]:
    tables = page.find_tables()
    bboxes = [t.bbox for t in tables]

    def outside_tables(obj: dict) -> bool:
        if obj.get("object_type") != "char":
            return True
        cx = (obj["x0"] + obj["x1"]) / 2
        cy = (obj["top"] + obj["bottom"]) / 2
        return not any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in bboxes)

    items: list[Line | TableItem] = [
        TableItem(rows=t.extract(), top=t.bbox[1], page=page_no) for t in tables
    ]
    for raw in page.filter(outside_tables).extract_text_lines(return_chars=True):
        if raw["top"] > page.height - FOOTER_BAND or PAGE_NUMBER.match(raw["text"].strip()):
            continue
        fontname, size = _line_font(raw["chars"])
        items.append(
            Line(
                text=raw["text"],
                top=raw["top"],
                bottom=raw["bottom"],
                x0=raw["x0"],
                size=size,
                bold="Bold" in fontname,
                italic="Oblique" in fontname or "Italic" in fontname,
                mono="Courier" in fontname,
                page=page_no,
                chars=raw["chars"],
            )
        )
    items.sort(key=lambda item: item.top)
    return items


def _list_marker(line: Line) -> str | None:
    """Return the line text without its bullet/number if it starts a list item."""
    text = line.text.strip()
    if m := BULLET.match(text):
        if m.group(0).strip():
            return text[m.end() :]
    if m := NUMBERED.match(text):
        # A list number sits in a bullet column: a wider gap than a space separates it from
        # the item text. "2025. Transportation" at the start of a wrapped line is not a list.
        marker = m.group(0).strip()
        chars = [c for c in line.chars if c["text"].strip()]
        if len(chars) > len(marker):
            gap = chars[len(marker)]["x0"] - chars[len(marker) - 1]["x1"]
            if gap >= LIST_GAP:
                return text[m.end() :]
    return None


@dataclass
class _Pending:
    kind: str
    parts: list[str]
    page: int  # page of the first line
    bottom: float
    last_page: int
    level: int | None = None
    italic: bool = False


def load_pdf(raw: bytes, meta: DocumentMeta) -> tuple[list[Block], int | None]:
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        page_count = len(pdf.pages)
        items = [item for n, page in enumerate(pdf.pages, 1) for item in _page_items(page, n)]

    lines = [item for item in items if isinstance(item, Line)]
    if not lines and not items:
        return [], page_count
    body_size = Counter(ln.size for ln in lines if not ln.bold).most_common(1)[0][0] if lines else 0
    margin = Counter(ln.x0 for ln in lines if not ln.bold).most_common(1)[0][0] if lines else 0

    # Title: the leading run of largest-size lines on page 1.
    title_lines: set[int] = set()
    first_page = [ln for ln in lines if ln.page == 1]
    if first_page:
        title_size = max(ln.size for ln in first_page)
        for ln in first_page:
            if ln.size != title_size:
                break
            title_lines.add(id(ln))
    heading_sizes = sorted(
        {
            ln.size
            for ln in lines
            if ln.bold and ln.size > body_size + 0.5 and id(ln) not in title_lines
        },
        reverse=True,
    )
    level_of = {size: min(i + 1, 2) for i, size in enumerate(heading_sizes)}

    b = BlockBuilder()
    cur: _Pending | None = None
    last_table_page: int | None = None  # page of the most recent table block, if it is last

    def flush() -> None:
        nonlocal cur
        if cur is None:
            return
        if cur.kind == "heading":
            text = " ".join(cur.parts)
            if clean(text) != clean(meta.title):
                b.heading(text, cur.level or 1, page=cur.page)
        elif cur.kind == "code":
            b.add("code", "\n".join(cur.parts), page=cur.page)
        else:
            b.add(cur.kind, " ".join(cur.parts), page=cur.page)  # type: ignore[arg-type]
        cur = None

    for item in items:
        if isinstance(item, TableItem):
            flush()
            rows = [[clean(c or "") for c in row] for row in item.rows]
            prev = b.blocks[-1] if b.blocks else None
            if (
                prev is not None
                and prev.kind == "table"
                and last_table_page == item.page - 1
                and prev.rows
                and rows
                and rows[0] == prev.rows[0]
            ):
                prev.rows.extend(r for r in rows[1:] if any(r))
                prev.text = table_text(prev.rows)
            else:
                b.table(rows, page=item.page)
            last_table_page = item.page
            continue

        ln = item
        if id(ln) in title_lines:
            continue
        last_table_page = None  # text between two tables means they are separate tables
        new_page = cur is not None and ln.page != cur.last_page
        gap = ln.top - cur.bottom if cur is not None and ln.page == cur.last_page else None
        close = gap is not None and gap < PARAGRAPH_GAP
        text = ln.text.strip()

        if ln.bold and ln.size in level_of:
            level = level_of[ln.size]
            if cur and cur.kind == "heading" and cur.level == level and close:
                cur.parts.append(text)
            else:
                flush()
                cur = _Pending("heading", [text], ln.page, ln.bottom, ln.page, level=level)
        elif ln.mono:
            if cur and cur.kind == "code" and (close or new_page):
                cur.parts.append(ln.text.rstrip())
            else:
                flush()
                cur = _Pending("code", [ln.text.rstrip()], ln.page, ln.bottom, ln.page)
        elif (item_text := _list_marker(ln)) is not None:
            flush()
            cur = _Pending("list_item", [item_text], ln.page, ln.bottom, ln.page)
        else:
            continues_item = (
                cur is not None
                and cur.kind == "list_item"
                and ln.x0 > margin + INDENT
                and (close or new_page)
            )
            continues_para = (
                cur is not None
                and cur.kind == "paragraph"
                and cur.italic == ln.italic
                and (close or (new_page and not cur.parts[-1].endswith(SENTENCE_END)))
            )
            if continues_item or continues_para:
                cur.parts.append(text)  # type: ignore[union-attr]
            else:
                flush()
                cur = _Pending("paragraph", [text], ln.page, ln.bottom, ln.page, italic=ln.italic)
        if cur is not None:
            cur.bottom = ln.bottom
            cur.last_page = ln.page
    flush()
    return b.blocks, page_count
