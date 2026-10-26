"""Loaders for line-based formats: Markdown, plain text and CSV."""

import csv
import io
import re

from app.dataset.schemas import DocumentMeta
from app.ingestion.blocks import BlockBuilder, clean
from app.ingestion.models import Block

LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?$")


def _split_pipe_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def load_markdown(raw: bytes, meta: DocumentMeta) -> tuple[list[Block], int | None]:
    """`#` = document title (skipped), `##` = level 1, `###` and deeper = level 2."""
    b = BlockBuilder()
    lines = raw.decode("utf-8").splitlines()
    para: list[str] = []

    def flush() -> None:
        if para:
            text = " ".join(para)
            # The metadata line is written as `_..._` (emphasis); drop the markers.
            if text.startswith("_") and text.endswith("_"):
                text = text[1:-1]
            b.add("paragraph", text)
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            b.add("code", "\n".join(code))
        elif m := re.match(r"^(#{1,6})\s+(.*)$", stripped):
            flush()
            depth, text = len(m.group(1)), m.group(2).strip()
            if depth == 1 and clean(text) == clean(meta.title):
                pass  # the in-file title is not a block
            else:
                b.heading(text, level=1 if depth <= 2 else 2)
        elif stripped.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not TABLE_SEPARATOR.match(lines[i].strip()):
                    rows.append(_split_pipe_row(lines[i]))
                i += 1
            b.table(rows)
            continue
        elif LIST_ITEM.match(line):
            flush()
            b.add("list_item", LIST_ITEM.sub("", line, count=1))
        elif not stripped:
            flush()
        else:
            para.append(stripped)
        i += 1
    flush()
    return b.blocks, None


def load_text(raw: bytes, meta: DocumentMeta) -> tuple[list[Block], int | None]:
    """Plain text: a line underlined with `-` is a level-1 heading; `=` rules and the
    NIMBUS LOGISTICS banner and upper-case title are boilerplate."""
    b = BlockBuilder()
    lines = [line.rstrip() for line in raw.decode("utf-8").splitlines()]
    para: list[str] = []

    def flush() -> None:
        if para:
            b.add("paragraph", " ".join(para))
            para.clear()

    def is_rule(line: str, char: str) -> bool:
        return len(line) >= 3 and set(line) == {char}

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if i < 3 and (stripped == "NIMBUS LOGISTICS" or stripped.upper() == meta.title.upper()):
            pass  # banner and title
        elif is_rule(stripped, "="):
            flush()
        elif stripped and is_rule(nxt, "-"):
            flush()
            b.heading(stripped, level=1)
            i += 1
        elif " | " in stripped and " | " in nxt:
            flush()
            rows = []
            while i < len(lines) and " | " in lines[i]:
                rows.append([c.strip() for c in lines[i].split(" | ")])
                i += 1
            b.table(rows)
            continue
        elif LIST_ITEM.match(line):
            flush()
            b.add("list_item", LIST_ITEM.sub("", line, count=1))
        elif not stripped:
            flush()
        else:
            para.append(stripped)
        i += 1
    flush()
    return b.blocks, None


def load_csv(raw: bytes, meta: DocumentMeta) -> tuple[list[Block], int | None]:
    """A CSV file is one table block, header row first."""
    b = BlockBuilder()
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    b.table(rows)
    return b.blocks, None
