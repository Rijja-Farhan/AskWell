"""Shared helpers for loaders: text cleaning and a builder that tracks section paths."""

import re
import unicodedata

from app.ingestion.models import Block, BlockKind

_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    """NFC-normalize and collapse runs of whitespace into single spaces."""
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def clean_code(text: str) -> str:
    """NFC-normalize code, keeping its line breaks and indentation."""
    lines = unicodedata.normalize("NFC", text).splitlines()
    return "\n".join(line.rstrip() for line in lines).strip("\n")


def table_text(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(row) for row in rows)


class BlockBuilder:
    """Collects blocks in document order and assigns each one its section path."""

    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self._path: list[str] = []

    def heading(self, text: str, level: int, page: int | None = None) -> None:
        text = clean(text)
        if not text:
            return
        self._path = [*self._path[: level - 1], text]
        self.blocks.append(
            Block(kind="heading", text=text, level=level, section_path=self._path, page=page)
        )

    def add(self, kind: BlockKind, text: str, page: int | None = None) -> None:
        text = clean_code(text) if kind == "code" else clean(text)
        if text:
            self.blocks.append(Block(kind=kind, text=text, section_path=self._path, page=page))

    def table(self, rows: list[list[str]], page: int | None = None) -> None:
        rows = [[clean(cell or "") for cell in row] for row in rows]
        rows = [row for row in rows if any(row)]
        if rows:
            self.blocks.append(
                Block(
                    kind="table",
                    text=table_text(rows),
                    section_path=self._path,
                    page=page,
                    rows=rows,
                )
            )
