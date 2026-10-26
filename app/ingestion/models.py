"""Common parsed-document format produced by every loader."""

from typing import Literal

from pydantic import BaseModel

from app.dataset.schemas import DocumentMeta

BlockKind = Literal["heading", "paragraph", "list_item", "table", "code"]


class Block(BaseModel):
    kind: BlockKind
    # Clean text. Tables: one " | "-joined row per line, header first.
    text: str
    # Headings only: 1 = top-level section, 2 = subsection.
    level: int | None = None
    # Enclosing heading texts, outermost first. A heading's path includes itself.
    section_path: list[str] = []
    # 1-based page number (PDF only).
    page: int | None = None
    # Tables only: cell texts, header row first.
    rows: list[list[str]] | None = None


class ParsedDocument(BaseModel):
    meta: DocumentMeta
    content_hash: str  # sha256 hex of the raw file bytes
    title: str
    blocks: list[Block]
    text: str  # block texts joined by a blank line
    page_count: int | None = None  # PDF only
