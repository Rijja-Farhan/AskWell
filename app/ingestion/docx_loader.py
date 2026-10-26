"""DOCX loader: walks the document body in order, mapping Word styles to block kinds."""

import io

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.dataset.schemas import DocumentMeta
from app.ingestion.blocks import BlockBuilder, clean
from app.ingestion.models import Block

CODE_FONTS = {"Courier New", "Courier", "Consolas"}


def load_docx(raw: bytes, meta: DocumentMeta) -> tuple[list[Block], int | None]:
    doc = Document(io.BytesIO(raw))
    b = BlockBuilder()
    for item in doc.iter_inner_content():
        if isinstance(item, Table):
            b.table([[cell.text for cell in row.cells] for row in item.rows])
        elif isinstance(item, Paragraph):
            _add_paragraph(b, item, meta)
    return b.blocks, None


def _add_paragraph(b: BlockBuilder, p: Paragraph, meta: DocumentMeta) -> None:
    style = p.style.name if p.style is not None else ""
    text = p.text
    if style == "Title":
        if clean(text) != clean(meta.title):
            b.heading(text, level=1)
        return
    if style.startswith("Heading"):
        number = style.removeprefix("Heading").strip()
        level = int(number) if number.isdigit() else 1
        b.heading(text, level=min(level, 2))
    elif style.startswith("List"):
        b.add("list_item", text)
    elif p.runs and all(r.font.name in CODE_FONTS for r in p.runs if r.text.strip()):
        b.add("code", text)
    else:
        b.add("paragraph", text)
