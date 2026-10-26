"""Chunking: fixed-size character windows (the Phase 3 baseline) or structure-aware chunks.

`chunk_document` ignores document structure and is kept unchanged so the baseline stays
reproducible. `chunk_structured` follows the parsed blocks: sections, headings and tables.
`make_chunks` picks one according to `settings.chunk_strategy`.
"""

import re
from dataclasses import dataclass

from pydantic import BaseModel

from app.config import Settings
from app.ingestion.models import Block, ParsedDocument

Scalar = str | int | float | bool


class Chunk(BaseModel):
    chunk_id: str  # f"{doc_id}::{chunk_index:04d}"
    doc_id: str
    chunk_index: int
    # Fixed chunks: == doc.text[start_char:end_char]. Structured chunks: a context header plus
    # the block texts; start_char/end_char give the span of doc.text they cover.
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Scalar]


def chunk_metadata(doc: ParsedDocument) -> dict[str, Scalar]:
    """Scalar document metadata stored with every chunk (Chroma accepts scalars only)."""
    m = doc.meta
    return {
        "doc_id": m.doc_id,
        "title": m.title,
        "department": m.department,
        "doc_type": m.doc_type,
        "version": m.version,
        "status": m.status,
        "access_level": m.access_level,
        "effective_date": m.effective_date.isoformat(),
        "path": m.path,
    }


def chunk_document(doc: ParsedDocument, size: int, overlap: int) -> list[Chunk]:
    """Split `doc.text` into windows of `size` characters, each overlapping the previous one
    by `overlap` characters. The last window may be shorter. Windows cover the whole text."""
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"chunk overlap must be >= 0 and < size ({size}), got {overlap}")

    text = doc.text
    metadata = chunk_metadata(doc)
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.meta.doc_id}::{index:04d}",
                doc_id=doc.meta.doc_id,
                chunk_index=index,
                text=text[start:end],
                start_char=start,
                end_char=end,
                metadata=metadata,
            )
        )
        if end == len(text):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------- structure-aware chunking

CHUNKER_VERSION = 1  # bump when chunk_structured output changes, so indexes rebuild
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
MIN_BUDGET = 200  # characters; floor for the body budget after the header is subtracted


@dataclass
class _Atom:
    """A piece of a section that is never split further: one block (or one part of an
    oversized block), with any headings directly above it attached in front."""

    text: str
    start: int
    end: int
    blocks: list[Block]
    is_table: bool = False


def block_offsets(doc: ParsedDocument) -> list[int]:
    """Start of each block in `doc.text` (block texts are joined by a blank line)."""
    offsets, pos = [], 0
    for block in doc.blocks:
        offsets.append(pos)
        pos += len(block.text) + 2
    return offsets


def top_level_sections(doc: ParsedDocument) -> list[list[int]]:
    """Block indexes per top-level section. Text before the first heading (the document code
    line) joins the first section instead of becoming a chunk of its own."""
    groups: list[list[int]] = [[]]
    for i, block in enumerate(doc.blocks):
        is_top = block.kind == "heading" and len(block.section_path) == 1
        if is_top and any(doc.blocks[j].kind == "heading" for j in groups[-1]):
            groups.append([])
        groups[-1].append(i)
    return [g for g in groups if g]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for match in SENTENCE_END.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if e > s]


def _split_text(text: str, budget: int, overlap: int) -> list[tuple[int, int]]:
    """Spans of `text` of at most `budget` chars, cut only between sentences. Each piece after
    the first repeats the previous piece's last sentences, up to `overlap` chars."""
    sentences = _sentence_spans(text)
    pieces: list[tuple[int, int]] = []
    i = 0
    while i < len(sentences):
        j = i + 1  # always take at least one sentence, even an oversized one
        while j < len(sentences) and sentences[j][1] - sentences[i][0] <= budget:
            j += 1
        pieces.append((sentences[i][0], sentences[j - 1][1]))
        if j == len(sentences):
            break
        back = j  # step back over sentences that fit in the overlap, keeping progress
        while back - 1 > i and sentences[j - 1][1] - sentences[back - 1][0] <= overlap:
            back -= 1
        i = back
    return pieces


def _split_table(text: str, budget: int) -> list[tuple[str, int, int]]:
    """Split a table by rows, repeating the header row in every piece."""
    lines = text.split("\n")
    starts, pos = [], 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
    header, pieces = lines[0], []
    i = 1
    while i < len(lines):
        j = i + 1
        while j < len(lines) and len("\n".join([header, *lines[i : j + 1]])) <= budget:
            j += 1
        span_start = 0 if not pieces else starts[i]
        pieces.append(
            ("\n".join([header, *lines[i:j]]), span_start, starts[j - 1] + len(lines[j - 1]))
        )
        i = j
    return pieces or [(text, 0, len(text))]


def _section_atoms(
    doc: ParsedDocument, indexes: list[int], offsets: list[int], budget: int, overlap: int
) -> list[_Atom]:
    atoms: list[_Atom] = []
    pending: list[int] = []  # headings waiting for the content below them
    for i in indexes:
        block = doc.blocks[i]
        if block.kind == "heading":
            pending.append(i)
            continue
        prefix = "\n\n".join(doc.blocks[h].text for h in pending)
        room = max(budget - len(prefix) - 2, MIN_BUDGET) if prefix else budget
        if len(block.text) <= room:
            parts = [(block.text, 0, len(block.text))]
        elif block.kind == "table":
            parts = _split_table(block.text, room)
        else:
            parts = [(block.text[s:e], s, e) for s, e in _split_text(block.text, room, overlap)]
        for n, (text, s, e) in enumerate(parts):
            heads = pending if n == 0 else []
            atoms.append(
                _Atom(
                    text=f"{prefix}\n\n{text}" if heads else text,
                    start=offsets[heads[0]] if heads else offsets[i] + s,
                    end=offsets[i] + e,
                    blocks=[doc.blocks[h] for h in heads] + [block],
                    is_table=block.kind == "table",
                )
            )
        pending = []
    if pending:  # headings with no content below them: keep them with the previous piece
        text = "\n\n".join(doc.blocks[h].text for h in pending)
        heads = [doc.blocks[h] for h in pending]
        if atoms and not atoms[-1].is_table:
            last = atoms[-1]
            last.text, last.end = f"{last.text}\n\n{text}", offsets[pending[-1]] + len(text)
            last.blocks += heads
        else:
            atoms.append(_Atom(text, offsets[pending[0]], offsets[pending[-1]] + len(text), heads))
    return atoms


def _pack(atoms: list[_Atom], budget: int) -> list[list[_Atom]]:
    """Merge consecutive atoms up to `budget` chars. Tables always stand alone."""
    groups: list[list[_Atom]] = []
    size = 0
    for atom in atoms:
        current = groups[-1] if groups else None
        if (
            current is None
            or atom.is_table
            or current[-1].is_table
            or size + 2 + len(atom.text) > budget
        ):
            groups.append([atom])
            size = len(atom.text)
        else:
            current.append(atom)
            size += 2 + len(atom.text)
    return groups


def _common_path(blocks: list[Block]) -> list[str]:
    paths = [b.section_path for b in blocks if b.section_path]
    if not paths:
        return []
    common = paths[0]
    for path in paths[1:]:
        n = 0
        while n < min(len(common), len(path)) and common[n] == path[n]:
            n += 1
        common = common[:n]
    return common


def structured_metadata(
    doc: ParsedDocument, blocks: list[Block], section: str
) -> dict[str, Scalar]:
    path = _common_path(blocks)
    kinds = list(dict.fromkeys(b.kind for b in blocks))
    metadata: dict[str, Scalar] = {
        **chunk_metadata(doc),
        "family_id": doc.meta.family_id,
        "last_updated": doc.meta.last_updated.isoformat(),
        "section": section,
        "section_path": " > ".join(path),
        "block_kinds": ",".join(kinds),
        "has_table": "table" in kinds,
        "chunk_strategy": "structure",
    }
    pages = [b.page for b in blocks if b.page is not None]
    if doc.meta.format == "pdf" and pages:
        metadata["page_start"], metadata["page_end"] = min(pages), max(pages)
    return metadata


def chunk_structured(doc: ParsedDocument, max_chars: int, overlap: int) -> list[Chunk]:
    """Split a document along its structure.

    - A chunk stays inside one top-level section; small subsections are merged up to
      `max_chars`, and a heading always travels with the content below it.
    - A table is its own chunk; an oversized table splits by rows, repeating the header.
    - An oversized paragraph splits between sentences, overlapping by up to `overlap` chars.
    - Each chunk starts with a context header, "Title > Section", so the embedding knows what
      the text is about. The header counts towards `max_chars`.
    """
    if max_chars <= 0:
        raise ValueError(f"chunk size must be positive, got {max_chars}")
    if not 0 <= overlap < max_chars:
        raise ValueError(f"chunk overlap must be >= 0 and < size ({max_chars}), got {overlap}")

    offsets = block_offsets(doc)
    chunks: list[Chunk] = []
    for indexes in top_level_sections(doc):
        tops = [doc.blocks[i] for i in indexes if len(doc.blocks[i].section_path) >= 1]
        section = tops[0].section_path[0] if tops else ""
        header = f"{doc.title} > {section}" if section else doc.title
        budget = max(max_chars - len(header) - 2, MIN_BUDGET)
        for group in _pack(_section_atoms(doc, indexes, offsets, budget, overlap), budget):
            blocks = [b for atom in group for b in atom.blocks]
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.meta.doc_id}::{index:04d}",
                    doc_id=doc.meta.doc_id,
                    chunk_index=index,
                    text=header + "\n\n" + "\n\n".join(atom.text for atom in group),
                    start_char=group[0].start,
                    end_char=group[-1].end,
                    metadata=structured_metadata(doc, blocks, section),
                )
            )
    return chunks


def make_chunks(doc: ParsedDocument, settings: Settings) -> list[Chunk]:
    """Chunk a document with the configured strategy ("fixed" or "structure")."""
    if settings.chunk_strategy == "fixed":
        return chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
    return chunk_structured(doc, settings.chunk_size, settings.chunk_overlap)
