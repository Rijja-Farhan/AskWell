"""Load corpus files into `ParsedDocument`s, one loader per `DocumentMeta.format`."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from app.config import Settings
from app.dataset.schemas import DocumentMeta
from app.ingestion.docx_loader import load_docx
from app.ingestion.html_loader import load_html
from app.ingestion.models import Block, ParsedDocument
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.text_loaders import load_csv, load_markdown, load_text

Loader = Callable[[bytes, DocumentMeta], tuple[list[Block], int | None]]

LOADERS: dict[str, Loader] = {
    "pdf": load_pdf,
    "docx": load_docx,
    "md": load_markdown,
    "txt": load_text,
    "html": load_html,
    "csv": load_csv,
}


def load_document(path: Path, meta: DocumentMeta) -> ParsedDocument:
    """Parse one file. Raises on unreadable, corrupt or unsupported files."""
    loader = LOADERS.get(meta.format)
    if loader is None:
        raise ValueError(f"unsupported format {meta.format!r}")
    raw = path.read_bytes()
    blocks, page_count = loader(raw, meta)
    return ParsedDocument(
        meta=meta,
        content_hash=hashlib.sha256(raw).hexdigest(),
        title=meta.title,
        blocks=blocks,
        text="\n\n".join(block.text for block in blocks),
        page_count=page_count,
    )


def load_metadata(root: Path, settings: Settings) -> list[DocumentMeta]:
    path = root / settings.data_metadata_dir / "documents.json"
    return [DocumentMeta.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def load_corpus(root: Path, settings: Settings) -> tuple[list[ParsedDocument], list[str]]:
    """Parse every corpus file under `root`. Returns documents sorted by doc_id, and errors.

    Problems with single files are reported as errors; the rest of the corpus still loads.
    """
    errors: list[str] = []
    try:
        metas = load_metadata(root, settings)
    except (OSError, ValueError, ValidationError) as exc:
        return [], [f"cannot read documents.json: {exc}"]

    by_path = {m.path: m for m in metas}
    raw_dir = root / settings.data_raw_dir
    on_disk = sorted(
        p.relative_to(root).as_posix()
        for p in raw_dir.rglob("*")
        if p.is_file() and p.name != ".gitkeep"
    )

    docs: list[ParsedDocument] = []
    for rel in on_disk:
        ext = rel.rsplit(".", 1)[-1].lower() if "." in rel else ""
        if ext not in LOADERS:
            errors.append(f"unsupported extension: {rel}")
            continue
        meta = by_path.get(rel)
        if meta is None:
            errors.append(f"file has no metadata entry: {rel}")
            continue
        try:
            docs.append(load_document(root / rel, meta))
        except Exception as exc:  # noqa: BLE001 - any parser failure is reported, not fatal
            errors.append(f"cannot parse {rel}: {type(exc).__name__}: {exc}")

    present = set(on_disk)
    errors += [f"metadata entry has no file: {m.path}" for m in metas if m.path not in present]
    return sorted(docs, key=lambda d: d.meta.doc_id), errors
