import hashlib
import re
import shutil
from pathlib import Path

import pytest

from app.config import Settings
from app.dataset.schemas import DEPT_DIRS
from app.dataset.validate import INJECTION_PATTERN
from app.ingestion.blocks import clean
from app.ingestion.models import ParsedDocument
from app.ingestion.pipeline import load_corpus, load_metadata

ROOT = Path(__file__).resolve().parents[1]
BOILERPLATE = ["Nimbus Logistics Intranet", "All rights reserved", "Need more help?"]
PAGE_LINE = re.compile(r"^Page \d+$", re.M)


@pytest.fixture(scope="module")
def settings() -> Settings:
    # Explicit defaults so a local `.env` cannot redirect the paths.
    return Settings(_env_file=None)


@pytest.fixture(scope="module")
def corpus(settings) -> tuple[list[ParsedDocument], list[str]]:
    return load_corpus(ROOT, settings)


@pytest.fixture(scope="module")
def docs(corpus) -> list[ParsedDocument]:
    return corpus[0]


def ids(docs: list[ParsedDocument]) -> list[str]:
    return [d.meta.doc_id for d in docs]


def test_no_errors_and_one_doc_per_entry(corpus, settings):
    docs, errors = corpus
    assert errors == []
    assert ids(docs) == sorted(m.doc_id for m in load_metadata(ROOT, settings))


def test_level1_headings_match_metadata(docs):
    for d in docs:
        if d.meta.format == "csv":
            continue
        headings = [b.text for b in d.blocks if b.kind == "heading" and b.level == 1]
        assert headings == d.meta.section_headings, d.meta.doc_id


def test_tables_match_metadata(docs):
    for d in docs:
        tables = [b for b in d.blocks if b.kind == "table"]
        assert bool(tables) == d.meta.has_tables, d.meta.doc_id
        if tables:
            largest = max(len(t.rows) - 1 for t in tables)
            assert largest == d.meta.max_table_rows, d.meta.doc_id
            for t in tables:
                assert t.text == "\n".join(" | ".join(r) for r in t.rows)


def test_blocks_are_clean(docs):
    for d in docs:
        assert d.title == d.meta.title
        for b in d.blocks:
            assert b.text.strip(), d.meta.doc_id
            assert b.text != d.meta.title, d.meta.doc_id
            if b.kind != "code":
                assert "  " not in b.text, (d.meta.doc_id, b.text)


def test_metadata_line_is_first_paragraph(docs):
    for d in docs:
        if d.meta.format == "csv":
            continue
        first = d.blocks[0]
        assert first.kind == "paragraph" and first.section_path == []
        assert first.text.startswith(f"Document code: {d.meta.family_id.upper()}"), d.meta.doc_id


def test_key_terms_survive(docs):
    for d in docs:
        for term in d.meta.key_terms:
            assert term in d.text, (d.meta.doc_id, term)


def test_injection_text_is_preserved(docs):
    injected = [d for d in docs if d.meta.contains_injection]
    assert injected
    for d in injected:
        assert INJECTION_PATTERN.search(d.text), d.meta.doc_id


def test_boilerplate_removed(docs):
    for d in docs:
        for phrase in BOILERPLATE:
            assert phrase not in d.text, (d.meta.doc_id, phrase)
        assert not PAGE_LINE.search(d.text), d.meta.doc_id
        assert f"Nimbus Logistics · {d.meta.family_id.upper()} ·" not in d.text, d.meta.doc_id


def test_page_numbers(docs):
    for d in docs:
        if d.meta.format == "pdf":
            assert d.page_count and d.page_count >= 1
            assert all(1 <= b.page <= d.page_count for b in d.blocks), d.meta.doc_id
        else:
            assert d.page_count is None
            assert all(b.page is None for b in d.blocks), d.meta.doc_id


def test_multi_page_pdfs_use_later_pages(docs):
    multi = [d for d in docs if d.meta.format == "pdf" and d.page_count > 1]
    assert multi
    for d in multi:
        assert max(b.page for b in d.blocks) == d.page_count, d.meta.doc_id


def test_content_hash(docs):
    for d in docs:
        raw = (ROOT / d.meta.path).read_bytes()
        assert d.content_hash == hashlib.sha256(raw).hexdigest()


def test_section_paths(docs):
    for d in docs:
        path: list[str] = []
        for b in d.blocks:
            if b.kind == "heading":
                assert b.section_path[-1] == b.text
                assert len(b.section_path) <= b.level
                path = b.section_path
            else:
                assert b.section_path == path, (d.meta.doc_id, b.text[:40])


def test_pdf_table_text_not_repeated(docs):
    for d in docs:
        if d.meta.format != "pdf":
            continue
        for t in (b for b in d.blocks if b.kind == "table"):
            header = " ".join(t.rows[0])
            others = [b.text for b in d.blocks if b.kind != "table"]
            assert not any(header in text for text in others), d.meta.doc_id


def expected_blocks(doc: ParsedDocument) -> list[tuple]:
    """Blocks the generator rendered, rebuilt from the Phase 1 content source (read only)."""
    gen = pytest.importorskip("scripts.generate_dataset")  # needs dev deps (reportlab)
    meta = doc.meta
    src = gen.parse_source(gen.CONTENT_DIR / DEPT_DIRS[meta.department] / f"{meta.doc_id}.txt")
    out: list[tuple] = []
    if meta.format != "csv":
        out.append(("paragraph", clean(gen.meta_line(src, meta.access_level)), None))
    for kind, val in src.blocks:
        if kind == "h2":
            out.append(("heading", clean(val), 1))
        elif kind == "h3":
            # Plain text has no marker for subsections: they read as paragraphs.
            out.append(
                ("paragraph", clean(val), None)
                if meta.format == "txt"
                else ("heading", clean(val), 2)
            )
        elif kind == "p":
            out.append(("paragraph", clean(val), None))
        elif kind in ("ul", "ol"):
            out += [("list_item", clean(item), None) for item in val]
        elif kind == "table":
            out.append(("table", [[clean(c) for c in row] for row in val], None))
        elif kind == "code":
            out.append(("code", val.strip("\n"), None))
    return out


def test_structure_matches_source(docs):
    """Every format recovers exactly the blocks, order and levels the generator wrote."""
    for d in docs:
        got = [(b.kind, b.rows if b.kind == "table" else b.text, b.level) for b in d.blocks]
        assert got == expected_blocks(d), d.meta.doc_id


def test_text_joins_blocks(docs):
    for d in docs:
        assert d.text == "\n\n".join(b.text for b in d.blocks)


@pytest.fixture
def copy_root(tmp_path, settings) -> Path:
    root = tmp_path / "repo"
    for rel in (settings.data_raw_dir, settings.data_metadata_dir):
        shutil.copytree(ROOT / rel, root / rel)
    return root


def test_corrupt_pdf(copy_root, settings, docs):
    (copy_root / "data/raw/hr/hr-pol-003-v1.pdf").write_bytes(b"\x00garbage, not a PDF\xff" * 50)
    loaded, errors = load_corpus(copy_root, settings)
    assert any("cannot parse data/raw/hr/hr-pol-003-v1.pdf" in e for e in errors)
    assert len(loaded) == len(docs) - 1


def test_unlisted_file(copy_root, settings, docs):
    (copy_root / "data/raw/it/extra-notes.md").write_text("# Stray\n\nNot in metadata.\n")
    loaded, errors = load_corpus(copy_root, settings)
    assert errors == ["file has no metadata entry: data/raw/it/extra-notes.md"]
    assert len(loaded) == len(docs)


def test_unsupported_extension(copy_root, settings, docs):
    (copy_root / "data/raw/finance/budget.xlsx").write_bytes(b"PK")
    loaded, errors = load_corpus(copy_root, settings)
    assert errors == ["unsupported extension: data/raw/finance/budget.xlsx"]
    assert len(loaded) == len(docs)


def test_missing_file(copy_root, settings, docs):
    (copy_root / "data/raw/operations/ops-tbl-001-v1.csv").unlink()
    loaded, errors = load_corpus(copy_root, settings)
    assert errors == ["metadata entry has no file: data/raw/operations/ops-tbl-001-v1.csv"]
    assert len(loaded) == len(docs) - 1


def test_pdf_split_table_and_paragraph(tmp_path):
    """A table repeated across pages becomes one block; a paragraph split by a page break
    stays one paragraph. The corpus has neither, so build such a PDF here."""
    platypus = pytest.importorskip("reportlab.platypus")
    from reportlab.lib.styles import getSampleStyleSheet

    from app.dataset.schemas import DocumentMeta
    from app.ingestion.pipeline import load_document

    styles = getSampleStyleSheet()
    words = " ".join(f"word{i}" for i in range(1500))  # no sentence ends: one long paragraph
    rows = [["Code", "Rate"]] + [[f"R-{i:03d}", f"{i}.00"] for i in range(90)]
    table = platypus.Table(rows, repeatRows=1)
    table.setStyle(platypus.TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "grey")]))
    path = tmp_path / "split.pdf"
    platypus.SimpleDocTemplate(str(path), invariant=1).build(
        [
            platypus.Paragraph("Split Test", styles["Title"]),
            platypus.Paragraph("Rates", styles["Heading2"]),
            platypus.Paragraph(words, styles["BodyText"]),
            table,
        ]
    )
    meta = DocumentMeta.model_construct(doc_id="x", title="Split Test", format="pdf")
    doc = load_document(path, meta)

    paragraphs = [b for b in doc.blocks if b.kind == "paragraph"]
    tables = [b for b in doc.blocks if b.kind == "table"]
    assert len(paragraphs) == 1
    assert paragraphs[0].text == words
    assert paragraphs[0].page == 1 and doc.page_count > 2  # the paragraph crosses a page
    assert len(tables) == 1
    assert tables[0].rows == rows  # header kept once, all 90 data rows in order
    assert tables[0].page < doc.page_count  # the table continues on later pages
