import pytest

from app.ingestion.models import Block, ParsedDocument
from app.rag.chunking import (
    SENTENCE_END,
    block_offsets,
    chunk_document,
    chunk_structured,
    make_chunks,
    top_level_sections,
)
from tests.conftest import make_settings

MAX = 800
BASE_KEYS = {
    "doc_id",
    "title",
    "department",
    "doc_type",
    "version",
    "status",
    "access_level",
    "effective_date",
    "path",
    "family_id",
    "last_updated",
    "section",
    "section_path",
    "block_kinds",
    "has_table",
    "chunk_strategy",
}


@pytest.fixture(scope="module")
def chunked(corpus) -> list[tuple[ParsedDocument, list]]:
    return [(doc, chunk_structured(doc, MAX, 100)) for doc in corpus]


def by_id(corpus: list[ParsedDocument], doc_id: str) -> ParsedDocument:
    return next(d for d in corpus if d.meta.doc_id == doc_id)


def body(chunk) -> str:
    """Chunk text without its context header."""
    return chunk.text.split("\n\n", 1)[1]


def test_ids_order_header_and_size(chunked):
    for doc, chunks in chunked:
        assert chunks, doc.meta.doc_id
        for i, c in enumerate(chunks):
            assert c.chunk_id == f"{doc.meta.doc_id}::{i:04d}" and c.chunk_index == i
            assert c.text.startswith(doc.title)
            assert len(c.text) <= MAX
            assert 0 <= c.start_char < c.end_char <= len(doc.text)
            if i:
                assert c.start_char >= chunks[i - 1].start_char


def test_every_sentence_and_table_row_is_kept_whole(chunked):
    for doc, chunks in chunked:
        texts = [c.text for c in chunks]
        for block in doc.blocks:
            if block.kind == "table":
                units = block.text.split("\n")
            elif block.kind == "heading":
                units = [block.text]
            else:
                units = SENTENCE_END.split(block.text)
            for unit in units:
                assert any(unit in t for t in texts), (doc.meta.doc_id, unit[:60])


def test_small_blocks_are_never_split(chunked):
    for doc, chunks in chunked:
        for block in doc.blocks:
            if block.kind != "heading" and len(block.text) < MAX // 2:
                assert any(block.text in c.text for c in chunks), (doc.meta.doc_id, block.text)


def test_chunk_stays_in_one_top_level_section(chunked):
    for doc, chunks in chunked:
        offsets = block_offsets(doc)
        section_of = {}
        for indexes in top_level_sections(doc):
            for i in indexes:
                section_of[i] = indexes[0]
        for c in chunks:
            inside = [
                i
                for i, b in enumerate(doc.blocks)
                if offsets[i] >= c.start_char and offsets[i] + len(b.text) <= c.end_char
            ]
            assert len({section_of[i] for i in inside}) <= 1, c.chunk_id


def test_heading_never_ends_a_chunk(chunked):
    for doc, chunks in chunked:
        headings = {b.text for b in doc.blocks if b.kind == "heading"}
        followed = {
            b.text
            for b, nxt in zip(doc.blocks, doc.blocks[1:], strict=False)
            if b.kind == "heading" and nxt.kind != "heading"
        }
        for c in chunks:
            last = c.text.rsplit("\n\n", 1)[-1]
            if last in headings:
                assert last not in followed, c.chunk_id


def test_tables_stand_alone(chunked):
    for _, chunks in chunked:
        for c in chunks:
            if c.metadata["has_table"]:
                assert set(c.metadata["block_kinds"].split(",")) <= {"table", "heading"}


def test_large_table_splits_by_rows_with_header(corpus):
    doc = by_id(corpus, "it-tbl-001-v1")  # 1,320-character table
    table = next(b for b in doc.blocks if b.kind == "table")
    header = table.text.split("\n")[0]
    chunks = [c for c in chunk_structured(doc, MAX, 100) if c.metadata["has_table"]]
    assert len(chunks) >= 2
    for c in chunks:
        assert body(c).startswith(header)
    rows = table.text.split("\n")[1:]
    stored = [line for c in chunks for line in body(c).split("\n")[1:]]
    assert stored == rows  # every row exactly once, in order


def test_metadata_keys_and_types(chunked):
    for doc, chunks in chunked:
        for c in chunks:
            keys = set(c.metadata)
            assert BASE_KEYS <= keys
            assert all(isinstance(v, str | int | float | bool) for v in c.metadata.values())
            pages = {"page_start", "page_end"} & keys
            assert pages == ({"page_start", "page_end"} if doc.meta.format == "pdf" else set())
            assert c.metadata["chunk_strategy"] == "structure"


def test_pdf_page_range(corpus):
    doc = by_id(corpus, "hr-tbl-002-v1")
    for c in chunk_structured(doc, MAX, 100):
        assert 1 <= c.metadata["page_start"] <= c.metadata["page_end"] <= (doc.page_count or 1)


def test_q003_dates_share_a_chunk_with_their_question(corpus):
    chunks = chunk_structured(by_id(corpus, "hr-faq-001-v1"), MAX, 100)
    [dates] = [c for c in chunks if "November 1 to November 15" in c.text]
    assert "When is open enrollment?" in dates.text
    assert dates.metadata["section"] == "Health and Insurance Benefits"
    assert dates.text.startswith("New Hire Onboarding and Benefits FAQ > Health and Insurance")
    for other in ("Your First Week", "Retirement and Wellbeing", "Pay and Time Off"):
        assert other not in dates.text
    # The coverage sentence is no longer cut in half.
    assert "the first day of the month after your start date" in dates.text


def test_long_paragraph_splits_between_sentences_with_overlap(corpus):
    base = by_id(corpus, "hr-faq-001-v1")
    sentences = [f"Sentence number {i} explains one rule in some detail." for i in range(40)]
    text = " ".join(sentences)
    blocks = [
        Block(kind="heading", text="Rules", level=1, section_path=["Rules"]),
        Block(kind="paragraph", text=text, section_path=["Rules"]),
    ]
    doc = base.model_copy(update={"blocks": blocks, "text": "Rules\n\n" + text})
    chunks = chunk_structured(doc, 400, 120)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)
    assert body(chunks[0]).startswith("Rules\n\n")  # the heading travels with its content
    for c in chunks:
        assert all(part in sentences for part in SENTENCE_END.split(body(c).split("Rules\n\n")[-1]))
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        first_next = SENTENCE_END.split(body(nxt))[0]
        assert first_next in body(prev)  # overlap repeats at least one sentence


def test_deterministic(corpus):
    assert [chunk_structured(d, MAX, 100) for d in corpus] == [
        chunk_structured(d, MAX, 100) for d in corpus
    ]


def test_make_chunks_dispatches(corpus, tmp_path):
    doc = by_id(corpus, "hr-faq-001-v1")
    fixed = make_settings(tmp_path, chunk_strategy="fixed")
    structured = make_settings(tmp_path, chunk_strategy="structure")
    assert make_chunks(doc, fixed) == chunk_document(doc, 800, 100)
    assert make_chunks(doc, structured) == chunk_structured(doc, 800, 100)


@pytest.mark.parametrize(("size", "overlap"), [(100, 100), (100, -1), (0, 0)])
def test_bad_settings(corpus, size, overlap):
    with pytest.raises(ValueError):
        chunk_structured(corpus[0], size, overlap)
