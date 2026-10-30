import importlib.util
import sys

import pytest

from app.ingestion.models import Block, ParsedDocument
from app.rag.chunking import make_chunks
from app.rag.errors import IndexNotReadyError
from app.rag.retrieve import retrieve
from app.rag.store import get_client, open_collection, sync_index
from tests.conftest import ROOT, FakeEmbedder, make_settings


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)  # structured chunks, the default


@pytest.fixture
def synced(corpus, settings):
    """A freshly built structured index."""
    stats = sync_index(corpus, FakeEmbedder(), settings)
    assert stats.full_rebuild and stats.reason == "no index yet"
    return stats


def replace(corpus: list[ParsedDocument], doc: ParsedDocument) -> list[ParsedDocument]:
    return [doc if d.meta.doc_id == doc.meta.doc_id else d for d in corpus]


def doc_chunk_ids(settings, doc_id: str) -> list[str]:
    return sorted(open_collection(settings).get(where={"doc_id": doc_id})["ids"])


def with_note(doc: ParsedDocument, note: str) -> ParsedDocument:
    """The document with one paragraph appended to its last section, as a re-parse would."""
    last = doc.blocks[-1]
    block = Block(kind="paragraph", text=note, section_path=last.section_path, page=last.page)
    return doc.model_copy(update={"blocks": [*doc.blocks, block], "text": f"{doc.text}\n\n{note}"})


def test_second_sync_embeds_nothing(corpus, settings, synced):
    embedder = FakeEmbedder()
    stats = sync_index(corpus, embedder, settings)
    assert not stats.full_rebuild
    assert stats.chunks_embedded == 0 and embedder.calls == 0
    assert (stats.added, stats.updated, stats.removed) == ([], [], [])
    assert len(stats.unchanged) == 52 and stats.chunks == synced.chunks


@pytest.mark.parametrize("change", ["content_hash", "text", "metadata"])
def test_one_changed_doc_is_reembedded(corpus, settings, synced, change):
    doc = next(d for d in corpus if d.meta.doc_id == "hr-mtg-001-v1")
    if change == "content_hash":
        edited = doc.model_copy(update={"content_hash": "0" * 64})
    elif change == "text":
        edited = with_note(doc, "Note: temporary edit.")
    else:  # a status change in data/metadata/documents.json must re-index (Phase 8 filters on it)
        edited = doc.model_copy(
            update={"meta": doc.meta.model_copy(update={"status": "superseded"})}
        )
    embedder = FakeEmbedder()
    stats = sync_index(replace(corpus, edited), embedder, settings)

    assert stats.updated == ["hr-mtg-001-v1"]
    assert stats.added == stats.removed == [] and len(stats.unchanged) == 51
    assert stats.chunks_embedded == len(make_chunks(edited, settings)) and embedder.calls == 1
    stored = open_collection(settings).get(where={"doc_id": "hr-mtg-001-v1"}, include=["metadatas"])
    assert len(stored["ids"]) == stats.chunks_embedded
    if change == "metadata":
        assert {m["status"] for m in stored["metadatas"]} == {"superseded"}
    # Syncing the original back re-embeds that document again, and only it.
    assert sync_index(corpus, FakeEmbedder(), settings).updated == ["hr-mtg-001-v1"]


def test_removed_doc_loses_its_chunks_and_comes_back(corpus, settings, synced):
    before = doc_chunk_ids(settings, "it-tbl-001-v1")
    assert before
    stats = sync_index(
        [d for d in corpus if d.meta.doc_id != "it-tbl-001-v1"], FakeEmbedder(), settings
    )
    assert stats.removed == ["it-tbl-001-v1"] and stats.chunks_embedded == 0
    assert stats.chunks == synced.chunks - len(before)
    assert doc_chunk_ids(settings, "it-tbl-001-v1") == []
    stats = sync_index(corpus, FakeEmbedder(), settings)
    assert stats.added == ["it-tbl-001-v1"] and stats.chunks_embedded == len(before)


def test_config_change_triggers_full_rebuild(corpus, settings, synced):
    smaller = settings.model_copy(update={"chunk_size": 600})
    stats = sync_index(corpus, FakeEmbedder(), smaller)
    assert stats.full_rebuild and "chunk_size 800 -> 600" in stats.reason
    assert stats.chunks > synced.chunks
    assert not sync_index(corpus, FakeEmbedder(), smaller).full_rebuild


def test_full_flag_rebuilds(corpus, settings, synced):
    stats = sync_index(corpus, FakeEmbedder(), settings, full=True)
    assert stats.full_rebuild and stats.chunks_embedded == synced.chunks == stats.chunks


class FailingEmbedder(FakeEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("provider down")


def test_failed_embedding_leaves_old_chunks(corpus, settings, synced):
    doc = next(d for d in corpus if d.meta.doc_id == "hr-mtg-001-v1")
    before = open_collection(settings).get(where={"doc_id": "hr-mtg-001-v1"})
    with pytest.raises(RuntimeError, match="provider down"):
        sync_index(replace(corpus, with_note(doc, "Lost edit.")), FailingEmbedder(), settings)
    after = open_collection(settings).get(where={"doc_id": "hr-mtg-001-v1"})
    assert after["ids"] == before["ids"] and after["documents"] == before["documents"]
    assert open_collection(settings).count() == synced.chunks
    # The change is still pending, so the next sync with a working embedder applies it.
    stats = sync_index(replace(corpus, with_note(doc, "Lost edit.")), FakeEmbedder(), settings)
    assert stats.updated == ["hr-mtg-001-v1"]


def test_strategy_mismatch_is_reported(settings, synced):
    with pytest.raises(IndexNotReadyError, match="CHUNK_STRATEGY"):
        open_collection(settings.model_copy(update={"chunk_strategy": "fixed"}))


def test_collection_records_config(settings, synced):
    meta = get_client(settings).get_collection(settings.vector_collection).metadata
    assert meta["chunk_strategy"] == "structure" and meta["chunk_size"] == 800
    assert meta["embedding_model"] == FakeEmbedder.model and meta["dimension"] == 512


def test_q003_dates_chunk_in_top_5(settings, synced):
    hits = retrieve("When is open enrollment for benefits?", 5, FakeEmbedder(), settings)
    assert any("November 1 to November 15" in h.text for h in hits)


def test_build_index_script_refuses_parse_errors(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "build_index", ROOT / "scripts" / "build_index.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    # The script and POST /ingest share app.rag.ingest, which holds the parse and sync calls.
    monkeypatch.setattr(
        "app.rag.ingest.load_corpus", lambda root, s: ([], ["cannot parse x.pdf: bad"])
    )
    monkeypatch.setattr("app.rag.ingest.sync_index", lambda *a, **k: pytest.fail("index touched"))
    monkeypatch.setattr(sys, "argv", ["build_index.py"])
    assert script.main() == 1
    out = capsys.readouterr().out
    assert "cannot parse x.pdf" in out and "not changed" in out
