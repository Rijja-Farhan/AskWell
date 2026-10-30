"""Parse the corpus and sync the index: the one flow behind `scripts/build_index.py` and
`POST /ingest`, so the script and the API cannot diverge."""

from collections.abc import Callable
from pathlib import Path

from app.config import Settings
from app.ingestion.pipeline import load_corpus
from app.rag.embeddings import Embedder
from app.rag.errors import CorpusParseError
from app.rag.store import SyncStats, sync_index


def ingest_corpus(
    root: Path, embedder_factory: Callable[[], Embedder], settings: Settings, full: bool = False
) -> SyncStats:
    """Parse `root`'s corpus in-process, then sync the index with it.

    The embedder is built only after parsing succeeds, so parse errors are reported even
    without an API key. Raises CorpusParseError (index untouched) if any file fails to parse.
    Embedding failures propagate and also leave the index as it was (see `sync_index`).
    """
    docs, errors = load_corpus(root, settings)
    if errors:
        raise CorpusParseError(errors)
    return sync_index(docs, embedder_factory(), settings, full=full)
