"""Persistent Chroma vector store: build or sync the index and open it for queries.

Embeddings are always passed in explicitly. No Chroma embedding function is configured,
because Chroma's default one downloads a model, and tests must stay offline.

Incremental sync keeps its state in the collection itself: each chunk carries its document's
fingerprint, and the collection metadata records the chunking config and embedding model.
"""

import hashlib
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel

from app.config import Settings
from app.ingestion.models import ParsedDocument
from app.rag.chunking import CHUNKER_VERSION, Chunk, make_chunks
from app.rag.embeddings import Embedder
from app.rag.errors import IndexNotReadyError

REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_VECTOR_DBS = ("chroma",)
ADD_BATCH = 1000  # Chroma limits the number of records per add() call


class IndexStats(BaseModel):
    documents: int
    chunks: int
    dimension: int
    embedding_model: str
    collection: str
    path: str


def resolve(path: Path) -> Path:
    """Resolve a settings path against the repo root (absolute paths are kept)."""
    return path if path.is_absolute() else REPO_ROOT / path


def get_client(settings: Settings) -> chromadb.ClientAPI:
    if settings.vector_db not in SUPPORTED_VECTOR_DBS:
        raise ValueError(
            f"unsupported vector_db {settings.vector_db!r}; "
            f"supported: {', '.join(SUPPORTED_VECTOR_DBS)}"
        )
    return chromadb.PersistentClient(
        path=str(resolve(settings.vector_db_path)),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


class SyncStats(IndexStats):
    added: list[str] = []
    updated: list[str] = []
    removed: list[str] = []
    unchanged: list[str] = []
    chunks_embedded: int = 0
    full_rebuild: bool = False
    reason: str = ""  # why a full rebuild happened


def index_config(settings: Settings, embedding_model: str) -> dict[str, str | int]:
    """Everything that changes every chunk. A difference forces a full rebuild."""
    return {
        "embedding_model": embedding_model,
        "chunk_strategy": settings.chunk_strategy,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunker_version": CHUNKER_VERSION,
    }


def doc_fingerprint(doc: ParsedDocument, settings: Settings) -> str:
    """Changes when the file, its parsed text, its metadata entry or the chunking config do."""
    payload = {
        "content_hash": doc.content_hash,
        "text_sha256": hashlib.sha256(doc.text.encode()).hexdigest(),
        "meta": doc.meta.model_dump(mode="json"),
        "chunking": [settings.chunk_strategy, settings.chunk_size, settings.chunk_overlap],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _doc_chunks(doc: ParsedDocument, settings: Settings) -> list[Chunk]:
    fingerprint = doc_fingerprint(doc, settings)
    return [
        c.model_copy(update={"metadata": {**c.metadata, "doc_fingerprint": fingerprint}})
        for c in make_chunks(doc, settings)
    ]


def _embed(chunks: list[Chunk], embedder: Embedder) -> list[list[float]]:
    vectors = embedder.embed([c.text for c in chunks]) if chunks else []
    if len(vectors) != len(chunks):
        raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks")
    return vectors


def _add(collection, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    for i in range(0, len(chunks), ADD_BATCH):
        batch = chunks[i : i + ADD_BATCH]
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors[i : i + ADD_BATCH],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )


def _rebuild_reason(collection, config: dict) -> str:
    """Why the existing collection cannot be updated incrementally ("" if it can)."""
    if collection is None:
        return "no index yet"
    saved = collection.metadata or {}
    changed = [k for k, v in config.items() if saved.get(k) != v]
    if changed:
        return "config changed: " + ", ".join(
            f"{k} {saved.get(k)!r} -> {config[k]!r}" for k in changed
        )
    return ""


def sync_index(
    docs: list[ParsedDocument], embedder: Embedder, settings: Settings, full: bool = False
) -> SyncStats:
    """Bring the collection in line with `docs`, embedding only new or changed documents.

    New chunks are always embedded before anything is deleted, so an embedding failure leaves
    the index as it was. A full rebuild happens when asked, when there is no index yet, or
    when the embedding model or chunking config differs from the one the index was built with.
    """
    client = get_client(settings)
    try:
        collection = client.get_collection(settings.vector_collection, embedding_function=None)
    except Exception:  # noqa: BLE001 - Chroma raises NotFoundError for a missing collection
        collection = None
    config = index_config(settings, embedder.model)
    reason = "requested (--full)" if full else _rebuild_reason(collection, config)
    by_id = {d.meta.doc_id: d for d in docs}

    if reason:
        chunks = [c for d in docs for c in _doc_chunks(d, settings)]
        vectors = _embed(chunks, embedder)
        if collection is not None:
            client.delete_collection(settings.vector_collection)
        collection = client.create_collection(
            name=settings.vector_collection,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={**config, "dimension": len(vectors[0]) if vectors else 0},
        )
        _add(collection, chunks, vectors)
        return _stats(
            collection,
            settings,
            embedder,
            docs,
            added=sorted(by_id),
            chunks_embedded=len(chunks),
            full_rebuild=True,
            reason=reason,
        )

    # Incremental: compare each document's fingerprint with the one stored on its chunks.
    stored = collection.get(include=["metadatas"])
    indexed: dict[str, tuple[str, list[str]]] = {}
    for chunk_id, meta in zip(stored["ids"], stored["metadatas"], strict=True):
        fingerprint, ids = indexed.get(str(meta["doc_id"]), (str(meta.get("doc_fingerprint")), []))
        indexed[str(meta["doc_id"])] = (fingerprint, [*ids, chunk_id])
    added = sorted(d for d in by_id if d not in indexed)
    updated = sorted(
        d for d in by_id if d in indexed and indexed[d][0] != doc_fingerprint(by_id[d], settings)
    )
    removed = sorted(d for d in indexed if d not in by_id)
    unchanged = sorted(d for d in by_id if d in indexed and d not in updated)

    chunks = [c for d in added + updated for c in _doc_chunks(by_id[d], settings)]
    vectors = _embed(chunks, embedder)  # may raise: nothing has been changed yet
    stale = [i for d in updated + removed for i in indexed[d][1]]
    if stale:
        collection.delete(ids=stale)
    _add(collection, chunks, vectors)
    return _stats(
        collection,
        settings,
        embedder,
        docs,
        added=added,
        updated=updated,
        removed=removed,
        unchanged=unchanged,
        chunks_embedded=len(chunks),
    )


def _stats(collection, settings, embedder, docs, **fields) -> SyncStats:
    return SyncStats(
        documents=len(docs),
        chunks=collection.count(),
        dimension=int((collection.metadata or {}).get("dimension", 0)),
        embedding_model=embedder.model,
        collection=settings.vector_collection,
        path=str(resolve(settings.vector_db_path)),
        **fields,
    )


def build_index(docs: list[ParsedDocument], embedder: Embedder, settings: Settings) -> SyncStats:
    """Delete and recreate the collection, then add every chunk with its embedding."""
    return sync_index(docs, embedder, settings, full=True)


def open_collection(settings: Settings):
    """Return the built collection, or raise IndexNotReadyError with what to do."""
    hint = "Run `python scripts/build_index.py` first."
    client = get_client(settings)
    try:
        collection = client.get_collection(settings.vector_collection, embedding_function=None)
    except Exception as exc:  # Chroma raises NotFoundError for a missing collection
        raise IndexNotReadyError(
            f"vector collection {settings.vector_collection!r} not found. {hint}"
        ) from exc
    if collection.count() == 0:
        raise IndexNotReadyError(
            f"vector collection {settings.vector_collection!r} is empty. {hint}"
        )
    built_with = (collection.metadata or {}).get("embedding_model")
    if built_with != settings.embedding_model:
        raise IndexNotReadyError(
            f"the index was built with embedding model {built_with!r}, but the current model is "
            f"{settings.embedding_model!r}. Rerun `python scripts/build_index.py`."
        )
    # Indexes built before Phase 5 have no chunk_strategy key; they hold fixed chunks.
    built_strategy = (collection.metadata or {}).get("chunk_strategy", "fixed")
    if built_strategy != settings.chunk_strategy:
        raise IndexNotReadyError(
            f"collection {settings.vector_collection!r} holds {built_strategy!r} chunks, but "
            f"CHUNK_STRATEGY is {settings.chunk_strategy!r}. Use the matching collection "
            "(VECTOR_COLLECTION) or rebuild with `python scripts/build_index.py`."
        )
    return collection
