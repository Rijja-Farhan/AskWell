"""Top-k retrieval over the Chroma index: dense, BM25 or hybrid, optionally reranked,
restricted by metadata filters (see `app/rag/filters.py`).

- dense: the k chunks whose embeddings are closest to the question's (cosine).
- bm25: the k chunks with the best keyword (BM25) score; no embedding call.
- hybrid: `hybrid_candidates` chunks from each list, merged with weighted reciprocal rank
  fusion (RRF). RRF uses only ranks, so the two lists' incomparable scores never mix.

With a reranker configured, the mode's top `rerank_candidates` are rescored by a
cross-encoder (see `app/rag/rerank.py`) and the best k by that score are returned.
"""

from pydantic import BaseModel

from app.config import Settings
from app.rag.bm25 import get_index
from app.rag.chunking import Scalar
from app.rag.embeddings import Embedder
from app.rag.filters import RetrievalFilter, allowed_doc_ids, effective_policy, prefer_active
from app.rag.rerank import Reranker, rerank
from app.rag.store import open_collection


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    rank: int  # 1-based
    # The mode's own score, higher is better: dense 1 - cosine distance; bm25 the BM25 score;
    # hybrid the RRF score; when reranked, the reranker score. Scores from different modes
    # are not comparable.
    score: float
    text: str
    metadata: dict[str, Scalar]
    # Hybrid only: the chunk's rank in each candidate list (None if it was not in that list).
    dense_rank: int | None = None
    bm25_rank: int | None = None
    # Reranked only: the chunk's rank before reranking (None when not reranked).
    first_stage_rank: int | None = None


def _dense(
    collection, question: str, k: int, embedder: Embedder, doc_ids: set[str] | None = None
) -> list[RetrievedChunk]:
    [query_vector] = embedder.embed([question])
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        where={"doc_id": {"$in": sorted(doc_ids)}} if doc_ids is not None else None,
        include=["documents", "metadatas", "distances"],
    )
    hits = zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
        strict=True,
    )
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=str(metadata["doc_id"]),
            rank=rank,
            score=1.0 - distance,
            text=text,
            metadata=dict(metadata),
        )
        for rank, (chunk_id, text, metadata, distance) in enumerate(hits, start=1)
    ]


def _bm25(
    collection, question: str, k: int, doc_ids: set[str] | None = None
) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=hit.chunk_id,
            doc_id=str(hit.metadata["doc_id"]),
            rank=rank,
            score=hit.score,
            text=hit.text,
            metadata=hit.metadata,
        )
        for rank, hit in enumerate(get_index(collection).search(question, k, doc_ids), start=1)
    ]


def fuse(
    dense: list[RetrievedChunk],
    lexical: list[RetrievedChunk],
    k: int,
    rrf_k: int,
    bm25_weight: float,
) -> list[RetrievedChunk]:
    """Weighted reciprocal rank fusion: score = 1/(rrf_k + dense_rank)
    + bm25_weight/(rrf_k + bm25_rank), summed over the lists a chunk appears in."""
    chunks = {c.chunk_id: c for c in lexical} | {c.chunk_id: c for c in dense}
    dense_ranks = {c.chunk_id: c.rank for c in dense}
    bm25_ranks = {c.chunk_id: c.rank for c in lexical}

    def score(chunk_id: str) -> float:
        total = 0.0
        if chunk_id in dense_ranks:
            total += 1.0 / (rrf_k + dense_ranks[chunk_id])
        if chunk_id in bm25_ranks:
            total += bm25_weight / (rrf_k + bm25_ranks[chunk_id])
        return total

    ranked = sorted(chunks, key=lambda cid: (-score(cid), cid))[:k]
    return [
        chunks[cid].model_copy(
            update={
                "rank": rank,
                "score": score(cid),
                "dense_rank": dense_ranks.get(cid),
                "bm25_rank": bm25_ranks.get(cid),
            }
        )
        for rank, cid in enumerate(ranked, start=1)
    ]


def retrieve(
    question: str,
    k: int,
    embedder: Embedder,
    settings: Settings,
    reranker: Reranker | None = None,
    filters: RetrievalFilter | None = None,
) -> list[RetrievedChunk]:
    """Return the k best chunks for the question, best first, using
    `settings.retrieval_mode` and, unless `settings.reranker` is "none", the reranker
    (build it once with `get_reranker(settings)`). Only chunks the filters allow are ranked;
    the version policy defaults to `settings.version_policy`. Several hits may come from one
    document; no matching chunk gives []."""
    if settings.reranker != "none" and reranker is None:
        raise ValueError(
            f"reranker is {settings.reranker!r} but no reranker was passed to retrieve(); "
            "build one with get_reranker(settings), or set RERANKER=none"
        )
    collection = open_collection(settings)
    policy = effective_policy(filters, settings.version_policy)
    doc_ids = allowed_doc_ids(get_index(collection).metadatas, filters, policy)
    if doc_ids is not None and not doc_ids:
        return []
    if settings.reranker == "none":
        hits = _first_stage(collection, question, k, k, embedder, settings, doc_ids)
    else:
        pool = max(k, settings.rerank_candidates)
        candidates = _first_stage(collection, question, k, pool, embedder, settings, doc_ids)
        hits = rerank(question, candidates, reranker, len(candidates))
    if policy == "prefer_active":
        hits = prefer_active(hits)
    return [h.model_copy(update={"rank": rank}) for rank, h in enumerate(hits[:k], start=1)]


def _first_stage(
    collection,
    question: str,
    k: int,
    pool: int,
    embedder: Embedder,
    settings: Settings,
    doc_ids: set[str] | None,
) -> list[RetrievedChunk]:
    """The mode's top `pool` chunks. Hybrid fuses max(k, hybrid_candidates) from each list,
    so a pool of 2 * hybrid_candidates keeps the whole union, BM25-only hits included."""
    mode = settings.retrieval_mode
    if mode == "dense":
        return _dense(collection, question, pool, embedder, doc_ids)
    if mode == "bm25":
        return _bm25(collection, question, pool, doc_ids)
    n = max(k, settings.hybrid_candidates)
    return fuse(
        _dense(collection, question, n, embedder, doc_ids),
        _bm25(collection, question, n, doc_ids),
        pool,
        settings.rrf_k,
        settings.bm25_weight,
    )
