"""Doc-level retrieval metrics over chunk-level hits.

Gold labels are doc ids, so each doc is scored at its best (first) chunk rank. k counts chunks,
not unique docs: k chunks is what the LLM sees. Every metric returns None when there is nothing
to score (no gold docs, or no superseded docs), so those questions drop out of the means.
"""

from collections.abc import Iterable
from typing import Protocol


class RankedHit(Protocol):
    doc_id: str
    rank: int


def doc_ranks(chunks: Iterable[RankedHit]) -> dict[str, int]:
    """Each doc's best (lowest) chunk rank."""
    ranks: dict[str, int] = {}
    for c in chunks:
        if c.doc_id not in ranks or c.rank < ranks[c.doc_id]:
            ranks[c.doc_id] = c.rank
    return ranks


def recall_at_k(gold: list[str], ranks: dict[str, int], k: int) -> float | None:
    """Share of gold docs with a chunk in the top k."""
    if not gold:
        return None
    return sum(1 for d in set(gold) if ranks.get(d, k + 1) <= k) / len(set(gold))


def reciprocal_rank(gold: list[str], ranks: dict[str, int]) -> float | None:
    """1 / rank of the best-ranked gold doc, 0 if none was retrieved."""
    if not gold:
        return None
    best = first_gold_rank(gold, ranks)
    return 0.0 if best is None else 1.0 / best


def superseded_hit(superseded: list[str], ranks: dict[str, int], k: int) -> bool | None:
    """True if any superseded (outdated) version has a chunk in the top k."""
    if not superseded:
        return None
    return any(ranks.get(d, k + 1) <= k for d in superseded)


def first_gold_rank(gold: list[str], ranks: dict[str, int]) -> int | None:
    found = [ranks[d] for d in gold if d in ranks]
    return min(found) if found else None
