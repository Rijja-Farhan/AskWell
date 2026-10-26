"""BM25 keyword search over the same chunks the dense index holds.

Dense embeddings capture meaning but blur exact identifiers: "NX-2150" and "NX-2107" embed
almost alike. BM25 scores exact token matches, so it finds the one chunk that names the code.

The index is built in memory from the Chroma collection itself (ids, texts, metadata), so it
always matches the dense index: there is no side file to drift. It is cached per process and
rebuilt when the collection's chunk ids or document fingerprints change.
"""

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

from app.rag.chunking import Scalar

BM25_VERSION = 1  # bump when tokenization or scoring changes
K1 = 1.5
B = 0.75
# Lowercase alphanumeric runs; "-", "." or "_" between two of them keeps an identifier whole:
# nx-2150, hr-pol-005, wh-dal-02, 8.2. Trailing punctuation is dropped ("policy." -> "policy").
TOKEN = re.compile(r"[a-z0-9]+(?:[-._][a-z0-9]+)*")
STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in is it its me my
    of on or our so that the their them there these they this to was we what when where
    which who why will with you your""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class BM25Hit:
    chunk_id: str
    text: str
    metadata: dict[str, Scalar]
    score: float


class BM25Index:
    """Okapi BM25 over a fixed list of chunks."""

    def __init__(self, ids: list[str], texts: list[str], metadatas: list[dict[str, Scalar]]):
        self.ids, self.texts, self.metadatas = ids, texts, metadatas
        self.term_freqs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        doc_freq = Counter(term for tf in self.term_freqs for term in tf)
        n = len(texts)
        # Lucene's idf: always positive, even for a term in more than half the chunks.
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def score(self, query_terms: list[str], i: int) -> float:
        tf, length, total = self.term_freqs[i], self.lengths[i], 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f:
                norm = K1 * (1 - B + B * length / self.avg_length)
                total += self.idf[term] * f * (K1 + 1) / (f + norm)
        return total

    def search(self, query: str, n: int, doc_ids: set[str] | None = None) -> list[BM25Hit]:
        """The n best chunks with a score above 0, best first (ties broken by chunk id).
        With `doc_ids`, only chunks of those documents are ranked; IDF and lengths stay
        corpus-wide, so a kept chunk scores the same with or without the filter."""
        terms = tokenize(query)
        scored = [
            (self.score(terms, i), self.ids[i], i)
            for i in range(len(self.ids))
            if doc_ids is None or self.metadatas[i].get("doc_id") in doc_ids
        ]
        best = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))[:n]
        return [BM25Hit(cid, self.texts[i], self.metadatas[i], score) for score, cid, i in best]


_cache: dict[str, tuple[str, BM25Index]] = {}


def index_key(collection_id: str, ids: list[str], metadatas: list[dict]) -> str:
    """Changes whenever a sync adds, removes or re-embeds a chunk, or the collection is
    recreated (a full rebuild gives it a new id)."""
    digest = hashlib.sha256()
    for chunk_id, meta in sorted(zip(ids, metadatas, strict=True)):
        digest.update(f"{chunk_id}\0{meta.get('doc_fingerprint', '')}\n".encode())
    return f"{collection_id}:{BM25_VERSION}:{digest.hexdigest()}"


def get_index(collection) -> BM25Index:
    """The BM25 index for a Chroma collection, rebuilt only when the collection changed."""
    stored = collection.get(include=["metadatas"])  # cheap: no chunk texts
    key = index_key(str(collection.id), stored["ids"], stored["metadatas"])
    cached = _cache.get(collection.name)
    if cached is None or cached[0] != key:
        full = collection.get(include=["documents", "metadatas"])
        metadatas = [dict(m) for m in full["metadatas"]]
        cached = (key, BM25Index(full["ids"], full["documents"], metadatas))
        _cache[collection.name] = cached
    return cached[1]
