"""Offline access probe: prove that retrieval never returns a chunk the user may not read.

Usage: python scripts/run_access_probe.py [-k N]
Uses BM25 only (no embedding or LLM call, no reranker) on the existing `chroma_db/`, which is
only read. Two checks, every returned chunk asserted against `accessible_doc_ids(user)`:
  1. the access-denied eval questions, each asked as its listed user;
  2. a matrix of every user (and the guest) against fixed queries: each document's title and
     its key terms.
Prints a users x leaks table and writes eval/probes/results/access_probe.json.
Exit codes: 0 no leaks, 1 at least one leak, 2 setup problem (no index or users).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.dataset.schemas import UserProfile  # noqa: E402
from app.evaluation.runner import load_eval_set  # noqa: E402
from app.rag.access import ACCESS_VERSION, GUEST, accessible_doc_ids, load_users  # noqa: E402
from app.rag.bm25 import get_index  # noqa: E402
from app.rag.errors import IndexNotReadyError  # noqa: E402
from app.rag.filters import RetrievalFilter  # noqa: E402
from app.rag.retrieve import retrieve  # noqa: E402
from app.rag.store import open_collection, resolve  # noqa: E402

OUT = ROOT / "eval/probes/results/access_probe.json"


class NoEmbedder:
    """BM25 needs no embeddings; any call is a bug in this probe."""

    model = "none"

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("the access probe must not embed")


def fixed_queries(documents: list[dict]) -> list[str]:
    queries: list[str] = []
    for doc in documents:
        queries.append(doc["title"])
        if doc.get("key_terms"):
            queries.append(" ".join(doc["key_terms"]))
    return list(dict.fromkeys(queries))


def leaks_for(
    user: UserProfile, queries: list[str], allowed: set[str], settings, k: int
) -> tuple[int, int]:
    """(chunks returned, chunks outside `allowed`) over the queries."""
    returned = leaked = 0
    for query in queries:
        hits = retrieve(query, k, NoEmbedder(), settings, None, RetrievalFilter(user=user))
        returned += len(hits)
        leaked += sum(h.doc_id not in allowed for h in hits)
    return returned, leaked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-k", type=int, default=10, help="chunks retrieved per query")
    args = parser.parse_args()

    base = get_settings()
    settings = base.model_copy(
        update={"retrieval_mode": "bm25", "reranker": "none", "version_policy": "all"}
    )
    try:
        users = {**load_users(settings), GUEST.user_id: GUEST}
        collection = open_collection(settings)
    except (IndexNotReadyError, OSError, ValueError) as exc:
        print(f"setup problem: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    metadatas = get_index(collection).metadatas
    allowed = {uid: accessible_doc_ids(u, metadatas) for uid, u in users.items()}

    documents = json.loads(
        (resolve(settings.data_metadata_dir) / "documents.json").read_text(encoding="utf-8")
    )
    queries = fixed_queries(documents)

    rows = []
    for uid, user in users.items():
        returned, leaked = leaks_for(user, queries, allowed[uid], settings, args.k)
        rows.append(
            {
                "user_id": uid,
                "clearance": user.clearance,
                "departments": user.departments,
                "readable_docs": len(allowed[uid]),
                "queries": len(queries),
                "chunks_returned": returned,
                "leaks": leaked,
            }
        )

    denied_rows = []
    questions = [
        q for q in load_eval_set(resolve(settings.eval_set_path)) if q.category == "access-denied"
    ]
    for q in questions:
        user = users.get(q.user_id, GUEST)
        returned, leaked = leaks_for(user, [q.question], allowed[user.user_id], settings, args.k)
        denied_rows.append(
            {"id": q.id, "user_id": user.user_id, "chunks_returned": returned, "leaks": leaked}
        )

    total = sum(r["leaks"] for r in rows) + sum(r["leaks"] for r in denied_rows)
    print(
        f"{'user':<7}{'clearance':<14}{'depts':<7}{'docs':>5}{'queries':>9}{'chunks':>8}{'leaks':>7}"
    )
    for r in rows:
        print(
            f"{r['user_id']:<7}{r['clearance']:<14}{len(r['departments']):<7}"
            f"{r['readable_docs']:>5}{r['queries']:>9}{r['chunks_returned']:>8}{r['leaks']:>7}"
        )
    print(
        f"access-denied questions: {len(denied_rows)}, leaks {sum(r['leaks'] for r in denied_rows)}"
    )
    print(f"TOTAL LEAKS: {total}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "access_version": ACCESS_VERSION,
                "retrieval_mode": "bm25",
                "reranker": "none",
                "k": args.k,
                "total_leaks": total,
                "users": rows,
                "access_denied_questions": denied_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
