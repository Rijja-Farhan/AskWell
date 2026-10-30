"""Parse the corpus, then sync the Chroma index: embed only new or changed documents.

Usage: python scripts/build_index.py [--full]
Parses `data/raw/` in-process, so an edited file is picked up without a separate parse step.
A full rebuild happens with --full, when no index exists yet, or when the embedding model or
chunking config changed.
Exit codes: 0 success, 1 a file failed to parse (index unchanged), 2 API key missing,
3 provider unavailable (index unchanged).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openai  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.rag.errors import CorpusParseError, MissingAPIKeyError  # noqa: E402
from app.rag.ingest import ingest_corpus  # noqa: E402
from app.rag.llm import get_embedder  # noqa: E402

UNAVAILABLE = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync the vector index with data/raw/.")
    parser.add_argument("--full", action="store_true", help="re-embed every document")
    args = parser.parse_args()
    settings = get_settings()

    try:
        stats = ingest_corpus(ROOT, lambda: get_embedder(settings), settings, full=args.full)
    except CorpusParseError as exc:
        print("Parse errors; the index was not changed:")
        for error in exc.errors:
            print(f"  {error}")
        return 1
    except MissingAPIKeyError as exc:
        print(exc)
        return 2
    except UNAVAILABLE as exc:
        print(
            f"Embedding provider unavailable, try again later ({type(exc).__name__}). "
            "The index was not changed."
        )
        return 3

    if stats.full_rebuild:
        print(f"Full rebuild ({stats.reason})")
    for label, ids in [
        ("added", stats.added),
        ("updated", stats.updated),
        ("removed", stats.removed),
    ]:
        shown = f" {ids}" if ids and not stats.full_rebuild else ""
        print(f"  {label}: {len(ids)}{shown}")
    print(f"  unchanged: {len(stats.unchanged)}")
    print(f"Embedded {stats.chunks_embedded} chunks")
    print(
        f"Index: {stats.path} (collection {stats.collection!r}, {settings.chunk_strategy} "
        f"chunks): {stats.documents} documents, {stats.chunks} chunks, {stats.dimension} dims"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
