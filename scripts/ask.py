"""Ask Askwell a question from the command line.

Usage: python scripts/ask.py "question" [-k N] [--show-chunks] [--department D ...]
       [--as-of YYYY-MM-DD] [--version-policy active_only|prefer_active|all]
Exit codes: 0 answered or refused (including no chunk matching the filters), 1 other provider
error, 2 setup problem (API key or index), 3 provider unavailable after retries.
"""

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import get_args

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openai  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.dataset.schemas import Department  # noqa: E402
from app.rag.answer import answer_from_chunks  # noqa: E402
from app.rag.errors import IndexNotReadyError, MissingAPIKeyError  # noqa: E402
from app.rag.filters import RetrievalFilter, VersionPolicy, effective_policy  # noqa: E402
from app.rag.llm import get_embedder, get_llm  # noqa: E402
from app.rag.rerank import get_reranker  # noqa: E402
from app.rag.retrieve import retrieve  # noqa: E402

UNAVAILABLE = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ask Askwell a question.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=settings.retrieval_top_k, help="chunks to retrieve")
    parser.add_argument("--show-chunks", action="store_true", help="print retrieved chunk texts")
    parser.add_argument(
        "--department",
        action="append",
        choices=get_args(Department),
        help="only this department's documents (repeatable)",
    )
    parser.add_argument(
        "--as-of", type=date.fromisoformat, help="the versions in force on YYYY-MM-DD"
    )
    parser.add_argument(
        "--version-policy",
        choices=get_args(VersionPolicy),
        help=f"default {settings.version_policy}",
    )
    args = parser.parse_args()
    filters = RetrievalFilter(
        departments=args.department, as_of=args.as_of, version_policy=args.version_policy
    )

    try:
        embedder = get_embedder(settings)
        llm = get_llm(settings)
        reranker = get_reranker(settings)
        chunks = retrieve(args.question, args.k, embedder, settings, reranker, filters)
        if not chunks and settings.answer_mode == "baseline":
            print("No document matches these filters, so there is nothing to answer from.")
            return 0
        result = answer_from_chunks(args.question, chunks, llm, settings)
    except (MissingAPIKeyError, IndexNotReadyError) as exc:
        print(exc, file=sys.stderr)
        return 2
    except UNAVAILABLE as exc:
        print(
            f"The {settings.llm_provider} provider is unavailable, try again later "
            f"({type(exc).__name__}).",
            file=sys.stderr,
        )
        return 3
    except openai.APIStatusError as exc:
        print(f"Provider error {exc.status_code}: {exc.message}", file=sys.stderr)
        return 1

    print(result.answer.strip())
    if result.refused:
        reasons = {
            "no_context": "no document matches the question and filters",
            "low_score": f"best match scored below {settings.refusal_min_score}, no LLM call",
            "model": "the retrieved sources do not answer it",
        }
        print(f"\n(Refused: {result.refusal_reason}, {reasons[result.refusal_reason or 'model']})")
    elif result.citations:
        print("\nSources:")
        for c in result.citations:
            print(f"  [{c.number}] {c.title}, {c.section} ({c.doc_id})")
    if result.invalid_citations:
        print(f"\n({result.invalid_citations} invalid citation marker(s) removed)")
    method = settings.retrieval_mode
    if reranker:
        method += f", reranked by {reranker.model}"
    if filters.as_of:
        method += f", versions in force on {filters.as_of}"
    else:
        method += f", versions {effective_policy(filters, settings.version_policy)}"
    if filters.departments:
        method += f", departments {'/'.join(filters.departments)}"
    print(f"\nRetrieved ({method}):")
    for c in result.chunks:
        parts = []
        if args.show_chunks and reranker:
            parts.append(f"first stage #{c.first_stage_rank}")
        if args.show_chunks and settings.retrieval_mode == "hybrid":
            parts.append(f"dense #{c.dense_rank or '-'}, bm25 #{c.bm25_rank or '-'}")
        if args.show_chunks:
            m = c.metadata
            version = f"v{m.get('version')} {m.get('status')}"
            parts.append(f"{version}, effective {m.get('effective_date')}")
        ranks = "  " + ", ".join(parts) if parts else ""
        print(f"  {c.rank}. {c.chunk_id} ({c.score:.3f}){ranks}")
        if args.show_chunks:
            print("     " + c.text.replace("\n", "\n     "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
