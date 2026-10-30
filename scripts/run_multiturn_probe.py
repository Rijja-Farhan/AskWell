"""Measure follow-up retrieval with and without query rewriting (retrieval only, no answers).

Usage: python scripts/run_multiturn_probe.py [--offline] [-k N] [--probe PATH]
Per conversation turn, retrieve twice: with the raw question and with the standalone query.
Prints recall@k and MRR for both and writes eval/probes/results/multiturn[-offline].json.

--offline needs no network: it builds a throw-away index in a temp folder with a hashed
bag-of-words embedder, and uses each turn's `scripted_query` as the rewrite (a fixture). Numbers
are then only a smoke test of the pipeline. Live mode uses the configured index, embedder and
LLM; the history it feeds the rewriter holds the probe questions and a placeholder answer.
Exit codes: 0 done, 2 setup problem (API key or index), 3 provider unavailable.
"""

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openai  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402
from app.evaluation.metrics import doc_ranks, recall_at_k, reciprocal_rank  # noqa: E402
from app.rag.errors import IndexNotReadyError, MissingAPIKeyError  # noqa: E402
from app.rag.memory import Turn  # noqa: E402
from app.rag.retrieve import retrieve  # noqa: E402
from app.rag.rewrite import rewrite_query  # noqa: E402

PROBE = ROOT / "eval/probes/multiturn.jsonl"
RESULTS = ROOT / "eval/probes/results"
UNAVAILABLE = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)
TOKEN = re.compile(r"[a-z0-9]+")


class HashedEmbedder:
    """Deterministic offline embedder (hashed bag-of-words), for --offline only."""

    model = "offline-hashed-bow"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * 512
            for token in TOKEN.findall(text.lower()):
                vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % 512] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm == 0:
                vec[0], norm = 1.0, 1.0
            out.append([v / norm for v in vec])
        return out


def load_probe(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: list[float | None]) -> float | None:
    got = [v for v in values if v is not None]
    return round(sum(got) / len(got), 4) if got else None


def score(gold: list[str], chunks: list, k: int) -> dict:
    ranks = doc_ranks(chunks)
    return {"recall": recall_at_k(gold, ranks, k), "mrr": reciprocal_rank(gold, ranks)}


def run_probe(
    conversations: list[dict],
    k: int,
    search: Callable[[str], list],
    rewrite: Callable[[dict, int, list[Turn]], str],
) -> dict:
    """Pure driver (tests call it with fakes). `search(query)` returns ranked chunks."""
    rows = []
    for conv in conversations:
        history: list[Turn] = []
        for i, turn in enumerate(conv["turns"]):
            question, gold = turn["question"], turn["gold_doc_ids"]
            query = rewrite(conv, i, history) if i else question
            raw, new = score(gold, search(question), k), score(gold, search(query), k)
            rows.append(
                {
                    "conversation": conv["id"],
                    "kind": conv.get("kind"),
                    "turn": i + 1,
                    "question": question,
                    "standalone_query": query,
                    "raw": raw,
                    "rewritten": new,
                }
            )
            history.append(Turn(question, query, "(answered)", False, tuple(gold)))
    return summarize(rows, k, len(conversations))


def print_table(result: dict) -> None:
    k = result["k"]
    print(f"{'conv':<6}{'turn':<5}{'raw R@' + str(k):<10}{'new R@' + str(k):<10}"
          f"{'raw MRR':<9}{'new MRR':<9}query")  # fmt: skip
    for r in result["turns"]:
        print(
            f"{r['conversation']:<6}{r['turn']:<5}{r['raw']['recall']:<10.2f}"
            f"{r['rewritten']['recall']:<10.2f}{r['raw']['mrr']:<9.2f}"
            f"{r['rewritten']['mrr']:<9.2f}{r['standalone_query'][:60]}"
        )
    for scope in ("followups", "all_turns"):
        raw, new = (result["summary"][n][scope] for n in ("raw", "rewritten"))
        print(
            f"{scope:<10} recall@{k}: raw {raw['recall']} -> rewritten {new['recall']}; "
            f"MRR: raw {raw['mrr']} -> rewritten {new['mrr']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Raw vs rewritten follow-up retrieval.")
    parser.add_argument("--offline", action="store_true", help="no network: scripted rewrites")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--probe", type=Path, default=PROBE)
    args = parser.parse_args()
    conversations = load_probe(args.probe)
    base = get_settings()

    try:
        if args.offline:
            from app.ingestion.pipeline import load_corpus
            from app.rag.store import build_index

            tmp = tempfile.TemporaryDirectory()
            settings: Settings = base.model_copy(
                update={
                    "vector_db_path": Path(tmp.name) / "chroma",
                    "reranker": "none",
                    "embedding_model": HashedEmbedder.model,
                }
            )
            embedder = HashedEmbedder()
            docs, errors = load_corpus(ROOT, settings)
            if errors:
                print("corpus parse errors; cannot build the offline index", file=sys.stderr)
                return 2
            build_index(docs, embedder, settings)

            def rewrite(conv: dict, i: int, history: list[Turn]) -> str:
                return conv["turns"][i]["scripted_query"]  # the fixture stands in for the LLM

            result = run_probe(
                conversations,
                args.k,
                lambda q: retrieve(q, args.k, embedder, settings, None),
                rewrite,
            )
        else:
            from app.rag.llm import get_embedder, get_llm
            from app.rag.rerank import get_reranker

            settings = base
            embedder, llm = get_embedder(settings), get_llm(settings)
            reranker = get_reranker(settings)

            def rewrite(conv: dict, i: int, history: list[Turn]) -> str:
                question = conv["turns"][i]["question"]
                return rewrite_query(question, history, llm, settings).query

            result = run_probe(
                conversations,
                args.k,
                lambda q: retrieve(q, args.k, embedder, settings, reranker),
                rewrite,
            )
    except (MissingAPIKeyError, IndexNotReadyError) as exc:
        print(exc, file=sys.stderr)
        return 2
    except UNAVAILABLE as exc:
        print(f"Provider unavailable ({type(exc).__name__}); live probe pending.", file=sys.stderr)
        return 3

    result["mode"] = "offline" if args.offline else "live"
    print_table(result)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / ("multiturn-offline.json" if args.offline else "multiturn.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {out.relative_to(ROOT)}")
    return 0


def summarize(rows: list[dict], k: int, n: int) -> dict:
    followups = [r for r in rows if r["turn"] > 1]
    summary = {
        name: {
            "followups": {
                "recall": mean([r[name]["recall"] for r in followups]),
                "mrr": mean([r[name]["mrr"] for r in followups]),
            },
            "all_turns": {
                "recall": mean([r[name]["recall"] for r in rows]),
                "mrr": mean([r[name]["mrr"] for r in rows]),
            },
        }
        for name in ("raw", "rewritten")
    }
    return {"k": k, "conversations": n, "turns": rows, "summary": summary}


if __name__ == "__main__":
    sys.exit(main())
