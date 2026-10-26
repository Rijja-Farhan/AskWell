"""Score the RAG pipeline on the eval set and write a report under eval/reports/<name>/.

Usage: python scripts/run_eval.py --name NAME [--stages retrieval,generate,judge] [--fresh]
       [--limit N] [--compare OTHER_NAME]
Exit codes: 0 all requested stages complete, 1 incomplete for another reason (errors, config
mismatch), 2 setup problem (API key or index), 3 provider unavailable (progress and a partial
report are saved; rerun later to finish).
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.evaluation.runner import STAGES, ConfigMismatchError, judge_model, run_eval  # noqa: E402
from app.rag.errors import IndexNotReadyError, MissingAPIKeyError  # noqa: E402
from app.rag.llm import get_embedder, get_llm  # noqa: E402
from app.rag.rerank import get_reranker  # noqa: E402


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate Askwell on the eval set.")
    parser.add_argument("--name", required=True, help="run name, the folder under eval/reports/")
    parser.add_argument("--stages", default=",".join(STAGES), help="comma-separated stages")
    parser.add_argument("--fresh", action="store_true", help="delete the run and start over")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--compare", metavar="NAME", help="add deltas versus another run")
    args = parser.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if not stages or set(stages) - set(STAGES):
        parser.error(f"--stages must be a comma-separated subset of {','.join(STAGES)}")

    try:
        # Build only the clients the requested stages use: retrieval needs no chat model.
        embedder = get_embedder(settings) if "retrieval" in stages else None
        llm = get_llm(settings) if "generate" in stages else None
        judge = (
            get_llm(settings.model_copy(update={"llm_model": judge_model(settings)}))
            if "judge" in stages
            else None
        )
        reranker = None
        if "retrieval" in stages and settings.reranker != "none":
            start = time.perf_counter()
            reranker = get_reranker(settings)
            print(f"Reranker {settings.rerank_model} loaded in {time.perf_counter() - start:.1f}s")
        summary = run_eval(
            args.name,
            stages,
            embedder,
            llm,
            judge,
            settings,
            fresh=args.fresh,
            limit=args.limit,
            compare=args.compare,
            log=print,
            reranker=reranker,
        )
    except (MissingAPIKeyError, IndexNotReadyError) as exc:
        print(exc, file=sys.stderr)
        return 2
    except (ConfigMismatchError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1

    overall = summary.report["categories"]["overall"]
    print(f"\nReport: {summary.run_dir / 'report.md'}")
    print("Done: " + ", ".join(f"{s} {summary.done[s]}/{summary.n}" for s in summary.stages))
    print(
        f"Overall: recall@5 {overall['recall@5']}, MRR {overall['mrr']} "
        f"({overall['n_scored']} scored); faithfulness {overall['faithfulness']}, "
        f"relevance {overall['relevance']}, behavior pass {overall['behavior_pass']} "
        f"({overall['n_judged']} judged)"
    )
    if summary.stopped_stage:
        print(
            f"The {settings.llm_provider} provider stayed unavailable during "
            f"{summary.stopped_stage!r}. Progress is saved; rerun the same command later.",
            file=sys.stderr,
        )
        return 3
    if not summary.complete:
        print("Some questions are not done (see Errors in the report); rerun to retry them.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
