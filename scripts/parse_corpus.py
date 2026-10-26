"""Parse every corpus file into the common document format.

Writes one `ParsedDocument` JSON object per line to `settings.parsed_documents_path`.
Exit code 0 when every file parsed, 1 otherwise.
"""

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.ingestion.pipeline import load_corpus  # noqa: E402


def main() -> int:
    settings = get_settings()
    docs, errors = load_corpus(ROOT, settings)

    out = ROOT / settings.parsed_documents_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(d.model_dump_json() + "\n" for d in docs), encoding="utf-8")

    blocks = [b for d in docs for b in d.blocks]
    print(f"Parsed {len(docs)} documents -> {settings.parsed_documents_path}")
    print("By format:", dict(sorted(Counter(d.meta.format for d in docs).items())))
    print("Blocks:   ", len(blocks), dict(sorted(Counter(b.kind for b in blocks).items())))
    print("Tables:   ", sum(b.kind == "table" for b in blocks))
    if errors:
        print(f"\n{len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
