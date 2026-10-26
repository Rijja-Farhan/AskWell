"""Validate the synthetic dataset and print a coverage summary.

Exit code 0 when valid, 1 when any check fails.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.dataset.validate import dataset_summary, validate_dataset  # noqa: E402


def main() -> int:
    summary = dataset_summary(ROOT)
    for key, value in summary.items():
        print(f"{key:20} {value}")

    errors = validate_dataset(ROOT)
    print()
    if errors:
        print(f"INVALID: {len(errors)} error(s)")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("VALID: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
