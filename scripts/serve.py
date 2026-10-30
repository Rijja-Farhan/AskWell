"""Run the Askwell API server.

Usage: python scripts/serve.py [--host 127.0.0.1] [--port 8000]
Defaults come from API_HOST / API_PORT. Run one server at a time: the local Chroma store is
single-writer, and no other script should use chroma_db/ while it runs.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from app.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Serve the Askwell API.")
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    args = parser.parse_args()
    uvicorn.run("app.api.main:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
