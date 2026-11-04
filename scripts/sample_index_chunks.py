"""Randomly sample chunks of an extension's search index for question generation.

Questions written only from chunks the retriever already returns make the
retrieval scores look better than they are.  This script lists the whole index
through the local evaluation API (``POST /api/rag-eval/sample``) and picks
chunks at random, so the questions cover the index evenly.

Usage
-----
    export RAG_EVAL_API_KEY=...          # same key as the app's /api/rag-eval
    python scripts/sample_index_chunks.py \\
        --extension-id <extension id> \\
        --output datasets/<project>/chunks/sampled_chunks.jsonl --num 120

Add ``--append`` to sample more chunks later: chunks already in ``--output`` are
skipped and the new ones are appended.

Each output line: ``{"position", "metadata", "text", "usable"}``.  ``usable`` is
False for chunks that are only page headers/footers, URLs or other boilerplate;
questions should be written from usable chunks only.  Add the headers / footers of
your own documents with ``--boilerplate-pattern REGEX`` (repeatable) or
``--boilerplate-file FILE`` (one regex per line, ``#`` for comments).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KEY_HEADER = "x-rag-eval-key"

# Text commonly repeated on every page of PDFs; extend it per corpus with
# --boilerplate-pattern / --boilerplate-file.
DEFAULT_BOILERPLATE_PATTERNS = [
    r"Print Date:\s*\S+",
    r"Page\s+\d+\s+of\s+\d+",
    r"https?://\S+",
]
MIN_CONTENT_CHARS = 150


def content_without_boilerplate(
    text: str, patterns: list[str] = DEFAULT_BOILERPLATE_PATTERNS
) -> str:
    for pattern in patterns:
        text = re.sub(pattern, " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_usable(text: str, patterns: list[str] = DEFAULT_BOILERPLATE_PATTERNS) -> bool:
    """True when the chunk has enough document text besides boilerplate."""
    return len(content_without_boilerplate(text, patterns)) >= MIN_CONTENT_CHARS


def read_patterns(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def fetch_all(
    url: str, key: str, extension_id: str, page_size: int, timeout: int
) -> list[dict[str, Any]]:
    import requests

    chunks: list[dict[str, Any]] = []
    total = None
    while total is None or len(chunks) < total:
        resp = requests.post(
            url,
            json={"extensionId": extension_id, "skip": len(chunks), "top": page_size},
            headers={KEY_HEADER: key},
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        total = data["total"]
        if not data["chunks"]:
            break
        chunks.extend(data["chunks"])
        logger.info("Fetched %d / %d chunks", len(chunks), total)
    return chunks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--extension-id", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--num", type=int, default=120, help="Chunks to sample (default: 120).")
    p.add_argument(
        "--append", action="store_true",
        help="Sample only chunks not yet in --output and append them to it.",
    )
    p.add_argument(
        "--boilerplate-pattern", action="append", default=[], metavar="REGEX",
        help="Regex of repeated header/footer text to ignore (repeatable).",
    )
    p.add_argument(
        "--boilerplate-file", type=Path,
        help="File with one boilerplate regex per line ('#' starts a comment).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--url", default="http://localhost:3000/api/rag-eval/sample")
    p.add_argument("--page-size", type=int, default=500)
    p.add_argument("--request-timeout", type=int, default=120)
    return p


def main() -> None:
    args = build_parser().parse_args()
    key = os.environ.get("RAG_EVAL_API_KEY", "")
    if not key:
        logger.error("Set env RAG_EVAL_API_KEY (same key as the app's /api/rag-eval).")
        sys.exit(1)
    patterns = DEFAULT_BOILERPLATE_PATTERNS + args.boilerplate_pattern
    if args.boilerplate_file:
        patterns += read_patterns(args.boilerplate_file)

    chunks = fetch_all(args.url, key, args.extension_id, args.page_size, args.request_timeout)
    usable = [c for c in chunks if is_usable(c["pageContent"], patterns)]
    logger.info(
        "Index has %d chunks; %d (%.0f%%) are boilerplate only (headers/footers/URLs).",
        len(chunks),
        len(chunks) - len(usable),
        100 * (len(chunks) - len(usable)) / max(len(chunks), 1),
    )

    sampled_before: set[int] = set()
    if args.append and args.output.exists():
        with args.output.open(encoding="utf-8") as f:
            sampled_before = {json.loads(line)["position"] for line in f if line.strip()}
        usable = [c for c in usable if c["position"] not in sampled_before]
        logger.info("Skipping %d chunks already in %s", len(sampled_before), args.output)

    sampled = random.Random(args.seed).sample(usable, min(args.num, len(usable)))
    sampled.sort(key=lambda c: c["position"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.append else "w", encoding="utf-8") as f:
        for c in sampled:
            record = {
                "position": c["position"],
                "metadata": c["metadata"],
                "text": c["pageContent"],
                "usable": True,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d randomly sampled usable chunks to %s", len(sampled), args.output)


if __name__ == "__main__":
    main()
