"""Write a CSV sheet for reviewing a sample of a question bank by hand.

The automated checks (``prepare_dataset.py``) only show that the labels are
consistent; whether a question is clear and its answer really supported needs a
person.  This script picks ``--size`` questions of the chosen sets, spread over
source documents (one question per source parent, ordered by a seeded hash, so the
sample is reproducible), and writes one row per question with:

* the question, answer, evidence and source document;
* the source chunk text (from ``chunks/sampled_chunks.jsonl``);
* optionally the other chunks a saved run retrieved (``--run``), to judge whether
  other valid evidence exists — these are candidates for pooled qrels;
* empty review columns with the allowed values in the header.

Usage
-----
    python scripts/build_review_sheet.py --dataset datasets/my-project --size 40 \\
        --run output/<run>/top10/retrieval.jsonl --output output/<review>/review-sheet.csv

The sheet contains document text: keep it out of Git (write it under ``output/``).
Holdout questions are left out by default so that reviewing does not tune on them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_eval_framework.datasets.layout import BANK, CHUNKS  # noqa: E402

REVIEW_COLUMNS = {
    "review_question_clear": "ok / ambiguous / unclear",
    "review_answer_supported": "full / partial / not_supported",
    "review_other_evidence": "yes / no / not_checked",
    "review_realistic": "yes / no",
    "review_decision": "keep / fix / exclude",
    "reviewer": "",
    "review_notes": "",
}
SNIPPET = 400


def _rank(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def sample(
    rows: list[dict[str, str]], sets: set[str], size: int, seed: str
) -> list[dict[str, str]]:
    """One question per source parent, in seeded-hash order, up to *size*."""
    seen: set[str] = set()
    picked: list[dict[str, str]] = []
    candidates = [r for r in rows if (r.get("metadata_set") or "") in sets]
    for row in sorted(candidates, key=lambda r: _rank(seed, r["id"])):
        parent = row.get("metadata_source_parent_hash") or row["id"]
        if parent in seen:
            continue
        seen.add(parent)
        picked.append(row)
        if len(picked) == size:
            break
    return sorted(picked, key=lambda r: r["id"])


def _other_chunks(record: dict[str, Any] | None, gold_text: str, k: int) -> str:
    if not record:
        return ""
    lines = []
    for rank, doc in enumerate(record.get("retrieved_documents", [])[:k], start=1):
        content = str(doc.get("content", ""))
        if content.strip() == gold_text.strip():
            lines.append(f"[{rank}] (source chunk)")
            continue
        snippet = " ".join(content.split())[:SNIPPET]
        lines.append(f"[{rank}] {doc.get('title', '')}: {snippet}")
    return "\n".join(lines)


def build_rows(
    picked: list[dict[str, str]],
    chunks: dict[str, str],
    run: dict[str, dict[str, Any]],
    k: int,
) -> list[dict[str, str]]:
    out = []
    for row in picked:
        gold = chunks.get(row["id"], "")
        out.append(
            {
                "id": row["id"],
                "question": row.get("question", ""),
                "ground_truth_answer": row.get("ground_truth_answer", ""),
                "ground_truth_evidence": row.get("ground_truth_evidence", ""),
                "ground_truth_documents": row.get("ground_truth_documents", ""),
                "source_chunk": gold,
                f"retrieved_top{k}": _other_chunks(run.get(row["id"]), gold, k),
                **{column: "" for column in REVIEW_COLUMNS},
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", required=True, type=Path, help="Dataset directory.")
    p.add_argument("--size", type=int, default=40)
    p.add_argument("--sets", default="smoke,golden", help="metadata_set values to sample from.")
    p.add_argument("--seed", default="review-1")
    p.add_argument("--run", type=Path, default=None, help="Saved retrieval JSONL of the same bank.")
    p.add_argument("--top-k", type=int, default=5, help="Retrieved chunks shown per question.")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    with (args.dataset / BANK).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    chunks: dict[str, str] = {}
    chunk_file = args.dataset / CHUNKS
    if chunk_file.exists():
        for line in chunk_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunk = json.loads(line)
                for qid in chunk.get("question_ids", []):
                    chunks[qid] = chunk.get("text", "")
    run: dict[str, dict[str, Any]] = {}
    if args.run:
        for line in args.run.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                run[record["id"]] = record

    sets = {s.strip() for s in args.sets.split(",") if s.strip()}
    picked = sample(rows, sets, args.size, args.seed)
    sheet = build_rows(picked, chunks, run, args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so that Excel opens the Japanese text correctly.
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        fields = list(sheet[0]) if sheet else ["id"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({c: REVIEW_COLUMNS.get(c, "") for c in fields})  # allowed values
        writer.writerows(sheet)
    print(f"Wrote {len(sheet)} questions to {args.output} (row 2 lists the allowed values)")


if __name__ == "__main__":
    main()
