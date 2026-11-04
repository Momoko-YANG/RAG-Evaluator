"""Compare two runs question by question on one metric.

Reads two JSON reports written by ``rag-eval`` (e.g. baseline and knn50 scored with
the same config) and writes a Markdown table of how each question moved:

* both hit / only A / only B / neither — a question "hits" when its value is above 0
  (``--min-score`` to change), so for Hit@k / Recall@k (single gold) it is "in the top k";
* for graded metrics (MRR, nDCG) also improved / worse / unchanged among the questions
  scored in both runs.

Questions skipped or ``not_available`` in either run are counted, never treated as misses.
With single-gold labels a miss can still be a different chunk that answers the question:
check the "only A" / "neither" lists by hand.

Usage
-----
    python scripts/compare_runs.py \\
        --a output/<baseline>/<project>/eval-report-<project>.json --label-a baseline \\
        --b output/<knn50>/<project>/eval-report-<project>.json --label-b knn50 \\
        --evaluator retrieval_hit_rate --dataset <retrieval.jsonl> --output compare.md

``--k 10`` picks one k of a per-k evaluator (``retrieval_recall``, ``retrieval_ndcg``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

BUCKETS = {
    (True, True): "hit in both",
    (True, False): "only A hits (worse in B)",
    (False, True): "only B hits (better in B)",
    (False, False): "hit in neither",
}


def _values(report: dict[str, Any], evaluator: str, k: int | None) -> dict[str, float | None]:
    """Record id → value, or None when skipped / not available / errored."""
    values: dict[str, float | None] = {}
    for rr in report["record_results"]:
        result = rr["evaluator_results"].get(evaluator)
        if result is None:
            continue
        value: float | None = None
        if result.get("status") == "success":
            if k is None:
                value = float(result["score"])
            else:
                item = ((result.get("metadata") or {}).get("per_k") or {}).get(str(k))
                if item is None:
                    raise ValueError(f"{evaluator} has no per-k value for k={k}")
                value = item["value"] if item.get("status") == "ok" else None
        values[rr["record_id"]] = value
    return values


def compare(
    a: dict[str, Any],
    b: dict[str, Any],
    evaluator: str,
    *,
    k: int | None = None,
    min_score: float | None = None,
) -> dict[str, Any]:
    va, vb = _values(a, evaluator, k), _values(b, evaluator, k)
    if not va or not vb:
        raise ValueError(f"evaluator '{evaluator}' is missing from a report")

    def hit(v: float) -> bool:
        return v >= min_score if min_score is not None else v > 0

    common = sorted(set(va) & set(vb))
    scored = [i for i in common if va[i] is not None and vb[i] is not None]
    buckets: dict[tuple[bool, bool], list[str]] = {key: [] for key in BUCKETS}
    moves = {"improved": 0, "worse": 0, "unchanged": 0}
    for i in scored:
        x, y = va[i], vb[i]
        assert x is not None and y is not None
        buckets[(hit(x), hit(y))].append(i)
        moves["improved" if y > x else "worse" if y < x else "unchanged"] += 1
    return {
        "metric": f"{evaluator}@{k}" if k is not None else evaluator,
        "a": va,
        "b": vb,
        "only_in_a": sorted(set(va) - set(vb)),
        "only_in_b": sorted(set(vb) - set(va)),
        "unscored": [i for i in common if va[i] is None or vb[i] is None],
        "scored": scored,
        "buckets": buckets,
        "moves": moves,
        "mean_a": mean(va[i] for i in scored) if scored else None,  # type: ignore[misc]
        "mean_b": mean(vb[i] for i in scored) if scored else None,  # type: ignore[misc]
    }


def render(
    result: dict[str, Any],
    label_a: str,
    label_b: str,
    questions: dict[str, str] | None = None,
) -> str:
    questions = questions or {}
    va, vb = result["a"], result["b"]

    def mean_text(value: float | None) -> str:
        return f"{value:.4f}" if value is not None else "-"

    lines = [
        f"# Per-question comparison: {label_a} (A) vs {label_b} (B) — {result['metric']}",
        "",
        f"- Questions scored in both runs: {len(result['scored'])}",
        f"- Skipped / not_available in one run: {len(result['unscored'])}"
        f" (only in A {len(result['only_in_a'])}, only in B {len(result['only_in_b'])})",
        f"- Mean (questions scored in both): A {mean_text(result['mean_a'])} → "
        f"B {mean_text(result['mean_b'])}",
        "",
        "| Bucket | Questions |",
        "|---|---:|",
    ]
    for key, label in BUCKETS.items():
        lines.append(f"| {label} | {len(result['buckets'][key])} |")
    moves = result["moves"]
    lines += [
        "",
        f"Value changes: improved {moves['improved']} / worse {moves['worse']} / "
        f"unchanged {moves['unchanged']}",
        "",
        "With single-gold labels a miss may still be another chunk that answers the question; "
        "check the lists below by hand.",
    ]
    for key in [(True, False), (False, True), (False, False)]:
        ids = result["buckets"][key]
        if not ids:
            continue
        lines += [
            "",
            f"## {BUCKETS[key]} ({len(ids)})",
            "",
            "| id | A | B | Question |",
            "|---|---:|---:|---|",
        ]
        for i in ids:
            question = questions.get(i, "").replace("|", "\\|").replace("\n", " ")[:80]
            lines.append(f"| {i} | {va[i]:.4f} | {vb[i]:.4f} | {question} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--a", required=True, type=Path, help="JSON report of run A (before).")
    p.add_argument("--b", required=True, type=Path, help="JSON report of run B (after).")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--evaluator", required=True, help="e.g. retrieval_hit_rate")
    p.add_argument("--k", type=int, default=None, help="One k of a per-k evaluator.")
    p.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="A question hits when its value >= this (default: value > 0).",
    )
    p.add_argument("--dataset", type=Path, default=None, help="Scored JSONL, for question text.")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (args.a, args.b)]
    try:
        result = compare(*reports, args.evaluator, k=args.k, min_score=args.min_score)
    except ValueError as exc:
        p.error(str(exc))
    questions: dict[str, str] = {}
    if args.dataset:
        for line in args.dataset.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                questions[record["id"]] = record.get("question", "")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(result, args.label_a, args.label_b, questions), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
