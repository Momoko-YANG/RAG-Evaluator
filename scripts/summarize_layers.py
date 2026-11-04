"""Summarize a layered RAG evaluation report (retrieval / generation / end-to-end).

Reads the JSON report written by ``rag-eval`` and the dataset it scored, and
writes a Markdown summary that shows, per question, which layer failed:

* retrieval  — ``retrieval_hit_rate`` (was a labelled chunk in the top k?)
* generation — ``groundedness`` / ``hallucination_judge`` (did the answer stick to the chunks?)
* end-to-end — ``accuracy_judge`` (was the answer correct against the ground truth?)

Usage
-----
    python scripts/summarize_layers.py \\
        --report output/<run>/<project>/eval-report-<project>.json \\
        --dataset output/<run>/e2e.jsonl \\
        --output output/<run>/layers.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

CORRECT = 0.75  # accuracy_judge 4/5 or better (scores are normalised 1-5 -> 0-1)
FAITHFUL = 0.75  # groundedness / hallucination_judge 4/5 or better

QUADRANTS = {
    (True, True): "retrieval hit, answer correct — OK",
    (True, False): "retrieval hit, answer wrong — generation problem (prompt / model)",
    (False, True): "retrieval miss, answer correct — answered from another chunk",
    (False, False): "retrieval miss, answer wrong — retrieval problem",
}


def _score(results: dict[str, Any], name: str) -> float | None:
    r = results.get(name)
    if not r or r.get("status") != "success":
        return None
    return float(r["score"])


def merge_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-record evaluator results of several reports on the same dataset."""
    merged: dict[str, dict[str, Any]] = {}
    for report in reports:
        for rr in report["record_results"]:
            entry = merged.setdefault(
                rr["record_id"], {"record_id": rr["record_id"], "evaluator_results": {}}
            )
            entry["evaluator_results"].update(rr["evaluator_results"])
    return {"record_results": list(merged.values())}


def summarize(report: dict[str, Any], records: dict[str, dict[str, Any]]) -> str:
    rows = []
    for rr in report["record_results"]:
        res = rr["evaluator_results"]
        rec = records.get(rr["record_id"], {})
        rows.append(
            {
                "id": rr["record_id"],
                "question": rec.get("question", ""),
                "answerable": bool(
                    rec.get("ground_truth_evidence") or rec.get("ground_truth_documents")
                ),
                "hit": _score(res, "retrieval_hit_rate"),
                "mrr": _score(res, "retrieval_mrr"),
                "ndcg": _score(res, "document_retrieval"),
                "retrieval_llm": _score(res, "retrieval"),
                "grounded": _score(res, "groundedness"),
                "hallucination": _score(res, "hallucination_judge"),
                "relevance": _score(res, "relevance"),
                "accuracy": _score(res, "accuracy_judge"),
                "completeness": _score(res, "response_completeness"),
            }
        )

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    def avg(items: list[dict[str, Any]], key: str) -> str:
        vals = [r[key] for r in items if r[key] is not None]
        return f"{mean(vals):.2f}" if vals else "-"

    def rate(items: list[dict[str, Any]], key: str, threshold: float) -> str:
        vals = [r[key] for r in items if r[key] is not None]
        return f"{sum(v >= threshold for v in vals) / len(vals):.0%}" if vals else "-"

    layer_rows = [
        ("1 Retrieval", "retrieval_hit_rate (gold chunk in the top k)", "hit", None),
        ("1 Retrieval", "retrieval_mrr (rank of the gold chunk)", "mrr", None),
        ("1 Retrieval", "document_retrieval (Azure SDK nDCG@3)", "ndcg", None),
        ("1 Retrieval", "retrieval (Azure SDK, LLM-judged chunk relevance)", "retrieval_llm",
         FAITHFUL),
        ("2 Generation", "groundedness (answer based on the chunks)", "grounded", FAITHFUL),
        ("2 Generation", "hallucination_judge (no unsupported claims)", "hallucination", FAITHFUL),
        ("2 Generation", "relevance (answers the question)", "relevance", FAITHFUL),
        ("3 E2E", "accuracy_judge (matches the ground truth)", "accuracy", CORRECT),
        ("3 E2E", "response_completeness (covers the ground truth)", "completeness", CORRECT),
    ]
    lines = [
        "# Layered RAG evaluation summary",
        "",
        f"- Questions: {len(rows)}"
        f" (answerable {len(answerable)} / not answered by the documents {len(unanswerable)})",
        "",
        "## Scores per layer (answerable questions)",
        "",
        "| Layer | Metric | Mean | Pass rate |",
        "|---|---|---|---|",
    ]
    for layer, label, key, threshold in layer_rows:
        passed = rate(answerable, key, threshold) if threshold is not None else "-"
        lines.append(f"| {layer} | {label} | {avg(answerable, key)} | {passed} |")
    lines += ["", f"Pass: 4 or better on the 1-5 scale ({CORRECT} normalised).", ""]

    if unanswerable:
        refused = sum(1 for r in unanswerable if (r["accuracy"] or 0) >= CORRECT)
        lines += [
            "## Questions the documents do not answer",
            "",
            f"- Correctly declined: {refused} / {len(unanswerable)}",
            "- relevance is ignored here: it scores a correct refusal low.",
            "",
        ]

    quad = Counter()
    by_quad: dict[tuple[bool, bool], list[dict[str, Any]]] = {}
    for r in answerable:
        if r["hit"] is None or r["accuracy"] is None:
            continue
        key = (r["hit"] >= 1.0, r["accuracy"] >= CORRECT)
        quad[key] += 1
        by_quad.setdefault(key, []).append(r)

    total = sum(quad.values()) or 1
    lines += [
        "## Which layer failed (retrieval x answer)",
        "",
        "| Bucket | Questions | Share |",
        "|---|---|---|",
    ]
    for key, label in QUADRANTS.items():
        lines.append(f"| {label} | {quad[key]} | {quad[key] / total:.0%} |")
    lines.append("")

    unfaithful = [
        r for r in answerable
        if r["grounded"] is not None and min(r["grounded"], r["hallucination"] or 1) < FAITHFUL
    ]
    lines += [
        f"- Unsupported claims (groundedness / hallucination below 4): {len(unfaithful)}",
        "",
    ]

    for key in [(True, False), (False, False)]:
        items = by_quad.get(key, [])
        if not items:
            continue
        lines += [f"### {QUADRANTS[key]}", ""]
        for r in items:
            lines.append(
                f"- {r['id']} {r['question']} (accuracy {r['accuracy']:.2f}, groundedness "
                f"{r['grounded'] if r['grounded'] is not None else '-'})"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--report", required=True, type=Path, action="append",
        help="rag-eval JSON report; repeat to merge runs of other evaluators on the same dataset.",
    )
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    report = merge_reports([json.loads(r.read_text(encoding="utf-8")) for r in args.report])
    records = {}
    for line in args.dataset.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            records[rec["id"]] = rec
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summarize(report, records), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
