"""Tests for rag_eval_framework.datasets.splits."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rag_eval_framework.datasets import splits as split


def _bank(chunks: int, per_chunk: int = 2, no_answer: int = 0) -> list[dict[str, str]]:
    rows = [
        {"id": f"g-{c:03d}-{q}", "question": f"q{c}-{q}", "ground_truth_evidence": f"e{c}",
         "metadata_position": str(c)}
        for c in range(chunks)
        for q in range(per_chunk)
    ]
    rows += [{"id": f"x-{i:03d}", "question": f"x{i}", "ground_truth_evidence": ""}
             for i in range(no_answer)]
    return rows


def _sets(rows: list[dict[str, str]]) -> dict[str, str]:
    return {r["id"]: r["metadata_set"] for r in rows}


def test_questions_of_one_chunk_share_a_set() -> None:
    rows = _bank(chunks=40)
    split.assign_sets(rows, holdout_ratio=0.3, smoke_size=10, seed="0")
    by_chunk: dict[str, set[str]] = {}
    for r in rows:
        side = "holdout" if r["metadata_set"] == "holdout" else "golden"
        by_chunk.setdefault(r["metadata_position"], set()).add(side)
    assert all(len(sides) == 1 for sides in by_chunk.values())


def test_questions_of_one_document_share_a_set() -> None:
    rows = _bank(chunks=40)
    for r in rows:  # 4 chunks per document
        r["metadata_source_parent_hash"] = f"doc{int(r['metadata_position']) // 4}"
    split.assign_sets(rows, holdout_ratio=0.3, smoke_size=10, seed="0")
    by_doc: dict[str, set[str]] = {}
    for r in rows:
        side = "holdout" if r["metadata_set"] == "holdout" else "golden"
        by_doc.setdefault(r["metadata_source_parent_hash"], set()).add(side)
    assert all(len(sides) == 1 for sides in by_doc.values())
    assert any("holdout" in sides for sides in by_doc.values())


def test_sizes_follow_ratio_and_smoke_size() -> None:
    rows = _bank(chunks=50, per_chunk=1, no_answer=10)
    split.assign_sets(rows, holdout_ratio=0.2, smoke_size=12, seed="0")
    counts = {s: sum(1 for r in rows if r["metadata_set"] == s) for s in split.SETS}
    assert counts["holdout"] == 12  # 20% of 50 answerable + 20% of 10 not answerable
    assert counts["smoke"] == 12
    no_answer_sets = {r["metadata_set"] for r in rows if r["id"].startswith("x-")}
    assert no_answer_sets == {"smoke", "golden", "holdout"}


def test_existing_sets_are_kept_when_questions_are_added() -> None:
    rows = _bank(chunks=30)
    split.assign_sets(rows, holdout_ratio=0.3, smoke_size=10, seed="0")
    before = _sets(rows)

    added = _bank(chunks=60)[60:]  # chunks 30-59
    added.append({"id": "g-000-9", "question": "new q on chunk 0", "ground_truth_evidence": "e0",
                  "metadata_position": "0"})
    rows += added
    split.assign_sets(rows, holdout_ratio=0.3, smoke_size=10, seed="0")
    after = _sets(rows)

    assert {i: after[i] for i in before} == before
    chunk0 = "holdout" if before["g-000-0"] == "holdout" else "golden"
    assert after["g-000-9"] == chunk0


def test_rerun_gives_the_same_split() -> None:
    a, b = _bank(chunks=30), _bank(chunks=30)
    split.assign_sets(a, holdout_ratio=0.3, smoke_size=10, seed="0")
    split.assign_sets(b, holdout_ratio=0.3, smoke_size=10, seed="0")
    assert _sets(a) == _sets(b)


def test_chunk_in_holdout_and_golden_is_rejected() -> None:
    rows = _bank(chunks=1)
    rows[0]["metadata_set"], rows[1]["metadata_set"] = "holdout", "golden"
    with pytest.raises(ValueError, match="holdout and golden"):
        split.assign_sets(rows, holdout_ratio=0.3, smoke_size=1, seed="0")


def test_read_bank_tolerates_bom_and_adds_set_column(tmp_path: Path) -> None:
    bank = tmp_path / "questions.csv"
    with bank.open("w", encoding="utf-8-sig", newline="") as f:  # BOM, as Excel saves
        writer = csv.DictWriter(f, fieldnames=["id", "question"])
        writer.writeheader()
        writer.writerow({"id": "q1", "question": "?"})
    fields, rows = split.read_bank(bank)
    assert fields == ["id", "question", "metadata_set"]
    assert rows[0]["id"] == "q1"

    bank.write_text("question\n?\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs 'id'"):
        split.read_bank(bank)
