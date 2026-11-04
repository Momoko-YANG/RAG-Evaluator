"""Tests for scripts/prepare_dataset.py."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from rag_eval_framework.datasets.qrels import content_sha256, load_qrels

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_dataset", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prep = _load_module()

FIELDS = [
    "id",
    "question",
    "ground_truth_answer",
    "ground_truth_evidence",
    "metadata_set",
    "metadata_position",
    "ground_truth_documents",
    "metadata_source_key_hash",
    "metadata_content_hash",
    "metadata_source_parent_hash",
    "metadata_label_status",
]


def _dataset(tmp_path: Path, n: int = 10) -> Path:
    d = tmp_path / "ds"
    d.mkdir()
    rows, chunks = [], []
    for i in range(n):
        text = f"Chunk {i}: the answer is {i} days."
        rows.append(
            {
                "id": f"q-{i:03d}",
                "question": f"How many days for case {i}?",
                "ground_truth_answer": f"{i} days",
                "ground_truth_evidence": f"the answer is {i} days",
                "metadata_set": "",
                "metadata_position": str(i),
                "ground_truth_documents": f"doc{i}.pdf",
                "metadata_source_key_hash": f"key{i}",
                "metadata_content_hash": content_sha256(text),
                "metadata_source_parent_hash": f"parent{i}",
                "metadata_label_status": "draft",
            }
        )
        chunks.append(
            {
                "position": i,
                "metadata": f"doc{i}.pdf",
                "text": text,
                "source_key_hash": f"key{i}",
                "source_parent_hash": f"parent{i}",
                "question_ids": [f"q-{i:03d}"],
            }
        )
    (d / "bank").mkdir()
    (d / "chunks").mkdir()
    with (d / "bank" / "questions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (d / "chunks" / "sampled_chunks.jsonl").write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8"
    )
    manifest = {
        "service": "svc",
        "index": "idx",
        "sampleSeed": 7,
        "revisions": [{"version": 1}],
        "files": {"questions.csv": {"sha256": "old flat layout"}},
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["prepare_dataset.py", *argv])
    with pytest.raises(SystemExit) as exc:
        prep.main()
        raise SystemExit(0)
    return int(exc.value.code or 0)


def test_prepare_writes_derived_files_and_check_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _dataset(tmp_path)
    assert _run(monkeypatch, "--dataset", str(d), "--check", "--qrels-version", "sg-v1") == 1

    assert _run(monkeypatch, "--dataset", str(d), "--qrels-version", "sg-v1") == 0
    for name in ("smoke.csv", "golden.csv", "holdout.csv"):
        assert (d / "sets" / name).exists()
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["singleGoldQrels"] == "sg-v1"
    assert manifest["revisions"] == [{"version": 1}]  # hand-written fields kept
    assert manifest["validation"]["problems"] == []
    assert set(manifest["files"]) == {
        "bank/questions.csv",
        "chunks/sampled_chunks.jsonl",
        "sets/smoke.csv",
        "sets/golden.csv",
        "sets/holdout.csv",
        "qrels/sg-v1.jsonl",
    }  # the entry of the old flat path is dropped
    qrels = load_qrels(d / "qrels" / "sg-v1.jsonl")
    holdout = (d / "sets" / "holdout.csv").open(encoding="utf-8")
    holdout_ids = {r["id"] for r in csv.DictReader(holdout)}
    assert holdout_ids and not holdout_ids & set(qrels)  # holdout is kept out of qrels

    # The version is remembered; a second run changes nothing.
    assert _run(monkeypatch, "--dataset", str(d), "--check") == 0
    before = {p.name: p.read_bytes() for p in d.rglob("*") if p.is_file()}
    assert _run(monkeypatch, "--dataset", str(d)) == 0
    assert before == {p.name: p.read_bytes() for p in d.rglob("*") if p.is_file()}


def test_split_settings_are_stored_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _dataset(tmp_path)
    assert _run(monkeypatch, "--dataset", str(d), "--smoke-size", "2") == 0
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == {"holdoutRatio": 0.2, "smokeSize": 2, "seed": "7"}
    # A later run without options keeps smoke at 2 (the default 30 would grow it).
    assert _run(monkeypatch, "--dataset", str(d), "--check") == 0


def test_unanswerable_question_needs_no_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _dataset(tmp_path)
    bank = d / "bank" / "questions.csv"
    with bank.open("a", encoding="utf-8", newline="") as f:
        row = dict.fromkeys(FIELDS, "") | {"id": "q-999", "question": "Is there a gym?"}
        csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n").writerow(row)
    assert _run(monkeypatch, "--dataset", str(d), "--qrels-version", "sg-v1") == 0
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation"]["unanswerable_questions"] == 1
    assert "q-999" not in load_qrels(d / "qrels" / "sg-v1.jsonl")


def test_without_manifest_or_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _dataset(tmp_path)
    (d / "manifest.json").unlink()
    (d / "chunks" / "sampled_chunks.jsonl").unlink()
    assert _run(monkeypatch, "--dataset", str(d)) == 0
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert "evidence_matches" not in manifest["validation"]
    assert not (d / "qrels").exists()


def test_missing_bank_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _dataset(tmp_path)
    (d / "bank" / "questions.csv").unlink()
    assert _run(monkeypatch, "--dataset", str(d)) == 1


def test_validation_failure_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _dataset(tmp_path)
    bank = d / "bank" / "questions.csv"
    text = bank.read_text(encoding="utf-8")
    bank.write_text(text.replace("the answer is 3 days", "not in the chunk"), encoding="utf-8")
    assert _run(monkeypatch, "--dataset", str(d)) == 1
    assert not (d / "sets").exists()

    args = argparse.Namespace(qrels_version=None, holdout_ratio=0.2, smoke_size=3, seed=None)
    out = tmp_path / "out"
    out.mkdir()
    validation, _ = prep.build(d, out, args)
    assert any("q-003: evidence_match failed" in p for p in validation["problems"])


def test_reviewed_qrels_are_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _dataset(tmp_path)
    (d / "qrels").mkdir()
    reviewed = (
        json.dumps(
            {
                "question_id": "q-000",
                "relevant_chunks": [{"chunk_id": "k", "grade": 2}],
                "annotation_status": "reviewed",
                "judgment_scope": "pooled",
                "annotation_version": "p-v1",
            }
        )
        + "\n"
    )
    (d / "qrels" / "p-v1.jsonl").write_text(reviewed, encoding="utf-8")
    assert _run(monkeypatch, "--dataset", str(d), "--qrels-version", "p-v1") == 0
    assert (d / "qrels" / "p-v1.jsonl").read_text(encoding="utf-8") == reviewed


def test_excluded_boilerplate_and_golden_holdout_overlap_are_reported(tmp_path: Path) -> None:
    d = _dataset(tmp_path)
    with (d / "bank" / "excluded.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ground_truth_evidence", "exclusion_reason"])
        writer.writeheader()
        writer.writerow(
            {
                "id": "x-1",
                "ground_truth_evidence": "The answer is 5 days",
                "exclusion_reason": "boilerplate: shared footer",
            }
        )
    rows = list(csv.DictReader((d / "bank" / "questions.csv").open(encoding="utf-8")))
    rows[0]["metadata_set"], rows[1]["metadata_set"] = "golden", "holdout"
    rows[1]["metadata_source_parent_hash"] = rows[0]["metadata_source_parent_hash"]
    for r in rows[2:]:
        r["metadata_set"] = "golden"
    excluded = list(csv.DictReader((d / "bank" / "excluded.csv").open(encoding="utf-8")))

    validation = prep.validate(rows, None, excluded)
    assert validation["boilerplate_evidence_remaining"] == 1
    assert validation["golden_holdout_parent_overlap"] == 1
    assert len(validation["problems"]) == 2
