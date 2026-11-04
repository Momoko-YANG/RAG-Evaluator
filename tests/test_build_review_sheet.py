"""Tests for scripts/build_review_sheet.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_review_sheet.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_review_sheet", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sheet = _load_module()


def _rows() -> list[dict[str, str]]:
    rows = [
        {
            "id": f"q-{i:02d}",
            "question": f"q{i}",
            "metadata_set": "golden",
            "metadata_source_parent_hash": f"p{i // 2}",
        }
        for i in range(20)
    ]
    rows += [
        {
            "id": "h-01",
            "question": "h",
            "metadata_set": "holdout",
            "metadata_source_parent_hash": "ph",
        }
    ]
    return rows


def test_sample_is_reproducible_one_per_parent_and_skips_holdout() -> None:
    picked = sheet.sample(_rows(), {"smoke", "golden"}, size=6, seed="s")
    assert picked == sheet.sample(_rows(), {"smoke", "golden"}, size=6, seed="s")
    assert len(picked) == 6
    parents = [r["metadata_source_parent_hash"] for r in picked]
    assert len(set(parents)) == 6
    assert all(r["metadata_set"] != "holdout" for r in picked)
    assert len(sheet.sample(_rows(), {"golden"}, size=50, seed="s")) == 10  # 10 parents


def test_rows_mark_the_source_chunk_among_retrieved() -> None:
    picked = [{"id": "q-01", "question": "q"}]
    run = {
        "q-01": {
            "retrieved_documents": [
                {"title": "a.pdf", "content": "other  text\nhere"},
                {"title": "b.pdf", "content": "gold text"},
            ]
        }
    }
    (row,) = sheet.build_rows(picked, {"q-01": "gold text"}, run, k=5)
    assert row["source_chunk"] == "gold text"
    assert row["retrieved_top5"] == "[1] a.pdf: other text here\n[2] (source chunk)"
    assert set(sheet.REVIEW_COLUMNS) <= set(row)
