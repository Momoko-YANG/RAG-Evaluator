"""File layout of a collected dataset directory (e.g. ``datasets/my-project/``).

::

    <dataset>/
    ├── README.md
    ├── manifest.json              provenance, file hashes, validation
    ├── bank/                      edited by hand (the only files to edit)
    │   ├── questions.csv
    │   └── excluded.csv
    ├── sets/                      written by scripts/prepare_dataset.py
    │   ├── smoke.csv
    │   ├── golden.csv             smoke + golden
    │   └── holdout.csv
    ├── qrels/                     relevance judgments (generated single-gold or reviewed)
    └── chunks/
        └── sampled_chunks.jsonl   index sample the questions were written from

Paths are relative to the dataset directory.
"""

from __future__ import annotations

MANIFEST = "manifest.json"
BANK = "bank/questions.csv"
EXCLUDED = "bank/excluded.csv"
CHUNKS = "chunks/sampled_chunks.jsonl"
QRELS_DIR = "qrels"
SETS_DIR = "sets"
# Set file name → metadata_set values it contains.
SET_FILES: dict[str, tuple[str, ...]] = {
    "smoke.csv": ("smoke",),
    "golden.csv": ("smoke", "golden"),
    "holdout.csv": ("holdout",),
}


def set_path(name: str) -> str:
    """Path of a set file, e.g. ``sets/golden.csv``."""
    return f"{SETS_DIR}/{name}"


def qrels_path(version: str) -> str:
    """Path of a qrels file, e.g. ``qrels/single-gold-v2.jsonl``."""
    return f"{QRELS_DIR}/{version}.jsonl"
