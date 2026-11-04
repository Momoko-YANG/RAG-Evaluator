"""Assign the questions of a bank to smoke / golden / holdout sets.

The set of each question is stored in its ``metadata_set`` column, and
:func:`assign_sets` fills it in only for questions that have none, so adding
questions never moves existing ones:

* ``holdout`` — not used for tuning; scored now and then to check that the
  settings are not just fitted to the golden set (``holdout_ratio`` of new questions).
* ``golden``  — all other questions; scored for every settings decision.
* ``smoke``   — a fixed part of golden (``smoke_size`` questions), run after every
  code or prompt change.  ``golden.csv`` includes the smoke questions.

Questions from the same source document (``metadata_source_parent_hash``), or else the
same chunk (``metadata_position``), always get the same set, so holdout questions never
share a document (or chunk) with golden ones.  Questions the documents do not answer
(no ``ground_truth_evidence`` / ``ground_truth_documents``) are split separately, so that
every set gets some.  The split is a hash of the group and the seed, so it is reproducible.

``scripts/prepare_dataset.py`` uses this to write ``sets/`` of a dataset directory.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

SET_COLUMN = "metadata_set"
SETS = ("smoke", "golden", "holdout")
GOLDEN_SIDE = ("smoke", "golden")


def _rank(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def group_key(row: dict[str, str]) -> str:
    """Questions of one source document (else one chunk) always share a set."""
    parent = (row.get("metadata_source_parent_hash") or "").strip()
    if parent:
        return f"doc:{parent}"
    position = (row.get("metadata_position") or "").strip()
    return f"chunk:{position}" if position else f"id:{row['id']}"


def is_answerable(row: dict[str, str]) -> bool:
    """A bank row is answerable when it has evidence or source-document labels."""
    return bool(
        (row.get("ground_truth_evidence") or "").strip()
        or (row.get("ground_truth_documents") or "").strip()
    )


def assign_sets(
    rows: list[dict[str, str]], holdout_ratio: float, smoke_size: int, seed: str
) -> None:
    """Fill in ``metadata_set`` for the rows that have none (in place)."""
    ids = Counter(r["id"] for r in rows)
    duplicates = [i for i, n in ids.items() if n > 1]
    if duplicates:
        raise ValueError(f"Duplicate ids: {duplicates}")
    for r in rows:
        r[SET_COLUMN] = (r.get(SET_COLUMN) or "").strip()
        if r[SET_COLUMN] and r[SET_COLUMN] not in SETS:
            raise ValueError(f"{r['id']}: unknown {SET_COLUMN} {r[SET_COLUMN]!r}")

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)

    # 1. A group that already has questions in a set keeps new questions in that set.
    new_groups: list[tuple[str, list[dict[str, str]]]] = []
    for key, members in groups.items():
        assigned = {r[SET_COLUMN] for r in members if r[SET_COLUMN]}
        if not assigned:
            new_groups.append((key, members))
            continue
        if "holdout" in assigned and assigned & set(GOLDEN_SIDE):
            raise ValueError(f"{key}: questions of one group are in holdout and golden")
        for r in members:
            r[SET_COLUMN] = r[SET_COLUMN] or ("holdout" if "holdout" in assigned else "golden")

    # 2. New groups: holdout_ratio of the questions, separately for answerable or not.
    for answerable in (True, False):
        stratum = [(k, m) for k, m in new_groups if any(map(is_answerable, m)) == answerable]
        target = round(holdout_ratio * sum(len(m) for _, m in stratum))
        count = 0
        for _, members in sorted(stratum, key=lambda g: _rank(seed, g[0])):
            in_holdout = count < target
            count += len(members) if in_holdout else 0
            for r in members:
                r[SET_COLUMN] = "holdout" if in_holdout else "golden"

    # 3. Smoke: top up from golden, in proportion to answerable / not (at least one
    #    question the documents do not answer, if golden has one).
    smoke = [r for r in rows if r[SET_COLUMN] == "smoke"]
    side = [r for r in rows if r[SET_COLUMN] in GOLDEN_SIDE]
    if len(smoke) >= smoke_size or not side:
        return
    no_answer = sum(1 for r in side if not is_answerable(r))
    targets = {False: min(no_answer, max(1, round(smoke_size * no_answer / len(side))))}
    targets[True] = smoke_size - targets[False]
    for answerable, target in targets.items():
        candidates = [r for r in side if is_answerable(r) == answerable]
        have = sum(1 for r in smoke if is_answerable(r) == answerable)
        pool = sorted(
            (r for r in candidates if r[SET_COLUMN] == "golden"),
            key=lambda r: _rank(seed, r["id"]),
        )
        for r in pool[: max(0, target - have)]:
            r[SET_COLUMN] = "smoke"


def read_bank(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a bank CSV (BOM tolerated); add the ``metadata_set`` column if missing."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "id" not in fields or "question" not in fields:
        raise ValueError(f"{path}: needs 'id' and 'question' columns, found {fields}")
    if SET_COLUMN not in fields:
        fields.append(SET_COLUMN)
    return fields, rows
