"""Rebuild every derived file of a dataset from its question bank, or check them.

The layout is defined in ``rag_eval_framework.datasets.layout``: ``bank/questions.csv``
(and ``bank/excluded.csv``) are the only files to edit by hand.  This script derives
the rest, so they never drift apart:

1. assigns a set to new questions and writes ``sets/smoke.csv`` / ``golden.csv`` /
   ``holdout.csv`` (``datasets.splits.assign_sets``; existing sets never move);
2. writes the single-gold qrels ``qrels/<version>.jsonl`` from the bank's hashes
   (``datasets.single_gold``) — only while that file is still generated
   (every entry ``single_gold`` + ``model_draft``); reviewed qrels are never overwritten;
3. validates the bank against ``chunks/sampled_chunks.jsonl`` and the split;
4. updates ``manifest.json`` (created if missing): file SHA-256, counts, ``validation``,
   the qrels version and the split settings (``split``: holdout ratio, smoke size, seed —
   used by later runs unless given on the command line).  The manifest's own hash is
   not recorded; hand-written fields (``revisions``, …) are kept; ``files`` entries of
   paths that no longer exist are dropped.

Usage
-----
    python scripts/prepare_dataset.py --dataset datasets/my-project --qrels-version single-gold-v2
    python scripts/prepare_dataset.py --dataset datasets/my-project --check

``--check`` writes nothing and exits 1 when a derived file is out of date or a
validation check fails.  Without it, the files are rewritten only when they change
(exit 1 if validation fails; nothing is written then).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_eval_framework.datasets.layout import (  # noqa: E402
    BANK,
    CHUNKS,
    EXCLUDED,
    MANIFEST,
    SET_FILES,
    qrels_path,
    set_path,
)
from rag_eval_framework.datasets.qrels import (  # noqa: E402
    QrelsLoadError,
    content_sha256,
    load_qrels,
    write_qrels,
)
from rag_eval_framework.datasets.single_gold import build_single_gold_entries  # noqa: E402
from rag_eval_framework.datasets.splits import (  # noqa: E402
    GOLDEN_SIDE,
    SET_COLUMN,
    assign_sets,
    is_answerable,
    read_bank,
)
from rag_eval_framework.utils.normalisation import normalise_for_match  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QRELS_SETS = {"smoke", "golden"}
DEFAULT_HOLDOUT_RATIO = 0.2
DEFAULT_SMOKE_SIZE = 30




def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]], newline: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator=newline)
        writer.writeheader()
        writer.writerows(rows)


def _split_values(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split("|") if v.strip()]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(
    rows: list[dict[str, str]],
    chunks: list[dict[str, Any]] | None,
    excluded: list[dict[str, str]],
) -> dict[str, Any]:
    """Structure and consistency checks of the bank (not label quality)."""
    problems: list[str] = []
    sets = Counter(r[SET_COLUMN] for r in rows)

    ids = Counter(r["id"] for r in rows)
    problems += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    questions = Counter(normalise_for_match(r["question"]) for r in rows)
    problems += [f"duplicate question: {q[:40]}" for q, n in questions.items() if n > 1]

    parents = {
        s: {r.get("metadata_source_parent_hash", "") for r in rows if r[SET_COLUMN] in g}
        for s, g in (("golden", GOLDEN_SIDE), ("holdout", ("holdout",)))
    }
    overlap = sorted((parents["golden"] & parents["holdout"]) - {""})
    problems += [f"source parent {p[:12]} is in golden and holdout" for p in overlap]

    excluded_ids = {r["id"] for r in excluded}
    problems += [f"{i} is in {BANK} and {EXCLUDED}" for i in sorted(excluded_ids & set(ids))]
    boilerplate = {
        normalise_for_match(e)
        for r in excluded
        if (r.get("exclusion_reason") or "").lower().startswith("boilerplate")
        for e in _split_values(r.get("ground_truth_evidence"))
    }
    boilerplate_left = [
        r["id"]
        for r in rows
        if any(
            normalise_for_match(e) in boilerplate for e in _split_values(r["ground_truth_evidence"])
        )
    ]
    problems += [f"{i}: evidence is excluded boilerplate" for i in boilerplate_left]

    matches = {"evidence_matches": 0, "title_matches": 0, "content_hash_matches": 0}
    if chunks is not None:
        by_question: dict[str, dict[str, Any]] = {}
        for c in chunks:
            for q in c.get("question_ids", []):
                by_question[q] = c
        stale = sorted(set(by_question) - set(ids))
        problems += [f"{CHUNKS} refers to unknown question {q}" for q in stale]
        for r in rows:
            chunk = by_question.get(r["id"])
            if chunk is None and not is_answerable(r):
                continue  # unanswerable questions have no source chunk
            if chunk is None:
                problems.append(f"{r['id']}: no chunk in {CHUNKS}")
                continue
            text = normalise_for_match(chunk["text"])
            checks = {
                "evidence_matches": all(
                    normalise_for_match(e) in text
                    for e in _split_values(r["ground_truth_evidence"])
                ),
                "title_matches": normalise_for_match(r.get("ground_truth_documents", ""))
                == normalise_for_match(chunk.get("metadata", "")),
                "content_hash_matches": content_sha256(chunk["text"])
                == r.get("metadata_content_hash"),
            }
            for key, ok in checks.items():
                matches[key] += ok
                if not ok:
                    problems.append(f"{r['id']}: {key.removesuffix('es')} failed")

    per_chunk = Counter(r.get("metadata_source_key_hash") or r["id"] for r in rows)
    return {
        "actual": {
            "questions": len(rows),
            "smoke": sets["smoke"],
            "golden": sets["smoke"] + sets["golden"],
            "holdout": sets["holdout"],
            "sampled_chunks": len(chunks) if chunks is not None else None,
            "excluded": len(excluded),
        },
        **(matches if chunks is not None else {}),
        "unique_ids": len(ids),
        "unique_normalized_questions": len(questions),
        "unique_source_parents": len(
            {r.get("metadata_source_parent_hash") for r in rows} - {"", None}
        ),
        "golden_holdout_parent_overlap": len(overlap),
        "smoke_subset_of_golden": True,  # golden.csv is written from smoke + golden
        "boilerplate_evidence_remaining": len(boilerplate_left),
        "label_statuses": dict(
            Counter((r.get("metadata_label_status") or "").strip() or "(none)" for r in rows)
        ),
        "unanswerable_questions": sum(not is_answerable(r) for r in rows),
        "question_group_sizes": {str(k): v for k, v in sorted(Counter(per_chunk.values()).items())},
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _qrels_is_generated(path: Path) -> bool:
    try:
        entries = load_qrels(path)
    except QrelsLoadError:
        return False
    return all(
        e.judgment_scope == "single_gold" and e.annotation_status == "model_draft"
        for e in entries.values()
    )


def build(dataset: Path, out: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    """Write every derived file into *out*; return the validation and the qrels notes."""
    manifest: dict[str, Any] = {}
    if (dataset / MANIFEST).exists():
        manifest = json.loads((dataset / MANIFEST).read_text(encoding="utf-8"))
    if not (dataset / BANK).exists():
        raise ValueError(f"{dataset / BANK} not found (see rag_eval_framework.datasets.layout)")
    fields, rows = read_bank(dataset / BANK)
    # Split settings: CLI, else the manifest's, else the defaults; stored so reruns agree.
    saved = manifest.get("split") or {}
    settings = {
        "holdoutRatio": args.holdout_ratio
        if args.holdout_ratio is not None
        else saved.get("holdoutRatio", DEFAULT_HOLDOUT_RATIO),
        "smokeSize": args.smoke_size
        if args.smoke_size is not None
        else saved.get("smokeSize", DEFAULT_SMOKE_SIZE),
        "seed": str(args.seed or saved.get("seed") or manifest.get("sampleSeed", "0")),
    }
    if not 0 <= settings["holdoutRatio"] < 1:
        raise ValueError("holdout ratio must be in [0, 1)")
    assign_sets(rows, settings["holdoutRatio"], settings["smokeSize"], settings["seed"])
    manifest["split"] = settings

    # Keep the bank's line endings so unchanged files stay byte-identical.
    newline = "\r\n" if b"\r\n" in (dataset / BANK).read_bytes() else "\n"
    _write_csv(out / BANK, fields, rows, newline)
    for name, sets in SET_FILES.items():
        selected = [r for r in rows if r[SET_COLUMN] in sets]
        _write_csv(out / set_path(name), fields, selected, newline)

    chunks = None
    if (dataset / CHUNKS).exists():
        lines = (dataset / CHUNKS).read_text(encoding="utf-8").splitlines()
        chunks = [json.loads(line) for line in lines if line.strip()]
    excluded = _read_csv(dataset / EXCLUDED) if (dataset / EXCLUDED).exists() else []
    validation = validate(rows, chunks, excluded)

    notes: list[str] = []
    version = args.qrels_version or manifest.get("singleGoldQrels")
    if version:
        target = dataset / qrels_path(version)
        if target.exists() and not _qrels_is_generated(target):
            notes.append(
                f"{qrels_path(version)} holds reviewed or non-single-gold judgments; left as is"
            )
        else:
            index_identity = {
                k: manifest[k]
                for k in ("service", "index", "indexCount", "schemaSha256")
                if k in manifest
            } or None
            entries, missing = build_single_gold_entries(
                rows,
                sets=QRELS_SETS,
                annotation_version=version,
                annotation_status="model_draft",
                index_identity=index_identity,
            )
            answerable = {r["id"] for r in rows if is_answerable(r)}
            missing = [i for i in missing if i in answerable]  # unanswerable: no judgments
            if missing:
                validation["problems"].append(f"no chunk hashes for qrels: {', '.join(missing)}")
            (out / qrels_path(version)).parent.mkdir(parents=True, exist_ok=True)
            write_qrels(out / qrels_path(version), entries)
        manifest["singleGoldQrels"] = version
    else:
        notes.append("no --qrels-version and none in the manifest; qrels not generated")

    # Keep the old timestamp when the validation result did not change.
    old = manifest.get("validation") or {}
    unchanged = {k: v for k, v in old.items() if k != "checked_at"} == validation
    checked_at = old.get("checked_at") if unchanged else datetime.now(timezone.utc).isoformat()

    hashed = [BANK, *(set_path(n) for n in SET_FILES), EXCLUDED, CHUNKS]
    if version:
        hashed.append(qrels_path(version))
    files: dict[str, Any] = {}
    for rel, entry in (manifest.get("files") or {}).items():
        if (dataset / rel).exists():  # drop entries of moved / deleted files
            files[rel] = entry
    for rel in hashed:
        source = out / rel if (out / rel).exists() else dataset / rel
        if source.exists():
            files[rel] = {"sha256": _sha256(source)}
    manifest["files"] = files
    manifest["questionCount"] = len(rows)
    manifest["excludedCount"] = len(excluded)
    if chunks is not None:
        manifest["sourceCount"] = len(chunks)
    manifest["validation"] = {"checked_at": checked_at, **validation}
    # LF on every OS: the files are hashed and compared byte for byte by --check.
    (out / MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return validation, notes


def _changed(dataset: Path, out: Path) -> list[str]:
    changed = []
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        rel = path.relative_to(out)
        target = dataset / rel
        if not target.exists() or target.read_bytes() != path.read_bytes():
            changed.append(rel.as_posix())
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset directory.")
    parser.add_argument(
        "--check", action="store_true", help="Write nothing; exit 1 if out of date."
    )
    parser.add_argument(
        "--qrels-version",
        default=None,
        help="Single-gold qrels version, i.e. qrels/<version>.jsonl (default: from the manifest).",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=None,
        help=f"Share of new questions put in holdout (default: manifest split, else "
        f"{DEFAULT_HOLDOUT_RATIO}).",
    )
    parser.add_argument(
        "--smoke-size",
        type=int,
        default=None,
        help=f"Smoke questions (default: manifest split, else {DEFAULT_SMOKE_SIZE}).",
    )
    parser.add_argument(
        "--seed", default=None, help="Split seed (default: manifest split, else sampleSeed)."
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        try:
            validation, notes = build(args.dataset, out, args)
        except (ValueError, OSError, KeyError) as exc:
            logger.error("%s", exc)
            sys.exit(1)
        for note in notes:
            logger.info("%s", note)
        for problem in validation["problems"]:
            logger.error("validation: %s", problem)
        changed = _changed(args.dataset, out)

        if args.check:
            for rel in changed:
                logger.error("out of date: %s", rel)
            ok = not changed and not validation["problems"]
            logger.info("check %s", "passed" if ok else "failed")
            sys.exit(0 if ok else 1)

        if validation["problems"]:
            logger.error("validation failed; nothing written")
            sys.exit(1)
        for rel in changed:
            target = args.dataset / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out / rel, target)
            logger.info("wrote %s", rel)
        if not changed:
            logger.info("all derived files are up to date")


if __name__ == "__main__":
    main()
