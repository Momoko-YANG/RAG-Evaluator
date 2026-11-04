"""Tests for scripts/sample_index_chunks.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sample_index_chunks.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sample_index_chunks", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sampler = _load_module()


def _run(monkeypatch: pytest.MonkeyPatch, output: Path, *extra: str) -> None:
    chunks = [
        {"position": i, "metadata": "", "pageContent": f"規程本文 {i} " + "あ" * 200}
        for i in range(30)
    ]
    monkeypatch.setattr(sampler, "fetch_all", lambda *a, **k: chunks)
    monkeypatch.setenv("RAG_EVAL_API_KEY", "test")
    argv = ["sample_index_chunks.py", "--extension-id", "ext", "--output", str(output), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    sampler.main()


def _positions(path: Path) -> list[int]:
    return [json.loads(line)["position"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_append_samples_only_new_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "sampled_chunks.jsonl"
    _run(monkeypatch, output, "--num", "10")
    first = _positions(output)

    _run(monkeypatch, output, "--num", "10", "--append")
    positions = _positions(output)

    assert positions[:10] == first
    assert len(positions) == 20
    assert len(set(positions)) == 20


def test_without_append_the_file_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "sampled_chunks.jsonl"
    _run(monkeypatch, output, "--num", "10")
    _run(monkeypatch, output, "--num", "5")
    assert len(_positions(output)) == 5


def test_boilerplate_patterns_extend_the_defaults(tmp_path: Path) -> None:
    footer = "ACME Handbook Rev 3 " * 10
    assert sampler.is_usable(footer)
    patterns = [*sampler.DEFAULT_BOILERPLATE_PATTERNS, r"ACME Handbook Rev \d+"]
    assert not sampler.is_usable(footer, patterns)

    pattern_file = tmp_path / "boilerplate.txt"
    pattern_file.write_text("# footer\nACME Handbook Rev \\d+\n\n", encoding="utf-8")
    assert sampler.read_patterns(pattern_file) == [r"ACME Handbook Rev \d+"]
