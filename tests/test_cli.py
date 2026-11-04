"""Smoke tests for the CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from rag_eval_framework.cli.run import main
from tests.conftest import make_config, make_record


def _write_sample_project(tmp_path: Path) -> Path:
    """Create a minimal config + dataset in tmp_path, return config path."""
    import yaml

    dataset_dir = tmp_path / "datasets" / "test"
    dataset_dir.mkdir(parents=True)
    dataset_path = dataset_dir / "eval.jsonl"
    records = [
        make_record(id="c-001"),
        make_record(id="c-002", contexts=[]),
    ]
    dataset_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    config_data = make_config(
        dataset_path=str(dataset_path),
        output_dir=str(tmp_path / "output"),
        thresholds={"answer_presence": 0.5},
    )
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml.dump(config_data), encoding="utf-8")
    return config_path


class TestCLIRun:
    def test_successful_run(self, tmp_path: Path) -> None:
        config_path = _write_sample_project(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["--config", str(config_path)])
        assert result.exit_code == 0
        assert "PASSED" in result.output

    def test_reports_written(self, tmp_path: Path) -> None:
        config_path = _write_sample_project(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["--config", str(config_path)])
        assert result.exit_code == 0
        output_dir = tmp_path / "output" / "test-project"
        assert (output_dir / "eval-report-test-project.json").exists()
        assert (output_dir / "eval-report-test-project.md").exists()

    def test_output_override(self, tmp_path: Path) -> None:
        config_path = _write_sample_project(tmp_path)
        custom_output = tmp_path / "custom-output"
        runner = CliRunner()
        result = runner.invoke(main, ["--config", str(config_path), "--output", str(custom_output)])
        assert result.exit_code == 0
        assert (custom_output / "test-project").exists()

    def test_missing_config_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--config", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_verbose_flag(self, tmp_path: Path) -> None:
        config_path = _write_sample_project(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["--config", str(config_path), "--verbose"])
        assert result.exit_code == 0

    def test_threshold_breach_exits_3(self, tmp_path: Path) -> None:
        import yaml

        dataset_dir = tmp_path / "datasets" / "strict"
        dataset_dir.mkdir(parents=True)
        dataset_path = dataset_dir / "eval.jsonl"
        # All records have empty contexts → context_presence = 0.0
        records = [make_record(id="c-001", contexts=[])]
        dataset_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        config_data = make_config(
            dataset_path=str(dataset_path),
            output_dir=str(tmp_path / "output"),
            evaluators=["context_presence"],
            thresholds={"context_presence": 0.9},
        )
        config_path = tmp_path / "strict.yaml"
        config_path.write_text(yaml.dump(config_data), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["--config", str(config_path)])
        assert result.exit_code == 3
        assert "FAILED" in result.output


class TestCLIDatasetOverride:
    def test_dataset_option_replaces_config_dataset(self, tmp_path: Path) -> None:
        config_path = _write_sample_project(tmp_path)
        other = tmp_path / "datasets" / "test" / "run-2.jsonl"
        other.write_text(json.dumps(make_record(id="r2-001")), encoding="utf-8")

        result = CliRunner().invoke(
            main, ["--config", str(config_path), "--dataset", str(other)]
        )

        assert result.exit_code == 0
        assert f"Dataset : {other}" in result.output
        report = tmp_path / "output" / "test-project" / "eval-report-test-project.json"
        assert "r2-001" in report.read_text(encoding="utf-8")
