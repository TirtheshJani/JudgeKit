"""Tests for the Typer CLI. Phase 2."""

from __future__ import annotations

import json
from unittest.mock import patch

from typer.testing import CliRunner

from judgekit.cli import app

cli_runner = CliRunner()


def test_run_dry_run_prints_estimate_and_exits_zero(tmp_path):
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text("""
run_id: test_smoke
seed: 42
budget_usd: 1.0
judges:
  - vendor: groq
    model: llama-3.3-70b-versatile
    max_tokens: 256
benchmarks:
  - name: pubmedqa
    n: 10
""")
    result = cli_runner.invoke(app, ["run", "--config", str(config_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "groq" in result.output.lower()
    assert "10" in result.output  # 10 items
    # Total calls = 10 items × 1 judge = 10
    assert "10" in result.output


def test_estimate_command_prints_cost_breakdown(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("""
run_id: r1
seed: 42
budget_usd: 25.0
judges:
  - vendor: anthropic
    model: claude-sonnet-4-5
    max_tokens: 256
benchmarks:
  - name: pubmedqa
    n: 100
""")
    result = cli_runner.invoke(app, ["estimate", "--config", str(config_path)])
    assert result.exit_code == 0
    assert "anthropic" in result.output.lower()
    assert "100" in result.output
    # Should show a non-zero cost estimate (Anthropic isn't free)
    assert "$" in result.output or "usd" in result.output.lower()


def test_run_command_calls_run_eval_with_correct_args(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("""
run_id: myrun
seed: 42
budget_usd: 1.0
judges:
  - vendor: groq
    model: llama-3.3-70b-versatile
    max_tokens: 256
benchmarks:
  - name: pubmedqa
    n: 5
""")
    expected_output = tmp_path / "myrun" / "judgments.jsonl"

    with patch("judgekit.cli.run_eval", return_value=expected_output) as mock_run:
        result = cli_runner.invoke(
            app, ["run", "--config", str(config_path), "--output-dir", str(tmp_path)]
        )

    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[0][0].run_id == "myrun"  # config
    assert call_args[0][1] == tmp_path  # output_dir


def test_report_command_prints_label_distribution(tmp_path):
    # Create a fake JSONL run
    run_dir = tmp_path / "myrun"
    run_dir.mkdir()
    jsonl = run_dir / "judgments.jsonl"
    records = [
        {
            "run_id": "myrun",
            "benchmark_id": "pubmedqa",
            "item_id": str(i),
            "judge_id": "groq/llama-3.3-70b-versatile",
            "label": label,
            "raw_text": "text",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "est_cost_usd": 0.0,
            "latency_ms": 100.0,
            "error": None,
        }
        for i, label in enumerate(["CORRECT", "CORRECT", "INCORRECT", "UNCERTAIN"])
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in records))

    result = cli_runner.invoke(app, ["report", "myrun", "--results-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "CORRECT" in result.output
    assert "2" in result.output  # 2 CORRECT
    assert "INCORRECT" in result.output


def test_report_exits_nonzero_when_run_not_found(tmp_path):
    result = cli_runner.invoke(app, ["report", "nonexistent_run", "--results-dir", str(tmp_path)])
    assert result.exit_code != 0


def _write_golden_jsonl(path):
    """Deterministic 3-judge x 6-item fixture for golden-CSV test."""
    judges = [
        "anthropic/claude-sonnet-4-5",
        "groq/llama-3.3-70b-versatile",
        "sambanova/DeepSeek-R1",
    ]
    labels_by_judge = {
        judges[0]: ["CORRECT", "CORRECT", "INCORRECT", "CORRECT", "UNCERTAIN", "CORRECT"],
        judges[1]: ["CORRECT", "INCORRECT", "INCORRECT", "CORRECT", "UNCERTAIN", "CORRECT"],
        judges[2]: ["CORRECT", "CORRECT", "INCORRECT", "INCORRECT", "UNCERTAIN", "CORRECT"],
    }
    costs_by_judge = {judges[0]: 0.004, judges[1]: 0.0, judges[2]: 0.0}
    benchmarks = ["pubmedqa", "pubmedqa", "medqa", "medqa", "humaneval", "humaneval"]
    records = []
    for jg in judges:
        for i, lbl in enumerate(labels_by_judge[jg]):
            records.append(
                {
                    "run_id": "golden",
                    "benchmark_id": benchmarks[i],
                    "item_id": str(i),
                    "judge_id": jg,
                    "label": lbl,
                    "raw_text": lbl,
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "est_cost_usd": costs_by_judge[jg],
                    "latency_ms": 100.0,
                    "error": None,
                }
            )
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def test_report_golden_csv(tmp_path):
    from pathlib import Path

    run_dir = tmp_path / "golden"
    run_dir.mkdir()
    _write_golden_jsonl(run_dir / "judgments.jsonl")

    out = tmp_path / "report.csv"
    result = cli_runner.invoke(
        app, ["report", "golden", "--results-dir", str(tmp_path), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()

    golden = Path("tests/fixtures/golden_report_v1.csv").read_text(encoding="utf-8")
    actual = out.read_text(encoding="utf-8")
    assert actual == golden, (
        "report.csv drifted from golden fixture. "
        "If the change is intentional, update tests/fixtures/golden_report_v1.csv."
    )
