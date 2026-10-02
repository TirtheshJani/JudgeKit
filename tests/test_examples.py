"""The offline demo in examples/ must keep working without API keys."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from typer.testing import CliRunner

from judgekit.agreement import compute_all
from judgekit.cli import app

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SAMPLE = EXAMPLES / "sample_run" / "judgments.jsonl"


def test_sample_run_has_five_judges_and_twelve_items():
    report = compute_all(SAMPLE)
    assert report.n_items == 12
    assert len(report.judge_ids) == 5
    assert report.fleiss_kappa_score == pytest.approx(0.4383, abs=1e-4)
    assert report.krippendorff_alpha_score == pytest.approx(0.4477, abs=1e-4)


@pytest.mark.filterwarnings("ignore::Warning")
def test_agreement_demo_runs_and_finds_disagreement_clusters(capsys):
    module = runpy.run_path(str(EXAMPLES / "agreement_demo.py"))
    n_clusters = module["main"](SAMPLE)
    out = capsys.readouterr().out
    assert n_clusters >= 1
    assert "Fleiss kappa" in out
    assert "Disagreement clusters" in out


def test_report_cli_on_sample_run(tmp_path):
    out = tmp_path / "report.csv"
    result = CliRunner().invoke(
        app, ["report", "sample_run", "--results-dir", str(EXAMPLES), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "vendor_spend_usd,total,0.028800" in out.read_text(encoding="utf-8")
