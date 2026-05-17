from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(name="judgekit", help="Multi-judge LLM evaluation framework.")


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to eval YAML config."),
    run_id: str | None = typer.Option(None, "--run-id", help="Override run ID."),
    output_dir: Path = typer.Option(Path("results"), "--output-dir"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost; do not call APIs."),
) -> None:
    """Run an evaluation against the configured judges and benchmarks."""
    raise NotImplementedError


@app.command()
def estimate(
    config: Path = typer.Option(..., "--config", "-c", help="Path to eval YAML config."),
) -> None:
    """Estimate token cost before running."""
    raise NotImplementedError


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run ID to generate report for."),
    results_dir: Path = typer.Option(Path("results"), "--results-dir"),
    out: Path | None = typer.Option(None, "--out", help="Output CSV path."),
) -> None:
    """Generate agreement statistics report from a completed run."""
    raise NotImplementedError
