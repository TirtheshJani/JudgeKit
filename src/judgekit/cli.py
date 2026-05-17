from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import typer

from judgekit.budget.pricing import lookup
from judgekit.config import EvalConfig, load_config
from judgekit.runner import run_eval

app = typer.Typer(name="judgekit", help="Multi-judge LLM evaluation framework.")


def _print_estimate(config: EvalConfig) -> None:
    total_calls = 0
    total_cost = 0.0
    for jc in config.judges:
        pricing = lookup(jc.vendor, jc.model)
        n_items = sum(bc.n or 100 for bc in config.benchmarks)
        est_cost = n_items * (600 / 1000 * pricing.input_per_1k + 150 / 1000 * pricing.output_per_1k)
        total_calls += n_items
        total_cost += est_cost
        typer.echo(f"  {jc.vendor}/{jc.model}: {n_items} calls, ~${est_cost:.4f} USD")
    typer.echo(f"Total: {total_calls * len(config.judges)} calls, ~${total_cost:.4f} USD")


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to eval YAML config."),
    run_id: str | None = typer.Option(None, "--run-id", help="Override run ID."),
    output_dir: Path = typer.Option(Path("results"), "--output-dir"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost; do not call APIs."),
) -> None:
    """Run an evaluation against the configured judges and benchmarks."""
    cfg = load_config(config)
    if run_id is not None:
        cfg = cfg.model_copy(update={"run_id": run_id})
    if dry_run:
        typer.echo(f"Dry-run estimate for run_id={cfg.run_id}:")
        _print_estimate(cfg)
        return
    out_path = run_eval(cfg, output_dir)
    typer.echo(f"Run complete: {out_path}")


@app.command()
def estimate(
    config: Path = typer.Option(..., "--config", "-c", help="Path to eval YAML config."),
) -> None:
    """Estimate token cost before running."""
    cfg = load_config(config)
    typer.echo(f"Estimate for run_id={cfg.run_id}:")
    _print_estimate(cfg)


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run ID to generate report for."),
    results_dir: Path = typer.Option(Path("results"), "--results-dir"),
    out: Path | None = typer.Option(None, "--out", help="Output CSV path."),
) -> None:
    """Generate agreement statistics report from a completed run."""
    jsonl_path = results_dir / run_id / "judgments.jsonl"
    if not jsonl_path.exists():
        typer.echo(f"Error: {jsonl_path} not found.", err=True)
        raise typer.Exit(1)

    records: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))

    total = len(records)
    per_judge: Counter[str] = Counter(str(r["judge_id"]) for r in records)
    label_dist: Counter[str] = Counter(str(r["label"]) for r in records)

    typer.echo(f"Run ID:        {run_id}")
    typer.echo(f"Total records: {total}")
    typer.echo("")
    typer.echo("Records per judge:")
    for judge, count in sorted(per_judge.items()):
        typer.echo(f"  {judge}: {count}")
    typer.echo("")
    typer.echo("Label distribution:")
    for label, count in sorted(label_dist.items()):
        typer.echo(f"  {label}: {count}")

    if out is not None:
        with out.open("w", newline="", encoding="utf-8") as csv_fh:
            writer = csv.writer(csv_fh)
            writer.writerow(["label", "count"])
            for label, count in sorted(label_dist.items()):
                writer.writerow([label, count])
        typer.echo(f"\nReport written to {out}")
