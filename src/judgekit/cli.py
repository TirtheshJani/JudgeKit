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
        est_cost = n_items * (
            600 / 1000 * pricing.input_per_1k + 150 / 1000 * pricing.output_per_1k
        )
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
    """Generate agreement statistics + spend report from a completed run."""
    from judgekit.agreement import compute_all  # local: keep CLI startup fast

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
    per_benchmark: Counter[str] = Counter(str(r["benchmark_id"]) for r in records)
    vendor_spend: dict[str, float] = {}
    for r in records:
        vendor = str(r["judge_id"]).split("/", 1)[0]
        vendor_spend[vendor] = vendor_spend.get(vendor, 0.0) + float(r.get("est_cost_usd", 0.0))

    agreement = compute_all(jsonl_path)

    typer.echo(f"Run ID:        {run_id}")
    typer.echo(f"Total records: {total}")
    typer.echo(f"Items:         {agreement.n_items}")
    typer.echo(f"Judges:        {len(agreement.judge_ids)}")
    typer.echo("")
    typer.echo(f"Fleiss kappa:        {agreement.fleiss_kappa_score:.4f}")
    typer.echo(f"Krippendorff alpha:  {agreement.krippendorff_alpha_score:.4f}")
    typer.echo("")
    typer.echo("Records per judge:")
    for judge, count in sorted(per_judge.items()):
        typer.echo(f"  {judge}: {count}")
    typer.echo("")
    typer.echo("Label distribution:")
    for label, count in sorted(label_dist.items()):
        typer.echo(f"  {label}: {count}")

    if out is None:
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as csv_fh:
        writer = csv.writer(csv_fh)
        writer.writerow(["section", "key", "value"])
        writer.writerow(["meta", "run_id", run_id])
        writer.writerow(["meta", "total_records", total])
        writer.writerow(["meta", "n_items", agreement.n_items])
        writer.writerow(["meta", "n_judges", len(agreement.judge_ids)])
        writer.writerow(["agreement", "fleiss_kappa", f"{agreement.fleiss_kappa_score:.6f}"])
        writer.writerow(
            ["agreement", "krippendorff_alpha", f"{agreement.krippendorff_alpha_score:.6f}"]
        )
        for (a, b), kappa in sorted(agreement.pairwise_kappa.items()):
            writer.writerow(["pairwise_kappa", f"{a}|{b}", f"{kappa:.6f}"])
        for label, count in sorted(label_dist.items()):
            writer.writerow(["label_distribution", label, count])
        for judge, count in sorted(per_judge.items()):
            writer.writerow(["per_judge_count", judge, count])
        for bench, count in sorted(per_benchmark.items()):
            writer.writerow(["per_benchmark_count", bench, count])
        for vendor, spend in sorted(vendor_spend.items()):
            writer.writerow(["vendor_spend_usd", vendor, f"{spend:.6f}"])
        writer.writerow(["vendor_spend_usd", "total", f"{sum(vendor_spend.values()):.6f}"])
    typer.echo(f"\nReport written to {out}")
