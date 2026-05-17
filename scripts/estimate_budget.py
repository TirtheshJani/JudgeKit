"""Pre-flight cost estimator. Usage: uv run python scripts/estimate_budget.py --config <path>"""

from __future__ import annotations

import argparse
from pathlib import Path

from judgekit.budget.estimator import estimate_config
from judgekit.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate JudgeKit eval cost before running.")
    parser.add_argument("--config", required=True, type=Path, help="Path to eval YAML config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = estimate_config(cfg)
    total_cost = 0.0
    total_calls = 0
    for row in rows:
        print(f"  {row['vendor']}/{row['model']}: {row['n_items']} calls, ~${row['est_cost_usd']:.4f} USD")
        total_cost += row['est_cost_usd']
        total_calls += row['n_items']
    print(f"Total: {total_calls * len(cfg.judges)} calls, ~${total_cost:.4f} USD")


if __name__ == "__main__":
    main()
