"""Pre-flight cost estimator. Usage: uv run python scripts/estimate_budget.py --config <path>"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate JudgeKit eval cost before running.")
    parser.add_argument("--config", required=True, type=Path, help="Path to eval YAML config.")
    args = parser.parse_args()
    raise NotImplementedError(f"estimate_budget not yet implemented (config={args.config})")


if __name__ == "__main__":
    main()
