"""Offline JudgeKit demo: agreement statistics and disagreement clusters, no API keys.

The bundled `examples/sample_run/judgments.jsonl` is SYNTHETIC. It mimics the
shape of a real run (5 judges x 12 items across 4 benchmarks) so you can see
the analysis layer work without calling any vendor. None of its numbers are
results about real judges.

Usage (from the repo root):

    uv run python examples/agreement_demo.py
"""

from __future__ import annotations

import math
from pathlib import Path

from judgekit.agreement import cluster_disagreements, compute_all

SAMPLE = Path(__file__).parent / "sample_run" / "judgments.jsonl"


def main(jsonl_path: Path = SAMPLE) -> int:
    report = compute_all(jsonl_path)
    print(f"Sample file:         {jsonl_path}")
    print(f"Items x judges:      {report.n_items} x {len(report.judge_ids)}")
    print(f"Fleiss kappa:        {report.fleiss_kappa_score:.4f}")
    print(f"Krippendorff alpha:  {report.krippendorff_alpha_score:.4f}")
    print("\nPairwise Cohen's kappa:")
    for (a, b), kappa in sorted(report.pairwise_kappa.items()):
        shown = "nan" if math.isnan(kappa) else f"{kappa:.4f}"
        print(f"  {a:<50} vs {b:<50} {shown}")

    clusters = cluster_disagreements(jsonl_path)
    print(f"\nDisagreement clusters: {len(clusters)}")
    for c in clusters:
        print(f"  cluster {c.cluster_id} (n={c.size}) e.g. {', '.join(c.representative_item_ids)}")
        print(f"    dominant pattern: {c.dominant_pattern}")
    return len(clusters)


if __name__ == "__main__":
    main()
