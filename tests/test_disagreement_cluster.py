"""Tests for disagreement clustering. Phase 6."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from judgekit.agreement import AgreementReport, compute_all
from judgekit.agreement.disagreement import DisagreementCluster, cluster_disagreements
from judgekit.agreement.krippendorff import krippendorff_alpha

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LABELS = ["CORRECT", "INCORRECT", "UNCERTAIN"]
LABEL_MAP = {"CORRECT": 0, "INCORRECT": 1, "UNCERTAIN": 2, "ERROR": 3}


def _make_record(
    item_id: str,
    judge_id: str,
    label: str,
    run_id: str = "r1",
    benchmark_id: str = "pubmedqa",
) -> dict:
    return {
        "run_id": run_id,
        "benchmark_id": benchmark_id,
        "item_id": item_id,
        "judge_id": judge_id,
        "label": label,
        "raw_text": "",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "est_cost_usd": 0.0,
        "latency_ms": 100.0,
        "error": None,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# 1. Krippendorff alpha — perfect agreement
# ---------------------------------------------------------------------------


def test_krippendorff_alpha_perfect_agreement():
    """3 raters × 5 items, all same labels → alpha = 1.0."""
    ratings = np.array(
        [
            [0, 1, 2, 0, 1],  # rater 1
            [0, 1, 2, 0, 1],  # rater 2
            [0, 1, 2, 0, 1],  # rater 3
        ],
        dtype=np.float64,
    )
    result = krippendorff_alpha(ratings, level_of_measurement="nominal")
    assert math.isclose(result, 1.0, abs_tol=1e-9), f"Expected 1.0, got {result}"


# ---------------------------------------------------------------------------
# 2. Krippendorff alpha — known value cross-checked with krippendorff package
# ---------------------------------------------------------------------------


def test_krippendorff_alpha_known_value():
    """Standard example from Wikipedia / krippendorff PyPI docs.

    Cross-checked: krippendorff.alpha(data, level_of_measurement='nominal') == 0.6914
    """
    # Example from Krippendorff (2004), widely cited; verified with the PyPI package.
    data = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan, 3, 4, 1, 2, 1, 1, 3, 3, np.nan, 3],
            [1, np.nan, 2, 1, 3, 3, 4, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 2, 1, 3, 4, 4, np.nan, 2, 1, 1, 3, 3, np.nan, 4],
        ],
        dtype=np.float64,
    )
    result = krippendorff_alpha(data, level_of_measurement="nominal")
    # Reference value from `import krippendorff; krippendorff.alpha(data, level_of_measurement='nominal')`
    # = 0.691358024691358
    assert math.isclose(result, 0.6914, abs_tol=1e-4), f"Expected ~0.6914, got {result}"


# ---------------------------------------------------------------------------
# 3. Krippendorff alpha — with missing data (NaN)
# ---------------------------------------------------------------------------


def test_krippendorff_alpha_with_missing_data():
    """One NaN in the ratings matrix — must not crash and return finite float."""
    ratings = np.array(
        [
            [0, 1, 2, 0, 1],
            [0, np.nan, 2, 0, 1],
            [0, 1, 2, 0, 1],
        ],
        dtype=np.float64,
    )
    result = krippendorff_alpha(ratings, level_of_measurement="nominal")
    assert math.isfinite(result), f"Expected finite float, got {result}"


# ---------------------------------------------------------------------------
# 4. compute_all — returns AgreementReport with correct fields
# ---------------------------------------------------------------------------


def test_compute_all_returns_agreement_report(tmp_path):
    """2 judges × 5 items, all CORRECT → fleiss=1.0, alpha=1.0, n_items=5, 2 judges."""
    judges = ["judge_a", "judge_b"]
    items = [str(i) for i in range(5)]
    records = [_make_record(item_id=it, judge_id=jg, label="CORRECT") for jg in judges for it in items]
    path = tmp_path / "judgments.jsonl"
    _write_jsonl(path, records)

    report = compute_all(path)

    assert isinstance(report, AgreementReport)
    assert report.n_items == 5
    assert len(report.judge_ids) == 2
    assert math.isclose(report.fleiss_kappa_score, 1.0, abs_tol=1e-6), (
        f"fleiss_kappa_score={report.fleiss_kappa_score}"
    )
    assert math.isclose(report.krippendorff_alpha_score, 1.0, abs_tol=1e-6), (
        f"krippendorff_alpha_score={report.krippendorff_alpha_score}"
    )


# ---------------------------------------------------------------------------
# 5. compute_all — pairwise_kappa keys and values
# ---------------------------------------------------------------------------


def test_compute_all_pairwise_kappa_keys(tmp_path):
    """2 judges → exactly 1 pair in pairwise_kappa dict, value == 1.0."""
    judges = ["judge_a", "judge_b"]
    items = [str(i) for i in range(5)]
    records = [_make_record(item_id=it, judge_id=jg, label="CORRECT") for jg in judges for it in items]
    path = tmp_path / "judgments.jsonl"
    _write_jsonl(path, records)

    report = compute_all(path)

    assert len(report.pairwise_kappa) == 1, f"Expected 1 pair, got {report.pairwise_kappa}"
    kappa_val = next(iter(report.pairwise_kappa.values()))
    assert math.isclose(kappa_val, 1.0, abs_tol=1e-6), f"Expected kappa=1.0, got {kappa_val}"


# ---------------------------------------------------------------------------
# 6. compute_all — perfect disagreement
# ---------------------------------------------------------------------------


def test_compute_all_mixed_labels(tmp_path):
    """Judge A always CORRECT, judge B always INCORRECT → kappa near -1.0."""
    items = [str(i) for i in range(4)]
    records = [
        *[_make_record(item_id=it, judge_id="judge_a", label="CORRECT") for it in items],
        *[_make_record(item_id=it, judge_id="judge_b", label="INCORRECT") for it in items],
    ]
    path = tmp_path / "judgments.jsonl"
    _write_jsonl(path, records)

    report = compute_all(path)

    kappa_val = next(iter(report.pairwise_kappa.values()))
    # Perfect disagreement: kappa = -1.0 (or near it depending on marginals)
    assert kappa_val <= -0.9, f"Expected kappa near -1.0, got {kappa_val}"


# ---------------------------------------------------------------------------
# 7. cluster_disagreements — returns non-empty list
# ---------------------------------------------------------------------------


def test_cluster_disagreements_returns_list(tmp_path):
    """3 judges × 20 items, mixed labels → returns a non-empty list of DisagreementCluster."""
    judges = ["judge_a", "judge_b", "judge_c"]
    rng = np.random.default_rng(42)
    records = []
    for i in range(20):
        item_id = str(i)
        # First 10 items: all agree on CORRECT; last 10: random disagreement
        if i < 10:
            label_choices = ["CORRECT", "CORRECT", "CORRECT"]
        else:
            label_choices = rng.choice(["CORRECT", "INCORRECT", "UNCERTAIN"], size=3).tolist()
        for j, judge in enumerate(judges):
            records.append(_make_record(item_id=item_id, judge_id=judge, label=label_choices[j]))
    path = tmp_path / "judgments.jsonl"
    _write_jsonl(path, records)

    clusters = cluster_disagreements(path, k_min=2, k_max=4)

    assert isinstance(clusters, list)
    assert len(clusters) > 0, "Expected at least one DisagreementCluster"


# ---------------------------------------------------------------------------
# 8. cluster_disagreements — cluster field validity
# ---------------------------------------------------------------------------


def test_cluster_disagreements_cluster_fields(tmp_path):
    """Each DisagreementCluster has valid cluster_id, size, dominant_pattern, representative_item_ids."""
    judges = ["judge_a", "judge_b", "judge_c"]
    rng = np.random.default_rng(7)
    records = []
    for i in range(20):
        item_id = str(i)
        if i < 5:
            label_choices = ["CORRECT", "CORRECT", "CORRECT"]
        else:
            label_choices = rng.choice(["CORRECT", "INCORRECT", "UNCERTAIN"], size=3).tolist()
        for j, judge in enumerate(judges):
            records.append(_make_record(item_id=item_id, judge_id=judge, label=label_choices[j]))
    path = tmp_path / "judgments.jsonl"
    _write_jsonl(path, records)

    clusters = cluster_disagreements(path, k_min=2, k_max=4)

    assert len(clusters) > 0, "Expected at least one cluster"
    for cluster in clusters:
        assert isinstance(cluster, DisagreementCluster)
        assert cluster.cluster_id >= 0, f"cluster_id={cluster.cluster_id} must be >= 0"
        assert cluster.size > 0, f"size={cluster.size} must be > 0"
        assert isinstance(cluster.dominant_pattern, str) and len(cluster.dominant_pattern) > 0, (
            f"dominant_pattern must be non-empty string, got {cluster.dominant_pattern!r}"
        )
        assert isinstance(cluster.representative_item_ids, list), (
            f"representative_item_ids must be list, got {type(cluster.representative_item_ids)}"
        )
