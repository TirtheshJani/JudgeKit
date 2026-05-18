from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from judgekit.agreement.cohens_kappa import pairwise_kappa
from judgekit.agreement.disagreement import cluster_disagreements
from judgekit.agreement.fleiss_kappa import fleiss_kappa
from judgekit.agreement.krippendorff import krippendorff_alpha

# Label encoding for JudgmentRecord label fields
_LABEL_MAP: dict[str, int] = {
    "CORRECT": 0,
    "INCORRECT": 1,
    "UNCERTAIN": 2,
    "ERROR": 3,
    "UNPARSABLE": 3,
}
_N_CATEGORIES = 4


@dataclass
class AgreementReport:
    pairwise_kappa: dict[tuple[str, str], float] = field(default_factory=dict)
    fleiss_kappa_score: float = 0.0
    krippendorff_alpha_score: float = 0.0
    n_items: int = 0
    judge_ids: list[str] = field(default_factory=list)


def compute_all(jsonl_path: Path) -> AgreementReport:
    """Compute all agreement statistics from a JSONL file of JudgmentRecords.

    Each line must have at minimum: item_id, judge_id, label.
    Returns an AgreementReport with pairwise kappa, Fleiss kappa, and
    Krippendorff's alpha.
    """
    # Load records
    records: list[dict[str, Any]] = []
    with Path(jsonl_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return AgreementReport()

    # Collect unique judge_ids and item_ids (preserve insertion order → sorted for reproducibility)
    judge_ids: list[str] = sorted({r["judge_id"] for r in records})
    item_ids: list[str] = sorted({r["item_id"] for r in records})
    n_judges = len(judge_ids)
    n_items = len(item_ids)

    judge_idx = {j: i for i, j in enumerate(judge_ids)}
    item_idx = {it: i for i, it in enumerate(item_ids)}

    # Build ratings matrix (n_items, n_judges) of integer label codes
    # Missing entries remain -1 (will be treated as NaN by krippendorff_alpha)
    ratings = np.full((n_items, n_judges), fill_value=-1, dtype=np.int64)
    for rec in records:
        ii = item_idx[rec["item_id"]]
        ji = judge_idx[rec["judge_id"]]
        label_str = str(rec.get("label", "ERROR")).upper()
        ratings[ii, ji] = _LABEL_MAP.get(label_str, 3)

    # --- Pairwise Cohen's kappa ---
    pw_kappa: dict[tuple[str, str], float] = {}
    for j_a, j_b in itertools.combinations(judge_ids, 2):
        col_a = ratings[:, judge_idx[j_a]]
        col_b = ratings[:, judge_idx[j_b]]
        # Only use items where both judges have a rating
        mask = (col_a >= 0) & (col_b >= 0)
        if mask.sum() == 0:
            pw_kappa[(j_a, j_b)] = float("nan")
            continue
        labels_a = [str(v) for v in col_a[mask]]
        labels_b = [str(v) for v in col_b[mask]]
        try:
            pw_kappa[(j_a, j_b)] = pairwise_kappa(labels_a, labels_b)
        except Exception:
            pw_kappa[(j_a, j_b)] = float("nan")

    # --- Fleiss' kappa ---
    # Requires (n_items, n_judges) integer matrix; use items with all judges present
    fleiss_score = 0.0
    try:
        # Filter to items where all judges have valid ratings
        valid_mask = (ratings >= 0).all(axis=1)
        if valid_mask.sum() >= 2 and n_judges >= 2:
            fleiss_score = fleiss_kappa(ratings[valid_mask], n_categories=_N_CATEGORIES)
    except NotImplementedError:
        fleiss_score = 0.0
    except Exception:
        fleiss_score = 0.0

    # --- Krippendorff's alpha ---
    kripp_score = 0.0
    try:
        # Build (n_judges, n_items) float array with NaN for missing
        kripp_ratings = ratings.T.astype(np.float64)
        kripp_ratings[kripp_ratings < 0] = np.nan
        result = krippendorff_alpha(kripp_ratings, level_of_measurement="nominal")
        kripp_score = float(result) if not np.isnan(result) else 0.0
    except NotImplementedError:
        kripp_score = 0.0
    except Exception:
        kripp_score = 0.0

    return AgreementReport(
        pairwise_kappa=pw_kappa,
        fleiss_kappa_score=fleiss_score,
        krippendorff_alpha_score=kripp_score,
        n_items=n_items,
        judge_ids=judge_ids,
    )


__all__ = [
    "AgreementReport",
    "cluster_disagreements",
    "compute_all",
    "fleiss_kappa",
    "krippendorff_alpha",
    "pairwise_kappa",
]
