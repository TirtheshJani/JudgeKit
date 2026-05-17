from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from judgekit.agreement.cohens_kappa import pairwise_kappa
from judgekit.agreement.disagreement import cluster_disagreements
from judgekit.agreement.fleiss_kappa import fleiss_kappa
from judgekit.agreement.krippendorff import krippendorff_alpha


@dataclass
class AgreementReport:
    pairwise_kappa: dict[tuple[str, str], float] = field(default_factory=dict)
    fleiss_kappa_score: float = 0.0
    krippendorff_alpha_score: float = 0.0
    n_items: int = 0
    judge_ids: list[str] = field(default_factory=list)


def compute_all(jsonl_path: Path) -> AgreementReport:
    raise NotImplementedError


__all__ = [
    "AgreementReport",
    "cluster_disagreements",
    "compute_all",
    "fleiss_kappa",
    "krippendorff_alpha",
    "pairwise_kappa",
]
