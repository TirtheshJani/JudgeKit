from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DisagreementCluster:
    cluster_id: int
    size: int
    dominant_pattern: str
    representative_item_ids: list[str] = field(default_factory=list)


def cluster_disagreements(
    jsonl_path: Path,
    k_min: int = 3,
    k_max: int = 8,
) -> list[DisagreementCluster]:
    """KMeans clustering of per-question disagreement vectors.

    Uses elbow method to select k in [k_min, k_max].
    """
    raise NotImplementedError
