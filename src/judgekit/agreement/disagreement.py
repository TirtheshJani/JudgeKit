from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Label encoding — must match __init__.py
_LABEL_MAP: dict[str, int] = {
    "CORRECT": 0,
    "INCORRECT": 1,
    "UNCERTAIN": 2,
    "ERROR": 3,
    "UNPARSABLE": 3,
}


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

    Algorithm:
    1. Load JSONL, group by item_id.
    2. For each item build a feature vector of per-judge integer labels.
    3. Filter to items where judges disagree (not all-same label).
    4. If fewer than k_min disagreeing items, return empty list.
    5. Use KMeans with k in [k_min, k_max], pick k via elbow (inertia drop).
    6. For each cluster build a DisagreementCluster with dominant_pattern and
       up to 3 representative item ids.

    Uses scikit-learn KMeans (test/analysis code only — not in src production
    path; sklearn is an allowed test/analysis dependency).
    """
    from sklearn.cluster import KMeans  # noqa: PLC0415 — deferred import (analysis only)

    # --- Load records ---
    records: list[dict] = []
    with Path(jsonl_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return []

    # Collect ordered judge IDs and item IDs
    judge_ids: list[str] = sorted({r["judge_id"] for r in records})
    item_ids: list[str] = sorted({r["item_id"] for r in records})
    n_judges = len(judge_ids)
    n_items = len(item_ids)

    judge_idx = {j: i for i, j in enumerate(judge_ids)}
    item_idx = {it: i for i, it in enumerate(item_ids)}

    # Build (n_items, n_judges) label matrix; missing → -1
    matrix = np.full((n_items, n_judges), fill_value=-1, dtype=np.int64)
    for rec in records:
        ii = item_idx[rec["item_id"]]
        ji = judge_idx[rec["judge_id"]]
        label_str = str(rec.get("label", "ERROR")).upper()
        matrix[ii, ji] = _LABEL_MAP.get(label_str, 3)

    # --- Filter to disagreeing items (at least 2 judges with different labels) ---
    disagree_mask = np.array(
        [
            not _all_same(matrix[i][matrix[i] >= 0])
            for i in range(n_items)
        ]
    )
    disagree_indices = np.where(disagree_mask)[0]

    if len(disagree_indices) < k_min:
        return []

    X = matrix[disagree_indices].astype(np.float64)
    disagree_item_ids = [item_ids[i] for i in disagree_indices]

    # --- Select k via elbow on inertia ---
    k_max_actual = min(k_max, len(disagree_indices))
    k_min_actual = min(k_min, k_max_actual)

    if k_min_actual >= k_max_actual:
        best_k = k_min_actual
    else:
        inertias: list[float] = []
        k_range = range(k_min_actual, k_max_actual + 1)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(float(km.inertia_))

        # Elbow: pick k where the second derivative of inertia is maximum
        if len(inertias) >= 3:
            drops = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
            best_idx = int(np.argmax(drops))
            best_k = list(k_range)[best_idx]
        else:
            best_k = k_min_actual

    # --- Final clustering ---
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km_final.fit_predict(X)

    # --- Build DisagreementCluster objects ---
    clusters: list[DisagreementCluster] = []
    for cluster_id in range(best_k):
        mask = labels == cluster_id
        cluster_item_ids = [disagree_item_ids[i] for i in range(len(disagree_item_ids)) if mask[i]]
        cluster_rows = X[mask].astype(np.int64)

        if len(cluster_rows) == 0:
            continue

        # Dominant pattern: most common (judge → label) mapping
        pattern_counter: Counter[str] = Counter()
        for row in cluster_rows:
            parts = []
            for ji, judge in enumerate(judge_ids):
                val = int(row[ji])
                if val >= 0:
                    label_name = _int_to_label(val)
                    parts.append(f"{judge}={label_name}")
            pattern_counter[",".join(parts)] += 1

        dominant_pattern = pattern_counter.most_common(1)[0][0] if pattern_counter else "unknown"

        clusters.append(
            DisagreementCluster(
                cluster_id=cluster_id,
                size=len(cluster_item_ids),
                dominant_pattern=dominant_pattern,
                representative_item_ids=cluster_item_ids[:3],
            )
        )

    return clusters


def _all_same(arr: np.ndarray) -> bool:
    """Return True if all elements of arr are equal (handles empty)."""
    if len(arr) == 0:
        return True
    return bool(np.all(arr == arr[0]))


def _int_to_label(val: int) -> str:
    _map = {0: "CORRECT", 1: "INCORRECT", 2: "UNCERTAIN", 3: "ERROR"}
    return _map.get(val, "UNKNOWN")
