from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fleiss_kappa(ratings: NDArray[np.int_], n_categories: int) -> float:
    """Fleiss's kappa for multi-rater categorical agreement.

    ratings: (n_items, n_raters) array of category indices (0-indexed integers).
    n_categories: total number of distinct categories.

    Algorithm:
    1. Convert to (n_items, n_categories) count matrix.
    2. Compute p_j = proportion of all assignments in category j.
    3. Compute P_i = per-item proportion of agreeing pairs.
    4. P_bar = mean(P_i).
    5. P_e = sum(p_j ** 2).
    6. kappa = (P_bar - P_e) / (1 - P_e).
    """
    ratings = np.asarray(ratings, dtype=np.int_)
    n_items, n_raters = ratings.shape

    # Build (n_items, n_categories) count matrix
    counts = np.zeros((n_items, n_categories), dtype=np.int_)
    for item in range(n_items):
        for cat_idx in ratings[item]:
            counts[item, cat_idx] += 1

    # p_j: proportion of all assignments in category j
    total_assignments = n_items * n_raters
    p_j = counts.sum(axis=0) / total_assignments  # shape (n_categories,)

    # P_i: per-item proportion of agreeing rater pairs
    # P_i = (1 / (n_raters * (n_raters - 1))) * sum_j n_ij * (n_ij - 1)
    n_ij = counts  # (n_items, n_categories)
    P_i = (n_ij * (n_ij - 1)).sum(axis=1) / (n_raters * (n_raters - 1))

    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()

    if P_e == 1.0:
        # Degenerate: all assignments in one category and all items agree.
        return 1.0

    return float((P_bar - P_e) / (1.0 - P_e))
