from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def krippendorff_alpha(
    ratings: NDArray[np.float64],
    level_of_measurement: str = "nominal",
) -> float:
    """Krippendorff's alpha inter-rater reliability.

    ratings: (n_raters, n_items) array; NaN indicates missing data.
    level_of_measurement: 'nominal' or 'interval'.

    Implementation uses the coincidence matrix approach (Krippendorff 2004):

        1. Build value_counts: (n_items, n_values) count matrix.
        2. Compute coincidence matrix o[k, l]:
               o = sum_u  n_uk * n_ul / (m_u - 1)   (off-diagonal)
               o = sum_u  n_uk * (n_uk - 1) / (m_u - 1)  (diagonal)
           equivalently: o = sum_u (n_u n_u^T - diag(n_u)) / (m_u - 1)
        3. n_v = o.sum(axis=0) — marginal pairable counts per value.
        4. Random coincidence matrix e[k, l] = (n_v[k]*n_v[l] - n_v[k]*I) / (n_total - 1)
        5. Distance matrix d[k, l] via metric.
        6. alpha = 1 - (o * d).sum() / (e * d).sum()

    NaN values are treated as missing and excluded from calculations.
    """
    ratings = np.asarray(ratings, dtype=np.float64)
    n_raters, n_items = ratings.shape

    # Determine unique value domain from non-NaN values
    all_vals = ratings.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals) < 2:
        return float("nan")

    value_domain = np.unique(all_vals)
    n_vals = len(value_domain)

    if n_vals <= 1:
        # Only one distinct value → perfect agreement
        return 1.0

    # Build value_counts: (n_items, n_vals) — how many raters assigned each value to each item
    value_counts = np.zeros((n_items, n_vals), dtype=np.float64)
    for u in range(n_items):
        col = ratings[:, u]
        valid = col[~np.isnan(col)]
        for v in valid:
            idx = int(np.searchsorted(value_domain, v))
            value_counts[u, idx] += 1

    # Number of raters per item (pairable = max(m_u, 2) for the formula,
    # but items with < 2 raters contribute 0 to coincidences)
    m = value_counts.sum(axis=1)  # (n_items,)

    # Build coincidence matrix: o[k, l] = sum_u n_uk*n_ul/(m_u-1) for k!=l
    #                                       sum_u n_uk*(n_uk-1)/(m_u-1) for k==l
    # combined: o = sum_u [ outer(n_u, n_u) - diag(n_u) ] / (m_u - 1)
    o = np.zeros((n_vals, n_vals), dtype=np.float64)
    for u in range(n_items):
        m_u = m[u]
        if m_u < 2:
            continue
        n_u = value_counts[u]  # shape (n_vals,)
        outer = np.outer(n_u, n_u)
        diag = np.diag(n_u)
        o += (outer - diag) / (m_u - 1)

    # Marginal pairable counts
    n_v = o.sum(axis=0)  # shape (n_vals,)
    n_total = n_v.sum()

    if n_total == 0.0:
        return float("nan")

    # Distance matrix
    d = _distance_matrix(value_domain, level_of_measurement)

    # Observed disagreement numerator
    o_d = (o * d).sum()

    # Expected (random) coincidence matrix: e[k,l] = (n_v[k]*n_v[l] - delta_kl*n_v[k]) / (n_total - 1)
    e = (np.outer(n_v, n_v) - np.diag(n_v)) / (n_total - 1)
    e_d = (e * d).sum()

    if e_d == 0.0:
        # No expected disagreement → perfect agreement (all values identical)
        return 1.0

    return float(1.0 - o_d / e_d)


def _distance_matrix(value_domain: NDArray[np.float64], level_of_measurement: str) -> NDArray[np.float64]:
    """Return (n_vals, n_vals) pairwise distance matrix."""
    n = len(value_domain)
    d = np.zeros((n, n), dtype=np.float64)
    if level_of_measurement == "nominal":
        for i in range(n):
            for j in range(n):
                d[i, j] = 0.0 if i == j else 1.0
    elif level_of_measurement == "interval":
        for i in range(n):
            for j in range(n):
                d[i, j] = (value_domain[i] - value_domain[j]) ** 2
    else:
        raise ValueError(f"Unsupported level_of_measurement: {level_of_measurement!r}")
    return d
