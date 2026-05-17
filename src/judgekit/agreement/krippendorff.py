from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def krippendorff_alpha(
    ratings: NDArray[np.float64],
    level_of_measurement: str = "nominal",
) -> float:
    """Krippendorff's alpha inter-rater reliability.

    ratings: (n_raters, n_items) array; NaN indicates missing data.
    level_of_measurement: 'nominal', 'ordinal', 'interval', or 'ratio'.

    Implementation follows the standard formula:
        alpha = 1 - D_o / D_e
    where D_o is observed disagreement and D_e is expected disagreement.
    For nominal: disagreement d(v_k, v_l) = 0 if v_k == v_l, else 1.
    NaN values are treated as missing and excluded from calculations.
    """
    ratings = np.asarray(ratings, dtype=np.float64)
    n_raters, n_items = ratings.shape

    if level_of_measurement == "nominal":
        metric = _nominal_metric
    elif level_of_measurement == "interval":
        metric = _interval_metric
    else:
        raise ValueError(f"Unsupported level_of_measurement: {level_of_measurement!r}")

    # --- Observed disagreement (D_o) ---
    # For each item: sum pairwise disagreements among non-NaN raters / (m_u * (m_u - 1))
    # where m_u is the number of non-NaN raters for item u.
    # D_o = (1 / n) * sum_u [ (1 / (m_u*(m_u-1))) * sum_{k!=l} d(r_uk, r_ul) ]

    do_sum = 0.0
    n_items_valid = 0  # items with at least 2 non-NaN raters

    for u in range(n_items):
        col = ratings[:, u]
        valid = col[~np.isnan(col)]
        m_u = len(valid)
        if m_u < 2:
            continue
        n_items_valid += 1
        pair_sum = 0.0
        for k in range(m_u):
            for ll in range(m_u):
                if k != ll:
                    pair_sum += metric(valid[k], valid[ll])
        do_sum += pair_sum / (m_u * (m_u - 1))

    if n_items_valid == 0:
        return float("nan")

    D_o = do_sum / n_items_valid

    # --- Expected disagreement (D_e) ---
    # Pool all non-NaN values. Treat each item's values as a coincidence matrix.
    # D_e = (1 / (n*(n-1))) * sum_{k,l: k!=l} d(v_k, v_l)
    # where n = total number of non-NaN ratings, v_k iterates over all non-NaN values.

    all_values = ratings.flatten()
    all_values = all_values[~np.isnan(all_values)]
    n_total = len(all_values)

    if n_total < 2:
        return float("nan")

    de_sum = 0.0
    for k in range(n_total):
        for ll in range(n_total):
            if k != ll:
                de_sum += metric(all_values[k], all_values[ll])

    D_e = de_sum / (n_total * (n_total - 1))

    if D_e == 0.0:
        # All values identical → perfect agreement
        return 1.0

    return 1.0 - D_o / D_e


def _nominal_metric(a: float, b: float) -> float:
    return 0.0 if a == b else 1.0


def _interval_metric(a: float, b: float) -> float:
    return (a - b) ** 2
