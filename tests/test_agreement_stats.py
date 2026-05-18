"""Tests for agreement statistics (kappa, alpha). Phase 3."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters
from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss_kappa

from judgekit.agreement.cohens_kappa import pairwise_kappa
from judgekit.agreement.fleiss_kappa import fleiss_kappa  # noqa: E402

# ---------------------------------------------------------------------------
# Cohen's kappa tests
# ---------------------------------------------------------------------------


def test_pairwise_kappa_perfect_agreement():
    """Identical labels from both raters => kappa = 1.0."""
    rater_a = ["CORRECT", "CORRECT", "INCORRECT"]
    rater_b = ["CORRECT", "CORRECT", "INCORRECT"]
    assert pairwise_kappa(rater_a, rater_b) == pytest.approx(1.0, abs=1e-9)


def test_pairwise_kappa_known_value():
    """Hand-computed and sklearn-verified kappa for a small example.

    rater_a = ["CORRECT", "CORRECT", "INCORRECT", "UNCERTAIN"]
    rater_b = ["CORRECT", "INCORRECT", "INCORRECT", "UNCERTAIN"]

    p_o = 3/4 = 0.75
    count_a: CORRECT=2, INCORRECT=1, UNCERTAIN=1
    count_b: CORRECT=1, INCORRECT=2, UNCERTAIN=1
    p_e = (2/4)(1/4) + (1/4)(2/4) + (1/4)(1/4) = 2/16+2/16+1/16 = 5/16 = 0.3125
    kappa = (0.75 - 0.3125) / (1 - 0.3125) = 0.4375 / 0.6875 ≈ 0.6364

    sklearn.metrics.cohen_kappa_score confirms: 0.6363636363636364
    """
    rater_a = ["CORRECT", "CORRECT", "INCORRECT", "UNCERTAIN"]
    rater_b = ["CORRECT", "INCORRECT", "INCORRECT", "UNCERTAIN"]
    expected = 0.6363636363636364
    result = pairwise_kappa(rater_a, rater_b)
    assert abs(result - expected) < 1e-6


def test_pairwise_kappa_all_same_label():
    """All items get the same label from both raters => kappa = 1.0.

    When p_e == 1.0 (degenerate case), the formula 0/0 is handled gracefully
    by returning 1.0 because observed agreement is also perfect.
    """
    rater_a = ["CORRECT", "CORRECT", "CORRECT"]
    rater_b = ["CORRECT", "CORRECT", "CORRECT"]
    result = pairwise_kappa(rater_a, rater_b)
    assert result == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "rater_a, rater_b",
    [
        (["A", "B", "A", "C"], ["A", "A", "A", "C"]),
        (["X", "Y", "X", "Y", "Z"], ["X", "Y", "Y", "X", "Z"]),
        (["1", "2", "3", "1", "2"], ["1", "1", "3", "2", "2"]),
    ],
)
def test_pairwise_kappa_cross_check_sklearn(rater_a, rater_b):
    """Result matches sklearn.metrics.cohen_kappa_score within 1e-6."""
    expected = cohen_kappa_score(rater_a, rater_b)
    result = pairwise_kappa(rater_a, rater_b)
    assert abs(result - expected) < 1e-6, (
        f"Got {result}, expected {expected} for {rater_a} vs {rater_b}"
    )


# ---------------------------------------------------------------------------
# Fleiss kappa tests
# ---------------------------------------------------------------------------


def test_fleiss_kappa_perfect_agreement():
    """3 items x 2 raters, all raters agree on each item => kappa = 1.0.

    ratings[i, j] = category index assigned by rater j to item i.
    """
    # Item 0: both raters choose cat 0
    # Item 1: both raters choose cat 1
    # Item 2: both raters choose cat 2
    ratings = np.array([[0, 0], [1, 1], [2, 2]])
    result = fleiss_kappa(ratings, n_categories=3)
    assert result == pytest.approx(1.0, abs=1e-9)


def test_fleiss_kappa_known_value():
    """Cross-check with statsmodels for a 14-item, 6-rater, 3-category example.

    statsmodels expects a (n_items, n_categories) count matrix; we convert
    from our (n_items, n_raters) raw ratings to compare.

    Reference value: fleiss_kappa ≈ 0.4060 (verified via statsmodels).
    """
    ratings = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 2],
        [1, 1, 1, 1, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [1, 1, 2, 2, 2, 2],
        [1, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2],
        [0, 0, 0, 1, 2, 2],
    ])
    # statsmodels reference
    counts, _ = aggregate_raters(ratings)
    expected = sm_fleiss_kappa(counts)  # ≈ 0.4060235704932346

    result = fleiss_kappa(ratings, n_categories=3)
    assert abs(result - expected) < 1e-4, (
        f"Got {result}, expected {expected}"
    )


def test_fleiss_kappa_cross_check_statsmodels():
    """Second cross-check with a different 5-item, 4-rater, 3-category ratings matrix.

    statsmodels reference: fleiss_kappa ≈ 0.1870.
    """
    ratings = np.array([
        [0, 0, 1, 0],
        [1, 1, 1, 0],
        [0, 1, 0, 0],
        [2, 2, 2, 1],
        [1, 0, 1, 1],
    ])
    counts, _ = aggregate_raters(ratings)
    expected = sm_fleiss_kappa(counts)  # ≈ 0.1869918699186991

    result = fleiss_kappa(ratings, n_categories=3)
    assert abs(result - expected) < 1e-4, (
        f"Got {result}, expected {expected}"
    )
