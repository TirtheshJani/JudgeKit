from __future__ import annotations

from collections import Counter


def pairwise_kappa(
    rater_a: list[str],
    rater_b: list[str],
    labels: list[str] | None = None,
) -> float:
    """Cohen's kappa for two raters over a categorical label set.

    kappa = (p_o - p_e) / (1 - p_e)

    where p_o is the observed agreement proportion and p_e is the
    expected agreement by chance (product of marginal proportions).

    Edge case: if p_e == 1.0 (all items have the same label), the
    formula is 0/0.  Both raters agree perfectly in this case so we
    return 1.0.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must have the same length")
    n = len(rater_a)
    if n == 0:
        raise ValueError("rater sequences must be non-empty")

    if labels is None:
        labels = sorted(set(rater_a) | set(rater_b))

    count_a = Counter(rater_a)
    count_b = Counter(rater_b)

    p_o = sum(a == b for a, b in zip(rater_a, rater_b, strict=True)) / n
    p_e = sum(count_a[lbl] / n * count_b[lbl] / n for lbl in labels)

    if p_e == 1.0:
        # Degenerate: all items share one label; p_o must also be 1.0.
        return 1.0

    return (p_o - p_e) / (1.0 - p_e)
