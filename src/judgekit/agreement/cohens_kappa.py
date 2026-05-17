from __future__ import annotations


def pairwise_kappa(
    rater_a: list[str],
    rater_b: list[str],
    labels: list[str] | None = None,
) -> float:
    """Cohen's kappa for two raters over a categorical label set."""
    raise NotImplementedError
