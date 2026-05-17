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
    """
    raise NotImplementedError
