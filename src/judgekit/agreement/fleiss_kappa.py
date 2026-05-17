from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fleiss_kappa(ratings: NDArray[np.int_], n_categories: int) -> float:
    """Fleiss's kappa for multi-rater categorical agreement.

    ratings: (n_items, n_raters) array of category indices.
    n_categories: total number of distinct categories.
    """
    raise NotImplementedError
