"""cooling_twin — a calibrated grey-box thermal digital twin.

This module owns the project-wide reproducibility contract: a single
seed constant and a single way to derive random generators from it.
No other module should call np.random.seed() or construct a Generator
from a literal integer — always go through set_seed() so every random
stream in a run traces back to one recorded value.
"""

from __future__ import annotations

import numpy as np

SEED: int = 42
"""The project's fixed random seed.

Changing this value changes every stochastic result in the project
(calibration parameters, sensitivity screening, conformal resampling).
If you change it, record why in an ADR in 07_PROGRESS.md — a silent
change here invalidates every previously reported number.
"""


def set_seed(seed: int = SEED) -> np.random.Generator:
    """Create a NumPy random Generator seeded for reproducibility.

    This is the ONLY sanctioned way to obtain randomness in this
    project. Never call np.random.seed() directly, and never build a
    Generator from a hardcoded literal elsewhere in the codebase - both
    bypass this single point of control and make runs unreproducible.

    Args:
        seed: The seed value. Defaults to the project-wide SEED
            constant. Override only in tests that need a different
            fixed value, never in production code paths.

    Returns:
        A np.random.Generator instance with independent internal state,
        safe to pass into any function that needs randomness.

    Example:
        >>> rng = set_seed()
        >>> rng.random(3)  # doctest: +SKIP
        array([0.773, 0.439, 0.858])
    """
    return np.random.default_rng(seed)