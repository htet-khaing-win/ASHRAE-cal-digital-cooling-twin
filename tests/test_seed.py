"""Determinism tests for the project's seed contract.

If any test here fails, do not proceed to any lesson that depends on
stochastic optimization (M6) - the reproducibility guarantee is broken
and every downstream calibration result becomes unverifiable.
"""

from __future__ import annotations

import numpy as np

from cooling_twin import SEED, set_seed


def test_set_seed_returns_generator() -> None:
    """set_seed() must return a np.random.Generator, not the legacy
    RandomState - the legacy API shares global state across instances
    in ways the new Generator API does not.
    """
    rng = set_seed()
    assert isinstance(rng, np.random.Generator)


def test_same_seed_gives_identical_sequence() -> None:
    """Two generators from the same seed must produce bit-identical
    output. This is the core reproducibility guarantee: if this test
    ever fails, calibration results are no longer verifiable.
    """
    rng1 = set_seed(SEED)
    rng2 = set_seed(SEED)

    sequence1 = rng1.random(100)
    sequence2 = rng2.random(100)

    np.testing.assert_array_equal(sequence1, sequence2)


def test_different_seeds_give_different_sequences() -> None:
    """Sanity check on the other direction: different seeds must NOT
    coincidentally collide. Guards against a degenerate implementation
    that ignores the seed argument entirely.
    """
    rng1 = set_seed(42)
    rng2 = set_seed(43)

    sequence1 = rng1.random(100)
    sequence2 = rng2.random(100)

    assert not np.array_equal(sequence1, sequence2)


def test_default_seed_matches_project_constant() -> None:
    """set_seed() with no argument must use the project SEED, not some
    other default - this is what makes calling set_seed() from any
    module produce results consistent with every other module.
    """
    rng_default = set_seed()
    rng_explicit = set_seed(SEED)

    sequence_default = rng_default.random(50)
    sequence_explicit = rng_explicit.random(50)

    np.testing.assert_array_equal(sequence_default, sequence_explicit)