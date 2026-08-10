"""Tests for ASHRAE G14 metrics.

These are **known-answer tests**: every expected value below was worked
out by hand from `03_DOMAIN_REFERENCE.md` SS4's formulas and is written
into the test as a literal, with the arithmetic shown in a comment. No
expected value is produced by calling the code under test.

That distinction matters here more than anywhere else in the repo. A
property test (L4.5's pattern) asks "is the output self-consistent?" --
it would happily pass on a `cvrmse()` that divides by `n` instead of
`n - p`, because the wrong formula is just as self-consistent as the
right one. Only an independently-derived number catches a transcription
error in the formula itself.

100% statement coverage is required on `metrics.py` (02_CURRICULUM.md
L6.2), so every `raise` branch has its own test.
"""

from __future__ import annotations

import numpy as np
import pytest

from cooling_twin.calibration.metrics import cvrmse, nmbe

# Worked by hand throughout this file:
#   measured  y     = [100, 200, 300, 400]        ybar = 250
#   predicted yhat  = [110, 190, 310, 390]
#   errors (y-yhat) = [-10, +10, -10, +10]        sum = 0, sum of squares = 400
MEASURED = [100.0, 200.0, 300.0, 400.0]
PREDICTED = [110.0, 190.0, 310.0, 390.0]


# --------------------------------------------------------------------
# Known-answer tests
# --------------------------------------------------------------------


def test_nmbe_known_answer_errors_cancel() -> None:
    """sum(errors) = 0, so NMBE = 0 / (4 * 250) * 100 = 0.0%."""
    assert nmbe(MEASURED, PREDICTED, n_params=0) == pytest.approx(0.0)


def test_cvrmse_known_answer() -> None:
    """sqrt(400 / 4) / 250 * 100 = 10 / 250 * 100 = 4.0%."""
    assert cvrmse(MEASURED, PREDICTED, n_params=0) == pytest.approx(4.0)


def test_nmbe_known_answer_uniform_under_prediction() -> None:
    """A model 25 low everywhere.

    errors = [25, 25, 25, 25], sum = 100.
    NMBE = 100 / (4 * 250) * 100 = 10.0%.
    """
    biased = [75.0, 175.0, 275.0, 375.0]
    assert nmbe(MEASURED, biased, n_params=0) == pytest.approx(10.0)


def test_nmbe_sign_convention_is_measured_minus_predicted() -> None:
    """Over-prediction must come out NEGATIVE.

    This is the standard's convention (`y - yhat`) and the opposite of
    the intuitive "error = prediction - truth". A sign flip here would
    invert every verdict in the final report while leaving CV(RMSE)
    -- which squares the errors -- completely unchanged, so nothing
    else in the suite would notice.
    """
    over_predicting = [125.0, 225.0, 325.0, 425.0]
    assert nmbe(MEASURED, over_predicting, n_params=0) == pytest.approx(-10.0)


def test_nmbe_is_zero_for_a_model_that_is_obviously_bad() -> None:
    """The reason NMBE alone is never reported (03_DOMAIN_REFERENCE SS4).

    A flat line at the measured mean has errors [-150, -50, 50, 150],
    which sum to exactly 0 -> NMBE = 0.0%, a perfect score. The same
    model's CV(RMSE) is sqrt(50000 / 4) / 250 * 100 = 44.7214%, which
    fails G14's 30% hourly threshold outright.
    """
    flat_at_mean = [250.0, 250.0, 250.0, 250.0]

    assert nmbe(MEASURED, flat_at_mean, n_params=0) == pytest.approx(0.0)
    assert cvrmse(MEASURED, flat_at_mean, n_params=0) == pytest.approx(
        44.72135955, rel=1e-9
    )


def test_perfect_model_scores_zero_on_both_metrics() -> None:
    assert nmbe(MEASURED, MEASURED, n_params=0) == pytest.approx(0.0)
    assert cvrmse(MEASURED, MEASURED, n_params=0) == pytest.approx(0.0)


# --------------------------------------------------------------------
# The n - p correction
# --------------------------------------------------------------------


def test_cvrmse_n_minus_p_known_answer() -> None:
    """Same errors, p = 2: sqrt(400 / 2) / 250 * 100 = 5.65685%.

    The naive `n` denominator would give 4.0%. This test is the one
    that fails if someone "simplifies" the formula.
    """
    assert cvrmse(MEASURED, PREDICTED, n_params=2) == pytest.approx(
        5.65685424, rel=1e-8
    )


def test_metrics_get_worse_as_claimed_parameters_increase() -> None:
    """More parameters must never make the score look better."""
    scores = [cvrmse(MEASURED, PREDICTED, n_params=p) for p in (0, 1, 2, 3)]
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]


def test_nmbe_n_minus_p_known_answer() -> None:
    """errors sum to 100, p = 2: 100 / (2 * 250) * 100 = 20.0%."""
    biased = [75.0, 175.0, 275.0, 375.0]
    assert nmbe(MEASURED, biased, n_params=2) == pytest.approx(20.0)


# --------------------------------------------------------------------
# Input types
# --------------------------------------------------------------------


def test_accepts_numpy_arrays_and_lists_identically() -> None:
    from_lists = cvrmse(MEASURED, PREDICTED, n_params=1)
    from_arrays = cvrmse(np.array(MEASURED), np.array(PREDICTED), n_params=1)
    assert from_lists == pytest.approx(from_arrays)


# --------------------------------------------------------------------
# Validation branches -- one test per raise, for 100% coverage
# --------------------------------------------------------------------


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same shape"):
        nmbe(MEASURED, PREDICTED[:3], n_params=0)


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        cvrmse([], [], n_params=0)


def test_rejects_non_finite_values() -> None:
    with_nan = [110.0, np.nan, 310.0, 390.0]
    with pytest.raises(ValueError, match="must be finite"):
        cvrmse(MEASURED, with_nan, n_params=0)


def test_rejects_negative_n_params() -> None:
    with pytest.raises(ValueError, match="n_params must be >= 0"):
        nmbe(MEASURED, PREDICTED, n_params=-1)


def test_rejects_no_remaining_degrees_of_freedom() -> None:
    """p == n leaves nothing to measure against."""
    with pytest.raises(ValueError, match=r"n - p must be > 0"):
        cvrmse(MEASURED, PREDICTED, n_params=4)


def test_rejects_zero_measured_mean() -> None:
    with pytest.raises(ValueError, match="mean of measured is zero"):
        nmbe([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], n_params=0)
