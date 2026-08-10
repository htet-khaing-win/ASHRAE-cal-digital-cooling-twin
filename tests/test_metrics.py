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

from cooling_twin.calibration.metrics import (
    DataInterval,
    G14Verdict,
    ashrae_g14_pass,
    cvrmse,
    nmbe,
)

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


# --------------------------------------------------------------------
# L6.3 -- the ASHRAE G14 acceptance gate
# --------------------------------------------------------------------


def _verdict(nmbe_pct: float, cvrmse_pct: float, interval: DataInterval) -> G14Verdict:
    """Build a verdict directly, to test the pass/fail logic in isolation.

    Constructing the dataclass rather than reverse-engineering data that
    happens to score these values keeps the threshold logic under test
    separate from the metric arithmetic already covered above -- a
    failure here then means the gate is wrong, not that the metrics are.
    """
    nmbe_limit, cvrmse_limit = {
        DataInterval.HOURLY: (10.0, 30.0),
        DataInterval.MONTHLY: (5.0, 15.0),
    }[interval]
    return G14Verdict(
        interval=interval,
        nmbe_pct=nmbe_pct,
        cvrmse_pct=cvrmse_pct,
        nmbe_limit_pct=nmbe_limit,
        cvrmse_limit_pct=cvrmse_limit,
        n_params=8,
        n_points=8760,
    )


def test_hourly_thresholds_match_the_standard() -> None:
    """03_DOMAIN_REFERENCE.md SS4: hourly is NMBE +/-10%, CV(RMSE) 30%."""
    verdict = ashrae_g14_pass(MEASURED, PREDICTED, n_params=0)

    assert verdict.interval is DataInterval.HOURLY
    assert verdict.nmbe_limit_pct == 10.0
    assert verdict.cvrmse_limit_pct == 30.0


def test_monthly_thresholds_match_the_standard() -> None:
    """03_DOMAIN_REFERENCE.md SS4: monthly is NMBE +/-5%, CV(RMSE) 15%."""
    verdict = ashrae_g14_pass(
        MEASURED, PREDICTED, n_params=0, interval=DataInterval.MONTHLY
    )

    assert verdict.nmbe_limit_pct == 5.0
    assert verdict.cvrmse_limit_pct == 15.0


def test_hourly_is_the_default_interval() -> None:
    """This project's declared target (03_DOMAIN_REFERENCE.md SS4)."""
    assert ashrae_g14_pass(MEASURED, PREDICTED, n_params=0).interval is (
        DataInterval.HOURLY
    )


@pytest.mark.parametrize(
    ("nmbe_pct", "cvrmse_pct", "expected"),
    [
        (0.0, 0.0, True),  # perfect
        (9.99, 29.99, True),  # just inside both
        (10.0, 30.0, True),  # exactly on both limits -- G14 says "<="
        (-10.0, 30.0, True),  # the limit is on |NMBE|, so symmetric
        (10.01, 29.99, False),  # NMBE alone fails it
        (9.99, 30.01, False),  # CV(RMSE) alone fails it
        (-10.01, 1.0, False),  # over-prediction fails the same way
    ],
)
def test_hourly_pass_fail_boundaries(
    nmbe_pct: float, cvrmse_pct: float, expected: bool
) -> None:
    """Both criteria required; the comparison is inclusive on both."""
    assert _verdict(nmbe_pct, cvrmse_pct, DataInterval.HOURLY).passed is expected


def test_monthly_is_stricter_than_hourly_on_identical_metrics() -> None:
    """The same numbers pass hourly and fail monthly.

    This is the whole reason the interval has to be an argument: a
    verdict is meaningless without it.
    """
    assert _verdict(7.0, 20.0, DataInterval.HOURLY).passed is True
    assert _verdict(7.0, 20.0, DataInterval.MONTHLY).passed is False


def test_verdict_reports_which_criterion_failed() -> None:
    """A bare bool would lose exactly this."""
    verdict = _verdict(nmbe_pct=15.0, cvrmse_pct=20.0, interval=DataInterval.HOURLY)

    assert verdict.passed is False
    assert verdict.nmbe_pass is False
    assert verdict.cvrmse_pass is True


def test_stretch_target_is_separate_from_the_g14_pass() -> None:
    """CV(RMSE) <= 20% is this project's target, not the standard's."""
    passes_g14_only = _verdict(1.0, 25.0, DataInterval.HOURLY)
    assert passes_g14_only.passed is True
    assert passes_g14_only.meets_stretch_target is False

    assert _verdict(1.0, 20.0, DataInterval.HOURLY).meets_stretch_target is True


def test_suspiciously_good_is_flagged_and_is_not_a_pass_signal() -> None:
    """06_ASSESSMENT.md: hourly CV(RMSE) < 5% is implausible."""
    too_good = _verdict(0.1, 4.9, DataInterval.HOURLY)

    assert too_good.passed is True  # it does pass the standard...
    assert too_good.is_suspiciously_good is True  # ...and still needs checking

    assert _verdict(0.1, 5.0, DataInterval.HOURLY).is_suspiciously_good is False


def test_suspiciously_good_does_not_apply_to_monthly_data() -> None:
    """The 5% plausibility floor is stated for hourly prediction only.

    Monthly averaging genuinely removes most of the variance that makes
    hourly prediction hard, so a low monthly CV(RMSE) is not the same
    warning sign.
    """
    assert _verdict(0.1, 4.9, DataInterval.MONTHLY).is_suspiciously_good is False


def test_suspiciously_good_logs_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The flag must reach a human, not just sit on the dataclass."""
    with caplog.at_level("WARNING"):
        ashrae_g14_pass(MEASURED, MEASURED, n_params=0)

    assert "plausibility floor" in caplog.text


def test_summary_names_the_failing_criterion() -> None:
    text = _verdict(15.0, 20.0, DataInterval.HOURLY).summary()

    assert text.startswith("FAIL")
    assert "NMBE +15.00%" in text
    assert "limit +/-10%" in text
    assert "n=8760" in text
    assert "p=8" in text


def test_verdict_is_immutable() -> None:
    """Frozen so a verdict cannot be edited after the fact.

    A mutable verdict is a verdict that can be "fixed" between being
    computed and being written into reports/02_calibration.md.
    """
    verdict = _verdict(15.0, 20.0, DataInterval.HOURLY)
    with pytest.raises(Exception):  # noqa: B017 -- FrozenInstanceError
        verdict.nmbe_pct = 1.0  # type: ignore[misc]


def test_rejects_an_unknown_interval() -> None:
    with pytest.raises(ValueError, match="must be a DataInterval"):
        ashrae_g14_pass(MEASURED, PREDICTED, n_params=0, interval="hourly")  # type: ignore[arg-type]


def test_gate_reports_n_and_p_it_actually_used() -> None:
    verdict = ashrae_g14_pass(MEASURED, PREDICTED, n_params=1)

    assert verdict.n_points == 4
    assert verdict.n_params == 1
    assert verdict.nmbe_pct == pytest.approx(nmbe(MEASURED, PREDICTED, n_params=1))
    assert verdict.cvrmse_pct == pytest.approx(cvrmse(MEASURED, PREDICTED, n_params=1))
