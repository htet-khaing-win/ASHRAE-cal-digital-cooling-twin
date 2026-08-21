"""Tests for the naive baselines.

Known-answer testing throughout (the L6.2 pattern): every expected
coefficient is a value ordinary least squares must return exactly for
the constructed data, derived by hand rather than by calling the code.
An OLS fit on data that genuinely lies on a line has an exact answer,
which makes these checks unusually sharp -- there is no "close enough"
to hide a wrong design matrix behind.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from cooling_twin import set_seed
from cooling_twin.calibration.baseline import (
    MIN_RELATIVE_IMPROVEMENT_PCT,
    BaselineFit,
    beats_baseline,
    fit_annual_mean,
    fit_change_point,
    fit_linear_regression,
    relative_cvrmse_improvement_pct,
)
from cooling_twin.calibration.metrics import cvrmse

# y = 3 + 2x exactly, so OLS must recover [3, 2] with zero residual.
FEATURE = [0.0, 1.0, 2.0, 3.0, 4.0]
ON_THE_LINE = [3.0, 5.0, 7.0, 9.0, 11.0]


# --------------------------------------------------------------------
# Annual mean
# --------------------------------------------------------------------


def test_annual_mean_known_answer() -> None:
    """mean([100, 200, 300, 400]) = 250."""
    fit = fit_annual_mean([100.0, 200.0, 300.0, 400.0])

    assert fit.coefficients == pytest.approx([250.0])
    assert fit.name == "annual mean"


def test_annual_mean_fits_exactly_one_parameter() -> None:
    """p = 1, not 0. The mean is itself a fitted parameter.

    Scoring it with p=0 would flatter the weakest baseline in the
    comparison, which is the one place an error would be least likely
    to be noticed.
    """
    assert fit_annual_mean([1.0, 2.0, 3.0]).n_params == 1


def test_annual_mean_predicts_a_constant_and_ignores_feature_values() -> None:
    fit = fit_annual_mean([100.0, 200.0, 300.0, 400.0])

    predicted = fit.predict([-50.0, 0.0, 12.5, 900.0])

    assert predicted == pytest.approx([250.0] * 4)


def test_annual_mean_uses_the_row_count_of_the_features_it_is_given() -> None:
    """Uniform signature: the same call shape works for any baseline."""
    fit = fit_annual_mean([100.0, 200.0, 300.0, 400.0])

    assert fit.predict(np.zeros(7)).shape == (7,)


# --------------------------------------------------------------------
# Linear regression
# --------------------------------------------------------------------


def test_linear_regression_known_answer_exact_line() -> None:
    """Data on y = 3 + 2x must give intercept 3, slope 2."""
    fit = fit_linear_regression(FEATURE, ON_THE_LINE)

    assert fit.coefficients == pytest.approx([3.0, 2.0])
    assert fit.n_params == 2


def test_linear_regression_reproduces_the_line_it_fitted() -> None:
    fit = fit_linear_regression(FEATURE, ON_THE_LINE)

    assert fit.predict(FEATURE) == pytest.approx(ON_THE_LINE)
    assert cvrmse(ON_THE_LINE, fit.predict(FEATURE), n_params=2) == pytest.approx(0.0)


def test_linear_regression_extrapolates_from_the_fitted_coefficients() -> None:
    """Predicting on unseen rows -- the reason coefficients are stored.

    Fitting on 2016 and predicting 2017 (L6.10) is only possible
    because `BaselineFit` holds coefficients, not predictions.
    """
    fit = fit_linear_regression(FEATURE, ON_THE_LINE)

    assert fit.predict([10.0, 20.0]) == pytest.approx([23.0, 43.0])


def test_linear_regression_counts_one_parameter_per_feature_plus_intercept() -> None:
    features = np.column_stack([FEATURE, np.array(FEATURE) ** 2])

    fit = fit_linear_regression(features, ON_THE_LINE)

    assert fit.n_params == 3


def test_linear_regression_beats_the_annual_mean_on_temperature_driven_load() -> None:
    """Sanity: the ordering of the two baselines is not accidental."""
    outdoor = np.linspace(5.0, 35.0, 200)
    load = 300.0 + 40.0 * outdoor

    mean_score = cvrmse(load, fit_annual_mean(load).predict(outdoor), n_params=1)
    regression_score = cvrmse(
        load, fit_linear_regression(outdoor, load).predict(outdoor), n_params=2
    )

    assert regression_score < mean_score


# --------------------------------------------------------------------
# The improvement requirement
# --------------------------------------------------------------------


def test_relative_improvement_known_answer() -> None:
    """(48 - 24) / 48 * 100 = 50.0%."""
    assert relative_cvrmse_improvement_pct(48.0, 24.0) == pytest.approx(50.0)


def test_relative_improvement_is_negative_when_the_model_is_worse() -> None:
    """(41.14 - 101.49) / 41.14 * 100 = -146.69%.

    This is L6.1's uncalibrated physics model against Fox 2016's
    linear-regression baseline: worse than a two-parameter regression.
    """
    assert relative_cvrmse_improvement_pct(41.14, 101.49) == pytest.approx(
        -146.694, rel=1e-4
    )


def test_relative_improvement_is_scale_free() -> None:
    """48 -> 24 and 10 -> 5 are both 50%, though the gaps differ 8-fold."""
    assert relative_cvrmse_improvement_pct(48.0, 24.0) == pytest.approx(
        relative_cvrmse_improvement_pct(10.0, 5.0)
    )


def test_beats_baseline_boundary_is_inclusive() -> None:
    """Exactly 30% relative improvement clears the bar."""
    baseline = 40.0
    exactly_30_pct_better = baseline * (1 - MIN_RELATIVE_IMPROVEMENT_PCT / 100.0)

    assert beats_baseline(baseline, exactly_30_pct_better) is True
    assert beats_baseline(baseline, exactly_30_pct_better + 0.01) is False


def test_beats_baseline_default_threshold_matches_the_assessment_file() -> None:
    """06_ASSESSMENT.md M6: ">= 30% relative CV(RMSE)"."""
    assert MIN_RELATIVE_IMPROVEMENT_PCT == 30.0


def test_beats_baseline_accepts_an_explicit_threshold() -> None:
    assert beats_baseline(40.0, 32.0, min_improvement_pct=20.0) is True
    assert beats_baseline(40.0, 32.0, min_improvement_pct=25.0) is False


def test_rejects_a_non_positive_baseline_score() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        relative_cvrmse_improvement_pct(0.0, 5.0)


# --------------------------------------------------------------------
# Validation branches
# --------------------------------------------------------------------


def test_annual_mean_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        fit_annual_mean([])


def test_annual_mean_rejects_two_dimensional_input() -> None:
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        fit_annual_mean([[1.0, 2.0], [3.0, 4.0]])


def test_annual_mean_rejects_non_finite_input() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        fit_annual_mean([1.0, np.nan, 3.0])


def test_regression_rejects_misaligned_target() -> None:
    with pytest.raises(ValueError, match="aligned observations"):
        fit_linear_regression(FEATURE, ON_THE_LINE[:3])


def test_regression_rejects_two_dimensional_target() -> None:
    with pytest.raises(ValueError, match="measured must be 1-dimensional"):
        fit_linear_regression(FEATURE, [[1.0], [2.0], [3.0], [4.0], [5.0]])


def test_regression_rejects_non_finite_target() -> None:
    bad = [3.0, 5.0, np.nan, 9.0, 11.0]
    with pytest.raises(ValueError, match="measured must be finite"):
        fit_linear_regression(FEATURE, bad)


def test_regression_rejects_non_finite_features() -> None:
    bad = [0.0, 1.0, np.inf, 3.0, 4.0]
    with pytest.raises(ValueError, match="features must be finite"):
        fit_linear_regression(bad, ON_THE_LINE)


def test_regression_rejects_empty_features() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        fit_linear_regression([], [])


def test_regression_rejects_more_than_two_dimensions() -> None:
    with pytest.raises(ValueError, match="1- or 2-dimensional"):
        fit_linear_regression(np.zeros((2, 2, 2)), [1.0, 2.0])


def test_regression_rejects_zero_feature_columns() -> None:
    """A (n, 0) array would silently become an annual mean."""
    with pytest.raises(ValueError, match="zero columns"):
        fit_linear_regression(np.zeros((5, 0)), ON_THE_LINE)


def test_regression_rejects_a_constant_feature() -> None:
    """A constant column is collinear with the intercept.

    lstsq would return one arbitrary solution from an infinite family,
    with coefficients that look fine and mean nothing.
    """
    with pytest.raises(ValueError, match="rank-deficient"):
        fit_linear_regression([2.0, 2.0, 2.0, 2.0, 2.0], ON_THE_LINE)


def test_regression_rejects_collinear_features() -> None:
    features = np.column_stack([FEATURE, np.array(FEATURE) * 3.0])
    with pytest.raises(ValueError, match="rank-deficient"):
        fit_linear_regression(features, ON_THE_LINE)


def test_predict_rejects_a_different_number_of_feature_columns() -> None:
    fit = fit_linear_regression(
        np.column_stack([FEATURE, np.array(FEATURE) ** 2]), ON_THE_LINE
    )
    with pytest.raises(ValueError, match="fitted on 2 feature column"):
        fit.predict(FEATURE)


def test_baseline_fit_is_immutable() -> None:
    fit = fit_annual_mean([1.0, 2.0, 3.0])
    with pytest.raises(Exception):  # noqa: B017 -- FrozenInstanceError
        fit.name = "tampered"  # type: ignore[misc]


def test_baseline_fit_can_be_constructed_directly() -> None:
    """The dataclass is part of the public surface, not an implementation detail."""
    fit = BaselineFit(name="hand-built", coefficients=np.array([10.0, 2.0]), n_params=2)

    assert fit.predict([1.0, 2.0]) == pytest.approx([12.0, 14.0])


# ---------------------------------------------------------------------
# 3P change-point baseline (ADR-015)
# ---------------------------------------------------------------------


def test_change_point_recovers_a_known_breakpoint() -> None:
    """Known-answer: load = 500 + 40 * max(0, T - 18), no noise.

    Built to lie EXACTLY on the model, so the fit has an exact answer
    and a wrong design matrix cannot hide inside a tolerance -- the same
    construction the OLS baselines in this file use.
    """
    rng = set_seed()
    temperature = rng.uniform(-10.0, 40.0, 8760)
    load = 500.0 + 40.0 * np.maximum(0.0, temperature - 18.0)

    fit = fit_change_point(temperature, load)

    assert fit.n_params == 3
    assert fit.change_point == pytest.approx(18.0, abs=0.5)
    assert fit.coefficients[0] == pytest.approx(500.0, abs=5.0)
    assert fit.coefficients[1] == pytest.approx(40.0, abs=0.5)
    assert fit.predict(temperature) == pytest.approx(load, abs=10.0)


def test_change_point_is_flat_below_the_breakpoint() -> None:
    """The property the RC model cannot express: a floor."""
    rng = set_seed()
    temperature = rng.uniform(-10.0, 40.0, 8760)
    load = 500.0 + 40.0 * np.maximum(0.0, temperature - 18.0)

    fit = fit_change_point(temperature, load)
    cold = fit.predict(np.array([-20.0, -10.0, 0.0, 10.0]))

    assert cold == pytest.approx(cold[0])  # identical at every cold point
    assert cold[0] == pytest.approx(500.0, abs=5.0)


def test_change_point_counts_its_breakpoint_as_a_parameter() -> None:
    """p = 3, not 2. Under-counting flatters the baseline.

    The breakpoint is fitted from the data, so it costs a degree of
    freedom in `cvrmse`'s n - p denominator. It does not appear in the
    design matrix, which is why the coefficient array is shorter than
    `n_params` -- the one place in this module where they differ.
    """
    rng = set_seed()
    temperature = rng.uniform(-10.0, 40.0, 3000)
    load = 300.0 + 25.0 * np.maximum(0.0, temperature - 15.0)

    fit = fit_change_point(temperature, load)

    assert fit.n_params == 3
    assert fit.coefficients.size == 2
    assert fit.predict(temperature).shape == (3000,)


def test_change_point_beats_a_straight_line_on_a_floored_load() -> None:
    """The comparison ADR-015 authorised this baseline to make."""
    rng = set_seed()
    temperature = rng.uniform(-20.0, 35.0, 8760)
    load = 450.0 + 45.0 * np.maximum(0.0, temperature - 12.0) + rng.normal(
        0.0, 40.0, 8760
    )

    change_point = fit_change_point(temperature, load)
    straight = fit_linear_regression(temperature, load)

    assert cvrmse(load, change_point.predict(temperature), n_params=3) < cvrmse(
        load, straight.predict(temperature), n_params=2
    )


def test_change_point_warns_when_it_degenerates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A purely linear load has no breakpoint, and must say so."""
    rng = set_seed()
    temperature = rng.uniform(-10.0, 40.0, 8760)
    load = 100.0 + 30.0 * temperature + rng.normal(0.0, 20.0, 8760)

    with caplog.at_level(logging.WARNING, logger="cooling_twin.calibration.baseline"):
        fit_change_point(temperature, load)

    assert "degenerated" in caplog.text


def test_change_point_warns_on_a_negative_slope(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cooling model sloping down with temperature is a meter finding."""
    rng = set_seed()
    temperature = rng.uniform(-10.0, 40.0, 8760)
    load = 2000.0 - 30.0 * np.maximum(0.0, temperature - 10.0)

    with caplog.at_level(logging.WARNING, logger="cooling_twin.calibration.baseline"):
        fit_change_point(temperature, load)

    assert "not positive" in caplog.text


def test_change_point_rejects_a_series_too_short_to_split() -> None:
    """No candidate can leave MIN_SEGMENT_HOURS above it."""
    rng = set_seed()
    temperature = rng.uniform(0.0, 30.0, 100)
    with pytest.raises(ValueError, match="no candidate change point"):
        fit_change_point(temperature, rng.normal(1000.0, 50.0, 100))


@pytest.mark.parametrize(
    ("temperature", "load", "match"),
    [
        (np.zeros((10, 2)), np.zeros(10), "1-dimensional"),
        (np.array([]), np.array([]), "at least one point"),
        (np.array([1.0, np.nan, 3.0]), np.zeros(3), "must be finite"),
    ],
)
def test_change_point_rejects_unusable_input(
    temperature: np.ndarray, load: np.ndarray, match: str
) -> None:
    """Same validation vocabulary as the other baselines."""
    with pytest.raises(ValueError, match=match):
        fit_change_point(temperature, load)


def test_existing_baselines_are_unchanged_by_the_new_field() -> None:
    """`change_point` defaults to None and must not alter old fits.

    The predict() path was generalised to support the 3P model; this
    pins that the two OLS baselines still predict exactly as before.
    """
    rng = set_seed()
    temperature = rng.uniform(0.0, 30.0, 500)
    load = 3.0 + 2.0 * temperature

    mean_fit = fit_annual_mean(load)
    regression = fit_linear_regression(temperature, load)

    assert mean_fit.change_point is None
    assert regression.change_point is None
    assert mean_fit.predict(temperature) == pytest.approx(load.mean())
    assert regression.predict(temperature) == pytest.approx(load)
