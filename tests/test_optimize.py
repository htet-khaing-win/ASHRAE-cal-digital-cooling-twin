"""Tests for the calibration objective.

Known-answer throughout (the L6.2 pattern). The reference case is:

    measured y     = [100, 200, 300, 400]          ybar = 250
    model A  yhat  = 0.92 * y = [92, 184, 276, 368]
             errors (y - yhat) = [8, 16, 24, 32]
             sum = 80         -> NMBE = 80 / (4*250) * 100 = 8.0%
             sum sq = 1920    -> CV(RMSE) = sqrt(1920/4)/250*100 = 8.7636%
    model B  yhat  = y + [-60, +60, -60, +60]
             errors = [60, -60, 60, -60]
             sum = 0          -> NMBE = 0.0%
             sum sq = 14400   -> CV(RMSE) = sqrt(14400/4)/250*100 = 24.0%

Under G14's hourly limits (NMBE 10%, CV(RMSE) 30%) that gives budget
terms of 0.8 / 0.29212 for A and 0.0 / 0.8 for B -- so A totals 1.09212
and B totals 0.8. Every expected number below is derived from those
lines, not from calling the code.
"""

from __future__ import annotations

import numpy as np
import pytest

from cooling_twin.calibration.metrics import DataInterval, cvrmse, nmbe
from cooling_twin.calibration.optimize import (
    DEFAULT_PENALTY_WEIGHT,
    INFEASIBLE_OBJECTIVE,
    ObjectiveBreakdown,
    clipping_violation,
    g14_objective,
    physical_penalty,
)

MEASURED = np.array([100.0, 200.0, 300.0, 400.0])
MODEL_A = MEASURED * 0.92  # biased low by 8%, but tight
MODEL_B = MEASURED + np.array([-60.0, 60.0, -60.0, 60.0])  # unbiased, loose


# --------------------------------------------------------------------
# The objective's known answers
# --------------------------------------------------------------------


def test_objective_reports_the_raw_metrics_unchanged() -> None:
    breakdown = g14_objective(MEASURED, MODEL_A, n_params=0)

    assert breakdown.nmbe_pct == pytest.approx(8.0)
    assert breakdown.cvrmse_pct == pytest.approx(8.76356, rel=1e-5)


def test_each_term_is_the_metric_divided_by_its_own_g14_limit() -> None:
    """8.0 / 10 = 0.8 and 8.76356 / 30 = 0.292119."""
    breakdown = g14_objective(MEASURED, MODEL_A, n_params=0)

    assert breakdown.nmbe_term == pytest.approx(0.8)
    assert breakdown.cvrmse_term == pytest.approx(0.292119, rel=1e-5)


def test_total_is_the_sum_of_the_budget_terms() -> None:
    """0.292119 + 0.8 = 1.092119."""
    assert g14_objective(MEASURED, MODEL_A, n_params=0).total == pytest.approx(
        1.092119, rel=1e-5
    )


def test_an_unbiased_but_loose_model_scores_its_cvrmse_budget_alone() -> None:
    """B: NMBE exactly 0, CV(RMSE) 24 / 30 = 0.8."""
    breakdown = g14_objective(MEASURED, MODEL_B, n_params=0)

    assert breakdown.nmbe_term == pytest.approx(0.0)
    assert breakdown.cvrmse_term == pytest.approx(0.8)
    assert breakdown.total == pytest.approx(0.8)


def test_the_budget_objective_and_a_naive_sum_disagree_about_which_model_is_better() -> (
    None
):
    """The whole reason the normalisation exists.

    `cvrmse + |nmbe|` implicitly prices one point of NMBE the same as
    one point of CV(RMSE), though G14 allows three times as much of the
    latter. On these two models the two objectives pick opposite
    winners -- so the choice of objective, not the data, decides the
    answer.
    """
    budget_a = g14_objective(MEASURED, MODEL_A, n_params=0).total
    budget_b = g14_objective(MEASURED, MODEL_B, n_params=0).total

    naive_a = cvrmse(MEASURED, MODEL_A, 0) + abs(nmbe(MEASURED, MODEL_A, 0))
    naive_b = cvrmse(MEASURED, MODEL_B, 0) + abs(nmbe(MEASURED, MODEL_B, 0))

    assert budget_b < budget_a  # budget objective prefers B
    assert naive_a < naive_b  # naive objective prefers A


def test_monthly_interval_normalises_against_the_stricter_limits() -> None:
    """Monthly limits are 5 / 15, so both terms double against hourly."""
    hourly = g14_objective(MEASURED, MODEL_A, n_params=0)
    monthly = g14_objective(
        MEASURED, MODEL_A, n_params=0, interval=DataInterval.MONTHLY
    )

    assert monthly.nmbe_term == pytest.approx(2 * hourly.nmbe_term)
    assert monthly.cvrmse_term == pytest.approx(2 * hourly.cvrmse_term)


def test_nmbe_weight_scales_only_the_nmbe_term() -> None:
    doubled = g14_objective(MEASURED, MODEL_A, n_params=0, nmbe_weight=2.0)
    base = g14_objective(MEASURED, MODEL_A, n_params=0)

    assert doubled.nmbe_term == pytest.approx(base.nmbe_term)  # reported unweighted
    assert doubled.total == pytest.approx(base.total + base.nmbe_term)


def test_objective_uses_n_minus_p_like_the_metrics_do() -> None:
    with_params = g14_objective(MEASURED, MODEL_A, n_params=2)

    assert with_params.cvrmse_pct == pytest.approx(cvrmse(MEASURED, MODEL_A, 2))
    assert with_params.nmbe_pct == pytest.approx(nmbe(MEASURED, MODEL_A, 2))
    assert with_params.total > g14_objective(MEASURED, MODEL_A, n_params=0).total


# --------------------------------------------------------------------
# The breakdown
# --------------------------------------------------------------------


def test_binding_criterion_names_the_largest_term() -> None:
    """A is bias-limited; B is scatter-limited."""
    assert g14_objective(MEASURED, MODEL_A, n_params=0).binding_criterion == "nmbe"
    assert g14_objective(MEASURED, MODEL_B, n_params=0).binding_criterion == "cvrmse"


def test_binding_criterion_reports_the_penalty_when_it_dominates() -> None:
    breakdown = g14_objective(
        MEASURED, MODEL_A, n_params=0, violations={"clipped_hours": 1.0}
    )

    assert breakdown.binding_criterion == "penalty"


def test_summary_names_the_binding_criterion() -> None:
    text = g14_objective(MEASURED, MODEL_A, n_params=0).summary()

    assert "binding: nmbe" in text
    assert "total=1.0921" in text


def test_breakdown_is_immutable() -> None:
    breakdown = g14_objective(MEASURED, MODEL_A, n_params=0)
    with pytest.raises(Exception):  # noqa: B017 -- FrozenInstanceError
        breakdown.total = 0.0  # type: ignore[misc]


def test_breakdown_can_be_constructed_directly() -> None:
    breakdown = ObjectiveBreakdown(
        cvrmse_pct=1.0,
        nmbe_pct=-2.0,
        cvrmse_term=0.1,
        nmbe_term=0.2,
        penalty=0.0,
        total=0.3,
    )

    assert breakdown.binding_criterion == "nmbe"


# --------------------------------------------------------------------
# Penalties
# --------------------------------------------------------------------


def test_no_violations_means_no_penalty() -> None:
    assert physical_penalty({}) == 0.0
    assert physical_penalty({"a": 0.0, "b": 0.0}) == 0.0


def test_penalty_scales_linearly_with_the_violation() -> None:
    """The property that gives the infeasible region a usable slope.

    A constant penalty would make every infeasible point look equally
    bad, leaving a differential-evolution population no direction to
    move in.
    """
    small = physical_penalty({"v": 0.5}, weight=10.0)
    large = physical_penalty({"v": 2.0}, weight=10.0)

    assert small == pytest.approx(5.0)
    assert large == pytest.approx(20.0)
    assert large == pytest.approx(4 * small)


def test_penalties_from_several_invariants_add() -> None:
    assert physical_penalty({"a": 0.5, "b": 1.5}, weight=2.0) == pytest.approx(4.0)


def test_default_penalty_weight_is_steep_relative_to_the_metric_budgets() -> None:
    """A full-allowance violation must outweigh any achievable metric term."""
    assert DEFAULT_PENALTY_WEIGHT == 10.0
    assert physical_penalty({"v": 1.0}) > g14_objective(
        MEASURED, MODEL_B, n_params=0
    ).total


def test_penalty_rejects_a_negative_violation() -> None:
    """A negative violation would pay the optimiser to break physics."""
    with pytest.raises(ValueError, match="is negative"):
        physical_penalty({"v": -0.5})


def test_penalty_rejects_a_non_finite_violation() -> None:
    with pytest.raises(ValueError, match="not finite"):
        physical_penalty({"v": float("nan")})


def test_penalty_rejects_a_negative_weight() -> None:
    with pytest.raises(ValueError, match="weight must be >= 0"):
        physical_penalty({"v": 1.0}, weight=-1.0)


def test_objective_rejects_a_negative_nmbe_weight() -> None:
    with pytest.raises(ValueError, match="nmbe_weight must be >= 0"):
        g14_objective(MEASURED, MODEL_A, n_params=0, nmbe_weight=-1.0)


# --------------------------------------------------------------------
# The clipping violation
# --------------------------------------------------------------------


def test_no_clipping_is_no_violation() -> None:
    assert clipping_violation(np.array([1.0, 2.0, 3.0, 4.0])) == 0.0


def test_clipping_within_the_allowance_is_no_violation() -> None:
    """Exactly 5% negative against a 5% allowance -> 0.0, inclusive."""
    load = np.ones(100)
    load[:5] = -1.0

    assert clipping_violation(load, max_clipped_fraction=0.05) == pytest.approx(0.0)


def test_clipping_violation_known_answer() -> None:
    """10% clipped against a 5% allowance: (0.10 - 0.05) / 0.05 = 1.0."""
    load = np.ones(100)
    load[:10] = -1.0

    assert clipping_violation(load, max_clipped_fraction=0.05) == pytest.approx(1.0)


def test_clipping_violation_grows_with_the_excess() -> None:
    """20% clipped: (0.20 - 0.05) / 0.05 = 3.0."""
    load = np.ones(100)
    load[:20] = -1.0

    assert clipping_violation(load, max_clipped_fraction=0.05) == pytest.approx(3.0)


def test_zero_is_not_clipped() -> None:
    """A load of exactly zero is met, not clipped -- the test is `< 0`."""
    assert clipping_violation(np.zeros(10)) == 0.0


def test_clipping_violation_rejects_an_out_of_range_allowance() -> None:
    with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
        clipping_violation(np.ones(10), max_clipped_fraction=0.0)
    with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
        clipping_violation(np.ones(10), max_clipped_fraction=1.5)


def test_clipping_violation_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        clipping_violation(np.array([]))


def test_clipping_violation_rejects_non_finite_input() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        clipping_violation(np.array([1.0, np.nan]))


# --------------------------------------------------------------------
# The infeasible sentinel
# --------------------------------------------------------------------


def test_infeasible_objective_is_finite_and_dominant() -> None:
    """Finite so Morris and the optimiser can still form differences."""
    assert np.isfinite(INFEASIBLE_OBJECTIVE)
    badly_infeasible = g14_objective(
        MEASURED, MODEL_A, n_params=0, violations={"v": 100.0}
    ).total
    assert badly_infeasible < INFEASIBLE_OBJECTIVE
