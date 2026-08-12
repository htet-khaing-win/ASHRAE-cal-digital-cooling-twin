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

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

from cooling_twin.calibration.metrics import DataInterval, cvrmse, nmbe
from cooling_twin.calibration.optimize import (
    DEFAULT_DE_POPSIZE,
    DEFAULT_PENALTY_WEIGHT,
    INFEASIBLE_OBJECTIVE,
    LOCAL_STEP_FRACTION,
    CalibrationResult,
    ObjectiveBreakdown,
    _FiniteObjective,
    calibrate,
    clipping_violation,
    find_pinned_parameters,
    g14_objective,
    physical_penalty,
    select_better_stage,
    write_artifact,
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


# --------------------------------------------------------------------
# L6.7 -- the two-stage search
# --------------------------------------------------------------------
#
# Known-answer surface for every test below:
#
#     rippled_bowl(x) = sum(z^2) + 0.15 * sum(1 - cos(2*pi*4*z))
#     z = (x - truth) / bound_width
#
# Minimum is EXACTLY 0.0 at x = truth, and there is a local minimum
# roughly every quarter of each bound width. A local optimiser started
# at the nominal guess lands in one of those and stops; a global one
# must not.


TRAP_BOUNDS = {"gain": (5.0, 200.0), "ua": (0.3, 3.0)}
TRAP_TRUTH = np.array([120.0, 1.7])
TRAP_WIDTH = np.array([195.0, 2.7])


def rippled_bowl(vector: np.ndarray) -> float:
    """Known-answer multi-modal objective; 0.0 at TRAP_TRUTH."""
    offset = (np.asarray(vector, dtype=float) - TRAP_TRUTH) / TRAP_WIDTH
    ripple = 1.0 - np.cos(2.0 * np.pi * 4.0 * offset)
    return float(np.sum(offset**2) + 0.15 * np.sum(ripple))


def test_calibrate_finds_the_known_optimum() -> None:
    """DE must cross the ripples that trap a local start."""
    result = calibrate(rippled_bowl, TRAP_BOUNDS, maxiter=40)

    assert result.objective_value < 1e-6
    np.testing.assert_allclose(result.best_parameters, TRAP_TRUTH, rtol=1e-3)


def test_local_only_is_trapped_by_the_same_surface() -> None:
    """The reason stage 1 exists: this is what a local-only run returns."""
    trapped = minimize(
        rippled_bowl,
        x0=np.array([15.0, 1.0]),  # L6.1's nominal guess
        method="L-BFGS-B",
        bounds=list(TRAP_BOUNDS.values()),
        options={"eps": LOCAL_STEP_FRACTION * TRAP_WIDTH},
    )
    assert float(trapped.fun) > 0.1
    assert abs(float(trapped.x[0]) - TRAP_TRUTH[0]) > 50.0


def test_calibrate_runs_in_parallel_without_a_pickling_error() -> None:
    """workers != 1 pickles the objective -- including our own wrapper.

    Regression test: `calibrate` used to wrap the objective in a nested
    closure to count evaluations, which made every parallel run fail
    with "Can't pickle local object" no matter how picklable the
    caller's own objective was.
    """
    result = calibrate(rippled_bowl, TRAP_BOUNDS, maxiter=3, workers=2)

    assert np.isfinite(result.objective_value)
    assert result.n_evaluations > 0


def test_evaluation_count_comes_from_the_optimisers() -> None:
    """A hand-rolled counter cannot see a worker process's calls."""
    result = calibrate(rippled_bowl, TRAP_BOUNDS, maxiter=5)

    # DE alone evaluates at least its initial population.
    assert result.n_evaluations >= DEFAULT_DE_POPSIZE * len(TRAP_BOUNDS)


def test_finite_objective_wrapper_is_picklable() -> None:
    """The property the parallel path depends on, tested directly."""
    wrapper = _FiniteObjective(rippled_bowl)
    restored = pickle.loads(pickle.dumps(wrapper))

    assert restored(TRAP_TRUTH) == wrapper(TRAP_TRUTH) == 0.0


def test_finite_objective_wrapper_substitutes_the_sentinel() -> None:
    assert _FiniteObjective(lambda _v: float("nan"))(TRAP_TRUTH) == INFEASIBLE_OBJECTIVE
    assert _FiniteObjective(lambda _v: float("inf"))(TRAP_TRUTH) == INFEASIBLE_OBJECTIVE


def test_calibrate_is_reproducible_under_the_same_seed() -> None:
    """L0.3's rule applied to the optimiser: same seed, same answer."""
    first = calibrate(rippled_bowl, TRAP_BOUNDS, maxiter=15, seed=7)
    second = calibrate(rippled_bowl, TRAP_BOUNDS, maxiter=15, seed=7)

    assert first.best_parameters == second.best_parameters
    assert first.n_evaluations == second.n_evaluations


def test_select_better_stage_keeps_the_local_refinement_when_it_helps() -> None:
    global_x, local_x = np.array([1.0]), np.array([2.0])
    x, value, stage = select_better_stage(global_x, 5.0, local_x, 4.0)

    assert (x, value, stage) == (local_x, 4.0, "local")


def test_select_better_stage_discards_a_local_refinement_that_hurts() -> None:
    """Finite-difference noise must not be allowed to degrade the run."""
    global_x, local_x = np.array([1.0]), np.array([2.0])
    x, value, stage = select_better_stage(global_x, 4.0, local_x, 5.0)

    assert (x, value, stage) == (global_x, 4.0, "global")


def test_calibrate_reports_the_better_stage() -> None:
    """The local stage is a candidate, not an override."""
    result = calibrate(rippled_bowl, TRAP_BOUNDS, maxiter=15)

    assert result.objective_value == min(result.global_objective, result.local_objective)
    expected_stage = "local" if result.local_objective <= result.global_objective else "global"
    assert result.accepted_stage == expected_stage


def test_calibrate_substitutes_the_sentinel_for_a_non_finite_objective() -> None:
    """A NaN must never reach the optimiser's comparison."""

    def sometimes_nan(vector: np.ndarray) -> float:
        return float("nan") if vector[0] < 100.0 else rippled_bowl(vector)

    result = calibrate(sometimes_nan, TRAP_BOUNDS, maxiter=20)

    assert np.isfinite(result.objective_value)
    assert result.best_parameters[0] >= 100.0


def test_calibrate_rejects_a_collapsed_bound() -> None:
    """A zero-width bound charges a degree of freedom and fits nothing."""
    with pytest.raises(ValueError, match="strictly less than"):
        calibrate(rippled_bowl, {"gain": (5.0, 200.0), "ua": (1.7, 1.7)})


def test_calibrate_rejects_empty_bounds() -> None:
    with pytest.raises(ValueError, match="at least one parameter"):
        calibrate(rippled_bowl, {})


def test_calibrate_rejects_non_finite_bounds() -> None:
    with pytest.raises(ValueError, match="finite"):
        calibrate(rippled_bowl, {"gain": (5.0, np.inf)})


def test_calibrate_records_the_breakdown_when_asked() -> None:
    """The artifact has to explain itself, not just report a number."""

    def breakdown_fn(vector: np.ndarray) -> ObjectiveBreakdown:
        return g14_objective(MEASURED, MODEL_A, n_params=len(vector))

    result = calibrate(rippled_bowl, TRAP_BOUNDS, breakdown_fn=breakdown_fn, maxiter=10)

    assert result.breakdown is not None
    assert result.breakdown.binding_criterion == "nmbe"


# --------------------------------------------------------------------
# Pinned parameters -- a finding, not a result
# --------------------------------------------------------------------


def test_pinned_parameters_detects_both_bounds() -> None:
    names = ("at_lower", "interior", "at_upper")
    values = np.array([0.0, 5.0, 10.0])
    bounds = ((0.0, 10.0), (0.0, 10.0), (0.0, 10.0))

    assert find_pinned_parameters(names, values, bounds) == ("at_lower", "at_upper")


def test_pinned_tolerance_is_relative_to_bound_width() -> None:
    """0.05 away from a bound is pinned on a width-1 range, not on width-100."""
    narrow = find_pinned_parameters(("p",), np.array([0.05]), ((0.0, 1.0),))
    wide = find_pinned_parameters(("p",), np.array([0.05]), ((0.0, 100.0),))

    assert narrow == ()
    assert wide == ("p",)


def test_pinned_parameters_rejects_an_absurd_tolerance() -> None:
    """A tolerance of 0.5 would call every point pinned."""
    with pytest.raises(ValueError, match="tolerance"):
        find_pinned_parameters(("p",), np.array([0.5]), ((0.0, 1.0),), tolerance=0.5)


def test_calibrate_flags_a_pinned_optimum() -> None:
    """A truth outside the box must come back as pinned, not as an answer.

    This is Q7's situation reproduced on a known-answer surface: the
    objective wants `gain = 120`, the bound stops it at 60, and the run
    must SAY so rather than reporting 60 as the calibrated value.
    """

    def unreachable_optimum(vector: np.ndarray) -> float:
        offset = (np.asarray(vector, dtype=float) - TRAP_TRUTH) / TRAP_WIDTH
        return float(np.sum(offset**2))

    result = calibrate(unreachable_optimum, {"gain": (5.0, 60.0), "ua": (0.3, 3.0)}, maxiter=25)

    assert "gain" in result.pinned_parameters
    assert "ua" not in result.pinned_parameters
    assert result.best_parameters[0] > 59.0


# --------------------------------------------------------------------
# Artifact logging
# --------------------------------------------------------------------


def test_write_artifact_round_trips(tmp_path: Path) -> None:
    result = calibrate(
        rippled_bowl,
        TRAP_BOUNDS,
        maxiter=10,
        metadata={"building_id": "Fox_education_Theodore", "year": 2016},
    )
    path = write_artifact(result, tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))

    assert path.parent == tmp_path
    assert record["seed"] == result.seed
    assert record["parameters"]["gain"] == result.parameters["gain"]
    assert record["metadata"]["building_id"] == "Fox_education_Theodore"
    assert record["stages"]["global"]["objective"] == result.global_objective


def make_result(**overrides: object) -> CalibrationResult:
    """A minimal result record, for testing the record itself."""
    fields: dict[str, object] = {
        "parameter_names": ("gain",),
        "best_parameters": (1.0,),
        "bounds": ((0.0, 2.0),),
        "objective_value": 1.0,
        "global_objective": 2.0,
        "local_objective": 1.0,
        "accepted_stage": "local",
        "n_evaluations": 10,
        "global_message": "converged",
        "local_message": "converged",
        "pinned_parameters": (),
        "seed": 42,
        "elapsed_seconds": 0.5,
        "timestamp_utc": "2026-08-11T00:00:00+00:00",
    }
    fields.update(overrides)
    return CalibrationResult(**fields)  # type: ignore[arg-type]


def test_local_improvement_reports_the_fraction_de_left_on_the_table() -> None:
    assert make_result().local_improvement == pytest.approx(0.5)


def test_local_improvement_is_zero_when_the_global_stage_scored_zero() -> None:
    """Guards the division -- a perfect global fit is not a 0/0 error."""
    result = make_result(global_objective=0.0, local_objective=0.0)

    assert result.local_improvement == 0.0


def test_artifact_record_includes_the_breakdown(tmp_path: Path) -> None:
    """The binding criterion is the most useful line in the log."""

    def breakdown_fn(vector: np.ndarray) -> ObjectiveBreakdown:
        return g14_objective(MEASURED, MODEL_A, n_params=len(vector))

    result = calibrate(rippled_bowl, TRAP_BOUNDS, breakdown_fn=breakdown_fn, maxiter=10)
    record = json.loads(write_artifact(result, tmp_path).read_text(encoding="utf-8"))

    assert record["breakdown"]["binding_criterion"] == "nmbe"
    # n = 4, p = 2 -> sqrt(1920 / 2) / 250 * 100 = 12.3935%
    assert record["breakdown"]["cvrmse_pct"] == pytest.approx(12.3935, abs=1e-4)


def test_write_artifact_creates_the_directory(tmp_path: Path) -> None:
    result = calibrate(rippled_bowl, TRAP_BOUNDS, maxiter=10)
    path = write_artifact(result, tmp_path / "runs" / "nested")

    assert path.exists()


def test_write_artifact_rejects_unserialisable_metadata(tmp_path: Path) -> None:
    """Caught here rather than as a half-written file on disk."""
    result = calibrate(rippled_bowl, TRAP_BOUNDS, maxiter=10, metadata={"index": object()})

    with pytest.raises(ValueError, match="JSON-serialisable"):
        write_artifact(result, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_two_runs_do_not_overwrite_each_other(tmp_path: Path) -> None:
    """Filenames carry timestamp and seed for exactly this reason."""
    first = calibrate(rippled_bowl, TRAP_BOUNDS, maxiter=10, seed=1)
    second = calibrate(rippled_bowl, TRAP_BOUNDS, maxiter=10, seed=2)

    assert write_artifact(first, tmp_path) != write_artifact(second, tmp_path)
    assert len(list(tmp_path.iterdir())) == 2
