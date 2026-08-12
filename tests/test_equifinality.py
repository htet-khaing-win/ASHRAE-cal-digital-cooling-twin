"""Tests for the equifinality probe (calibration/equifinality.py).

Known-answer throughout, against the module's own demo surface:

    objective(g, m, t, r) = FLOOR + (g/400 + m/180 - 1)^2 + (t - 22)^2

Every expectation below is derivable on paper from that formula, not
read off a run:

  * The minimum is exactly FLOOR, attained on the whole LINE
    g/400 + m/180 = 1 with t = 22 and r free. So the behavioural set is
    a known geometric object, not "whatever the optimiser found".
  * g and m are affine in each other along that line, so their
    correlation across behavioural sets must be -1 to floating point.
  * t costs (t - 22)^2, so a 5% relative tolerance on a FLOOR of 0.2
    admits |t - 22| <= 0.1 K exactly -- a span of at most 0.2 K in a
    6 K box, i.e. 3.3%, comfortably "identified".
  * r appears nowhere in the objective, so a bounded L-BFGS-B leaves it
    exactly where it started. Its span across Latin-hypercube starts is
    therefore large, and its correlation with everything else is zero
    up to sampling.

The distinction those last two points encode is the one the module
exists to make: a WIDE spread means "not measured", and only a wide
spread WITH a strong correlation means "traded off against something
else".
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from cooling_twin.calibration.equifinality import (
    DEMO_BOUNDS,
    DEMO_FLOOR,
    DEMO_GAIN_SCALE,
    DEMO_VENT_SCALE,
    IDENTIFIED_SPAN_FRACTION,
    MIN_SETS_FOR_SPREAD,
    UNIDENTIFIED_SPAN_FRACTION,
    BehaviouralSet,
    EquifinalityStudy,
    _demo_objective,
    _demo_retrofit_saving_pct,
    collect_behavioural_sets,
    compensating_pairs,
    most_divergent_pair,
    outcome_spread,
    parameter_correlations,
    parameter_spread,
)

TOLERANCE = 0.05
N_STARTS = 12

# A point exactly on the ridge: 200/400 + 90/180 = 1, t = 22.
ON_RIDGE = np.array([200.0, 90.0, 22.0, 11.0])


@pytest.fixture(scope="module")
def study() -> EquifinalityStudy:
    """One shared study -- the surface is deterministic, so it is reusable."""
    return collect_behavioural_sets(
        _demo_objective,
        DEMO_BOUNDS,
        ON_RIDGE,
        tolerance=TOLERANCE,
        n_starts=N_STARTS,
    )


def _study_from(vectors: list[list[float]], objectives: list[float]) -> EquifinalityStudy:
    """Build a study directly, for testing the analysis functions alone.

    The collection step is expensive and already tested; constructing
    the dataclass keeps a failure in `parameter_spread` from being
    reported as a failure of the optimiser.
    """
    sets = tuple(
        BehaviouralSet(
            parameters=tuple(vector),
            objective=value,
            start=tuple(vector),
            origin=f"restart-{index}",
            message="synthetic",
        )
        for index, (vector, value) in enumerate(zip(vectors, objectives, strict=True))
    )
    return EquifinalityStudy(
        parameter_names=tuple(DEMO_BOUNDS),
        bounds=tuple(DEMO_BOUNDS.values()),
        sets=sets,
        rejected=(),
        reference_objective=objectives[0],
        best_objective=min(objectives),
        tolerance=TOLERANCE,
        threshold=min(objectives) * (1.0 + TOLERANCE),
        seed=0,
        n_evaluations=0,
        elapsed_seconds=0.0,
    )


# --- collection -----------------------------------------------------------


def test_reference_point_is_included_and_refined(study: EquifinalityStudy) -> None:
    """The calibrated vector under test is itself one of the sets."""
    reference = study.reference_set
    assert reference is not None
    assert reference.origin == "reference"
    # It started exactly on the ridge, so refinement cannot improve it.
    assert reference.objective == pytest.approx(DEMO_FLOOR, abs=1e-9)


def test_every_behavioural_set_is_within_the_threshold(study: EquifinalityStudy) -> None:
    """Membership is the definition, so check it rather than assume it."""
    assert study.sets
    assert study.threshold == pytest.approx(study.best_objective * (1.0 + TOLERANCE))
    assert all(entry.objective <= study.threshold for entry in study.sets)
    assert all(entry.objective > study.threshold for entry in study.rejected)


def test_the_ridge_is_recovered(study: EquifinalityStudy) -> None:
    """Every behavioural set lies on the known minimising LINE.

    Hand-derived: within tolerance, `(g/400 + m/180 - 1)^2` cannot
    exceed `FLOOR * TOLERANCE`, so the scale error is bounded by its
    square root -- 0.1 for FLOOR 0.2 at 5%.
    """
    matrix = study.matrix
    scale_error = matrix[:, 0] / DEMO_GAIN_SCALE + matrix[:, 1] / DEMO_VENT_SCALE - 1.0
    assert np.all(np.abs(scale_error) <= np.sqrt(DEMO_FLOOR * TOLERANCE) + 1e-9)


def test_multiple_starts_find_genuinely_different_parameter_sets(
    study: EquifinalityStudy,
) -> None:
    """The point of the exercise: the fit does not pin the parameters.

    A study whose sets all coincide would be reassuring and would also
    mean the probe found nothing -- so this asserts the disagreement is
    large, not merely non-zero.
    """
    spreads = parameter_spread(study)
    assert spreads["internal_gain_w_per_m2"].span_fraction > 0.5
    assert spreads["vent_flow_kg_per_s"].span_fraction > 0.5


def test_identified_parameter_is_pinned_by_the_data(study: EquifinalityStudy) -> None:
    """`t_setpoint_c` costs the whole budget for a 1 K error.

    |t - 22| <= sqrt(FLOOR * TOLERANCE) = 0.1 K, so the span is at most
    0.2 K of a 6 K box = 3.3%, under the 10% "identified" cut.
    """
    spread = parameter_spread(study)["t_setpoint_c"]
    assert spread.span <= 2.0 * np.sqrt(DEMO_FLOOR * TOLERANCE) + 1e-9
    assert spread.span_fraction <= IDENTIFIED_SPAN_FRACTION
    assert spread.verdict == "identified"


def test_inert_parameter_is_reported_unidentified(study: EquifinalityStudy) -> None:
    """`r_internal_ratio` is absent from the objective, so it never moves."""
    spread = parameter_spread(study)["r_internal_ratio"]
    assert spread.span_fraction >= UNIDENTIFIED_SPAN_FRACTION
    assert spread.verdict == "unidentified"


def test_the_study_is_reproducible_under_a_fixed_seed() -> None:
    """Same seed, same surface, same behavioural sets (L0.3)."""
    first = collect_behavioural_sets(
        _demo_objective, DEMO_BOUNDS, ON_RIDGE, n_starts=4, seed=7
    )
    second = collect_behavioural_sets(
        _demo_objective, DEMO_BOUNDS, ON_RIDGE, n_starts=4, seed=7
    )
    assert first.matrix == pytest.approx(second.matrix)


def test_parallel_and_serial_agree() -> None:
    """Restarts are independent, so a process pool must not change the answer."""
    serial = collect_behavioural_sets(
        _demo_objective, DEMO_BOUNDS, ON_RIDGE, n_starts=4, seed=7, workers=1
    )
    parallel = collect_behavioural_sets(
        _demo_objective, DEMO_BOUNDS, ON_RIDGE, n_starts=4, seed=7, workers=2
    )
    assert serial.matrix == pytest.approx(parallel.matrix)


def test_zero_starts_gives_the_reference_alone() -> None:
    """The degenerate case still returns a valid study, with a warning path."""
    lonely = collect_behavioural_sets(
        _demo_objective, DEMO_BOUNDS, ON_RIDGE, n_starts=0
    )
    assert len(lonely.sets) == 1
    assert lonely.sets[0].origin == "reference"


def test_a_restart_beating_the_reference_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A calibration that is not the best fit in its own box is a finding.

    The reference here sits off the ridge (200/400 + 45/180 = 0.75),
    which every restart beats.
    """
    off_ridge = np.array([200.0, 45.0, 22.0, 11.0])
    with caplog.at_level(logging.WARNING):
        collect_behavioural_sets(_demo_objective, DEMO_BOUNDS, off_ridge, n_starts=4)
    assert "BEAT the calibrated optimum" in caplog.text


# --- input validation -----------------------------------------------------


def test_empty_bounds_rejected() -> None:
    with pytest.raises(ValueError, match="at least one parameter"):
        collect_behavioural_sets(_demo_objective, {}, [])


def test_inverted_bounds_rejected() -> None:
    with pytest.raises(ValueError, match="strictly less than"):
        collect_behavioural_sets(_demo_objective, {"a": (3.0, 1.0)}, [2.0])


def test_non_finite_bounds_rejected() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        collect_behavioural_sets(_demo_objective, {"a": (0.0, np.inf)}, [2.0])


def test_zero_tolerance_rejected() -> None:
    """Exact ties do not occur in floating point -- the study would look clean."""
    with pytest.raises(ValueError, match="tolerance must be > 0"):
        collect_behavioural_sets(
            _demo_objective, DEMO_BOUNDS, ON_RIDGE, tolerance=0.0, n_starts=2
        )


def test_negative_n_starts_rejected() -> None:
    with pytest.raises(ValueError, match="n_starts must be >= 0"):
        collect_behavioural_sets(
            _demo_objective, DEMO_BOUNDS, ON_RIDGE, n_starts=-1
        )


def test_wrong_length_reference_rejected() -> None:
    with pytest.raises(ValueError, match="describes 4 parameters"):
        collect_behavioural_sets(_demo_objective, DEMO_BOUNDS, [200.0, 90.0])


def test_reference_outside_the_bounds_rejected() -> None:
    """A study run in a different box measures a different calibration."""
    with pytest.raises(ValueError, match="outside the bounds"):
        collect_behavioural_sets(
            _demo_objective, DEMO_BOUNDS, [900.0, 90.0, 22.0, 11.0], n_starts=2
        )


def test_non_finite_objective_at_the_reference_rejected() -> None:
    with pytest.raises(ValueError, match="not finite at the reference"):
        collect_behavioural_sets(
            lambda _vector: float("nan"), DEMO_BOUNDS, ON_RIDGE, n_starts=0
        )


# --- spread and correlation ----------------------------------------------


def test_spread_fraction_is_relative_to_bound_width() -> None:
    """Known answer: half of each box, whatever the units."""
    lower = np.array([low for low, _ in DEMO_BOUNDS.values()])
    upper = np.array([high for _, high in DEMO_BOUNDS.values()])
    midpoint = lower + 0.5 * (upper - lower)
    synthetic = _study_from([list(lower), list(midpoint)], [1.0, 1.0])
    for spread in parameter_spread(synthetic).values():
        assert spread.span_fraction == pytest.approx(0.5)


def test_spread_warns_when_there_are_too_few_sets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One point has zero span, which reads as perfect identifiability."""
    synthetic = _study_from([[200.0, 90.0, 22.0, 11.0]], [1.0])
    with caplog.at_level(logging.WARNING):
        spreads = parameter_spread(synthetic)
    assert spreads["internal_gain_w_per_m2"].verdict == "identified"
    assert "not evidence of identifiability" in caplog.text


def test_middle_of_the_range_is_weakly_identified() -> None:
    """The third verdict: a 30% span is neither pinned nor free.

    Constructed exactly on 30% of each box so the branch is chosen by
    arithmetic, not by whatever a run happened to produce.
    """
    lower = np.array([low for low, _ in DEMO_BOUNDS.values()])
    upper = np.array([high for _, high in DEMO_BOUNDS.values()])
    synthetic = _study_from(
        [list(lower), list(lower + 0.3 * (upper - lower)), list(lower)], [1.0, 1.0, 1.0]
    )
    for spread in parameter_spread(synthetic).values():
        assert spread.span_fraction == pytest.approx(0.3)
        assert spread.verdict == "weakly identified"


def test_reference_is_absent_when_the_calibrated_point_is_not_behavioural() -> None:
    """A study can legitimately exclude the reported answer -- say so, not crash."""
    synthetic = _study_from(
        [
            [100.0, 90.0, 22.0, 11.0],
            [200.0, 80.0, 22.0, 11.0],
            [300.0, 70.0, 22.0, 11.0],
        ],
        [1.0, 1.0, 1.0],
    )
    assert synthetic.reference_set is None
    saving = outcome_spread(synthetic, _demo_retrofit_saving_pct)
    assert saving.reference is None
    assert "calibrated n/a" in saving.summary("%")


def test_outcome_summary_reports_the_range_not_a_point(
    study: EquifinalityStudy,
) -> None:
    """A single number is exactly the claim this module refuses to make."""
    text = outcome_spread(study, _demo_retrofit_saving_pct).summary("%")
    assert "behavioural sets" in text
    assert "spread" in text
    assert " %" in text


def test_compensating_pair_is_found_with_correlation_minus_one(
    study: EquifinalityStudy,
) -> None:
    """Gain and ventilation are affine along the ridge, so r -> -1.

    Not exactly -1: each refinement stops at L-BFGS-B's own convergence
    tolerance, a short distance off the line, and the behavioural
    threshold admits a band around it rather than the line itself. The
    residual departure is order 1e-4, which is the right size for
    solver tolerance and far too small to be a real second direction.
    """
    pairs = compensating_pairs(study)
    assert len(pairs) == 1
    name_a, name_b, correlation = pairs[0]
    assert {name_a, name_b} == {"internal_gain_w_per_m2", "vent_flow_kg_per_s"}
    assert correlation == pytest.approx(-1.0, abs=1e-3)


def test_inert_parameter_is_not_reported_as_compensating(
    study: EquifinalityStudy,
) -> None:
    """A wide spread alone is not a trade-off -- that separation is the point."""
    names = study.parameter_names
    correlations = parameter_correlations(study)
    inert = names.index("r_internal_ratio")
    gain = names.index("internal_gain_w_per_m2")
    assert abs(correlations[inert, gain]) < 0.8


def test_correlation_with_a_constant_parameter_is_nan() -> None:
    """Undefined, not zero: zero would read as "independent"."""
    synthetic = _study_from(
        [
            [100.0, 90.0, 22.0, 11.0],
            [200.0, 80.0, 22.0, 11.0],
            [300.0, 70.0, 22.0, 11.0],
        ],
        [1.0, 1.0, 1.0],
    )
    correlations = parameter_correlations(synthetic)
    setpoint = synthetic.parameter_names.index("t_setpoint_c")
    assert np.all(np.isnan(correlations[setpoint, :]))
    assert not np.isnan(correlations[0, 1])


def test_correlation_over_two_points_is_refused() -> None:
    """Any two points correlate perfectly -- that would invent the finding."""
    synthetic = _study_from(
        [[100.0, 90.0, 22.0, 11.0], [200.0, 80.0, 22.0, 11.0]], [1.0, 1.0]
    )
    with pytest.raises(ValueError, match=f"at least {MIN_SETS_FOR_SPREAD}"):
        parameter_correlations(synthetic)


@pytest.mark.parametrize("threshold", [0.0, 1.5, -0.2])
def test_compensation_threshold_out_of_range_rejected(
    study: EquifinalityStudy, threshold: float
) -> None:
    with pytest.raises(ValueError, match=r"threshold must be in \(0, 1\]"):
        compensating_pairs(study, threshold=threshold)


# --- the exhibit and the consequence --------------------------------------


def test_most_divergent_pair_brackets_the_behavioural_range(
    study: EquifinalityStudy,
) -> None:
    low, high = most_divergent_pair(study, "internal_gain_w_per_m2")
    spread = parameter_spread(study)["internal_gain_w_per_m2"]
    assert low.parameters[0] == pytest.approx(spread.minimum)
    assert high.parameters[0] == pytest.approx(spread.maximum)
    # Both are admissible: that is what makes the pair an exhibit.
    assert low.objective <= study.threshold
    assert high.objective <= study.threshold


def test_most_divergent_pair_rejects_an_unknown_parameter(
    study: EquifinalityStudy,
) -> None:
    with pytest.raises(KeyError, match="not a calibrated parameter"):
        most_divergent_pair(study, "chiller_cop")


def test_most_divergent_pair_needs_two_sets() -> None:
    synthetic = _study_from([[200.0, 90.0, 22.0, 11.0]], [1.0])
    with pytest.raises(ValueError, match="at least 2 behavioural sets"):
        most_divergent_pair(synthetic, "t_setpoint_c")


def test_outcome_spread_is_the_known_retrofit_range(study: EquifinalityStudy) -> None:
    """Hand-derived: on the ridge the saving is exactly 30% * vent_share.

    So the predicted saving from one identical retrofit is bounded by
    the behavioural range of `vent_flow_kg_per_s` and nothing else --
    and that range is most of the box.
    """
    saving = outcome_spread(study, _demo_retrofit_saving_pct)
    vent = parameter_spread(study)["vent_flow_kg_per_s"]
    assert saving.minimum == pytest.approx(
        30.0 * vent.minimum / DEMO_VENT_SCALE, rel=0.02
    )
    assert saving.maximum == pytest.approx(
        30.0 * vent.maximum / DEMO_VENT_SCALE, rel=0.02
    )
    # The reference set is on the ridge at vent_share = 0.5.
    assert saving.reference == pytest.approx(15.0, abs=1e-6)
    assert saving.spread > 20.0


def test_outcome_spread_refuses_a_non_finite_outcome(study: EquifinalityStudy) -> None:
    """Dropping the bad set would NARROW the spread -- the one unsafe direction."""
    with pytest.raises(ValueError, match="not finite at behavioural set"):
        outcome_spread(study, lambda _vector: float("inf"))


def test_outcome_spread_needs_a_behavioural_set() -> None:
    empty = _study_from([[200.0, 90.0, 22.0, 11.0]], [1.0])
    empty = EquifinalityStudy(**{**empty.__dict__, "sets": ()})
    with pytest.raises(ValueError, match="no behavioural sets"):
        outcome_spread(empty, _demo_retrofit_saving_pct)


def test_to_dict_is_json_serialisable(study: EquifinalityStudy) -> None:
    """The study travels in the run log, so it must survive `json.dumps`."""
    import json

    payload = json.loads(json.dumps(study.to_dict()))
    assert payload["n_behavioural"] == len(study.sets)
    assert payload["spread"]["t_setpoint_c"]["verdict"] == "identified"
