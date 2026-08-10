"""Tests for Morris screening.

The screening objective used here is deliberately one whose elementary
effects can be worked out on paper:

    y = 10*a + 1*b + 0*c + 8*a*d

An elementary effect is `(y(x + delta) - y(x)) / delta` for one moved
parameter, so:

    EE_a = 10 + 8d   ->  bounded in [10, 18], varies with d  -> large sigma
    EE_b = 1         ->  exactly 1 everywhere                -> sigma exactly 0
    EE_c = 0         ->  exactly 0 everywhere                -> sigma exactly 0
    EE_d = 8a        ->  bounded in [0, 8],  varies with a   -> large sigma

Those exact values for `b` and `c`, and the exact bounds for `a` and
`d`, are what the tests assert. This is the L6.2 known-answer pattern
applied to a stochastic-sampling algorithm: the sample points vary with
the seed, but the elementary effects of a known function do not.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from cooling_twin.calibration.sensitivity import (
    DEFAULT_TOP_K,
    MAX_CALIBRATED_PARAMS,
    MorrisResult,
    ParameterSpec,
    morris_screening,
    select_top_parameters,
)

SPECS = [
    ParameterSpec("a_strong", 0.0, 1.0, nominal=0.5),
    ParameterSpec("b_weak", 0.0, 1.0, nominal=0.5),
    ParameterSpec("c_inert", 0.0, 1.0, nominal=0.5),
    ParameterSpec("d_interacting", 0.0, 1.0, nominal=0.5),
]


def objective(x: npt.NDArray[np.float64]) -> float:
    """y = 10a + b + 8ad. See this module's docstring for the effects."""
    a, b, _c, d = x
    return float(10.0 * a + 1.0 * b + 8.0 * a * d)


@pytest.fixture(scope="module")
def result() -> MorrisResult:
    return morris_screening(SPECS, objective, n_trajectories=40)


def _mu_star(result: MorrisResult, name: str) -> float:
    return float(result.mu_star[result.names.index(name)])


def _sigma(result: MorrisResult, name: str) -> float:
    return float(result.sigma[result.names.index(name)])


# --------------------------------------------------------------------
# Known answers
# --------------------------------------------------------------------


def test_inert_parameter_has_exactly_zero_influence(result: MorrisResult) -> None:
    """c does not appear in the objective, so every EE_c is exactly 0."""
    assert _mu_star(result, "c_inert") == pytest.approx(0.0, abs=1e-12)
    assert _sigma(result, "c_inert") == pytest.approx(0.0, abs=1e-12)


def test_purely_linear_parameter_has_its_exact_coefficient(
    result: MorrisResult,
) -> None:
    """EE_b = 1 at every point, so mu* is exactly 1 and sigma exactly 0."""
    assert _mu_star(result, "b_weak") == pytest.approx(1.0, abs=1e-9)
    assert _sigma(result, "b_weak") == pytest.approx(0.0, abs=1e-9)


def test_influential_parameters_land_inside_their_analytic_bounds(
    result: MorrisResult,
) -> None:
    """EE_a lies in [10, 18] and EE_d in [0, 8], so mu* must too."""
    assert 10.0 <= _mu_star(result, "a_strong") <= 18.0
    assert 0.0 <= _mu_star(result, "d_interacting") <= 8.0


def test_ranking_recovers_the_known_ordering(result: MorrisResult) -> None:
    ranked = list(result.to_frame()["parameter"])

    assert ranked == ["a_strong", "d_interacting", "b_weak", "c_inert"]


def test_sigma_separates_interacting_from_additive_parameters(
    result: MorrisResult,
) -> None:
    """d only ever acts through a, so its effect is entirely conditional.

    This is what sigma is for: mu* alone would rank d as merely
    "moderately influential" and say nothing about the fact that its
    influence is zero whenever a is zero.
    """
    interacting = _sigma(result, "d_interacting") / _mu_star(result, "d_interacting")
    additive = _sigma(result, "b_weak") / _mu_star(result, "b_weak")

    assert interacting > 0.5
    assert additive == pytest.approx(0.0, abs=1e-9)


def test_cost_is_n_times_k_plus_one(result: MorrisResult) -> None:
    """Morris's defining economy: 40 trajectories, 4 parameters -> 200 runs."""
    assert result.n_model_runs == 40 * (len(SPECS) + 1)


# --------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------


def test_screening_is_reproducible_for_a_fixed_seed() -> None:
    first = morris_screening(SPECS, objective, n_trajectories=10, seed=123)
    second = morris_screening(SPECS, objective, n_trajectories=10, seed=123)

    assert first.names == second.names
    np.testing.assert_array_equal(first.mu_star, second.mu_star)
    np.testing.assert_array_equal(first.sigma, second.sigma)


def test_a_different_seed_changes_the_samples_but_not_the_conclusion() -> None:
    """The ordering is a property of the objective, not of the sample."""
    first = morris_screening(SPECS, objective, n_trajectories=20, seed=1)
    second = morris_screening(SPECS, objective, n_trajectories=20, seed=2)

    assert not np.array_equal(first.mu_star, second.mu_star)
    assert list(first.to_frame()["parameter"]) == list(second.to_frame()["parameter"])


# --------------------------------------------------------------------
# to_frame
# --------------------------------------------------------------------


def test_to_frame_is_sorted_by_influence_descending(result: MorrisResult) -> None:
    frame = result.to_frame()

    assert list(frame["mu_star"]) == sorted(frame["mu_star"], reverse=True)


def test_to_frame_reports_sigma_relative_to_mu_star(result: MorrisResult) -> None:
    """An absolute sigma cannot be compared across parameters of different scale."""
    frame = result.to_frame().set_index("parameter")

    assert frame.loc["b_weak", "sigma_over_mu_star"] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(frame.loc["c_inert", "sigma_over_mu_star"])  # mu* is 0


# --------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------


def test_select_top_parameters_keeps_the_most_influential(
    result: MorrisResult,
) -> None:
    assert select_top_parameters(result, top_k=2) == ("a_strong", "d_interacting")


def test_selection_returns_plain_strings(result: MorrisResult) -> None:
    """SALib hands back numpy str_; those compare oddly and print oddly."""
    assert all(type(name) is str for name in select_top_parameters(result, top_k=2))


def test_default_top_k_leaves_headroom_under_the_gate_cap() -> None:
    """06_ASSESSMENT.md caps calibration at 10 parameters; we select 8."""
    assert DEFAULT_TOP_K < MAX_CALIBRATED_PARAMS
    assert MAX_CALIBRATED_PARAMS == 10


def test_selection_warns_when_the_cut_is_inside_the_confidence_interval(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two parameters that overlap cannot be separated by this screening."""
    borderline = MorrisResult(
        names=("p1", "p2", "p3"),
        mu_star=np.array([10.0, 5.0, 4.9]),
        sigma=np.array([1.0, 1.0, 1.0]),
        mu_star_conf=np.array([0.5, 0.5, 0.5]),
        n_model_runs=100,
    )

    with caplog.at_level("WARNING"):
        selected = select_top_parameters(borderline, top_k=2)

    assert selected == ("p1", "p2")
    assert "not distinguishable" in caplog.text


def test_selection_is_quiet_when_the_cut_is_clean(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clean = MorrisResult(
        names=("p1", "p2", "p3"),
        mu_star=np.array([10.0, 5.0, 0.1]),
        sigma=np.array([1.0, 1.0, 1.0]),
        mu_star_conf=np.array([0.1, 0.1, 0.1]),
        n_model_runs=100,
    )

    with caplog.at_level("WARNING"):
        select_top_parameters(clean, top_k=2)

    assert caplog.text == ""


def test_selection_of_every_parameter_does_not_warn(
    result: MorrisResult, caplog: pytest.LogCaptureFixture
) -> None:
    """There is no cut to check when nothing is excluded."""
    with caplog.at_level("WARNING"):
        selected = select_top_parameters(result, top_k=len(SPECS))

    assert len(selected) == len(SPECS)
    assert caplog.text == ""


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


def test_parameter_spec_rejects_an_inverted_range() -> None:
    with pytest.raises(ValueError, match="strictly less than"):
        ParameterSpec("bad", 1.0, 0.0, nominal=0.5)


def test_parameter_spec_rejects_a_zero_width_range() -> None:
    with pytest.raises(ValueError, match="strictly less than"):
        ParameterSpec("bad", 1.0, 1.0, nominal=1.0)


def test_parameter_spec_rejects_a_nominal_outside_the_bounds() -> None:
    with pytest.raises(ValueError, match="must lie within"):
        ParameterSpec("bad", 0.0, 1.0, nominal=1.5)


def test_parameter_spec_accepts_a_nominal_on_the_boundary() -> None:
    assert ParameterSpec("edge", 0.0, 1.0, nominal=1.0).nominal == 1.0


def test_screening_rejects_fewer_than_two_parameters() -> None:
    with pytest.raises(ValueError, match="at least 2 parameters"):
        morris_screening(SPECS[:1], objective)


def test_screening_rejects_duplicate_parameter_names() -> None:
    duplicated = [*SPECS, ParameterSpec("a_strong", 0.0, 1.0, nominal=0.5)]
    with pytest.raises(ValueError, match="must be unique"):
        morris_screening(duplicated, objective)


def test_screening_rejects_too_few_trajectories() -> None:
    with pytest.raises(ValueError, match="n_trajectories must be >= 2"):
        morris_screening(SPECS, objective, n_trajectories=1)


def test_screening_rejects_too_few_levels() -> None:
    with pytest.raises(ValueError, match="num_levels must be >= 2"):
        morris_screening(SPECS, objective, num_levels=1)


def test_screening_rejects_a_non_finite_objective() -> None:
    """An infeasible parameter set must be penalised, not returned as inf.

    Morris forms differences between outputs; inf - inf is nan, which
    would silently poison an entire trajectory's effects.
    """

    def diverging(x: npt.NDArray[np.float64]) -> float:
        return float("inf") if x[0] > 0.5 else float(x[0])

    with pytest.raises(ValueError, match="non-finite output"):
        morris_screening(SPECS, diverging, n_trajectories=5)


def test_select_top_parameters_rejects_a_non_positive_k(result: MorrisResult) -> None:
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        select_top_parameters(result, top_k=0)


def test_select_top_parameters_rejects_exceeding_the_gate_cap(
    result: MorrisResult,
) -> None:
    with pytest.raises(ValueError, match="exceeds MAX_CALIBRATED_PARAMS"):
        select_top_parameters(result, top_k=MAX_CALIBRATED_PARAMS + 1)


def test_morris_result_is_immutable(result: MorrisResult) -> None:
    with pytest.raises(Exception):  # noqa: B017 -- FrozenInstanceError
        result.n_model_runs = 1  # type: ignore[misc]
